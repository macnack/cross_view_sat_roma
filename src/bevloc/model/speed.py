"""H100 throughput / precision switches for the frozen-encoder training (scripts/train_vigor.py --encoder-dtype /
--decoder-dtype / --compile).

Wrappers only: nothing in third_party or the sat-roma-infer package is edited, and no module is replaced, so every
state dict keeps its keys (the checkpoints stay loadable by every evaluator).

- `set_encoder_dtype(encoder, "bfloat16")`: the frozen DINOv3 encoder's forward (always under no_grad in this repo)
  runs under torch.autocast(bfloat16) and its outputs are cast back to float32, so the decoder (with its own float16
  autocast) sees float32 inputs exactly as before. bfloat16 also lets scaled_dot_product_attention use the flash
  kernel (float32 has only the memory-efficient / math kernels). "float32" (the default) leaves the forward alone.
- `set_decoder_dtype(decoder, "bfloat16")`: the Sat-RoMa decoder's own autocast dtype. The package opens
  `torch.autocast(device, dtype=self.amp_dtype)` in `Decoder.forward` (the proj convs), `TransformerDecoder.forward`
  (the coarse classifier) and `ConvRefiner.forward` (the stride-16 refiner), with `sat_roma.utils.get_autocast_params`
  forcing it ON on any CUDA device (OFF on CPU) whatever `amp` says; `amp_dtype` is float16 by default
  (`sat_roma.build.build_sat_roma`). The attribute is read at every forward, so setting it on those modules is the
  package's own hook (no forward is replaced; bevloc.model.refine.RefinerTap does the same for the refiner). An outer
  autocast of ours could not take precedence: the package's inner context re-enables its own dtype. Training has no
  GradScaler, so under float16 the backward flushes the small per-logit gradients ((p - y) / ~37k tokens ~ 1e-8)
  to zero; bfloat16 has float32's exponent range (no underflow) at 8 mantissa bits. "float32" = autocast with
  float32 target (every op in float32; the reference of the gradient check). "float16" = the released behaviour and
  the default: the modules are left untouched. A RefinerTap with its own precision (a refiner trained with the fine
  loss: float32) still wins for the refiner while it is active and restores this dtype on exit.
- `compile_modules(matcher, which)`: nn.Module.compile() in place (the `_orig_mod.` prefix of torch.compile's
  wrapper never reaches a state dict) on the encoder, the decoder or both; dynamic=False (the shapes are fixed:
  reference 896 px, panorama grid, per-rank batch).
"""
from __future__ import annotations

import torch

ENCODER_DTYPES = {"float32": None, "bfloat16": torch.bfloat16, "float16": torch.float16}
DECODER_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
DECODER_DTYPE_DEFAULT = "float16"               # the package's amp_dtype (the released configuration, every run so far)
COMPILE_CHOICES = ("none", "encoder", "decoder", "both")


def _to_float32(x):
    if torch.is_tensor(x):
        return x.float() if x.is_floating_point() and x.dtype != torch.float32 else x
    if isinstance(x, dict):
        return {k: _to_float32(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_float32(v) for v in x)
    return x


def set_encoder_dtype(encoder: torch.nn.Module, dtype: str = "float32"):
    """Run `encoder` under autocast(dtype) with float32 outputs; "float32" switches it back off. Installs the
    wrapper on the instance once; the active dtype is `encoder._bevloc_amp_dtype` (so a check can toggle it)."""
    if dtype not in ENCODER_DTYPES:
        raise ValueError(f"encoder dtype {dtype!r} not in {sorted(ENCODER_DTYPES)}")
    if not hasattr(encoder, "_bevloc_orig_forward"):
        orig = encoder.forward

        def forward(*args, **kw):
            amp = ENCODER_DTYPES[encoder._bevloc_amp_dtype]
            if amp is None:
                return orig(*args, **kw)
            dev = next((a.device.type for a in args if torch.is_tensor(a)), "cuda")
            with torch.autocast(dev, dtype=amp):
                out = orig(*args, **kw)
            return _to_float32(out)
        encoder._bevloc_orig_forward = orig
        encoder.forward = forward
    encoder._bevloc_amp_dtype = dtype
    return encoder


PRECISION_DEFAULTS = {"encoder_dtype": "float32", "compile": "none", "decoder_dtype": DECODER_DTYPE_DEFAULT}


def precision_record(encoder_dtype="float32", compile="none", decoder_dtype=DECODER_DTYPE_DEFAULT) -> dict:
    """The precision keys train_vigor.py writes into the checkpoints' train dict, the config snapshot and the resume
    counters: decoder_dtype always (older files carry none = float16); encoder_dtype + compile only when either is
    not the default (single-GPU default files keep their earlier keys)."""
    out = dict(decoder_dtype=decoder_dtype)
    if encoder_dtype != "float32" or compile != "none":
        out.update(encoder_dtype=encoder_dtype, compile=compile)
    return out


def precision_mismatch(recorded: dict, now: dict) -> list:
    """Warnings for a resume whose precision differs from the file's (a missing key = the default of its time)."""
    return [f"resume file trained with {k}={recorded.get(k, d)}, this segment {now.get(k, d)}"
            for k, d in PRECISION_DEFAULTS.items() if recorded.get(k, d) != now.get(k, d)]


def decoder_amp_modules(decoder: torch.nn.Module):
    """(name, module) of every decoder submodule whose forward reads `amp_dtype` (Decoder, TransformerDecoder,
    ConvRefiner in the released build)."""
    return [(n, m) for n, m in decoder.named_modules() if hasattr(m, "amp_dtype")]


def set_decoder_dtype(decoder: torch.nn.Module, dtype: str = DECODER_DTYPE_DEFAULT):
    """Set the autocast dtype the Sat-RoMa decoder's forwards open on CUDA (see the module doc). Returns the names of
    the modules set. "float16" restores the package default. Raises when the decoder has no such module (a package
    change would otherwise make the flag a silent no-op)."""
    if dtype not in DECODER_DTYPES:
        raise ValueError(f"decoder dtype {dtype!r} not in {sorted(DECODER_DTYPES)}")
    mods = decoder_amp_modules(decoder)
    if not mods:
        raise RuntimeError(f"{type(decoder).__name__} has no submodule with an `amp_dtype` attribute: the Sat-RoMa "
                           "package no longer reads it, --decoder-dtype would do nothing")
    for _, m in mods:
        m.amp_dtype = DECODER_DTYPES[dtype]
    decoder._bevloc_decoder_dtype = dtype
    return [n or "<decoder>" for n, _ in mods]


def decoder_dtype_of(decoder: torch.nn.Module) -> str:
    """The decoder's autocast dtype name; raises when its modules disagree (e.g. a RefinerTap is active)."""
    names = {n: m.amp_dtype for n, m in decoder_amp_modules(decoder)}
    got = set(names.values())
    if len(got) != 1:
        raise RuntimeError(f"decoder modules have different autocast dtypes: {names}")
    inv = {v: k for k, v in DECODER_DTYPES.items()}
    return inv[got.pop()]


def compile_modules(matcher, which: str = "none"):
    """In-place torch.compile (dynamic=False) of matcher.model.encoder and / or matcher.model.decoder."""
    if which not in COMPILE_CHOICES:
        raise ValueError(f"--compile {which!r} not in {COMPILE_CHOICES}")
    if which != "none":
        # dynamic=False specialises every input shape: training (per-rank batch; reference + panorama = 2 graphs),
        # validation (--val-batch and the last partial batch: 4 more) = 6 of dynamo's default 8 per function. Room
        # for a few more, and an error instead of the silent fall-back to eager when the limit is still hit
        # (a compile error itself already raises: suppress_errors is off by default)
        cfg = torch._dynamo.config
        for lim, fail in (("recompile_limit", "fail_on_recompile_limit_hit"),
                          ("cache_size_limit", "fail_on_cache_limit_hit")):
            if hasattr(cfg, lim):
                setattr(cfg, lim, max(int(getattr(cfg, lim)), 16))
                if hasattr(cfg, fail):
                    setattr(cfg, fail, True)
                break
    done = []
    if which in ("encoder", "both"):
        matcher.model.encoder.compile(dynamic=False)
        done.append("encoder")
    if which in ("decoder", "both"):
        matcher.model.decoder.compile(dynamic=False)
        done.append("decoder")
    return done


def attention_report(module: torch.nn.Module) -> dict:
    """Which attention path the timm / Sat-RoMa blocks take: counts of modules exposing timm's `fused_attn` flag
    (True = torch.nn.functional.scaled_dot_product_attention) by class name, and the SDPA backends enabled."""
    out = {}
    for m in module.modules():
        if hasattr(m, "fused_attn"):
            k = f"{type(m).__name__}.fused_attn={bool(m.fused_attn)}"
            out[k] = out.get(k, 0) + 1
    try:
        out["sdpa_flash_enabled"] = bool(torch.backends.cuda.flash_sdp_enabled())
        out["sdpa_mem_efficient_enabled"] = bool(torch.backends.cuda.mem_efficient_sdp_enabled())
    except AttributeError:
        pass
    return out


def encoder_diff(encoder, x):
    """float32 vs the encoder's current autocast dtype on one input: max / mean absolute difference of the output
    tokens (scale 16), the float32 tokens' mean |value| and the mean per-token cosine similarity."""
    cur = encoder._bevloc_amp_dtype
    with torch.no_grad():
        encoder._bevloc_amp_dtype = "float32"
        a = encoder(x)[16].float()
        encoder._bevloc_amp_dtype = cur
        b = encoder(x)[16].float()
    d = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()), ref_mean_abs=float(a.abs().mean()),
                rel_mean=float(d.mean() / a.abs().mean().clamp_min(1e-12)), cos_mean=float(cos.mean()),
                cos_min=float(cos.min()), shape=list(a.shape))
