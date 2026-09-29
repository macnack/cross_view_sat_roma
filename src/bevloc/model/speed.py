"""H100 throughput switches for the frozen-encoder training (scripts/train_vigor.py --encoder-dtype / --compile).

Wrappers only: nothing in third_party or the sat-roma-infer package is edited, and no module is replaced, so every
state dict keeps its keys (the checkpoints stay loadable by every evaluator).

- `set_encoder_dtype(encoder, "bfloat16")`: the frozen DINOv3 encoder's forward (always under no_grad in this repo)
  runs under torch.autocast(bfloat16) and its outputs are cast back to float32, so the decoder (with its own float16
  autocast) sees float32 inputs exactly as before. bfloat16 also lets scaled_dot_product_attention use the flash
  kernel (float32 has only the memory-efficient / math kernels). "float32" (the default) leaves the forward alone.
- `compile_modules(matcher, which)`: nn.Module.compile() in place (the `_orig_mod.` prefix of torch.compile's
  wrapper never reaches a state dict) on the encoder, the decoder or both; dynamic=False (the shapes are fixed:
  reference 896 px, panorama grid, per-rank batch).
"""
from __future__ import annotations

import torch

ENCODER_DTYPES = {"float32": None, "bfloat16": torch.bfloat16, "float16": torch.float16}
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


def compile_modules(matcher, which: str = "none"):
    """In-place torch.compile (dynamic=False) of matcher.model.encoder and / or matcher.model.decoder."""
    if which not in COMPILE_CHOICES:
        raise ValueError(f"--compile {which!r} not in {COMPILE_CHOICES}")
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
