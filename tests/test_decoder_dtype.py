"""--decoder-dtype (bevloc.model.speed.set_decoder_dtype): the Sat-RoMa decoder's autocast dtype, its default, the
records in the train dict / snapshot / resume counters, and the flags of the trainer and the evaluators.

CPU: the attribute hook and the package's dtype choice (sat_roma.utils.get_autocast_params) for a CUDA device. The
package disables its autocast on CPU, so the dtype of a real decoder intermediate is checked on CUDA only (skipped
without a GPU)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from bevloc.model.refine import RefinerTap
from bevloc.model.speed import (
    DECODER_DTYPE_DEFAULT, DECODER_DTYPES, decoder_amp_modules, decoder_dtype_of, precision_mismatch,
    precision_record, set_decoder_dtype,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_satroma import C_ENC, tiny_decoder  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _amp_dtypes(dec):
    return {n: m.amp_dtype for n, m in decoder_amp_modules(dec)}


def test_default_is_the_packages_float16_and_every_autocast_module_is_found():
    dec = tiny_decoder()
    d = _amp_dtypes(dec)
    # Decoder (proj), TransformerDecoder (classifier), ConvRefiner (stride-16 refiner): the three package forwards
    # that open torch.autocast(dtype=self.amp_dtype)
    assert set(d) == {"", "embedding_decoder", "conv_refiner.16"}
    assert set(d.values()) == {torch.float16} and DECODER_DTYPE_DEFAULT == "float16"
    assert decoder_dtype_of(dec) == "float16"


@pytest.mark.parametrize("name", ["bfloat16", "float32", "float16"])
def test_set_decoder_dtype_sets_every_module_and_the_package_uses_it_on_cuda(name):
    from sat_roma.utils import get_autocast_params
    dec = tiny_decoder()
    set_decoder_dtype(dec, name)
    assert set(_amp_dtypes(dec).values()) == {DECODER_DTYPES[name]}
    assert decoder_dtype_of(dec) == name and dec._bevloc_decoder_dtype == name
    for _, m in decoder_amp_modules(dec):
        # the package's own choice for a CUDA tensor (pure function: no GPU needed): autocast ON with our dtype
        dev, enabled, dt = get_autocast_params(torch.device("cuda:0"), enabled=getattr(m, "amp", False),
                                               dtype=m.amp_dtype)
        assert (dev, enabled, dt) == ("cuda", True, DECODER_DTYPES[name])
    set_decoder_dtype(dec, "float16")                  # reversible: back to the package default
    assert set(_amp_dtypes(dec).values()) == {torch.float16}


def test_bad_dtype_and_a_decoder_without_the_hook_raise():
    with pytest.raises(ValueError):
        set_decoder_dtype(tiny_decoder(), "int8")
    with pytest.raises(RuntimeError, match="amp_dtype"):
        set_decoder_dtype(torch.nn.Linear(2, 2), "bfloat16")


def test_a_refiner_tap_precision_wins_and_restores_the_decoder_dtype():
    dec = tiny_decoder()
    set_decoder_dtype(dec, "bfloat16")
    with RefinerTap(dec, precision="float32"):
        assert dec.conv_refiner["16"].amp_dtype == torch.float32
        assert dec.embedding_decoder.amp_dtype == torch.bfloat16
    assert decoder_dtype_of(dec) == "bfloat16"


def test_precision_record_and_resume_mismatch():
    assert precision_record() == dict(decoder_dtype="float16")          # default single-GPU runs: one new key only
    r = precision_record("bfloat16", "encoder", "bfloat16")
    assert r == dict(decoder_dtype="bfloat16", encoder_dtype="bfloat16", compile="encoder")
    # a file written before the flag (no decoder_dtype) counts as float16
    assert precision_mismatch({}, dict(decoder_dtype="float16", encoder_dtype="float32", compile="none")) == []
    w = precision_mismatch(dict(encoder_dtype="bfloat16", compile="encoder"), r)
    assert len(w) == 1 and "decoder_dtype=float16" in w[0] and "bfloat16" in w[0]
    assert precision_mismatch(r, r) == []


def test_trainer_and_evaluators_expose_the_flag_with_the_float16_default():
    src = (REPO / "scripts" / "train_vigor.py").read_text()
    # recorded in the checkpoints' train dict, the config snapshot and the resume counters (one record for all three)
    assert "train_meta.update(prec)" in src and "tf32=a.tf32, **prec)" in src and "val=v, **prec)" in src
    for script in ("train_vigor.py", "eval_vigor.py", "eval_panoroma_poznan.py"):
        out = subprocess.run([sys.executable, str(REPO / "scripts" / script), "--help"], capture_output=True,
                             text=True, cwd=REPO, timeout=300)
        assert out.returncode == 0, out.stderr[-2000:]
        assert "--decoder-dtype" in out.stdout, script


def _decode(dec, device):
    g = torch.Generator().manual_seed(3)
    f_q = torch.randn(1, C_ENC, 14, 14, generator=g).to(device)
    f_r = torch.randn(1, C_ENC, 56, 56, generator=g).to(device)
    dec.expose_intermediates = True
    seen = {}
    hooks = [dec.embedding_decoder.to_out.register_forward_hook(lambda m, i, o: seen.__setitem__("to_out", o.dtype)),
             dec.proj["16"][0].register_forward_hook(lambda m, i, o: seen.__setitem__("proj", o.dtype))]
    try:
        out = dec({16: f_q}, {16: f_r}, scale_factor=0.4)
    finally:
        for h in hooks:
            h.remove()
    return seen, out[16]["gm_cls"]


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="the package enables its decoder autocast on CUDA only (get_autocast_params disables it on "
                           "CPU), so the dtype of a decoder intermediate can only be observed on a GPU")
@pytest.mark.parametrize("name", ["float16", "bfloat16", "float32"])
def test_decoder_intermediates_run_in_the_chosen_dtype_on_cuda(name):
    dec = tiny_decoder().cuda()
    if name != "float16":
        set_decoder_dtype(dec, name)
    seen, gm = _decode(dec, "cuda")
    assert seen == {"to_out": DECODER_DTYPES[name], "proj": DECODER_DTYPES[name]}
    assert gm.dtype == DECODER_DTYPES[name]
