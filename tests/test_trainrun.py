"""Run control of the long VIGOR training (bevloc.model.trainrun, scripts/train_vigor.py --epochs/--resume/--profile)."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bevloc import config as C
from bevloc.model import trainrun as R
from bevloc.model.depth_query import ErpDepthQuery
from bevloc.model.refine import decoder_state

from tiny_satroma import tiny_decoder  # noqa: E402


# ---- epochs -> steps, names --------------------------------------------------------------------------------------

def test_epoch_steps_use_drop_last():
    assert R.steps_per_epoch(42087, 32) == 1315                    # 42087 // 32, the last 7 samples dropped
    assert R.epoch_steps(42087, 32, 100) == 131500
    assert R.epoch_steps(10, 4, 3) == 6
    with pytest.raises(ValueError):
        R.steps_per_epoch(3, 4)


def test_periodic_save_names_and_epochs():
    assert R.epoch_ckpt_path("checkpoints", "t", 10).name == "vigor_t_ep010.pt"
    assert R.epoch_ckpt_path("checkpoints", "t", 100).name == "vigor_t_ep100.pt"
    assert R.resume_path("checkpoints", "t").name == "vigor_t_resume.pt"
    spe = 7
    saved = [R.is_epoch_save(k, spe, 10) for k in range(1, 100 * spe + 1)]
    assert [e for e in saved if e] == list(range(10, 101, 10))
    assert R.is_epoch_save(10 * spe, spe, 0) == 0                  # K = 0: off
    assert R.is_epoch_save(10 * spe - 1, spe, 10) == 0             # mid-epoch


# ---- sampler ---------------------------------------------------------------------------------------------------

class _Idx(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def test_epoch_sampler_is_seeded_per_epoch_and_skip_resumes_mid_epoch():
    s = R.EpochSampler(50, seed=3)
    s.set_epoch(0)
    a0 = list(s)
    s.set_epoch(0)
    assert list(s) == a0 and sorted(a0) == list(range(50))
    s.set_epoch(1)
    assert list(s) != a0
    s.set_epoch(0, skip=20)
    assert list(s) == a0[20:] and len(s) == 30
    # through a DataLoader with drop_last: the resumed epoch yields exactly the remaining full batches
    dl = DataLoader(_Idx(50), batch_size=8, sampler=s, drop_last=True)
    s.set_epoch(0)
    full = [b.tolist() for b in dl]
    assert len(full) == R.steps_per_epoch(50, 8) == 6
    s.set_epoch(0, skip=2 * 8)
    assert [b.tolist() for b in dl] == full[2:]


# ---- checkpoint contents ---------------------------------------------------------------------------------------

def test_erp_depth_checkpoint_has_head_and_decoder_but_no_encoder():
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    cfg.lift.query_mode = "erp_depth"
    cfg.erp_depth.head = True
    q = ErpDepthQuery(cfg)
    enc = nn.Linear(4, 4)                                          # stands in for the frozen DINOv3 encoder
    dec = tiny_decoder()
    ck = {"query": q.state_dict(), "decoder": decoder_state(dec, include_refiner=False)}
    assert ck["query"] and all(k.startswith("head.") for k in ck["query"])
    assert not any("refiner" in k for k in ck["decoder"])
    R.assert_no_frozen_encoder(ck, enc)                            # passes
    rep = R.param_groups_report(ck)
    assert rep["query"][0] == sum(p.numel() for p in q.parameters())
    with pytest.raises(AssertionError):
        R.assert_no_frozen_encoder(dict(ck, decoder=dict(ck["decoder"], **{"encoder.w": enc.weight})))
    with pytest.raises(AssertionError):                            # by identity, whatever the key
        R.assert_no_frozen_encoder(dict(ck, query={"x": enc.weight}), enc)


# ---- resume ----------------------------------------------------------------------------------------------------

class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(3, 2)


def _run(steps, start=0, state=None, path=None, save_at=None):
    """A toy loop with the script's structure: seeded sampler, RNG noise in the loss, AdamW, resume file."""
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    q, d = _Toy(), _Toy()
    opt = torch.optim.AdamW(list(q.parameters()) + list(d.parameters()), lr=1e-2)
    data = torch.arange(20, dtype=torch.float32)
    s = R.EpochSampler(20, seed=0)
    g = torch.Generator().manual_seed(0)
    dl = DataLoader(data, batch_size=4, sampler=s, drop_last=True, generator=g)
    spe = R.steps_per_epoch(20, 4)
    k0 = 0
    if state:
        k0 = R.load_resume(state, q, d, opt, loader_gen=g)["step"]

    def batches(k):
        e, skip = divmod(k, spe)
        while True:
            s.set_epoch(e, skip * 4)
            for b in dl:
                yield b
            e, skip = e + 1, 0
    it = batches(k0)
    for k in range(k0 + 1, steps + 1):
        x = next(it)[:, None].repeat(1, 3) + torch.randn(4, 3) + float(np.random.rand()) + random.random()
        loss = (q.a(x) - d.a(x)).pow(2).mean() + q.a(x).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if save_at == k:
            R.atomic_save(R.make_resume({"query": q.state_dict(), "decoder": d.state_dict()}, opt,
                                        dict(step=k), loader_gen=g), path)
    return q, d, opt


def test_resume_restores_weights_optimizer_step_and_rng(tmp_path):
    p = tmp_path / "r.pt"
    q_full, d_full, _ = _run(12)                                   # 12 steps = 2.4 epochs straight
    _run(7, path=p, save_at=7)                                     # stop mid-epoch 1 at step 7
    r = torch.load(p, weights_only=False)
    assert r["counters"]["step"] == 7 and "state" in r["optimizer"] and set(r["rng"]) >= {"torch", "numpy", "python", "loader"}
    q_res, d_res, opt_res = _run(12, state=p)                      # fresh process-like start + resume
    for m1, m2 in ((q_full, q_res), (d_full, d_res)):
        for a, b in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(a, b, atol=1e-6)
    assert opt_res.state_dict()["state"][0]["step"].item() == 12


# ---- profile ---------------------------------------------------------------------------------------------------

def test_profile_summary_fields():
    d = R.profile_summary(8, [0.1, 0.1], [0.0, 0.0], [0.9, 0.9], 40 * 2 ** 30, 50 * 2 ** 30, 80 * 2 ** 30,
                          util=(95.0, 51000.0), extra=dict(workers=8))
    for k in ("batch", "steps", "step_s", "data_wait_s", "h2d_s", "compute_s", "data_wait_frac", "samples_per_s",
              "peak_alloc_gib", "peak_reserved_gib", "headroom_frac", "gpu_util_pct", "workers"):
        assert k in d
    assert d["data_wait_frac"] == pytest.approx(0.1)
    assert d["samples_per_s"] == pytest.approx(8.0)
    assert d["headroom_frac"] == pytest.approx(0.375)
    assert R.profile_summary(4, [], [], [], 0, 0, 0)["steps"] == 0


# ---- review fixes: atomic write, resume refusal, log de-duplication ----------------------------------------------

class _Boom:
    def __reduce__(self):
        raise RuntimeError("killed mid-write")


def test_atomic_save_keeps_previous_file_when_the_write_dies(tmp_path):
    p = tmp_path / "r.pt"
    R.atomic_save({"step": 1}, p)
    with pytest.raises(RuntimeError):
        R.atomic_save({"step": 2, "x": _Boom()}, p)                  # dies after the tmp file was opened
    assert torch.load(p, weights_only=False) == {"step": 1}
    R.atomic_save({"step": 3}, p)                                  # a stale .tmp does not get in the way
    assert torch.load(p, weights_only=False) == {"step": 3}


def test_check_resume_refuses_other_batch_or_training_set():
    cnt = dict(batch=32, steps_per_epoch=1315, n_train=42087, tag="t")
    R.check_resume(cnt, 32, 1315, 42087, "t")                      # passes
    R.check_resume(dict(batch=32, steps_per_epoch=1315), 32, 1315, 42090)   # older file: no n_train stored
    for args in ((16, 2630, 42087, "t"), (32, 1315, 42090, "t"), (32, 1315, 42087, "other")):
        with pytest.raises(SystemExit):
            R.check_resume(cnt, *args)


def test_truncate_log_drops_rows_after_the_resume_step(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("step,split,ce\n1,train,1\n2,train,1\n2,val,1\n3,train,1\n4,tra")   # killed at step 4 mid-line
    assert R.truncate_log(p, 2) == 2
    assert p.read_text() == "step,split,ce\n1,train,1\n2,train,1\n2,val,1\n"
    assert R.truncate_log(p, 2) == 0
    assert R.truncate_log(tmp_path / "missing.csv", 5) == 0
