"""Single-node data parallelism of scripts/train_vigor.py (bevloc.model.ddp, the sharded trainrun.EpochSampler) and
the H100 throughput switches (bevloc.model.speed). The gradient tests run 2 CPU processes on the gloo backend."""
from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bevloc.model import trainrun as R
from bevloc.model.ddp import Dist, assert_unwrapped, check_resume_world, per_rank_batch
from bevloc.model.speed import compile_modules, set_encoder_dtype


class _Idx(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def _single_batches(n, B, seed, epoch, skip=0):
    s = R.EpochSampler(n, seed=seed)
    s.set_epoch(epoch, skip)
    return [b.tolist() for b in DataLoader(_Idx(n), batch_size=B, sampler=s, drop_last=True)]


def _rank_batches(n, B, W, r, seed, epoch, skip=0):
    s = R.EpochSampler(n, seed=seed, rank=r, world=W, global_batch=B)
    s.set_epoch(epoch, skip)
    return [b.tolist() for b in DataLoader(_Idx(n), batch_size=B // W, sampler=s, drop_last=True)]


# ---- batch split, sampler sharding -----------------------------------------------------------------------------

def test_per_rank_batch_is_global_over_world():
    assert per_rank_batch(32, 4) == 8 and per_rank_batch(128, 4) == 32 and per_rank_batch(32, 1) == 32
    with pytest.raises(SystemExit):
        per_rank_batch(30, 4)
    # the optimisation of the reviewed single-GPU plan: steps per epoch depend on the GLOBAL batch only
    assert R.epoch_steps(42087, 32, 100) == 131500
    assert R.epoch_steps(42087, 128, 100) == 32800


@pytest.mark.parametrize("n,B,W", [(103, 8, 4), (64, 8, 2), (50, 12, 3)])
def test_sharded_sampler_union_per_step_is_the_single_gpu_batch(n, B, W):
    for epoch in (0, 1, 5):
        single = _single_batches(n, B, 7, epoch)
        ranks = [_rank_batches(n, B, W, r, 7, epoch) for r in range(W)]
        assert all(len(rb) == len(single) == n // B for rb in ranks)
        seen = []
        for j, sb in enumerate(single):
            parts = [ranks[r][j] for r in range(W)]
            assert all(len(p) == B // W for p in parts)
            u = [i for p in parts for i in p]
            assert len(set(u)) == B and sorted(u) == sorted(sb)          # disjoint, union = the global batch
            seen += u
        assert len(seen) == len(set(seen)) == (n // B) * B               # every kept sample once per epoch
        assert set(seen) == set(R.EpochSampler(n, seed=7).order(epoch)[:(n // B) * B].tolist())


def test_sharded_sampler_resume_mid_epoch_continues_every_shard():
    n, B, W, spe = 103, 8, 4, 103 // 8
    for r in range(W):
        full = _rank_batches(n, B, W, r, 3, 2)
        for done in (1, 5, spe - 1):
            assert _rank_batches(n, B, W, r, 3, 2, skip=done * B) == full[done:]
    s = R.EpochSampler(n, seed=3, rank=1, world=W, global_batch=B)
    s.set_epoch(0, 2 * B)
    assert len(s) == (96 - 16) // W == len(list(s))
    with pytest.raises(ValueError):
        s.set_epoch(0, 3)                                          # not a whole number of global batches
    with pytest.raises(ValueError):
        R.EpochSampler(n, rank=0, world=4, global_batch=6)


def test_single_gpu_sampler_is_unchanged():
    a = R.EpochSampler(50, seed=3)
    b = R.EpochSampler(50, seed=3, rank=0, world=1, global_batch=8)
    for s in (a, b):
        s.set_epoch(2, 5)
    assert list(a) == list(b) == R.EpochSampler(50, seed=3).order(2)[5:].tolist() and len(a) == 45


# ---- gradient averaging across processes (gloo, CPU) -----------------------------------------------------------

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3))
        self.unused = nn.Linear(2, 2)                              # trainable but never in the graph

    def forward(self, x):
        return self.f(x)


def _toy():
    torch.manual_seed(0)
    return _ToyNet()


def _data(B):
    g = torch.Generator().manual_seed(1)
    return torch.randn(B, 5, generator=g), torch.randn(B, 3, generator=g)


def _worker(rank, world, port, out, mode):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    D = Dist(rank, world, backend="gloo")
    try:
        m = _toy()
        x, y = _data(8)
        b = 8 // world
        xs, ys = x[rank * b:(rank + 1) * b], y[rank * b:(rank + 1) * b]
        loss = (m(xs) - ys).pow(2).mean()                          # per-rank mean over its b samples
        bad = mode == "bad" and rank == 1
        if not bad:
            loss.backward()
        st = dict(ce=float(rank + 1), acc=0.5, n=10 * (rank + 1), vce_m=float("nan") if rank else 2.0,
                  fine_skipped=rank)
        bad_any, st_g = D.reduce_step(list(m.parameters()), bad, st)
        out[rank] = dict(bad=bad_any, st=st_g, grads=[None if p.grad is None else p.grad.clone() for p in m.parameters()])
    finally:
        D.close()


def _spawn(mode, world=2):
    import torch.multiprocessing as mp
    out = mp.Manager().dict()
    mp.start_processes(_worker, args=(world, _free_port(), out, mode), nprocs=world, start_method="spawn")
    return [out[r] for r in range(world)]


def test_reduced_gradients_equal_the_single_process_global_batch():
    res = _spawn("ok")
    m = _toy()
    x, y = _data(8)
    (m(x) - y).pow(2).mean().backward()                            # one process, global batch 8
    for r in res:
        assert r["bad"] is False
        for g, p in zip(r["grads"], m.parameters()):
            if p.grad is None:
                assert g is None                                   # unused on every rank: stays None
            else:
                assert torch.allclose(g, p.grad, atol=1e-6)
    a, b = res
    for ga, gb in zip(a["grads"], b["grads"]):
        assert (ga is None and gb is None) or torch.equal(ga, gb)  # bitwise identical on every rank
    st = a["st"]
    assert str(st) == str(b["st"])                                # equal on every rank (nan-aware)
    assert st["ce"] == pytest.approx((1 * 10 + 2 * 20) / 30)      # n-weighted like validate's avg
    assert st["n"] == 30 and st["fine_skipped"] == 1
    assert st["vce_m"] == pytest.approx(2.0)                       # mean over the ranks with a finite value


def test_a_step_skipped_on_one_rank_is_skipped_on_all():
    res = _spawn("bad")
    assert [r["bad"] for r in res] == [True, True]


# ---- loss normalisation: the summed rank gradients of the real training step = the single-process union gradient ---

def _step_setup(B=4):
    """The erp_depth training step of tests/test_vce.py (small real decoder, projection head) on B samples whose
    matchable-token counts, depths and GT offsets differ per sample (reference partly black = unmatchable cells)."""
    import sys
    from bevloc import config as C
    from bevloc.data.vigor import R_NORTH
    from bevloc.model.coarse import vce_options
    from bevloc.model.query import build_query
    for d in ("scripts", "tests"):
        if str(C.REPO / d) not in sys.path:
            sys.path.insert(0, str(C.REPO / d))
    from test_vce import REF, _TinyMatcher
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    cfg.lift.query_mode = "erp_depth"
    cfg.erp_depth.head = True
    cfg.erp_depth.head_dim = 32
    cfg.erp_depth.vce_mode = "expect"                  # deterministic VCE (the "sample" draw differs per process)
    torch.manual_seed(0)
    q = build_query(cfg, "erp_depth")
    mt = _TinyMatcher().eval()                         # as FeatureQueryMatcher: the matcher stays in eval mode
    # A well-conditioned GP solve (K_yy + sigma I; the released 0.1 on a random decoder is ill-conditioned in float32
    # and amplifies 1e-7 rounding differences of the upstream gradient to ~1e-3 of the pos_conv / proj gradients,
    # between ANY two batch compositions, one process included): isolates the normalisation, which is what is tested
    mt.model.decoder.gps["16"].sigma_noise = 3000.0
    g = torch.Generator().manual_seed(5)
    dep = torch.rand(B, 28, 56, generator=g) * 25.0 + 5.0
    dep[:, :8] = 60.0
    H = torch.eye(3).repeat(B, 1, 1)
    H[:, 0, 2] = 336.0 + torch.tensor([0.0, 60.0, -90.0, 150.0])[:B]
    H[:, 1, 2] = 330.0 + torch.tensor([0.0, -40.0, 70.0, 120.0])[:B]
    ref = torch.rand(B, 3, REF, REF, generator=g)
    for i, frac in enumerate((0.0, 0.3, 0.6, 0.85)[:B]):   # 0 %, 30 %, 60 %, 85 % of the reference black
        ref[i, :, :int(frac * REF)] = 0.0
    batch = dict(erp=torch.rand(B, 1, 3, 448, 896, generator=g),
                 R_w2c=torch.from_numpy(R_NORTH.copy())[None, None].repeat(B, 1, 1, 1),
                 depth=dep.repeat_interleave(16, 1).repeat_interleave(16, 2)[:, None], ref=ref, H=H,
                 negative=torch.zeros(B, dtype=torch.bool))
    params = [p for p in list(q.parameters()) + list(mt.model.decoder.parameters()) if p.requires_grad]
    kw = dict(certainty_weight=0.01, pose_nll_weight=0.5, vce_weight=1.0, vce_opts=vce_options(cfg))
    return cfg, q, mt, batch, params, kw


def _run_step(q, mt, batch, cfg, kw, parts=None):
    from train_lift_splat import step
    return step(q, mt, batch, cfg, 0.05, 0, 4, 0.5, parts=parts, **kw)


def _step_worker(rank, world, port, out, mode):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    D = Dist(rank, world, backend="gloo")
    try:
        cfg, q, mt, batch, params, kw = _step_setup()
        b = batch["erp"].shape[0] // world
        mine = {k: v[rank * b:(rank + 1) * b] for k, v in batch.items()}
        parts = {}
        loss, st = _run_step(q, mt, mine, cfg, kw, parts)
        s_parts = sum(t for t, _ in parts.values() if t is not None)
        # "exact": reweighted to the global normalisers; "per_rank_mean": each rank's own means (before the fix)
        loss_r = D.global_loss(loss, parts) if mode == "exact" else loss
        loss_r.backward()
        bad, st_g = D.reduce_step(params, False, st, average=mode != "exact")
        out[rank] = dict(grads=[None if p.grad is None else p.grad.clone() for p in params], st=st_g, n=st["n"],
                         parts_sum_err=float((s_parts - loss).detach().abs()), bad=bad)
    finally:
        D.close()


def test_rank_mean_gradient_of_the_training_step_equals_the_single_process_union_gradient():
    import torch.multiprocessing as mp
    world = 2
    res = {}
    for mode in ("exact", "per_rank_mean"):
        out = mp.Manager().dict()
        mp.start_processes(_step_worker, args=(world, _free_port(), out, mode), nprocs=world, start_method="spawn")
        res[mode] = [out[r] for r in range(world)]
    cfg, q, mt, batch, params, kw = _step_setup()
    loss, st = _run_step(q, mt, batch, cfg, kw)                  # one process, the union batch
    loss.backward()
    ref = [p.grad for p in params]
    ex = res["exact"]
    assert ex[0]["n"] != ex[1]["n"] and min(r["n"] for r in ex) > 0   # uneven matchable-token counts
    assert all(r["parts_sum_err"] < 1e-5 and r["bad"] is False for r in ex)
    assert all((a is None) == (g is None) for a, g in zip(ex[0]["grads"], ref))
    scale = max(float(g.abs().max()) for g in ref if g is not None)
    err = max(float((a - g).abs().max()) for a, g in zip(ex[0]["grads"], ref) if g is not None)
    assert err <= 2e-5 * scale, (err, scale)                  # float32 (GP solve included): ~5e-6 relative
    assert all(torch.equal(a, b) for a, b in zip(ex[0]["grads"], ex[1]["grads"]) if a is not None)
    naive = max(float((a - g).abs().max()) for a, g in zip(res["per_rank_mean"][0]["grads"], ref) if g is not None)
    print(f"DDP-vs-union max grad error {err:.2e} (scale {scale:.3f}); per-rank means instead: {naive:.2e}")
    assert naive > 100 * max(err, 1e-9), (naive, err)             # the test is sensitive to the normalisation
    sg = ex[0]["st"]                                            # the logged statistics are the union's
    for k in ("ce", "acc", "cell_err", "pose_nll", "pose_err", "vce_m", "vce_pose_m"):
        assert sg[k] == pytest.approx(st[k], rel=1e-4, abs=1e-5), k
    assert sg["n"] == st["n"] and sg["n_pose"] == st["n_pose"] and sg["n_vce"] == st["n_vce"]


def test_global_loss_is_the_identity_on_one_process_and_refuses_unknown_terms():
    x = torch.tensor(2.5)
    assert Dist().global_loss(x, {"ce": (x, 3)}) is x
    with pytest.raises(KeyError):
        Dist(0, 2, init=False).global_loss(x, {"new_term": (x, 1)})


# ---- rank-0-only files, checkpoint keys, resume refusal --------------------------------------------------------

def test_only_rank_zero_writes(tmp_path):
    D0, D1 = Dist(0, 2, init=False), Dist(1, 2, init=False)
    assert D1.save({"x": 1}, tmp_path / "a.pt") is False and not (tmp_path / "a.pt").exists()
    assert D0.save({"x": 1}, tmp_path / "a.pt") is True and torch.load(tmp_path / "a.pt") == {"x": 1}
    D = Dist()                                                     # no torchrun: rank 0 of 1, all no-ops
    assert D.main and not D.on and D.reduce_step([], False, {"n": 1}) == (False, {"n": 1})
    assert D.gather_object(5) == [5]


def test_checkpoint_keys_have_no_module_prefix():
    m = _toy()
    assert_unwrapped({"query": m.state_dict(), "decoder": m.state_dict()})
    wrapped = {"module." + k: v for k, v in m.state_dict().items()}
    with pytest.raises(AssertionError):
        assert_unwrapped({"query": {}, "decoder": wrapped})


def test_resume_refused_on_world_size_or_split_change():
    cnt = dict(batch=32, world_size=4, batch_per_rank=8)
    check_resume_world(cnt, 4, 8)
    for w, b in ((2, 16), (1, 32), (4, 16)):
        with pytest.raises(SystemExit, match="resume refused"):
            check_resume_world(cnt, w, b)
    check_resume_world(dict(batch=32), 1, 32)                      # a single-GPU file on one GPU
    with pytest.raises(SystemExit):
        check_resume_world(dict(batch=32), 4, 8)                   # a single-GPU file under torchrun


def test_aggregate_profile_uses_the_global_batch():
    rows = [dict(steps=5, step_s=1.0, samples_per_s=8.0, data_wait_frac=0.01 * r, peak_reserved_gib=20.0 + r,
                 gpu_util_pct=90.0, oom=False) for r in range(4)]
    agg = R.aggregate_profiles(rows, 32)
    assert agg["samples_per_s"] == pytest.approx(32.0) and agg["world_size"] == 4
    assert agg["samples_per_s_sum_ranks"] == pytest.approx(32.0) and agg["data_wait_frac_max"] == pytest.approx(0.03)


# ---- encoder autocast / compile wrappers -----------------------------------------------------------------------

class _Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(8, 8)

    def forward(self, x):
        return {16: self.l(x)}


def test_encoder_autocast_wrapper_casts_back_and_keeps_keys():
    e = _Enc()
    x = torch.randn(4, 8)
    ref = e(x)[16]
    keys = list(e.state_dict())
    set_encoder_dtype(e, "bfloat16")
    y = e(x)[16]
    assert y.dtype == torch.float32 and not torch.equal(y, ref) and torch.allclose(y, ref, atol=0.05)
    set_encoder_dtype(e, "float32")
    assert torch.equal(e(x)[16], ref) and list(e.state_dict()) == keys
    with pytest.raises(ValueError):
        set_encoder_dtype(e, "int8")


def test_compile_keeps_state_dict_keys():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.encoder, self.model.decoder = _Enc(), _Enc()
    m = M()
    keys = list(m.model.decoder.state_dict())
    assert compile_modules(m, "decoder") == ["decoder"]            # lazy: nothing is compiled until called
    assert list(m.model.decoder.state_dict()) == keys
    with pytest.raises(ValueError):
        compile_modules(m, "all")
