"""Task 06 step 4: the pose-correctness head (shapes, parameter count, frame stream = logistic), its D4 augmentation,
the certainty cache on synthetic VIGOR with the reproduction gate, the training script's smoke path and the head rows
of certainty_vigor.py. CPU, planted toy decoders (tests/tiny_satroma.py), no weights, no data."""
from __future__ import annotations

import importlib.util
import json
import pickle
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from bevloc import config as C
from bevloc.eval import calibration as K
from bevloc.match import vote_stats as V
from bevloc.model import certainty_head as CH

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_two_pass import _setup  # noqa: E402
from vigor_labels import corrected  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(name, file):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CC = _load("bevloc_scripts_certainty_cache_vigor", "certainty_cache_vigor.py")
TR = _load("bevloc_scripts_train_certainty_head", "train_certainty_head.py")
CV = _load("bevloc_scripts_certainty_vigor", "certainty_vigor.py")
EV = _load("bevloc_scripts_eval_vigor", "eval_vigor.py")

KK, T = 56, 1568
HC = C.load().certainty_head


# ---- the head ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("streams", [("map", "tokens", "frame"), ("map",), ("tokens",), ("frame",), ("tokens", "frame")])
def test_head_shapes_parameter_count_and_missing_streams(streams):
    torch.manual_seed(0)
    head = CH.CertaintyHead(streams=streams, n_targets=3, n_frame=19)
    assert head.streams == CH.canonical_streams(streams) and CH.n_parameters(head) < 1_000_000
    B = 3
    maps = torch.rand(B, 4, KK, KK) if "map" in streams else None
    tokens = torch.rand(B, T, 8) if "tokens" in streams else None
    valid = (torch.rand(B, T) > 0.3) if "tokens" in streams else None
    frame = torch.randn(B, 19) if "frame" in streams else None
    if frame is not None:
        frame[0, 3] = float("nan")                                     # a missing statistic is imputed, not propagated
    z = head(maps=maps, tokens=tokens, token_valid=valid, frame=frame)
    assert z.shape == (B, 3) and torch.isfinite(z).all()
    p = head.probabilities(maps=maps, tokens=tokens, token_valid=valid, frame=frame)
    assert p.shape == (B, 3) and (p > 0).all() and (p < 1).all()
    if valid is not None:                                              # a sample without any valid token is finite
        valid[1] = False
        assert torch.isfinite(head(maps=maps, tokens=tokens, token_valid=valid, frame=frame)).all()
    full = CH.CertaintyHead()
    assert CH.n_parameters(full) < 1_000_000 and full.streams == CH.STREAMS
    with pytest.raises(ValueError):
        CH.CertaintyHead(streams=("maps",))


def _frame_cache(X, err, log_mask=None, n_map=4, k=4, t=2):
    """A minimal certainty cache whose only informative part is the frame statistics."""
    n, f = X.shape
    keys = [f"s{j}" for j in range(f)]
    frames = [dict(id=f"c/{i}", city="Chicago", pose_peak_m=None if not np.isfinite(err[i]) else float(err[i]),
                   inliers_peak=0.5, stats={kk: (None if np.isnan(X[i, j]) else float(X[i, j])) for j, kk in enumerate(keys)},
                   maps=np.zeros((n_map, k, k), np.float16), tokens=np.zeros((t, 8), np.float16),
                   token_valid=np.ones(t, bool)) for i in range(n)]
    meta = dict(frame_keys=keys, frame_log=list(log_mask if log_mask is not None else [False] * f), token_grid="erp",
                draw="calib", token_keys=list(CH.TOKEN_KEYS), map_keys=list(CH.MAP_KEYS), ckpt="x.pt", n=n)
    return dict(meta=meta, frames=frames)


def test_frame_stream_alone_reproduces_a_logistic_fit():
    rng = np.random.default_rng(0)
    n, f = 5000, 19
    X = rng.standard_normal((n, f))
    X[:, 2] = np.exp(X[:, 2])                                           # a log1p column, as the count statistics
    w = rng.standard_normal(f) * 0.8
    Xt = X.copy()
    Xt[:, 2] = np.log1p(Xt[:, 2])
    y = rng.random(n) < 1.0 / (1.0 + np.exp(-(Xt @ w - 0.3)))
    err = np.where(y, 1.0, 20.0)                                        # correct frames at 1 m, wrong ones at 20 m
    X[rng.random(n) < 0.02, 5] = np.nan                                 # a few missing values
    log = [j == 2 for j in range(f)]
    arrays = CH.CacheArrays(_frame_cache(X, err, log))
    n_val = 1000
    tr, va = np.arange(n - n_val), np.arange(n - n_val, n)
    lg = K.Logistic(l2=1.0).fit(np.where(np.isnan(Xt[tr]), np.nan, Xt[tr]), y[tr])
    p_lg = lg.predict(Xt[va])
    head, rec = TR.train(arrays, ("frame",), HC, [2.0, 5.0, 10.0], 5.0, "cpu", epochs=40, lr=1e-3, weight_decay=1e-4,
                         batch=256, patience=10, val_frames=n_val, seed=0, aug=False, log=lambda s: None)
    p_head = CH.predict(head, arrays.subset(va))[:, 1]
    a_lg, a_head = K.auroc(p_lg, y[va]), K.auroc(p_head, y[va])
    assert a_lg > 0.8 and abs(a_head - a_lg) < 0.02, (a_lg, a_head)
    assert K.reliability(p_head, y[va])[1] < 0.05                       # temperature-scaled: calibrated on val
    assert rec["n_params"] < 1_000_000 and rec["best_epoch"] >= 1 and len(rec["curve"]) == rec["epochs_run"]
    # the standardisation lives in buffers and survives a save / load round trip
    assert head.frame_log[2] and not head.frame_log[0] and abs(float(head.frame_mean[0])) < 0.1
    assert torch.isfinite(head.frame_fill).all()
    assert head.temperature.shape == (3,)


# ---- augmentation --------------------------------------------------------------------------------------------------------

def _raster_erp(tokens, k=KK, r=20.3):
    """Cells hit by a unit-circle rasterisation of every token's azimuth: alpha = 2 pi (a - 0.5) clockwise from north."""
    a = tokens[..., CH.AZIMUTH].numpy().reshape(-1)
    al = 2 * np.pi * (a - 0.5)
    c = (k - 1) / 2.0
    x, y = c + r * np.sin(al), c - r * np.cos(al)
    m = np.zeros((k, k))
    m[np.rint(y).astype(int), np.rint(x).astype(int)] = 1.0
    return m


def _raster_bev(tokens, k):
    """BEV grid: token (u, v) -> cell (v k - 0.5, u k - 0.5)."""
    u, v = tokens[..., CH.AZIMUTH].numpy().reshape(-1), tokens[..., CH.ROW].numpy().reshape(-1)
    m = np.zeros((k, k))
    m[np.rint(v * k - 0.5).astype(int), np.rint(u * k - 0.5).astype(int)] = 1.0
    return m


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("flip", [False, True])
def test_augmentation_moves_the_token_positions_with_the_maps(k, flip):
    w, h = 56, 28
    tok = torch.zeros(h * w, 8)
    rows, cols = np.divmod(np.arange(h * w), w)
    tok[:, CH.AZIMUTH] = torch.as_tensor((cols + 0.5) / w).float()
    tok[:, CH.ROW] = torch.as_tensor((rows + 0.5) / h).float()
    maps = torch.as_tensor(_raster_erp(tok))[None].repeat(4, 1, 1)
    m2 = CH.augment_maps(maps, k, flip)
    t2 = CH.augment_tokens(tok, k, flip, "erp")
    assert np.array_equal(_raster_erp(t2), m2[0].numpy())          # the map moved exactly where the azimuths say
    assert torch.equal(t2[:, CH.ROW], tok[:, CH.ROW])               # the row (elevation) never changes
    assert torch.equal(t2[:, :CH.AZIMUTH], tok[:, :CH.AZIMUTH])     # the other features are untouched
    assert (t2[:, CH.AZIMUTH] >= 0).all() and (t2[:, CH.AZIMUTH] < 1).all()
    # BEV grid (picture query): the (u, v) pair rotates and mirrors with the map
    kb = 14
    tb = torch.zeros(kb * kb, 8)
    rb, cb = np.divmod(np.arange(kb * kb), kb)
    tb[:, CH.AZIMUTH], tb[:, CH.ROW] = torch.as_tensor((cb + 0.5) / kb).float(), torch.as_tensor((rb + 0.5) / kb).float()
    mb = torch.zeros(kb, kb)
    mb[3, 10] = 1.0                                                  # one token's cell
    one = tb[3 * kb + 10][None]
    assert np.array_equal(_raster_bev(CH.augment_tokens(one, k, flip, "bev"), kb), CH.augment_maps(mb, k, flip).numpy())


def test_batched_augmentation_is_per_sample_and_the_identity_element_changes_nothing():
    g = torch.Generator().manual_seed(0)
    maps, tok = torch.rand(8, 4, KK, KK, generator=g), torch.rand(8, 40, 8, generator=g)
    k, flip = torch.arange(8) % 4, torch.arange(8) >= 4
    m2, t2 = CH.augment(maps, tok, k, flip, "erp")
    for i in range(8):
        assert torch.equal(m2[i], CH.augment_maps(maps[i], int(k[i]), bool(flip[i])))
        assert torch.equal(t2[i], CH.augment_tokens(tok[i], int(k[i]), bool(flip[i]), "erp"))
    assert torch.equal(m2[0], maps[0]) and torch.equal(t2[0], tok[0])
    m4, t4 = CH.augment(maps, tok, k + 4, flip, "erp")                # k mod 4
    assert torch.equal(m4, m2) and torch.equal(t4, t2)
    # a label depends on the error only, so the same cache entry rotated keeps its labels
    arrays = CH.CacheArrays(_frame_cache(np.zeros((3, 2)), np.array([1.0, 7.0, np.inf])))
    assert np.array_equal(arrays.labels([2.0, 5.0, 10.0]), [[1, 1, 1], [0, 0, 1], [0, 0, 0]])
    with pytest.raises(ValueError):
        CH.augment_tokens(tok[0], 1, False, "polar")


# ---- the cache on synthetic VIGOR ----------------------------------------------------------------------------------------

def _vigor_root(tmp_path, n_train=20, n_test=4):
    """Synthetic same-area VIGOR root: n_train panoramas in the train list, n_test others in the test list, all the
    same picture at the same label (planted toy decoders of test_two_pass), plus the models eval_vigor builds."""
    root = tmp_path / "vigor"
    _, ds, query, matcher, cons, cfg, _ = _setup(root, (352.0, 320.0))
    lab = root / "splits" / "VIGOR" / "Chicago"
    line = (lab / "pano_label_balanced.txt").read_text()
    src = root / "Chicago" / "panorama" / "p1,1.0,.jpg"
    for name, n0, n in (("same_area_balanced_train.txt", 1, n_train), ("same_area_balanced_test.txt", n_train + 1, n_train + n_test)):
        lines = []
        for k in range(n0, n + 1):
            if k != 1:
                shutil.copy(src, src.with_name(f"p{k},1.0,.jpg"))
            lines.append(line.replace("p1,", f"p{k},"))
        (lab / name).write_text("".join(lines))
    corrected(lab)
    tm = dict(val_frac=0.2, val_samples=0, cities=["Chicago"], split="samearea")
    M = dict(cfg=cfg, dev="cpu", mode="ipm", train_meta=tm, matcher=matcher, query=query, cons=cons)
    return root, M


def _args(tmp_path, root, *extra):
    return CC.build_parser().parse_args(["--ckpt", "x.pt", "--tag", "t", "--out", str(tmp_path), "--root", str(root),
                                         "--cities", "Chicago", "--split", "samearea", *extra])


@pytest.fixture(scope="module")
def caches(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cert")
    root, M = _vigor_root(tmp)
    calib = CC.cache_draw(_args(tmp, root, "--draw", "calib"), M)
    test = CC.cache_draw(_args(tmp, root, "--draw", "test", "--limit", "0"), M)
    return dict(tmp=tmp, root=root, M=M, calib=calib, test=test)


def test_cache_round_trip_reproduces_eval_vigor_and_holds_the_evidence(caches):
    with open(caches["test"], "rb") as f:
        c = pickle.load(f)
    m, frames = c["meta"], c["frames"]
    assert m["draw"] == "test" and m["n"] == len(frames) == 4 and m["gate"]["mismatches"] == 0 and m["gate"]["n"] == 4
    assert m["K"] == 56 and m["token_grid"] == "bev" and m["token_hw"] == [14, 14]
    assert len(m["frame_keys"]) == 19 and m["frame_keys"][0] == "inliers_peak" and set(V.STAT_KEYS) < set(m["frame_keys"])
    assert m["frame_log"][m["frame_keys"].index("n_modes")] and not m["frame_log"][m["frame_keys"].index("vote_top1")]
    r = frames[0]
    assert r["maps"].shape == (4, 56, 56) and r["maps"].dtype == np.float16
    assert r["tokens"].shape == (196, 8) and r["tokens"].dtype == np.float16 and r["token_valid"].shape == (196,)
    assert r["pose_peak_m"] == pytest.approx(3.62, abs=0.05) and r["H"] is not None   # the planted pose of test_two_pass
    # the planted decoder puts every valid token's mass at the pose-consistent cell: p_pose ~ 1, entropy ~ 0, peak at
    # the pose cell; the footprint marks the inlier tokens (+1) and nothing else
    vt = r["token_valid"]
    tok = r["tokens"].astype(np.float32)
    assert vt.sum() > 100 and (tok[~vt] == 0).all()
    assert tok[vt, CH.P_POSE].min() > 0.9 and tok[vt, CH.TOKEN_KEYS.index("entropy")].max() < 0.05
    assert tok[vt, CH.TOKEN_KEYS.index("peak_pose_cells")].max() < 0.75
    assert np.allclose(tok[vt, CH.AZIMUTH] * 14 - 0.5, np.arange(196)[vt] % 14, atol=1e-2)
    fp = r["maps"][CH.MAP_KEYS.index("footprint")].astype(np.float32)
    assert (fp == 1).sum() > 0 and (fp == -1).sum() == 0 and (fp == 1).sum() <= r["n_inlier_tokens"] == vt.sum()
    assert r["n_inlier_modes_recomputed"] == r["n_inliers"] == r["n_modes"] > 0
    ego = r["maps"][CH.MAP_KEYS.index("ego")].astype(np.float64)
    assert abs(ego.sum() - 1) < 1e-2 and r["stats"]["ego_mass_2cells"] > 0.99
    assert r["maps"][CH.MAP_KEYS.index("ref_valid")].max() == 1.0
    # the statistics and the error are eval_vigor.score's, key for key
    M = caches["M"]
    from bevloc.data.vigor import VigorPairs
    ds = VigorPairs(caches["root"], M["cfg"], cities=["Chicago"], split="samearea", train=False)
    rows = EV.score(ds, M["query"], M["matcher"], M["cons"], M["cfg"], "cpu")
    assert [x["id"] for x in rows] == m["ids"]
    for fr, row in zip(frames, rows):
        assert fr["pose_peak_m"] == pytest.approx(row["pose_peak_m"], abs=1e-9)
        for k in m["frame_keys"]:
            assert fr["stats"][k] == pytest.approx(row[k], abs=1e-9) if row[k] is not None else fr["stats"][k] is None
    arrays = CH.CacheArrays(c)
    assert arrays.maps.shape == (4, 4, 56, 56) and arrays.frame.shape == (4, 19) and np.isfinite(arrays.err).all()


def test_calib_draw_uses_the_heldout_rule_and_is_disjoint_from_the_test_draw(caches):
    with open(caches["calib"], "rb") as f:
        c = pickle.load(f)
    m = c["meta"]
    assert m["draw"] == "calib" and m["n"] == 4                        # 20 train labels, the last 20 % held out
    assert m["draw_info"]["coarse"]["source"] == "checkpoint" and m["draw_info"]["coarse"]["n_heldout"] == 4
    with open(caches["test"], "rb") as f:
        t = pickle.load(f)
    assert not set(m["ids"]) & set(t["meta"]["ids"])
    assert m["consensus"]["solver"] == "se2" and m["consensus_means"]["target"] == "means"
    # the cache keeps calib_split's order (the seed-0 permutation of the train list), so the trainer's "last
    # val_frames frames" is a random-like slice of the held-out frames, not the tail of one city's list
    from bevloc.data.vigor import read_labels
    labels = read_labels(caches["root"], ["Chicago"], "samearea", True)
    held = np.random.default_rng(0).permutation(len(labels))[16:]
    assert m["ids"] == [f"Chicago/{labels[i]['pano']}" for i in held] and list(held) != sorted(held)


def test_gate_stops_the_script_on_a_mismatch(caches, monkeypatch):
    frames = pickle.load(open(caches["test"], "rb"))["frames"]
    good = EV.score
    rows = None

    def perturbed(*args, **kw):
        nonlocal rows
        rows = good(*args, **kw)
        rows[1]["pose_peak_m"] += 0.01                               # off by more than 1e-3 m
        return rows
    monkeypatch.setattr(EV, "score", perturbed)
    a = _args(caches["tmp"], caches["root"], "--draw", "test", "--limit", "0", "--tag", "bad")
    with pytest.raises(SystemExit, match=frames[1]["id"]):
        CC.cache_draw(a, caches["M"])
    a = _args(caches["tmp"], caches["root"], "--draw", "test", "--limit", "0", "--tag", "bad", "--allow-mismatch")
    p = CC.cache_draw(a, caches["M"])
    g = pickle.load(open(p, "rb"))["meta"]["gate"]
    assert g["mismatches"] == 1 and g["ids"] == [frames[1]["id"]] and "pose_peak_m" in g["what"][0]
    # a statistic that differs is caught too, and --gate-frames limits the check
    monkeypatch.setattr(EV, "score", good)
    rows[1]["pose_peak_m"] -= 0.01                                   # the perturbed row, restored
    bad, n = CC.compare_rows(frames, [dict(r, vote_top1=0.0) if i == 2 else r for i, r in enumerate(rows)], pickle.load(
        open(caches["test"], "rb"))["meta"]["frame_keys"])
    assert n == 4 and [i for i, _ in bad] == [frames[2]["id"]]
    a = _args(caches["tmp"], caches["root"], "--draw", "test", "--limit", "0", "--tag", "g1", "--gate-frames", "1")
    assert pickle.load(open(CC.cache_draw(a, caches["M"]), "rb"))["meta"]["gate"]["n"] == 1


def test_evidence_of_a_placed_query_marks_inlier_and_outlier_tokens():
    """A placed (panorama-grid) query with planted outliers (test_consensus_sweep._sample): the footprint carries +1
    and -1 cells, the recomputed inlier set is the consensus's, the token rows follow the 14 x 28 grid."""
    from test_consensus_sweep import _cons, _sample
    from bevloc.match.satroma import consensus_for_query, query_tokens
    cons = _cons("se2")
    cons.stats_radius = 2.0
    query, batch, frac, gm, cert = _sample("placed", 0)
    m = consensus_for_query(cons, gm, query, batch, frac, 224, certainty=cert, stats=True)
    valid, xy = query_tokens(cons, query, batch, frac, 224, 0.05)
    valid_np = valid.numpy()
    gmm = gm.clone()
    gmm[:, ~valid] = 0.0
    rv = np.ones((56, 56), bool)
    maps, tok, tv, n_it, n_im = CC.evidence(cons, gmm, cert, valid_np, xy.numpy(), m.H, rv, 0.125)
    h, w = gm.shape[-2:]
    assert (h, w) == (14, 28) and tok.shape == (h * w, 8) and np.array_equal(tv, valid_np.reshape(-1))
    assert n_im == m.n_inliers and 0 < n_it <= n_im                 # the px -> cell round trip agrees here
    fp = maps[CH.MAP_KEYS.index("footprint")].astype(np.float32)
    assert (fp == 1).any() and (fp == -1).any() and (fp == 1).sum() <= n_it
    rows, cols = np.divmod(np.arange(h * w), w)
    t = tok.astype(np.float32)
    assert np.allclose(t[tv, CH.AZIMUTH], (cols[tv] + 0.5) / w, atol=1e-3) and np.allclose(t[tv, CH.ROW], (rows[tv] + 0.5) / h, atol=1e-3)
    assert (t[~tv] == 0).all() and (t[tv, CH.P_POSE] >= 0).all() and (t[tv, CH.P_POSE] <= 1).all()
    assert (t[tv, CH.TOKEN_KEYS.index("entropy")] >= 0).all() and (t[tv, CH.TOKEN_KEYS.index("entropy")] <= 1).all()
    assert np.allclose(t[tv, CH.TOKEN_KEYS.index("cert")], cert.numpy().reshape(-1)[tv], atol=2e-3)
    assert np.allclose(t[tv, CH.TOKEN_KEYS.index("depth_m")],
                       np.linalg.norm(xy.numpy().reshape(-1, 2)[tv] - 111.5, axis=1) * 0.125, rtol=2e-3)
    # the ego and vote maps are the statistics' maps: masses sum to one and agree with Match.stats
    ego, vote = maps[CH.MAP_KEYS.index("ego")].astype(np.float64), maps[CH.MAP_KEYS.index("vote")].astype(np.float64)
    assert abs(ego.sum() - 1) < 2e-2 and abs(vote.sum() - 1) < 2e-2
    assert V.vote_stats(vote)["vote_top1"] == pytest.approx(m.stats["vote_top1"], abs=2e-3)
    # without a pose: no footprint, no pose-consistency, the peak distance at its ceiling
    maps0, tok0, _, n0, n0m = CC.evidence(cons, gmm, cert, valid_np, xy.numpy(), None, rv, 0.125)
    assert (maps0[CH.MAP_KEYS.index("footprint")] == 0).all() and n0 == n0m == 0
    assert (tok0[tv, CH.P_POSE] == 0).all() and (tok0[tv, CH.TOKEN_KEYS.index("peak_pose_cells")] == 56).all()


def test_fine_pass_and_other_rows_are_refused():
    with pytest.raises(SystemExit, match="coarse default path only"):
        CC.main(["--ckpt", "x.pt", "--tag", "t", "--fine-config", "f.yaml", "--fine-ckpt", "f.pt"])
    with pytest.raises(SystemExit, match="coarse default path only"):
        CC.main(["--ckpt", "x.pt", "--tag", "t", "--hyp", "8"])


# ---- training script and the head rows of certainty_vigor.py ---------------------------------------------------------------

def test_train_script_smoke_and_head_rows_in_certainty_vigor(caches):
    tmp = caches["tmp"]
    pt = TR.main(["--cache", str(caches["calib"]), "--tag", "t", "--out", str(tmp), "--epochs", "2", "--val-frames", "1",
                  "--batch", "2", "--device", "cpu"])
    assert Path(pt).name == "certainty_head_t_map-tokens-frame.pt" and (tmp / "certainty_head_t_map-tokens-frame.json").is_file()
    js = json.loads((tmp / "certainty_head_t_map-tokens-frame.json").read_text())
    assert js["epochs_run"] == 2 and len(js["curve"]) == 2 and js["n_train"] == 3 and js["n_val"] == 1
    assert js["cache"]["draw"] == "calib" and js["frame_keys"][0] == "inliers_peak" and js["token_grid"] == "bev"
    head, ck = CH.load_head(pt)
    assert head.streams == CH.STREAMS and CH.n_parameters(head) < 1_000_000 and len(ck["val_ids"]) == 1
    arrays = CH.CacheArrays(pickle.load(open(caches["test"], "rb")))
    P = CH.predict(head, arrays)
    assert P.shape == (4, 3) and (P > 0).all() and (P < 1).all()
    pt_f = TR.main(["--cache", str(caches["calib"]), "--tag", "t", "--out", str(tmp), "--epochs", "1", "--val-frames", "1",
                    "--batch", "2", "--device", "cpu", "--streams", "frame", "--no-aug"])
    assert Path(pt_f).name == "certainty_head_t_frame.pt"
    with pytest.raises(SystemExit, match="not 'calib'"):                # never train on the test draw
        TR.main(["--cache", str(caches["test"]), "--tag", "x", "--out", str(tmp), "--epochs", "1", "--val-frames", "1"])
    # certainty_vigor: the heads next to the isotonic / logistic rows, which are fitted on the heads' TRAINING frames
    # (the last --val-frames of the fit cache, the heads' validation frames, are left out and are their conformal half)
    res = CV.main(["--cache", str(caches["test"]), "--fit-cache", str(caches["calib"]), "--val-frames", "1",
                   "--head", str(pt), str(pt_f), "--tag", "h", "--out", str(tmp)])
    names = list(res["models"])
    assert names[:2] == ["inlier_isotonic", "logistic"] and "head:t_map-tokens-frame" in names and "head:t_frame" in names
    hd = res["models"]["head:t_map-tokens-frame"]["5.0"]
    assert 0 <= hd["ece"] <= 1 and hd["conformal"] is not None and hd["conformal"]["n_cal"] == 1
    assert res["abstention"]["head:t_frame"]["at90"]["n"] == 4
    assert res["meta"]["protocol"]["mode"] == "calib->test" and res["meta"]["protocol"]["n_calib"] == 3
    assert res["meta"]["protocol"]["val_frames_left_out"] == 1
    assert (tmp / "certainty_h.json").is_file() and (tmp / "certainty_h.png").is_file()
    assert "head:t_frame" in res["gate"]
    # the cache path scores the cache header's 19 statistics (the frame stream's), the eval-json path FEATURES: the
    # reported logistic rows (2026-09-26) re-run with their own feature set
    feats = [f["key"] for f in res["meta"]["features"]]
    keys = pickle.load(open(caches["test"], "rb"))["meta"]["frame_keys"]
    assert set(feats) < set(keys) and "vote_offtile_mass" in feats and "spread_hyp_m" not in feats
    assert set(keys) - set(feats) == {"placed_depth_m"}                # None on a picture query: not a feature
    assert "vote_offtile_mass" not in {k for k, _ in CV.FEATURES} and "spread_hyp_m" in {k for k, _ in CV.FEATURES}
    # without a fit cache the head has no conformal half; a head needs the cache; a test-draw fit cache is refused; a
    # head validated on other frames than the left-out ones is refused (its conformal half would overlap the fit)
    res = CV.main(["--cache", str(caches["test"]), "--head", str(pt), "--tag", "h2", "--out", str(tmp)])
    assert res["models"]["head:t_map-tokens-frame"]["5.0"]["conformal"] is None and res["meta"]["protocol"]["mode"].startswith("5-fold")
    with pytest.raises(SystemExit, match="needs the test frames as a cache"):
        CV.main(["--eval-json", "x.json", "--head", str(pt), "--tag", "h3", "--out", str(tmp)])
    with pytest.raises(SystemExit, match="held-out training frames only"):
        CV.main(["--cache", str(caches["test"]), "--fit-cache", str(caches["test"]), "--val-frames", "1", "--tag", "h4",
                 "--out", str(tmp)])
    with pytest.raises(SystemExit, match="not the last 2 frames"):
        CV.main(["--cache", str(caches["test"]), "--fit-cache", str(caches["calib"]), "--val-frames", "2",
                 "--head", str(pt), "--tag", "h5", "--out", str(tmp)])
    with pytest.raises(SystemExit, match="must leave fit frames"):
        CV.main(["--cache", str(caches["test"]), "--fit-cache", str(caches["calib"]), "--val-frames", "4", "--tag", "h6",
                 "--out", str(tmp)])


def test_per_city_auroc_is_reported_when_the_test_frames_span_cities():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        good = rng.random() < 0.7
        rows.append(dict(id=f"f{i}", city="Chicago" if i % 2 else "Seattle", pose_peak_m=float(1.0 if good else 20.0),
                         inliers_peak=float(np.clip((0.8 if good else 0.5) + 0.1 * rng.standard_normal(), 0, 1))))
    cfg = C.load().certainty
    cfg.targets_m, cfg.rank_target_m = [2.0, 5.0, 10.0], 5.0
    names = CV.feature_names([rows], "peak")
    pred = CV.pool([CV.run_split(rows[:200], rows[200:], names, "peak", cfg, 0)], [np.arange(200)], 200)
    res = CV.evaluate(pred, CV.errors(rows[200:], "peak"), names, rows[200:], "peak", cfg)
    assert set(res["per_city"]) == {"Chicago", "Seattle"} and res["per_city"]["Chicago"]["n"] == 100
    assert 0.5 < res["per_city"]["Chicago"]["auroc"]["logistic"] <= 1.0
    assert "logistic" in CV.per_city_table(res, cfg)


def test_temperature_fit_lowers_the_nll_of_overconfident_logits():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((4000, 3)) * 3.0
    y = (rng.random((4000, 3)) < 1.0 / (1.0 + np.exp(-z / 3.0))).astype(np.float32)   # the true scale is 3
    T = TR.fit_temperature(z, y)
    assert np.all(np.abs(T - 3.0) < 0.5) and np.all(TR.bce(z / T[None], y) < TR.bce(z, y))
    y1 = np.ones((4000, 3), np.float32)
    assert np.array_equal(TR.fit_temperature(z, y1), np.ones(3))          # one class: nothing to calibrate
