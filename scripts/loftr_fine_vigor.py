"""LoFTR sanity check of the coarse-to-fine second pass on VIGOR (task 04): does an off-the-shelf dense matcher give
anything at 1 m cells?

  make loftr-fine CKPT=checkpoints/vigor_chicago_same_30k_cell0125_best.pt SPLIT=samearea TAG=... LIMIT=3000 \\
      CONFIG=configs/vigor_cell0125.yaml VIGOR_ARGS="--cities Chicago --solver se2"

Per sample: the coarse pass exactly as scripts/eval_vigor.py (--ckpt / --config: rows unchanged), then the fine window
of --fine-config (default configs/vigor_cell00625_fine.yaml: 56 m at 0.0625 m/px) centred on the coarse `peak` pose,
and kornia's pretrained LoFTR (outdoor) between the IPM picture rebuilt at the fine GSD (224 px = 14 m, greyscale,
invalid pixels black; matches on invalid picture pixels are dropped) and the window (greyscale, 896 px). Both images
share the GSD, so LoFTR only has to find a translation. The pose is solved on LoFTR's matches by the same consensus as
the other rows (`SatRoMa.refined_consensus`: the --solver, the threshold of cfg.matcher.reproj_cells reference cells,
the seed). Rows as eval_vigor.py --fine-*: pose_fine_m, yaw_fine_deg, inliers_fine, pose_fine_gated_m (--fine-gate),
plus nmatch_fine / nused_fine. Expected to be weak (cross-view: a ground picture against an orthophoto).

kornia is imported only when this script runs. Weights: LoFTR(pretrained="outdoor") reads
$TORCH_HOME/hub/checkpoints/loftr_outdoor.ckpt when it exists and downloads it otherwise (Eagle compute nodes have no
internet: put the file there first, or pass --loftr-weights <path>).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_vigor as EV  # noqa: E402

from bevloc.match.satroma import SatRoMa  # noqa: E402

LOFTR_URL = "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt"


def grey(img):
    """(3, H, W) RGB in [0, 1] -> (1, 1, H, W) luminance."""
    w = torch.tensor([0.299, 0.587, 0.114], dtype=img.dtype)[:, None, None]
    return (img * w).sum(0)[None, None]


def load_loftr(weights=None, dev="cpu"):
    try:
        import kornia.feature as KF
    except ImportError as e:                                       # not in the laptop env `bev-patch-pf`
        raise SystemExit(f"kornia is not importable ({e}); it is in the Eagle container") from e
    if weights:
        model = KF.LoFTR(pretrained=None)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False)["state_dict"])
    else:
        model = KF.LoFTR(pretrained="outdoor")
    return model.eval().to(dev)


def matches_to_pose(cons, k0, k1, conf, valid, min_conf=0.0, cells=56):
    """LoFTR matches (query px k0 (N, 2) -> reference px k1 (N, 2), confidence (N,)) -> (Match, n_used): matches on
    invalid picture pixels (`valid` (n, n) bool) or below min_conf are dropped, then the shared consensus."""
    k0, k1, conf = (np.asarray(v, np.float64) for v in (k0, k1, conf))
    k0, k1 = k0.reshape(-1, 2), k1.reshape(-1, 2)
    if len(k0):
        n = valid.shape[0]
        u = np.clip(np.round(k0[:, 0]).astype(int), 0, valid.shape[1] - 1)
        v = np.clip(np.round(k0[:, 1]).astype(int), 0, n - 1)
        keep = np.asarray(valid, bool)[v, u] & (conf >= float(min_conf))
        k0, k1 = k0[keep], k1[keep]
    return SatRoMa.refined_consensus(cons, k0, k1, cells=cells)


class LoftrFine:
    """Second-pass matcher for eval_vigor.fine_pass_row: LoFTR on (fine picture, fine window)."""

    def __init__(self, ds, cons, cfg, dev, model, min_conf=0.0):
        self.ds, self.cons, self.cfg, self.dev, self.model, self.min_conf = ds, cons, cfg, dev, model, min_conf
        self.last_info = {}

    def __call__(self, i, centre_en):
        s = self.ds.item(i, ref_centre_en=centre_en)
        q = grey(s["bev"] * s["bev_valid"][None].float())
        r = grey(s["ref"])
        with torch.no_grad():
            out = self.model({"image0": q.to(self.dev), "image1": r.to(self.dev)})
        k0, k1, conf = (out[k].detach().cpu().numpy() for k in ("keypoints0", "keypoints1", "confidence"))
        m, n_used = matches_to_pose(self.cons, k0, k1, conf, s["bev_valid"].numpy(), self.min_conf,
                                    cells=int(s["ref"].shape[-1]) // 16)
        self.last_info = dict(nmatch_fine=int(len(k0)), nused_fine=int(n_used))
        return m, s


def make_loftr_fine(weights, min_conf):
    def factory(a, cfg, ds, dev):
        cfg_f = EV.fine_config(a, cfg)
        cfg_f.lift.query_mode = "ipm"                              # the picture, rebuilt at the fine GSD
        ds_f = EV.fine_dataset(a, cfg_f, ds)
        n, S = int(cfg_f.grid.n), int(cfg_f.grid.n * cfg_f.reference.scale)
        cons = SatRoMa.from_wrapper(NS(im_a_size=n, im_b_size=S), cfg_f, use_means=False, min_valid_frac=0.05)
        model = load_loftr(weights, dev)
        print(f"fine pass: LoFTR outdoor ({weights or 'pretrained=outdoor via torch.hub'}), cell {cfg_f.grid.cell_m} m, "
              f"window {ds_f.ref_window_m} m, gate {a.fine_gate} m, solver {cfg_f.matcher.solver}, min conf {min_conf}",
              flush=True)
        meta = dict(config=a.fine_config, matcher="kornia LoFTR outdoor", weights=weights or LOFTR_URL,
                    min_conf=min_conf, cell_m=float(cfg_f.grid.cell_m), window_m=ds_f.ref_window_m, gate_m=a.fine_gate)
        return LoftrFine(ds_f, cons, cfg_f, dev, model, min_conf), meta
    return factory


def main():
    ap = EV.build_parser(__doc__)
    ap.set_defaults(fine_config=str(Path(__file__).resolve().parents[1] / "configs/vigor_cell00625_fine.yaml"),
                    out="experiments/10_loc2_matcher")
    ap.add_argument("--loftr-weights", default=None, help="loftr_outdoor.ckpt (default: torch.hub cache / download)")
    ap.add_argument("--loftr-min-conf", type=float, default=0.0, help="drop LoFTR matches below this confidence")
    a = ap.parse_args()
    EV.run(a, make_loftr_fine(a.loftr_weights, a.loftr_min_conf))


if __name__ == "__main__":
    main()
