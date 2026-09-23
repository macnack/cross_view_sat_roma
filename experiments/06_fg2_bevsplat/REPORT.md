# Task 02 — FG² / BevSplat transfer report

# Task 02 status — 23 Sep 2026

Protocol: [`docs/tasks/02_fg2_bevsplat.md`](../../docs/tasks/02_fg2_bevsplat.md).
Output root: `experiments/06_fg2_bevsplat/`.

## Progress vs steps

| Step | Result |
|------|--------|
| 1. Native reproducibility | **BLOCKED** (see below) |
| 2. Shared manifest + coordinate tests | **DONE** |
| 3. FG² zero-shot on Fixtor | deferred (task §1 stop rule) |
| 4. BevSplat zero-shot | deferred (checkpoints + §1) |
| 5–7. Overfit / fine-tune / report | not started |

## Step 1 — blockers (stop here per task doc)

### FG²
- Released VIGOR/KITTI checkpoints downloaded to `checkpoints/baselines/fg2/` (7× `model.pt`, 242 332 222 B each). Hashes: `fg2_checkpoint_hashes.txt`.
- License: GPL-3.0 (`third_party/FG2/LICENSE`).
- `third_party/FG2` present (read-only).
- **VIGOR dataset imagery is not on this machine** (only the empty `third_party/VIGOR` code tree). Cannot run `vigor_eval.py` or reproduce a published number.
- Local `bev-patch-pf` env lacks `mmcv` (FG² needs `mmcv.ops.MultiScaleDeformableAttention` + `mmcv.cnn.bricks.transformer.FFN`; authors pin torch 2.2.2 + `mim install mmcv-full`). Checkpoint `torch.load` of the state_dict succeeds; constructing `CVM` does not.
- Conda editable install of `sat_roma/.../bev-patch-pf` shadows FG²'s `utils` package; wrapper neutralises that at import time (`src/bevloc/baselines/fg2.py`).

### BevSplat
- Cloned read-only at `third_party/BevSplat` @ `92bebde1be2d50da5c679e11cb43c14ddc70544a`.
- License: MIT.
- OneDrive checkpoint folder returns HTTP 403 / reauth for non-interactive download:
  https://1drv.ms/f/c/86d953bfc66eb903/IgAP7P2tFzChR7rHeMuXIOq8AakOxR02eKMyI2Z7qsMjLxo?e=zaD0Fb
- Missing under `checkpoints/baselines/bevsplat/`: all four VIGOR `*.pth` files (plus KITTI).
- CUDA rasterizers not built (`scripts/bootstrap_cuda.sh`); also needs VIGOR for native eval.

**Per task §1: do not debug Fixtor adapters until a native checkpoint reproduces.**

## Step 2 — done

- Manifest: `manifest.json` — **200** held-out frames from `IcRzj0wTLZX874qitxVsQa`, years **2025+2024** (400 entries), local prior ±10% / ±10°, seed 0.
- Overview: `manifest_overview.jpg`.
- Coordinate unit tests: `tests/test_baseline_coordinates.py` (injected `(5 m, −3 m, 7°)` recovers with `err_m=0`, `err_deg=0`).
- Makefile targets: `baselines-manifest`, `fg2-smoke|eval|train`, `bevsplat-smoke|eval|train`, `baselines-report`.
- Wrappers (stubs): `src/bevloc/baselines/{common,fg2,bevsplat}.py`, `scripts/{build_baseline_manifest,eval_fg2,eval_bevsplat,train_*,report_baselines}.py`.

## What you need to unblock §1

1. **VIGOR dataset** at a path FG²/BevSplat configs can read (see each README).
2. **FG² env**: `conda env create -f third_party/FG2/environment.yml` then `mim install "mmcv-full>=1.7.1"` (or a known-good wheel for that torch/CUDA), activate, run authors' same-area known-ori eval; record one published metric + one qualitative image under `experiments/06_fg2_bevsplat/native_fg2/`.
3. **BevSplat checkpoints**: manual OneDrive download of the six `.pth` files into `checkpoints/baselines/bevsplat/`, then `bash third_party/BevSplat/scripts/bootstrap_cuda.sh` in a compatible env; run authors' VIGOR eval smoke.

After both native numbers land, resume §3 (FG² Fixtor zero-shot) then §4.


## Run metrics

### `bevsplat_zero`
```json
{
  "status": "blocked",
  "stage": "checkpoints",
  "bevsplat": {
    "repo": "third_party/BevSplat",
    "commit": "92bebde1be2d50da5c679e11cb43c14ddc70544a",
    "checkpoints_dir": "checkpoints/baselines/bevsplat",
    "available": {},
    "missing": [
      "VIGOR_same_no_GPS.pth",
      "VIGOR_same_GPS.pth.pth",
      "VIGOR_cross_no_GPS.pth.pth",
      "VIGOR_cross_GPS.pth.pth"
    ],
    "onedrive": "https://1drv.ms/f/c/86d953bfc66eb903/IgAP7P2tFzChR7rHeMuXIOq8AakOxR02eKMyI2Z7qsMjLxo?e=zaD0Fb",
    "license": "MIT (third_party/BevSplat/LICENSE)"
  },
  "manifest": "experiments/06_fg2_bevsplat/manifest.json",
  "n_smoke": 20,
  "blocker": "BevSplat VIGOR checkpoints are on OneDrive and were not downloadable non-interactively (https://1drv.ms/f/c/86d953bfc66eb903/IgAP7P2tFzChR7rHeMuXIOq8AakOxR02eKMyI2Z7qsMjLxo?e=zaD0Fb; HTTP 403 / reauth). Missing: ['VIGOR_same_no_GPS.pth', 'VIGOR_same_GPS.pth.pth', 'VIGOR_cross_no_GPS.pth.pth', 'VIGOR_cross_GPS.pth.pth']. Also needs CUDA rasterizer build (bash third_party/BevSplat/scripts/bootstrap_cuda.sh) and the VIGOR dataset for native reproduction (task \u00a71 stop rule)."
}
```

### `fg2_zero`
```json
{
  "status": "blocked",
  "stage": "load_cvm",
  "error": "No module named 'mmcv'"
}
```
