# Task 03 — Measure honestly, then densify the query, fix the solver, add the filter

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the n=48, self-selected Lift-Splat numbers with a defensible evaluation on Eagle, then answer in order: does learned depth beat flat IPM on Fixtor (Task 2), does a dense ground lift fix the starved query (Task 3), does a fixed-scale SE(2) solver cut the along-road tail (Task 4), and does a particle filter with a realistic prior reach the H6-style target (Task 5).

**Architecture:** One `build_query(cfg, mode)` factory produces the `(B, 1024, 14, 14)` query tokens the Sat-RoMa decoder expects from three interchangeable modes (`lift` = today's spherical Lift-Splat, `ipm` = RGB flat-ground IPM through the frozen `sat493m` encoder, `hybrid` = dense IPM ground features + learned depth for above-horizon tokens). One `scripts/eval_pose.py` scores any mode on an immutable manifest with bootstrap intervals and a chance row. The matcher wrapper gains a `solver` switch. A small SE(2) particle filter consumes the decoder's soft vote heatmap along a route.

**Tech Stack:** Python 3.12, torch 2.13+cu130 (Eagle container `sat_roma_2026-09-02.sif`), timm `vit_large_patch16_dinov3.sat493m`, sat-roma-infer (read-only), rasterio/pyproj (pip overlay), SLURM on `proxima` H100.

**Spec:** the assessment in this session (2026-09-23) summarised in `experiments/01_kickoff/REPORT.md`, `kickoff_bev_ortho_localization.html` §7–8, and `docs/decisions.md` entries 2026-09-22/23. Decision gates are restated per task below.

## Global constraints

- Every step is a script in `scripts/` + a `make` target + parameters in `configs/default.yaml`; each run writes a `config.yaml` snapshot next to its outputs (CLAUDE.md working rules; memory `rerun-via-makefile`).
- `third_party/` is read-only. Sat-RoMa is used through `bevloc.match.satroma` only.
- Never commit data or checkpoints. `data/`, `checkpoints/`, `.pydeps/`, `slurm/logs/` stay ignored.
- Ask before any design decision not in `docs/decisions.md`; append the agreed decision there with the date.
- Local GPU (12 GB) is for smoke tests only: `--steps 40`, `--n 5`. Every real run goes through `slurm/run.sbatch` on Eagle.
- `irAsBUKtGCfhPHuMbmOcLd` is never trained on (decision 2026-09-22). From this task on it is the **test** route; `IcRzj0wTLZX874qitxVsQa` is the **validation** route (checkpoint selection and dev numbers).
- Mapillary `computed_geometry` is a pose **proxy**; never call the green box GT in captions (decision 2026-09-23).
- Result images must be readable at a glance (memory `viz-must-be-readable`): bright, labelled, few tiles.
- Unit tests run with `make test` and need no data or GPU.

---

## File structure

| Path | Responsibility |
|---|---|
| `src/bevloc/data/mapillary.py` (modify) | `sat_data_root()`, `poznan_tiles(year)`; `MapillaryPairs._build()` + `sample_for()`; optional IPM query in samples |
| `src/bevloc/eval/report.py` (create) | bootstrap CI, pose summary, chance row |
| `scripts/eval_pose.py` (create) | manifest evaluation for any query mode and solver |
| `scripts/report_pose.py` (create) | collects `eval_*.json` into a Markdown table |
| `scripts/build_baseline_manifest.py` (modify) | `--seq` argument so the test manifest can be built |
| `src/bevloc/bev/ipm_sphere.py` (create) | ERP flat-ground IPM for a spherical camera, pixel-coordinate helper |
| `src/bevloc/model/query.py` (create) | `LiftQuery`, `IpmQuery`, `build_query` |
| `src/bevloc/model/hybrid_query.py` (create) | `HybridQuery`: dense ground grid_sample + learned above-horizon splat |
| `src/bevloc/model/lift_splat.py` (modify) | `min_elev_deg` band |
| `src/bevloc/match/se2.py` (create) | 2-point SE(2) RANSAC + rigid Procrustes |
| `src/bevloc/match/satroma.py` (modify) | `solver` switch |
| `src/bevloc/model/coarse.py` (modify) | `vote_heatmap()` factored out of `pose_heatmap_nll` |
| `src/bevloc/track/pf.py` (create) | SE(2) particle filter |
| `scripts/track_route.py` (create) | route tracking with PF vs per-frame RANSAC vs dead reckoning |
| `scripts/train_lift_splat.py` (modify) | `--query {lift,ipm,hybrid}`; uses `build_query` |
| `slurm/run.sbatch` (create), `slurm/README.md` (rewrite) | Eagle launcher |
| `configs/default.yaml` (modify) | `ipm`, `lift.query_mode`, `lift.min_elev_deg`, `matcher.solver`, `pf` sections |
| `Makefile` (modify) | targets listed per task |
| `tests/test_report.py`, `tests/test_ipm_sphere.py`, `tests/test_query.py`, `tests/test_hybrid_query.py`, `tests/test_se2.py`, `tests/test_pf.py`, `tests/test_mapillary_pairs.py` (create) | unit tests |

---

## Task 0: Eagle bootstrap (recipe from `~/Github/sat_roma`, verified live 2026-09-23)

Facts checked over `ssh eagle` this session (user `krupka.maciej`, key already registered, both grants visible):

| Item | Value |
|---|---|
| Scratch with Poznań tiles already present | `/mnt/storage_5/scratch/pl0467-01/mackop/sat_data` (14 geoportal folders incl. `geoportal_poznan_15km2_*`, years 2014–2025) |
| HF cache with `sat493m` already cached | `/mnt/storage_5/scratch/pl0467-01/mackop/hf_cache_global/hub` |
| Container to reuse | `/mnt/storage_6/project_data/pl0467-01/container_mackop/sat_roma_2026-09-02.sif` (py 3.12, torch 2.13.0+cu130, timm 1.0.28, opencv 4.13, einops, kornia). **No rasterio, no pyproj** in any container. |
| Internet from a compute node | yes (interactive node `e2023` reached huggingface.co) |
| Default SLURM account | `pl1269-01` |
| Home on compute nodes | broken (`$HOME` not mounted, sat_roma memory 2026-05-24). Never use `Path.home()` in code that runs on Eagle. |
| GPU partition | `proxima`, `--gpus-per-node=h100:1`; 5 idle + 40 mixed nodes at check time |

**Files:**
- Modify: `src/bevloc/data/mapillary.py` (add `sat_data_root`, `poznan_tiles`)
- Modify: `scripts/train_lift_splat.py:open_years`, `scripts/eval_lift_splat.py`, `scripts/build_baseline_manifest.py:open_year` (use `poznan_tiles`)
- Create: `slurm/run.sbatch`
- Rewrite: `slurm/README.md`
- Modify: `Makefile` (`eagle-submit`, `eagle-sync`)

**Interfaces:**
- Produces: `poznan_tiles(year: int) -> list[Path]` (raises `FileNotFoundError` with the searched root when < 9 tiles); env var `SAT_DATA_DIR`.

- [ ] **Step 1: Failing test for `poznan_tiles`**

`tests/test_mapillary_pairs.py`:
```python
"""Mapillary pairs helpers (no network, no GPU)."""
from __future__ import annotations

import pytest


def test_poznan_tiles_uses_env_root(tmp_path, monkeypatch):
    from bevloc.data.mapillary import poznan_tiles, sat_data_root
    monkeypatch.setenv("SAT_DATA_DIR", str(tmp_path))
    assert sat_data_root() == tmp_path
    for i in range(9):
        d = tmp_path / f"geoportal_poznan_15km2_e{i}_n{i}_gmix"
        d.mkdir()
        (d / "year_2025.tif").write_bytes(b"")
    assert len(poznan_tiles(2025)) == 9
    with pytest.raises(FileNotFoundError):
        poznan_tiles(2024)
```

- [ ] **Step 2: Run it, expect ImportError**

Run: `make test PYTEST_ARGS="tests/test_mapillary_pairs.py -v"` (or `PYTHONPATH=src:.pydeps $PY -m pytest tests/test_mapillary_pairs.py -v`)
Expected: FAIL, `cannot import name 'poznan_tiles'`.

- [ ] **Step 3: Implement in `src/bevloc/data/mapillary.py`** (top of file, after imports)

```python
import os


def sat_data_root() -> Path:
    """Root of the geoportal/lantmäteriet tile folders. Overridable for Eagle, where $HOME is not mounted."""
    return Path(os.environ.get("SAT_DATA_DIR", str(Path.home() / "Github/sat_data")))


def poznan_tiles(year: int) -> list[Path]:
    paths = sorted(sat_data_root().glob(f"geoportal_poznan_15km2_*/year_{int(year)}.tif"))
    if len(paths) < 9:
        raise FileNotFoundError(f"expected >= 9 Poznań tiles for {year} under {sat_data_root()}, found {len(paths)}")
    return paths
```

Replace the three `Path.home().glob(...)` call sites:
- `scripts/train_lift_splat.py:open_years`: `out[int(y)] = PoznanOrtho(poznan_tiles(y))`
- `scripts/eval_lift_splat.py`: `ortho = PoznanOrtho(poznan_tiles(a.year))`
- `scripts/build_baseline_manifest.py:open_year`: `return PoznanOrtho(poznan_tiles(year))`

- [ ] **Step 4: Run the test, expect PASS; run `make test` to confirm nothing else broke**

- [ ] **Step 5: Create `slurm/run.sbatch`**

```bash
#!/bin/bash
# Generic single-GPU job for this repo on Eagle (proxima H100), reusing sat_roma's container
# plus a pip overlay in ./.pydeps for rasterio/pyproj. Submit from the repo root on the login node:
#   sbatch --job-name=<name> --export=ALL,CMD="scripts/x.py --args" slurm/run.sbatch
#SBATCH --job-name=cvsr
#SBATCH --account=pl1269-01
#SBATCH --partition=proxima
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?run sbatch from the repo root}"
mkdir -p slurm/logs
S=/mnt/storage_5/scratch/pl0467-01/mackop
export SAT_DATA_DIR="${SAT_DATA_DIR:-$S/sat_data}"
export HF_HOME="${HF_HOME:-$S/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$S/hf_cache_global/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="src:.pydeps"
export SATROMA_INFER_DIR="$PWD/third_party/sat_roma_infer"
export PYTHONUNBUFFERED=1
SIF="${SIF:-/mnt/storage_6/project_data/pl0467-01/container_mackop/sat_roma_2026-09-02.sif}"
echo "[run] host=$(hostname) date=$(date -Is) CMD=$CMD"
singularity exec --nv --bind /mnt/storage_5,/mnt/storage_6 "$SIF" python $CMD
echo "[run] exit=$? date=$(date -Is)"
```

Makefile additions (next to `help`):
```make
EAGLE_DIR ?= /mnt/storage_5/scratch/pl0467-01/mackop/cross_view_sat_roma

eagle-sync: ## push Fixtor panoramas, manifests and best checkpoints to Eagle scratch (13 GB first time)
	rsync -avP data/mapillary/Fixtor eagle:$(EAGLE_DIR)/data/mapillary/
	rsync -avP experiments/06_fg2_bevsplat/manifest*.json eagle:$(EAGLE_DIR)/experiments/06_fg2_bevsplat/
	rsync -avP checkpoints/05_lift_splat_fixtor_*_best.pt eagle:$(EAGLE_DIR)/checkpoints/

eagle-submit: ## submit CMD="scripts/x.py ..." JOB=name as one H100 job (run on Eagle, repo root)
	mkdir -p slurm/logs   # SLURM does not create the --output directory itself
	sbatch --job-name=$(JOB) --export=ALL,CMD="$(CMD)" slurm/run.sbatch
```

- [ ] **Step 6: One-time cluster setup (login node; commands verbatim)**

```bash
# laptop: push the branch first
git push -u origin cursor/add-sat-roma-infer-submodule
ssh eagle
S=/mnt/storage_5/scratch/pl0467-01/mackop
cd $S && git clone --recurse-submodules git@github.com:macnack/cross_view_sat_roma.git
cd cross_view_sat_roma && git checkout cursor/add-sat-roma-infer-submodule
ls $S/sat_data/geoportal_poznan_15km2_*/year_2025.tif | wc -l      # expect 9
ls $S/sat_data/geoportal_poznan_15km2_*/year_2024.tif | wc -l      # expect 9
# pip overlay inside the container (interactive job, not the login node)
srun --account=pl1269-01 -p interactive --time=0:30:00 --ntasks=1 --cpus-per-task=4 --mem=16G --pty bash
cd $S/cross_view_sat_roma
SIF=/mnt/storage_6/project_data/pl0467-01/container_mackop/sat_roma_2026-09-02.sif
singularity exec --bind /mnt/storage_5,/mnt/storage_6 $SIF python -m pip install --no-cache-dir \
    --target .pydeps rasterio pyproj pyyaml einops matplotlib pytest
rm -rf .pydeps/numpy .pydeps/numpy-*.dist-info      # keep the container's numpy 2.2.6
PYTHONPATH=src:.pydeps singularity exec --bind /mnt/storage_5,/mnt/storage_6 $SIF \
    python -c "import rasterio, pyproj, cv2, timm, torch; print(torch.__version__, torch.cuda.is_available())"
PYTHONPATH=src:.pydeps SATROMA_INFER_DIR=$PWD/third_party/sat_roma_infer \
    singularity exec --bind /mnt/storage_5,/mnt/storage_6 $SIF python -m pytest tests -q
# pre-download the Sat-RoMa decoder into the shared cache (internet works on interactive nodes)
HF_HOME=$S/hf_home HUGGINGFACE_HUB_CACHE=$S/hf_cache_global/hub singularity exec $SIF python -c \
 "from huggingface_hub import hf_hub_download as d; [d('mackop102/sat_roma', f) for f in ('sat_roma_0t1q66hy_decoder.safetensors','sat_roma_0t1q66hy_config.json')]"
exit
```
Then from the laptop: `make eagle-sync` (13 GB, ~20 min on a good link).

- [ ] **Step 7: Smoke job on Eagle**

```bash
make eagle-submit JOB=smoke CMD="scripts/train_lift_splat.py --overfit 4 --steps 40 --out experiments/05_lift_splat/eagle_overfit4"
squeue -u $USER; tail -f slurm/logs/smoke_*.log
```
Expected: `step 40 TRAIN CE` well below the step-1 value (locally 8.0 → ~3), no `FileNotFoundError`, `torch.cuda.is_available()` True. Record seconds/step in `slurm/README.md`.

- [ ] **Step 8: Rewrite `slurm/README.md`** with the table above, the setup block, the submit pattern, the measured s/step, and a "what is different from sat_roma's submit.sh" paragraph (single-GPU, `CMD` string, pip overlay instead of a custom image). Keep the old `train_fusion.sbatch` untouched. Commit:

```bash
git add src/bevloc/data/mapillary.py scripts/train_lift_splat.py scripts/eval_lift_splat.py \
        scripts/build_baseline_manifest.py slurm/run.sbatch slurm/README.md Makefile tests/test_mapillary_pairs.py
git commit -m "Eagle launcher: container + pip overlay, SAT_DATA_DIR, generic run.sbatch"
```

---

## Task 1: Evaluation hygiene — manifests, bootstrap CIs, chance row, A-only run

**Why:** the 8.7 m result is n=48, the checkpoint was chosen by validation pose error on the same route it is reported on, and no chance level is printed. Centre-guess chance for the ±22.4 m window is median 17.9 m, R@5 4.0 %, R@10 15.6 %.

**Files:**
- Create: `src/bevloc/eval/report.py`, `scripts/eval_pose.py`, `scripts/report_pose.py`, `tests/test_report.py`
- Modify: `src/bevloc/data/mapillary.py` (`_build`, `sample_for`), `scripts/build_baseline_manifest.py` (`--seq`), `Makefile`, `docs/decisions.md`

**Interfaces:**
- Produces: `bootstrap_ci(values, stat, n_boot=1000, seed=0, alpha=0.05) -> (stat, (lo, hi))`; `summarise_pose(errors_m, n_boot=1000, seed=0) -> dict`; `centre_guess_errors(entries) -> list[float]`; `MapillaryPairs.sample_for(frame_id: str, ref_o: Oriented, year: int) -> dict` (same keys as `__getitem__`); `eval_*.json` schema `{"meta": {...}, "frames": [row...], "summary": {"peak": {...}, "means": {...}, "centre_guess": {...}}}`.

- [ ] **Step 1: Failing tests `tests/test_report.py`**

```python
"""Bootstrap summaries for pose evaluations."""
from __future__ import annotations

import numpy as np

from bevloc.eval.report import bootstrap_ci, centre_guess_errors, summarise_pose


def test_bootstrap_ci_brackets_the_statistic():
    v = np.random.default_rng(0).normal(10.0, 2.0, 500)
    med, (lo, hi) = bootstrap_ci(v, np.median, n_boot=300)
    assert lo <= med <= hi
    assert hi - lo < 1.0


def test_summarise_counts_misses_as_inf():
    s = summarise_pose([1.0, None, 3.0, 20.0], n_boot=50)
    assert s["n"] == 4 and s["matched"] == 3
    assert abs(s["recall@5m"] - 0.5) < 1e-9
    assert abs(s["recall@10m"] - 0.5) < 1e-9
    assert s["frac_gt_30m"] == 0.25          # the miss counts as > 30 m


def test_centre_guess_is_norm_of_crop_offset():
    assert centre_guess_errors([{"crop_offset_m": [3.0, 4.0]}]) == [5.0]
```

- [ ] **Step 2: Run, expect `ModuleNotFoundError: bevloc.eval.report`**

- [ ] **Step 3: Implement `src/bevloc/eval/report.py`**

```python
"""Summary statistics with percentile-bootstrap intervals for pose evaluations."""
from __future__ import annotations

import numpy as np


def _finite(values):
    return np.asarray([np.inf if (v is None or not np.isfinite(v)) else float(v) for v in values], float)


def bootstrap_ci(values, stat, n_boot=1000, seed=0, alpha=0.05):
    """(stat(values), (lo, hi)) percentile bootstrap. Misses (None/inf) stay inf, so they
    count against recall and pull the median up, never silently dropped."""
    v = _finite(values)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = np.array([stat(v[i]) for i in idx], float)
    return float(stat(v)), (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def summarise_pose(errors_m, n_boot=1000, seed=0):
    v = _finite(errors_m)
    med, med_ci = bootstrap_ci(v, np.median, n_boot, seed)
    out = {"n": int(len(v)), "matched": int(np.isfinite(v).sum()), "median_m": med, "median_ci": med_ci}
    for t in (5.0, 10.0):
        r, ci = bootstrap_ci(v, lambda a, t=t: float((a <= t).mean()), n_boot, seed)
        out[f"recall@{t:g}m"], out[f"recall@{t:g}m_ci"] = r, ci
    out["frac_gt_30m"] = float((v > 30.0).mean())
    return out


def centre_guess_errors(entries):
    """Error of 'predict the crop centre' per manifest entry; crop_offset_m is (right, up)."""
    return [float(np.hypot(*e["crop_offset_m"])) for e in entries]
```

- [ ] **Step 4: Tests pass. Commit** `git commit -am "eval: bootstrap summaries and centre-guess chance row"`

- [ ] **Step 5: Failing test for `MapillaryPairs.sample_for`** (append to `tests/test_mapillary_pairs.py`)

```python
import json

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin


def _synthetic_route(tmp_path):
    """One 500x500 m EPSG:2180 tile at 0.25 m (a 224 m crop fits inside) and one 64x32 ERP frame near its centre."""
    tile = tmp_path / "geoportal_poznan_15km2_e0_n0_gmix"
    tile.mkdir()
    img = np.random.default_rng(0).integers(1, 255, (3, 2000, 2000), np.uint8)
    with rasterio.open(tile / "year_2025.tif", "w", driver="GTiff", width=2000, height=2000, count=3,
                       dtype="uint8", crs="EPSG:2180", transform=from_origin(0.0, 500.0, 0.25, 0.25)) as ds:
        ds.write(img)
    seq = tmp_path / "seq"
    (seq / "images").mkdir(parents=True)
    cv2.imwrite(str(seq / "images" / "1.jpg"), np.full((32, 64, 3), 128, np.uint8))
    from pyproj import Transformer
    lon, lat = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True).transform(250.0, 250.0)
    (seq / "images.json").write_text(json.dumps([{
        "id": "1", "captured_at": 0, "computed_compass_angle": 0.0,
        "computed_geometry": {"coordinates": [lon, lat]}, "computed_rotation": [0.0, 0.0, 0.0]}]))
    return tile / "year_2025.tif", seq


def test_sample_for_uses_the_given_reference(tmp_path):
    from bevloc import config as C
    from bevloc.data.mapillary import MapillaryPairs, PoznanOrtho, load_frames
    from bevloc.data.ortho import Oriented, gt_homography
    tif, seq = _synthetic_route(tmp_path)
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    ortho = PoznanOrtho([tif])
    frames = load_frames([seq], ortho, margin_m=0.0)
    ds = MapillaryPairs(frames, {2025: ortho}, cfg, train=False, erp_size=(64, 32), years=[2025])
    ref_o = Oriented((240.0, 255.0), 7.0, 224 * 4, 0.25)     # deliberately off-centre and rotated
    s = ds.sample_for("1", ref_o, 2025)
    q = Oriented(frames[0]["_en"], ds._up_of(frames[0]), 224, 0.25)
    assert np.allclose(s["H"].numpy(), gt_homography(q, ref_o), atol=1e-4)
    assert s["erp"].shape == (1, 3, 32, 64) and s["ref"].shape == (3, 896, 896)
```

(`_up_of(fr)` is a one-line helper added below so the test does not re-derive grid convergence.)

- [ ] **Step 6: Refactor `MapillaryPairs._one` into `_build`, add `sample_for` and `_up_of`**

In `src/bevloc/data/mapillary.py`, replace the body of `_one` so that it only *samples* the reference and delegates:

```python
    def _up_of(self, fr):
        lon, lat = fr["computed_geometry"]["coordinates"]
        return grid_bearing(lon, lat, fr["computed_compass_angle"])[0]

    def _one(self, i):
        fr = self.frames[i]
        rng = (self.rng if self.train
               else np.random.default_rng([self.cfg.matcher.seed, int(fr["id"]) % (2**32)]))
        erp_q, R_q, up_q, en_q = self._load_erp_R_pose(fr, rng)
        g = self.cfg.grid
        query = Oriented(en_q, up_q, g.n, g.cell_m)
        year = int(rng.choice(self.years)) if self.train else int(self.years[0])
        scale, off, rot = self._pose_budget(rng)
        L = getattr(self.cfg, "lift", None)
        neg_frac = float(getattr(L, "neg_frac", 0.0) or 0.0) if L else 0.0
        negative = bool(self.train and neg_frac > 0 and rng.random() < neg_frac)
        if negative:
            sep = tuple(getattr(L, "neg_sep_frac", (0.70, 1.40)))
            ref_o = sample_negative_reference(query, rng, scale=scale, sep_frac=sep, max_rot_deg=rot)
        else:
            ref_o = sample_reference(query, rng, scale=scale, max_offset_frac=off, max_rot_deg=rot)
        return self._build(i, ref_o, year, rng, negative=negative, scale=scale,
                           loaded=(erp_q, R_q, up_q, en_q))

    def sample_for(self, frame_id, ref_o, year):
        """Deterministic sample for one frame with a GIVEN reference crop (manifest evaluation)."""
        i = next(k for k, fr in enumerate(self.frames) if str(fr["id"]) == str(frame_id))
        rng = np.random.default_rng([self.cfg.matcher.seed, int(frame_id) % (2**32)])
        return self._build(i, ref_o, int(year), rng, negative=False, scale=int(ref_o.size // self.cfg.grid.n))

    def _build(self, i, ref_o, year, rng, negative=False, scale=4, loaded=None):
        fr = self.frames[i]
        erp_q, R_q, up_q, en_q = loaded if loaded is not None else self._load_erp_R_pose(fr, rng)
        g = self.cfg.grid
        query = Oriented(en_q, up_q, g.n, g.cell_m)
        ref, valid = self.ortho[year].render(ref_o)
        if float(valid.mean()) < 0.5 or float((ref.sum(-1) > 0).mean()) < 0.5:
            raise RuntimeError(f"black / missing reference for {fr['id']} year {year} at {en_q}")
        ref = self._colour_jitter(ref, rng)
        H = gt_homography(query, ref_o).astype(np.float32)
        neigh = self._neighbor_indices(i)
        erps, Rs, se2s = [], [], []
        for j in neigh:
            if j == i:
                erp, R, up, en = erp_q, R_q, up_q, en_q
            else:
                erp, R, up, en = self._load_erp_R_pose(self.frames[j], rng)
            yaw, tx, ty = src_in_query_se2(en_q, up_q, en, up)
            erps.append(torch.from_numpy(erp).permute(2, 0, 1).float().div(255.0))
            Rs.append(torch.from_numpy(R))
            se2s.append(torch.tensor([yaw, tx, ty], dtype=torch.float32))
        return dict(
            id=fr["id"], year=year, scale=scale, negative=negative,
            erp=torch.stack(erps, 0), ref=torch.from_numpy(np.ascontiguousarray(ref)).permute(2, 0, 1).float().div(255.0),
            R_w2c=torch.stack(Rs, 0), se2=torch.stack(se2s, 0), H=torch.from_numpy(H),
            en=torch.tensor(en_q, dtype=torch.float64),
        )
```

- [ ] **Step 7: Tests pass; `make lift-overfit` locally reproduces the pre-refactor CE at step 40 (compare `experiments/05_lift_splat/overfit4/log.csv` last row before/after, same seed). Commit** `git commit -am "MapillaryPairs: _build/sample_for so a manifest can fix the reference crop"`

- [ ] **Step 8: `scripts/build_baseline_manifest.py --seq`**

Replace the `VAL_SEQ` constant usage with an argument; drop the `RESERVED in str(VAL_SEQ)` collision check and instead refuse training routes:

```python
    ap.add_argument("--seq", default="Fixtor/IcRzj0wTLZX874qitxVsQa",
                    help="route under data/mapillary used for this manifest")
    ...
    seq_dir = MAP_ROOT / a.seq
    from bevloc.data.mapillary import TRAIN_SEQS
    if seq_dir in TRAIN_SEQS:
        raise SystemExit(f"{a.seq} is a training route")
    all_frames = load_frames([seq_dir], ortho, margin_m=margin)
    meta["held_out_seq"] = seq_dir.name
```
(Do this first in the same step: move `MAP_ROOT`, `TRAIN_SEQS` and `VAL_SEQS` from `scripts/train_lift_splat.py` into `src/bevloc/data/mapillary.py`, add `TEST_SEQS = [MAP_ROOT / "Fixtor/irAsBUKtGCfhPHuMbmOcLd"]`, and import them in `train_lift_splat.py`, `eval_lift_splat.py` and this script; the eval script in Step 10 imports `MAP_ROOT` from the same place.)

Makefile:
```make
manifest-val:  ## 200-frame validation manifest on IcRzj (2025+2024)
	$(RUN) scripts/build_baseline_manifest.py --seq Fixtor/IcRzj0wTLZX874qitxVsQa \
		--out experiments/06_fg2_bevsplat/manifest.json --n 200
manifest-test: ## 200-frame TEST manifest on the reserved route irAsBUK (download it first: make mapillary IMAGE=<any image id of that sequence>)
	$(RUN) scripts/build_baseline_manifest.py --seq Fixtor/irAsBUKtGCfhPHuMbmOcLd \
		--out experiments/06_fg2_bevsplat/manifest_test.json --n 200
```
Download the test route (needs `MAPILLARY_TOKEN`; the CLI takes `--sequence`): `MAPILLARY_TOKEN=... $(RUN) -m mapillary_dl --sequence irAsBUKtGCfhPHuMbmOcLd --out data/mapillary/Fixtor`. Add a `mapillary-seq SEQ=` Makefile target for it. Then `make manifest-test`; check `manifest_overview.jpg` follows the roads. Rebuilding `manifest.json` must give byte-identical frames (per-frame RNG from id): `git diff --stat experiments/06_fg2_bevsplat/manifest.json` shows no change.

- [ ] **Step 9: Append to `docs/decisions.md`**

`- 2026-09-24 — Evaluation protocol: train = 4 Fixtor routes; validation/selection = IcRzj (manifest.json, 200 frames); TEST = irAsBUK (manifest_test.json, 200 frames, never used for selection). All tables report median, R@5, R@10 with 95 % bootstrap CIs and a centre-guess chance row. Reason: the 8.7 m result was n=48, selected on the same route it was reported on, and 1–3 m differences between runs were inside noise.`

- [ ] **Step 10: Write `scripts/eval_pose.py`**

```python
"""Pose evaluation of a query checkpoint on an immutable manifest (bootstrap CIs, chance row).

  make eval-pose CKPT=checkpoints/05_lift_splat_fixtor_seq_best.pt MANIFEST=experiments/06_fg2_bevsplat/manifest.json TAG=seq
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bevloc import config as C
from bevloc.baselines.common import load_manifest
from bevloc.data.mapillary import MAP_ROOT, MapillaryPairs, PoznanOrtho, load_frames, poznan_tiles
from bevloc.data.ortho import Oriented
from bevloc.eval.metrics import pose_errors
from bevloc.eval.report import centre_guess_errors, summarise_pose
from bevloc.match.satroma import SatRoMa
from bevloc.model.coarse import FeatureQueryMatcher, coarse_targets, ref_cell_validity
from bevloc.model.query import build_query, load_query_state

CELL_M = 4.0


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="experiments/06_fg2_bevsplat/manifest.json")
    ap.add_argument("--years", default="2025,2024")
    ap.add_argument("--n", type=int, default=0, help="0 = every entry; small values for smoke tests")
    ap.add_argument("--query", default="", help="lift | ipm | hybrid; default = the mode stored in the checkpoint")
    ap.add_argument("--solver", default="", help="srt | se2 (Task 4); default = cfg.matcher.solver")
    ap.add_argument("--seq-dists", default="", help="multi-frame splat distances, e.g. 0,2,5 (seq checkpoints)")
    ap.add_argument("--out", default="experiments/05_lift_splat/eval")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    cfg = C.load(a.config)
    L = cfg.lift
    if a.seq_dists.strip():
        L.seq_dists_m = [float(x) for x in a.seq_dists.split(",") if x.strip()]
    if a.solver:
        cfg.matcher.solver = a.solver
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    man = load_manifest(a.manifest)
    years = [int(y) for y in a.years.split(",")]
    entries = [e for e in man["frames"] if e["year"] in years]
    if a.n:
        entries = entries[: a.n]
    orthos = {y: PoznanOrtho(poznan_tiles(y)) for y in years}
    seq_dir = MAP_ROOT / "Fixtor" / man["meta"]["held_out_seq"]
    frames = load_frames([seq_dir], orthos[years[0]], margin_m=L.margin_m)
    ds = MapillaryPairs(frames, orthos, cfg, train=False, seed=cfg.train.seed, erp_size=tuple(L.erp_size), years=years)

    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    mode = a.query or state.get("mode", "lift")
    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=False)
    matcher.model.decoder.load_state_dict(state["decoder"], strict=False)
    query = build_query(cfg, mode).to(dev)
    load_query_state(query, state)
    query.eval()
    wraps = {}
    for tag, means in (("peak", False), ("means", True)):
        w = SatRoMa.from_config(cfg, use_means=means, min_valid_frac=L.min_patch_valid)
        w.m.model.decoder.load_state_dict(state["decoder"], strict=False)
        wraps[tag] = w

    rows = []
    for e in entries:
        ref_o = Oriented(tuple(e["crop_centre_en"]), e["crop_up_bearing_deg"], e["ref_size"], e["gsd_m"])
        s = ds.sample_for(e["frame_id"], ref_o, e["year"])
        batch = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        with torch.no_grad():
            f_q, frac = query(batch, matcher)
            f_s = matcher.reference_features(batch["ref"])
        H = np.asarray(e["H_gt"], float)
        rv = ref_cell_validity(batch["ref"])
        idx, use = coarse_targets(batch["H"], frac >= L.min_patch_valid, ref_valid=rv)
        mask = torch.nn.functional.interpolate(frac[:, None].float(), size=(cfg.grid.n, cfg.grid.n), mode="nearest")[0, 0]
        mask = (mask >= L.min_patch_valid).cpu().numpy()
        row = dict(frame_id=e["frame_id"], year=e["year"], centre_guess_m=float(np.hypot(*e["crop_offset_m"])))
        for tag, w in wraps.items():
            m = w.match_encoded(f_q, f_s[16], scale_factor=0.4, mask=mask, H_gt=H)
            err = pose_errors(m.H, H, cfg.grid.n, cfg.grid.cell_m) if m.H is not None else None
            row[f"pose_{tag}_m"] = None if err is None else err["position_m"]
            row[f"yaw_{tag}_deg"] = None if err is None else err["yaw_deg"]
            row[f"inliers_{tag}"] = m.inlier_ratio
            row["argmax_m"] = m.argmax_cells if m.argmax_cells is None else m.argmax_cells * CELL_M
        rows.append(row)
        print(f"  {row['frame_id']} y{row['year']}  peak {row['pose_peak_m']}  means {row['pose_means_m']}", flush=True)

    summary = {}
    for y in years:
        sub = [r for r in rows if r["year"] == y]
        summary[str(y)] = {
            "peak": summarise_pose([r["pose_peak_m"] for r in sub]),
            "means": summarise_pose([r["pose_means_m"] for r in sub]),
            "centre_guess": summarise_pose([r["centre_guess_m"] for r in sub]),
        }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval_{a.tag}_{Path(a.manifest).stem}.json"
    path.write_text(json.dumps({"meta": dict(ckpt=a.ckpt, manifest=a.manifest, mode=mode,
                                              solver=cfg.matcher.solver, seq_dists=list(L.seq_dists_m)),
                                "frames": rows, "summary": summary}, indent=2))
    for y, s in summary.items():
        p, c = s["peak"], s["centre_guess"]
        print(f"{y}: peak median {p['median_m']:.1f} m {p['median_ci']}  R@5 {p['recall@5m']:.2f} {p['recall@5m_ci']}  "
              f"R@10 {p['recall@10m']:.2f}  | centre-guess median {c['median_m']:.1f} m R@5 {c['recall@5m']:.2f}")
    print(f"wrote {path}")
    C.snapshot(cfg, out, dict(ckpt=a.ckpt, manifest=a.manifest, tag=a.tag))


if __name__ == "__main__":
    main()
```

`build_query` / `load_query_state` come from Task 2 Step 3; until then a two-line stub in `src/bevloc/model/query.py` wrapping `SphericalLiftSplat` with the `forward(batch, matcher)` contract is enough (Task 2 replaces it). `cfg.matcher.solver` defaults to `"srt"` (add to `configs/default.yaml` now; Task 4 reads it).

Makefile:
```make
eval-pose: ## score CKPT on MANIFEST (bootstrap CIs, centre-guess chance); TAG names the json
	$(RUN) scripts/eval_pose.py --config $(CONFIG) --ckpt $(CKPT) --manifest $(MANIFEST) --tag $(TAG) $(EVAL_ARGS)
pose-report: ## experiments/05_lift_splat/REPORT.md from every eval_*.json
	$(RUN) scripts/report_pose.py
```

- [ ] **Step 11: Local smoke**: `make eval-pose CKPT=checkpoints/05_lift_splat_fixtor_aug_best.pt MANIFEST=experiments/06_fg2_bevsplat/manifest.json TAG=smoke EVAL_ARGS="--n 5"`. Expected: 5 rows, a summary block per year, `centre_guess` median in the 10–20 m range.

- [ ] **Step 12: `scripts/report_pose.py`** — glob `experiments/05_lift_splat/eval/eval_*.json`, one Markdown row per (tag, manifest, year, solver) with `median [lo, hi]`, `R@5 [lo, hi]`, `R@10`, `>30 m`, `matched`, plus one `centre guess` row per manifest/year; write `experiments/05_lift_splat/REPORT.md` with a one-paragraph header stating the protocol from Step 9. Commit both scripts.

- [ ] **Step 13: Eagle runs (submit all at once; they are independent)**

```bash
# A-only: pose NLL, single frame, from aug best (separates A from B)
make eagle-submit JOB=lift_A CMD="scripts/train_lift_splat.py --out experiments/05_lift_splat/fixtor_pose_nll \
  --ckpt checkpoints/05_lift_splat_fixtor_aug_best.pt --years 2025,2024,2021 --val-year 2025 --steps 2000 \
  --neighbour-radius 4 --neighbour-weight 0.5 --pose-nll-weight 0.5"
# re-score the four existing checkpoints + A on both manifests
for T in multi years aug seq; do for M in manifest manifest_test; do
  make eagle-submit JOB=ev_${T}_${M} CMD="scripts/eval_pose.py --ckpt checkpoints/05_lift_splat_fixtor_${T}_best.pt \
     --manifest experiments/06_fg2_bevsplat/${M}.json --tag ${T} $([ $T = seq ] && echo --seq-dists 0,2,5)"
done; done
```
After `lift_A` finishes, submit its two evals with `--tag pose_nll`. Pull results: `rsync -avP eagle:$EAGLE_DIR/experiments/05_lift_splat/eval/ experiments/05_lift_splat/eval/` then `make pose-report`.

**Gate (write the verdict into REPORT.md):** a checkpoint "beats" another only if its median CI does not overlap the other's on the **test** manifest. If seq and pose_nll overlap, multi-frame is not established. If nothing beats the centre-guess row by more than the CI width, stop and say so before Task 2.

- [ ] **Step 14: Commit** `git add scripts/eval_pose.py scripts/report_pose.py experiments/05_lift_splat/REPORT.md experiments/05_lift_splat/eval/*.json docs/decisions.md Makefile configs/default.yaml && git commit -m "Manifest pose evaluation with bootstrap CIs; A-only pose-NLL run"`

---

## Task 2: Camera-only RGB-IPM baseline on Fixtor (does learned depth matter?)

**Why:** Durham camera IPM reached 16.7 m and Lift-Splat 8.7 m, but on different cities, data volumes and imagery years. No IPM query has ever been trained on Fixtor. If IPM through `sat493m` matches Lift-Splat, the depth head contributes nothing and H4 is answered.

**Files:**
- Create: `src/bevloc/bev/ipm_sphere.py`, `src/bevloc/model/query.py`, `tests/test_ipm_sphere.py`, `tests/test_query.py`
- Modify: `scripts/ipm_mapillary.py` (import from `ipm_sphere`), `src/bevloc/data/mapillary.py` (IPM sample keys), `scripts/train_lift_splat.py` (`--query`), `configs/default.yaml`, `Makefile`

**Interfaces:**
- `ground_pixel_coords(x, y, R_w2c, height_m, erp_hw) -> (mu, mv)` float32 arrays, ERP pixel coordinates of the ground point at ego `(x, y)` metres (x forward, y left). Same maths as `scripts/ipm_mapillary.py:ipm`.
- `ipm_erp(erp_rgb, R_w2c, height_m, n, cell_m, blind_radius_m=1.2) -> (img uint8 (n,n,3), valid bool (n,n))`.
- `build_query(cfg, mode: str) -> nn.Module` whose `forward(batch: dict, matcher: FeatureQueryMatcher) -> (f_q (B,1024,14,14), patch_frac (B,14,14))`; `load_query_state(query, state)` accepts old `{"lift": ...}` checkpoints and new `{"query": ..., "mode": ...}`.
- Sample keys added when `cfg.lift.query_mode == "ipm"`: `bev (3, n, n) float in [0,1]`, `bev_valid (n, n) bool`.

- [ ] **Step 1: Failing test `tests/test_ipm_sphere.py`**

```python
"""Flat-ground IPM for an equirectangular camera."""
from __future__ import annotations

import numpy as np

from bevloc.bev.ipm_sphere import ground_pixel_coords, ipm_erp

# R_w2c: cam x = east, cam y = -up, cam z = north (the convention of tests/test_lift_splat.py)
R_NORTH = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def test_ground_point_ahead_maps_to_erp_centre_column():
    h, W, H = 1.7, 1280, 640
    mu, mv = ground_pixel_coords(np.array([10.0]), np.array([0.0]), R_NORTH, h, (H, W))
    assert abs(mu[0] - W / 2) < 1e-6                       # straight ahead = centre column
    lat = -np.arctan2(h, 10.0)
    assert abs(mv[0] - (0.5 - lat / np.pi) * H) < 1e-6     # below the horizon by atan(h/10)


def test_ipm_marker_lands_in_the_forward_cell():
    h, W, H, n, cell = 1.7, 1280, 640, 64, 0.5
    erp = np.zeros((H, W, 3), np.uint8)
    mu, mv = ground_pixel_coords(np.array([10.0]), np.array([0.0]), R_NORTH, h, (H, W))
    erp[int(round(mv[0])) - 1: int(round(mv[0])) + 2, int(round(mu[0])) - 1: int(round(mu[0])) + 2] = 255
    img, valid = ipm_erp(erp, R_NORTH, h, n, cell)
    r, c = n // 2 - int(10.0 / cell), n // 2            # row 0 is forward, col 0 is left (cell centre 9.75 m)
    assert img[r, c].max() >= 200                       # bilinear sample inside the 3 px marker
    assert valid[r, c] and not valid[n // 2, n // 2]    # blind disc under the camera
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement `src/bevloc/bev/ipm_sphere.py`**

```python
"""Flat-ground IPM for a spherical (equirectangular) camera with a known height.

Camera axes: x right, y down, z forward. ``R_w2c`` maps world ENU to camera.
Ego BEV: x forward, y left; row 0 forward, column 0 left (bevloc.bev.grid convention).
"""
from __future__ import annotations

import cv2
import numpy as np


def ego_to_world_dirs(R_w2c):
    forward = R_w2c.T @ np.array([0.0, 0.0, 1.0])
    forward[2] = 0.0
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0], 0.0])
    return forward, left


def ground_pixel_coords(x, y, R_w2c, height_m, erp_hw):
    """ERP pixel (mu, mv) of the ground point at ego (x forward, y left) metres."""
    H, W = int(erp_hw[0]), int(erp_hw[1])
    forward, left = ego_to_world_dirs(np.asarray(R_w2c, float))
    x, y = np.asarray(x, float), np.asarray(y, float)
    p_w = np.stack([x * forward[0] + y * left[0], x * forward[1] + y * left[1],
                    np.full_like(x, -float(height_m))], -1).reshape(-1, 3)
    p_c = p_w @ np.asarray(R_w2c, float).T
    lon = np.arctan2(p_c[:, 0], p_c[:, 2])
    lat = np.arctan2(-p_c[:, 1], np.hypot(p_c[:, 0], p_c[:, 2]))
    mu = ((lon / (2 * np.pi) + 0.5) * W).astype(np.float32).reshape(x.shape)
    mv = ((0.5 - lat / np.pi) * H).astype(np.float32).reshape(x.shape)
    return mu, mv


def cell_centres(n, cell_m):
    c = (np.arange(n) - (n - 1) / 2.0) * cell_m
    x = -c[:, None] * np.ones((1, n))      # row 0 = forward
    y = -c[None, :] * np.ones((n, 1))      # col 0 = left
    return x, y


def ipm_erp(erp_rgb, R_w2c, height_m, n, cell_m, blind_radius_m=1.2):
    x, y = cell_centres(n, cell_m)
    mu, mv = ground_pixel_coords(x, y, R_w2c, height_m, erp_rgb.shape[:2])
    img = cv2.remap(erp_rgb, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    valid = np.hypot(x, y) >= blind_radius_m
    img[~valid] = 0
    return img, valid
```
Replace `ipm()` in `scripts/ipm_mapillary.py` with a call to `ipm_erp` (same output, `--height` unchanged) and re-run `make ipm-smoke`; `smoke.jpg` must look identical.

- [ ] **Step 4: Tests pass. Commit** `git commit -am "ipm_sphere: ERP flat-ground IPM as a library function"`

- [ ] **Step 5: Failing test `tests/test_query.py`**

```python
"""Query factory: every mode returns decoder-shaped tokens."""
from __future__ import annotations

import torch

from bevloc import config as C
from bevloc.model.query import IpmQuery, LiftQuery, build_query, load_query_state


class _FakeMatcher:
    """Stands in for FeatureQueryMatcher: encoder = 1x1 conv to 1024 at stride 16."""
    def __init__(self):
        self.conv = torch.nn.Conv2d(3, 1024, 16, stride=16)
    def image_query_features(self, img):
        with torch.no_grad():
            return self.conv(img)
    class model:  # noqa: N801 — mimics matcher.model.encoder(x)[16]
        @staticmethod
        def encoder(x):
            return {16: torch.nn.functional.avg_pool2d(x.repeat(1, 342, 1, 1)[:, :1024], 16)}


def test_build_query_modes_and_shapes():
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    m = _FakeMatcher()
    lift = build_query(cfg, "lift")
    assert isinstance(lift, LiftQuery)
    batch = dict(erp=torch.rand(1, 1, 3, 448, 896), R_w2c=torch.eye(3)[None, None], se2=torch.zeros(1, 1, 3))
    f_q, frac = lift(batch, m)
    assert f_q.shape == (1, 1024, 14, 14) and frac.shape == (1, 14, 14)
    ipm = build_query(cfg, "ipm")
    assert isinstance(ipm, IpmQuery)
    batch = dict(bev=torch.rand(1, 3, 224, 224), bev_valid=torch.ones(1, 224, 224, dtype=torch.bool))
    f_q, frac = ipm(batch, m)
    assert f_q.shape == (1, 1024, 14, 14) and float(frac.min()) == 1.0


def test_load_query_state_accepts_old_lift_checkpoints():
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    q = build_query(cfg, "lift")
    old = {"lift": q.lift.state_dict()}
    load_query_state(q, old)
    new = {"query": q.state_dict(), "mode": "lift"}
    load_query_state(q, new)
```

- [ ] **Step 6: Implement `src/bevloc/model/query.py`**

```python
"""Query builders: anything that turns a sample into the (B, 1024, 14, 14) tokens the decoder expects.

  lift   — spherical Lift-Splat (learned depth per ERP token)            [today's 05_lift_splat]
  ipm    — RGB flat-ground IPM picture through the frozen sat493m encoder [kick-off H2 lower bound]
  hybrid — dense IPM ground features + learned depth above the horizon   [Task 3]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bevloc.model.lift_splat import SphericalLiftSplat


def _lift_from_cfg(cfg, **kw):
    L, g = cfg.lift, cfg.grid
    args = dict(dim=L.dim, depth_bins=L.depth_bins, d_min=L.d_min, d_max=L.d_max,
                n=g.n, cell=g.cell_m, max_elev_deg=L.max_elev_deg)
    args.update(kw)
    return SphericalLiftSplat(**args)


class LiftQuery(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lift = _lift_from_cfg(cfg)

    def forward(self, batch, matcher):
        erp = batch["erp"]                                  # (B, T, 3, H, W)
        B, T = erp.shape[:2]
        with torch.no_grad():
            feats = [matcher.model.encoder(erp[:, t])[16] for t in range(T)]
        hw = erp.shape[-2:]
        if T == 1:
            return self.lift(feats[0], batch["R_w2c"][:, 0], erp_hw=hw)
        return self.lift.forward_multiframe(torch.stack(feats, 1), batch["R_w2c"], batch["se2"], erp_hw=hw)


class IpmQuery(nn.Module):
    """No parameters of its own: the picture goes through the frozen encoder; the decoder is what trains."""
    def __init__(self, cfg):
        super().__init__()
        self.patch = 16

    def forward(self, batch, matcher):
        f_q = matcher.image_query_features(batch["bev"])
        frac = F.avg_pool2d(batch["bev_valid"].float()[:, None], self.patch)[:, 0]
        return f_q, frac


def build_query(cfg, mode):
    if mode == "lift":
        return LiftQuery(cfg)
    if mode == "ipm":
        return IpmQuery(cfg)
    if mode == "hybrid":
        from bevloc.model.hybrid_query import HybridQuery
        return HybridQuery(cfg)
    raise ValueError(f"unknown query mode {mode!r}")


def load_query_state(query, state):
    if "query" in state:
        query.load_state_dict(state["query"])
    elif "lift" in state and hasattr(query, "lift"):
        query.lift.load_state_dict(state["lift"])
    elif "lift" in state and hasattr(query, "load_lift_weights"):
        query.load_lift_weights(state["lift"])
```

- [ ] **Step 7: Tests pass. Wire the trainer**

`scripts/train_lift_splat.py`: add `--query` (default `cfg.lift.query_mode`, config default `lift`); replace `lift = SphericalLiftSplat(...)` with `query = build_query(cfg, a.query).to(dev)`; in `step()` replace the two-branch encode block with `f_q, patch_frac = query(batch, matcher)`; checkpoints save `{"query": query.state_dict(), "mode": a.query, "decoder": ..., "step": ..., "val": ...}`; warm-start through `load_query_state`. Optimiser: `groups = [{"params": [p for p in query.parameters()], "lr": L.lr_lift}]` only if the list is non-empty. `MapillaryPairs._build`: when `cfg.lift.query_mode == "ipm"` (or `batch` needs it), compute from the **full-resolution** query ERP before the resize:

```python
        if getattr(self.cfg.lift, "query_mode", "lift") == "ipm":
            bev, bev_valid = ipm_erp(erp_full, R_q, self.cfg.ipm.height_m, g.n, g.cell_m, self.cfg.ipm.blind_radius_m)
            bev = self._colour_jitter(bev, rng)
            out["bev"] = torch.from_numpy(bev).permute(2, 0, 1).float().div(255.0)
            out["bev_valid"] = torch.from_numpy(bev_valid)
```
(`_load_erp_R_pose` must return the full-resolution image as well; keep the resized one for the lift path.) `collate` stacks `bev`/`bev_valid` when present. Config: `ipm: {height_m: 1.7, blind_radius_m: 1.2}`, `lift.query_mode: lift`.

Parity check: `make lift-overfit` reproduces the pre-refactor step-40 CE (same seed). Then commit: `git commit -am "Query factory: lift | ipm; trainer and eval take --query"`.

- [ ] **Step 8: Local smoke of the IPM path**: `$(RUN) scripts/train_lift_splat.py --query ipm --overfit 4 --steps 40 --out experiments/05_lift_splat/ipm_overfit4`. Expected: CE falls (the decoder alone can overfit 4 frames). Save one `bev` panel next to its reference for the record with `scripts/viz_lift_aug.py`-style tiling (label "IPM h=1.7 m, camera only").

- [ ] **Step 9: Eagle runs**

```bash
make eagle-submit JOB=ipm_train CMD="scripts/train_lift_splat.py --query ipm --out experiments/05_lift_splat/fixtor_ipm \
  --years 2025,2024,2021 --val-year 2025 --steps 3000 --neighbour-radius 4 --neighbour-weight 0.5 --pose-nll-weight 0.5"
# then, for M in manifest manifest_test:
make eagle-submit JOB=ev_ipm CMD="scripts/eval_pose.py --ckpt checkpoints/05_lift_splat_fixtor_ipm_best.pt --manifest experiments/06_fg2_bevsplat/manifest_test.json --tag ipm"
```
Makefile: `ipm-train`, reuse `eval-pose`.

**Gate:** compare `ipm` with the best `lift` row from Task 1 on the test manifest. Non-overlapping CIs in favour of lift → learned depth is doing work, keep it inside the hybrid (Task 3). Overlapping or IPM better → the depth head is dropped from the hybrid and H4's "learned lifting" row is filled with this result. Append the verdict to `docs/decisions.md` and to `REPORT.md`.

---

## Task 3: Dense hybrid query — IPM ground features at sub-patch resolution + learned lift above the horizon

**Why:** the layer sheet shows the Lift-Splat BEV as a radial star of dots (1568 tokens × 16 bins into 50k cells), `f_q` as a smooth blob and the certainty map as the same band every frame. The decoder is reading layout, not content. Ground is where the camera sees most and where IPM is exact; decision 2026-09-19 already says the lift must be `grid_sample` at sub-patch resolution.

**Files:**
- Create: `src/bevloc/model/hybrid_query.py`, `tests/test_hybrid_query.py`, `scripts/viz_hybrid.py`
- Modify: `src/bevloc/model/lift_splat.py` (`min_elev_deg`), `src/bevloc/model/query.py` (already dispatches), `configs/default.yaml` (`lift.min_elev_deg`, `lift.ground_max_range_m`), `Makefile`

**Interfaces:**
- `SphericalLiftSplat(..., min_elev_deg=-90.0)`: tokens with `elev < min_elev` are not splatted (default keeps today's behaviour).
- `HybridQuery(cfg)`: `forward(batch, matcher) -> (f_q, frac)`; `load_lift_weights(lift_state_dict)` copies `depth_head`, `feat_proj`, `bev_head` from an old Lift-Splat checkpoint.
- Internals: `dense_ground(f_proj (B,dim,h,w), R_w2c (B,3,3), erp_hw) -> (g (B,dim,n,n), ground_mask (B,n,n))`.

- [ ] **Step 1: Failing test `tests/test_hybrid_query.py`**

```python
"""Hybrid query: exact ground placement by IPM, learned placement above the horizon."""
from __future__ import annotations

import numpy as np
import torch

from bevloc import config as C
from bevloc.bev.ipm_sphere import ground_pixel_coords
from bevloc.model.hybrid_query import HybridQuery

R_NORTH = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]])


def _cfg():
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    cfg.lift.dim = 8
    return cfg


def test_dense_ground_samples_the_token_under_the_ground_point():
    cfg = _cfg()
    q = HybridQuery(cfg)
    h, w, H, W = 28, 56, 448, 896
    f = torch.zeros(1, 8, h, w)
    # put a one-hot channel on the token that sees the ground 10 m ahead
    mu, mv = ground_pixel_coords(np.array([10.0]), np.array([0.0]), R_NORTH[0].numpy(), cfg.ipm.height_m, (H, W))
    f[0, 3, int(mv[0] // 16), int(mu[0] // 16)] = 1.0
    g, mask = q.dense_ground(f, R_NORTH, (H, W))
    n, cell = cfg.grid.n, cfg.grid.cell_m
    r, c = n // 2 - int(round(10.0 / cell)), n // 2
    assert mask[0, r, c]
    assert g[0, 3, r, c] >= 0.25                    # bilinear weight of the one-hot token (>= 1/4 anywhere inside it)
    assert int(g[0, :, r, c].argmax()) == 3
    assert g[0, 3].sum() > 0 and g[0, :3].abs().sum() == 0


def test_forward_shapes_and_valid_fraction():
    cfg = _cfg()
    q = HybridQuery(cfg)
    class M:  # fake matcher: encoder returns zeros at stride 16 in the lift's input dim
        class model:
            @staticmethod
            def encoder(x):
                return {16: torch.zeros(x.shape[0], 1024, x.shape[-2] // 16, x.shape[-1] // 16)}
    batch = dict(erp=torch.rand(1, 1, 3, 448, 896), R_w2c=R_NORTH[None], se2=torch.zeros(1, 1, 3))
    f_q, frac = q(batch, M)
    assert f_q.shape == (1, 1024, 14, 14)
    assert 0.5 < float(frac.mean()) <= 1.0            # most of the 56 m grid is ground within range
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: `min_elev_deg` in `SphericalLiftSplat`**

Constructor gains `min_elev_deg=-90.0`, stores `self.min_elev = math.radians(min_elev_deg)`; both `forward` and `forward_multiframe` compute `elev_ok = (elev <= self.max_elev) & (elev >= self.min_elev)`. Existing tests must still pass unchanged.

- [ ] **Step 4: Implement `src/bevloc/model/hybrid_query.py`**

```python
"""Dense hybrid query: IPM places ground features exactly, a depth head places the rest.

Ground cells (within ``ground_max_range_m``) bilinearly sample the projected ERP tokens at the
sub-patch position of their ground point (decision 2026-09-19). Tokens looking above the ground
band, elev in [min_elev_deg, max_elev_deg], are splatted along learned depth bins exactly as in
SphericalLiftSplat (walls, facades). Both land in one accumulator; the BEV head is shared.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bevloc.bev.ipm_sphere import cell_centres, ground_pixel_coords
from bevloc.model.lift_splat import SphericalLiftSplat


class HybridQuery(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        L, g = cfg.lift, cfg.grid
        self.n, self.cell, self.height = int(g.n), float(g.cell_m), float(cfg.ipm.height_m)
        self.blind = float(cfg.ipm.blind_radius_m)
        self.max_range = float(getattr(L, "ground_max_range_m", g.n * g.cell_m / 2))
        # the ground band ends where a ray at max_range meets the ground
        band_lo = -math.degrees(math.atan2(self.height, self.max_range))
        self.lift = SphericalLiftSplat(dim=L.dim, depth_bins=L.depth_bins, d_min=L.d_min, d_max=L.d_max,
                                       n=g.n, cell=g.cell_m, max_elev_deg=L.max_elev_deg,
                                       min_elev_deg=max(band_lo, float(getattr(L, "min_elev_deg", band_lo))))
        x, y = cell_centres(self.n, self.cell)
        self.register_buffer("cell_x", torch.from_numpy(x).float(), persistent=False)
        self.register_buffer("cell_y", torch.from_numpy(y).float(), persistent=False)

    def load_lift_weights(self, lift_state):
        self.lift.load_state_dict(lift_state)

    def dense_ground(self, f_proj, R_w2c, erp_hw, se2=None):
        """Bilinear sample of projected tokens at every ground cell. Returns (B, dim, n, n), (B, n, n) mask."""
        B = f_proj.shape[0]
        H, W = int(erp_hw[0]), int(erp_hw[1])
        x, y = self.cell_x, self.cell_y
        if se2 is not None:                       # query-ego cell -> source-ego cell
            yaw, tx, ty = se2[:, 0], se2[:, 1], se2[:, 2]
            xs = x[None] - tx[:, None, None]
            ys = y[None] - ty[:, None, None]
            c, s = torch.cos(-yaw)[:, None, None], torch.sin(-yaw)[:, None, None]
            x, y = c * xs - s * ys, s * xs + c * ys
        else:
            x, y = x[None].expand(B, -1, -1), y[None].expand(B, -1, -1)
        grids, masks = [], []
        for b in range(B):
            mu, mv = ground_pixel_coords(x[b].cpu().numpy(), y[b].cpu().numpy(), R_w2c[b].cpu().numpy(),
                                         self.height, (H, W))
            u = torch.from_numpy(mu).to(f_proj.device) / W * 2 - 1
            v = torch.from_numpy(mv).to(f_proj.device) / H * 2 - 1
            grids.append(torch.stack([u, v], -1))
            rng = torch.hypot(x[b], y[b])
            masks.append((rng >= self.blind) & (rng <= self.max_range))
        grid = torch.stack(grids, 0).to(f_proj.dtype)
        g = F.grid_sample(f_proj, grid, mode="bilinear", padding_mode="border", align_corners=False)
        mask = torch.stack(masks, 0)
        return g * mask[:, None].to(g.dtype), mask

    def _accumulate(self, f_erp, R_w2c, erp_hw, se2=None):
        lift = self.lift
        B, _, h, w = f_erp.shape
        f_proj = lift.feat_proj(f_erp)
        g, gmask = self.dense_ground(f_proj, R_w2c, erp_hw, se2)
        alpha = lift.depth_head(f_erp).softmax(1).permute(0, 2, 3, 1).reshape(B, h * w, -1)
        feat = f_proj.reshape(B, -1, h * w)
        dir_cam, elev = lift.token_rays(h, w, int(erp_hw[0]), int(erp_hw[1]), f_erp.device, f_erp.dtype)
        x, y = lift.ego_xy(dir_cam, R_w2c, lift.depths.to(f_erp.dtype))
        if se2 is not None:
            x, y = lift.apply_se2(x, y, se2)
        elev_ok = ((elev <= lift.max_elev) & (elev >= lift.min_elev))[None].expand(B, -1)
        bev, valid = lift.splat(feat, alpha, x, y, elev_ok)
        mass = valid.to(g.dtype)[:, None]
        num = g + bev * mass
        den = gmask.to(g.dtype)[:, None] + mass
        return num, den

    def forward(self, batch, matcher):
        erp = batch["erp"]
        B, T = erp.shape[:2]
        hw = erp.shape[-2:]
        with torch.no_grad():
            feats = [matcher.model.encoder(erp[:, t])[16] for t in range(T)]
        num = den = None
        for t in range(T):
            se2 = None if T == 1 else batch["se2"][:, t]
            n_t, d_t = self._accumulate(feats[t], batch["R_w2c"][:, t], hw, se2)
            num = n_t if num is None else num + n_t
            den = d_t if den is None else den + d_t
        bev = num / den.clamp_min(1e-6)
        f_q = self.lift.bev_head(bev)
        frac = F.avg_pool2d((den[:, 0] > 1e-6).float()[:, None], 16)[:, 0]
        return f_q, frac
```
`SphericalLiftSplat.splat` is called with `elev_ok` of shape `(B, N)`; that matches its current signature. Config additions: `lift.min_elev_deg: -3.5`, `lift.ground_max_range_m: 28.0`.

- [ ] **Step 5: Tests pass; commit** `git commit -am "HybridQuery: dense IPM ground grid_sample + learned above-horizon splat"`

- [ ] **Step 6: `scripts/viz_hybrid.py`** — for 3 validation frames: panels `ERP | ground mask (n×n, white = IPM-placed) | BEV PCA (num/den) | f_q PCA | reference with pose boxes`, using `pca_rgb`, `fit`, `pose_overlay` imported from `scripts/viz_lift_layers.py` (move those helpers into `src/bevloc/viz.py` first so both scripts import them). Bright, labelled, 280 px per tile. Makefile `hybrid-viz`. Look at it before training: the ground panel must show road texture, not dots.

- [ ] **Step 7: Local smoke**: `--query hybrid --overfit 4 --steps 40`. Expected: CE falls at least as fast as `lift` on the same 4 frames.

- [ ] **Step 8: Eagle runs (two, independent)**

```bash
make eagle-submit JOB=hyb_scratch CMD="scripts/train_lift_splat.py --query hybrid --out experiments/05_lift_splat/fixtor_hybrid \
  --years 2025,2024,2021 --val-year 2025 --steps 4000 --neighbour-radius 4 --neighbour-weight 0.5 --pose-nll-weight 0.5"
make eagle-submit JOB=hyb_warm CMD="scripts/train_lift_splat.py --query hybrid --out experiments/05_lift_splat/fixtor_hybrid_warm \
  --ckpt checkpoints/05_lift_splat_fixtor_aug_best.pt --years 2025,2024,2021 --val-year 2025 --steps 3000 \
  --neighbour-radius 4 --neighbour-weight 0.5 --pose-nll-weight 0.5"
```
Then `eval_pose` on both manifests for both, `make pose-report`, `make hybrid-viz` on the best.

**Gate:** hybrid beats both `lift` and `ipm` on the test manifest with non-overlapping median CIs, and its `f_q` PCA shows scene structure (roads, kerbs) rather than a smooth gradient. If it only matches `ipm`, the depth branch is not contributing and Task 3 ends with the IPM query as the camera-only method. Record the verdict in `docs/decisions.md`.

---

## Task 4: Fixed-scale SE(2) solver (H7)

**Why:** scale is known from the camera height, but `SatRoMa._ransac` fits a 4-DoF similarity (`model="sRT"`, scale bounds 0.25–3.0). The visible failure mode is a slide along the road; removing the scale DoF is the cheapest test of whether a wrong mode is being absorbed.

**Files:**
- Create: `src/bevloc/match/se2.py`, `tests/test_se2.py`
- Modify: `src/bevloc/match/satroma.py` (`solver`), `configs/default.yaml` (`matcher.solver: srt`), `scripts/eval_pose.py` (already passes `--solver`), `Makefile`

**Interfaces:**
- `rigid_fit(a (N,2), b (N,2)) -> (theta_rad, t (2,))` least-squares rotation+translation, scale fixed at 1.
- `se2_ransac(a, b, thresh, n_iter=500, seed=0, min_inliers=4) -> (H (3,3) | None, inliers (N,) bool)`.
- `SatRoMa(..., solver="srt"|"se2")`; `cfg.matcher.solver`.

- [ ] **Step 1: Failing test `tests/test_se2.py`**

```python
"""2-point SE(2) RANSAC with fixed metric scale."""
from __future__ import annotations

import numpy as np

from bevloc.match.se2 import rigid_fit, se2_ransac


def _pairs(theta_deg=12.0, t=(3.0, -2.0), n=60, outliers=24, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 14, (n, 2))
    c, s = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
    b = a @ np.array([[c, -s], [s, c]]).T + np.asarray(t) + rng.normal(0, noise, (n, 2))
    b[:outliers] = rng.uniform(0, 56, (outliers, 2))
    return a, b


def test_rigid_fit_recovers_exact_transform():
    a, b = _pairs(outliers=0, noise=0.0)
    th, t = rigid_fit(a, b)
    assert abs(np.degrees(th) - 12.0) < 1e-6 and np.allclose(t, (3.0, -2.0), atol=1e-6)


def test_se2_ransac_rejects_outliers():
    a, b = _pairs()
    H, inl = se2_ransac(a, b, thresh=0.5, n_iter=300)
    assert H is not None and inl.sum() >= 30 and not inl[:24].any()
    assert abs(np.degrees(np.arctan2(H[1, 0], H[0, 0])) - 12.0) < 0.3
    assert np.allclose([H[0, 2], H[1, 2]], (3.0, -2.0), atol=0.1)
    assert abs(np.hypot(H[0, 0], H[1, 0]) - 1.0) < 1e-9           # scale exactly one
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement `src/bevloc/match/se2.py`**

```python
"""Fixed-scale SE(2) consensus on Sat-RoMa correspondences (kick-off H7).

Two correspondences determine (theta, tx, ty); at inlier ratio 0.3 and p=0.99 that is 49 trials
against 567 for a homography. Scale is one because query and reference share the GSD and the
query is metric from the camera height."""
from __future__ import annotations

import numpy as np


def rigid_fit(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ca, cb = a.mean(0), b.mean(0)
    Hm = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    t = cb - R @ ca
    return theta, t


def _H(theta, t):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, t[0]], [s, c, t[1]], [0.0, 0.0, 1.0]])


def se2_ransac(a, b, thresh, n_iter=500, seed=0, min_inliers=4):
    a, b = np.asarray(a, float), np.asarray(b, float)
    N = len(a)
    if N < 2:
        return None, np.zeros(N, bool)
    rng = np.random.default_rng(seed)
    best, best_inl = None, np.zeros(N, bool)
    for _ in range(n_iter):
        i, j = rng.choice(N, 2, replace=False)
        if np.linalg.norm(a[i] - a[j]) < 1e-9:
            continue
        theta, t = rigid_fit(a[[i, j]], b[[i, j]])
        H = _H(theta, t)
        err = np.linalg.norm((np.c_[a, np.ones(N)] @ H.T)[:, :2] - b, axis=1)
        inl = err <= thresh
        if inl.sum() > best_inl.sum():
            best, best_inl = H, inl
    if best is None or best_inl.sum() < min_inliers:
        return None, best_inl
    theta, t = rigid_fit(a[best_inl], b[best_inl])
    return _H(theta, t), best_inl
```

- [ ] **Step 4: Tests pass; wire the switch in `SatRoMa._ransac`**

Constructor: `solver="srt"`; `from_config` reads `m.solver`. In `_ransac`, after `r = estimate_homography(...)` and the `n_modes < 4` guard:

```python
        if self.solver == "se2":
            from bevloc.match.se2 import se2_ransac
            H2, inl2 = se2_ransac(r.pts_A, tgt, thresh=self.reproj, n_iter=500, seed=self.seed)
            if H2 is None:
                return Match(None, None, n_modes, n_patches, n_multi, 0.0, argmax_cells)
            Hf, inl = H2, float(inl2.mean())
```
(`tgt` is defined just above as `r.means_B if self.use_means else r.peaks_B`; move the `tgt` line before this block.) Everything downstream (`convert_to_pixel_homography`, corners) is unchanged because `Hf` lives in the same patch/cell coordinates as `r.H`. `use_means=True` keeps feeding the distinct GMM means, so the "peak" and "means" rows still exist for both solvers.

- [ ] **Step 5: Smoke**: `make smoke` (synthetic pair) with `matcher.solver: se2` gives yaw error < 1° and position within a few px, like `srt`. Commit `git commit -am "matcher: fixed-scale SE(2) RANSAC solver option (H7)"`.

- [ ] **Step 6: Eagle**: re-score the best checkpoint of Tasks 1–3 on both manifests with `--solver se2 --tag <name>_se2`; `make pose-report`.

**Gate:** H7 holds if the SE(2) rows lower the median or `frac_gt_30m` on the test manifest beyond the CI width on at least the best query. Record either way in `docs/decisions.md`; if it holds, `matcher.solver: se2` becomes the default and the kick-off's 3.3 text is confirmed by data.

---

## Task 5: Sequence tracking with an SE(2) particle filter (H6 observation model, realistic prior)

**Why:** every system in the sat_roma research note that reaches sub-5 m adds a temporal filter; per-frame RANSAC alone does not. The filter also replaces the artificial ±22 m random window with a prior that comes from the previous estimate, which is the deployment condition.

**Files:**
- Create: `src/bevloc/track/__init__.py`, `src/bevloc/track/pf.py`, `scripts/track_route.py`, `tests/test_pf.py`
- Modify: `src/bevloc/model/coarse.py` (`vote_heatmap`), `configs/default.yaml` (`pf` section), `Makefile`

**Interfaces:**
- `vote_heatmap(gm_cls (B,K2,h,w), gm_certainty, matchable) -> (B, K, K) log-probabilities` (normalised over the grid); `pose_heatmap_nll` calls it.
- `SE2ParticleFilter(n=512, seed=0)` with `init(en, yaw_deg, sigma_xy, sigma_yaw)`, `predict(dx, dy, dyaw_deg, sigma_xy, sigma_yaw)` (ego-frame motion), `update(loglik: (n,) array)`, `estimate() -> (en (2,), yaw_deg)`, `ess()`; resamples systematically when `ess < 0.5 n`.
- Output `experiments/07_track/<route>_<tag>.json` with per-frame `pf_err_m`, `ransac_err_m`, `dr_err_m` and `ate` summaries; a trajectory plot.

- [ ] **Step 1: Failing test `tests/test_pf.py`**

```python
"""SE(2) particle filter on a synthetic route."""
from __future__ import annotations

import numpy as np

from bevloc.track.pf import SE2ParticleFilter


def test_pf_tracks_a_straight_route_with_gaussian_likelihood():
    rng = np.random.default_rng(0)
    pf = SE2ParticleFilter(n=400, seed=0)
    truth = np.array([100.0, 200.0]); yaw = 30.0
    pf.init(truth + rng.normal(0, 5, 2), yaw + 3.0, sigma_xy=10.0, sigma_yaw=5.0)
    errs = []
    for _ in range(40):
        truth = truth + 2.0 * np.array([np.sin(np.radians(yaw)), np.cos(np.radians(yaw))])
        pf.predict(2.0 + rng.normal(0, 0.2), rng.normal(0, 0.2), rng.normal(0, 0.5), sigma_xy=0.5, sigma_yaw=1.0)
        d = np.linalg.norm(pf.particles[:, :2] - truth, axis=1)
        pf.update(-0.5 * (d / 3.0) ** 2)
        en, _ = pf.estimate()
        errs.append(np.linalg.norm(en - truth))
    assert np.median(errs[10:]) < 1.5
    assert pf.ess() > 0.2 * 400
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement `src/bevloc/track/pf.py`**

```python
"""Minimal SE(2) particle filter in the map frame (E, N, up-bearing degrees CW from grid north)."""
from __future__ import annotations

import numpy as np


class SE2ParticleFilter:
    def __init__(self, n=512, seed=0):
        self.n = int(n)
        self.rng = np.random.default_rng(seed)
        self.particles = np.zeros((self.n, 3))      # e, n, yaw_deg
        self.logw = np.zeros(self.n)

    def init(self, en, yaw_deg, sigma_xy=10.0, sigma_yaw=5.0):
        self.particles[:, :2] = np.asarray(en, float) + self.rng.normal(0, sigma_xy, (self.n, 2))
        self.particles[:, 2] = float(yaw_deg) + self.rng.normal(0, sigma_yaw, self.n)
        self.logw[:] = 0.0

    def predict(self, dx, dy, dyaw_deg, sigma_xy=1.0, sigma_yaw=1.0):
        b = np.radians(self.particles[:, 2])
        fwd = np.stack([np.sin(b), np.cos(b)], 1)
        left = np.stack([-np.cos(b), np.sin(b)], 1)
        dxs = dx + self.rng.normal(0, sigma_xy, self.n)
        dys = dy + self.rng.normal(0, sigma_xy, self.n)
        self.particles[:, :2] += dxs[:, None] * fwd + dys[:, None] * left
        self.particles[:, 2] = (self.particles[:, 2] + dyaw_deg + self.rng.normal(0, sigma_yaw, self.n)) % 360.0

    def update(self, loglik):
        self.logw += np.asarray(loglik, float)
        self.logw -= self.logw.max()
        if self.ess() < 0.5 * self.n:
            self._resample()

    def weights(self):
        w = np.exp(self.logw)
        return w / w.sum()

    def ess(self):
        w = self.weights()
        return 1.0 / float((w ** 2).sum())

    def _resample(self):
        w = self.weights()
        pos = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.searchsorted(np.cumsum(w), pos)
        self.particles = self.particles[np.minimum(idx, self.n - 1)].copy()
        self.logw[:] = 0.0

    def estimate(self):
        w = self.weights()
        en = (w[:, None] * self.particles[:, :2]).sum(0)
        b = np.radians(self.particles[:, 2])
        yaw = np.degrees(np.arctan2((w * np.sin(b)).sum(), (w * np.cos(b)).sum())) % 360.0
        return en, float(yaw)
```

- [ ] **Step 4: Tests pass; factor `vote_heatmap` out of `pose_heatmap_nll`**

```python
def vote_heatmap(gm_cls, gm_certainty=None, matchable=None):
    """Certainty-weighted soft vote of all query patches: (B, K, K) log-probabilities over the reference grid."""
    B, K2, h, w = gm_cls.shape
    k = int(round(K2 ** 0.5))
    log_p = F.log_softmax(gm_cls.float().permute(0, 2, 3, 1), dim=-1)
    if gm_certainty is not None:
        c = gm_certainty[:, 0] if gm_certainty.ndim == 4 else gm_certainty
        weight = torch.sigmoid(c.float())
    else:
        weight = gm_cls.new_ones(B, h, w)
    if matchable is not None:
        weight = weight * matchable.float()
    wsum = weight.reshape(B, -1).sum(-1).clamp_min(1e-6)
    heat = (log_p * weight[..., None]).reshape(B, -1, K2).sum(1) / wsum[:, None]
    return F.log_softmax(heat, dim=-1).reshape(B, k, k)
```
`pose_heatmap_nll` keeps its guards and uses `heat = vote_heatmap(...).reshape(B, -1)` as `pred` input (the extra `log_softmax` is a constant shift per row, so the CE and argmax are unchanged; verify with a test that `pose_heatmap_nll` on a fixed random input returns the same value before and after the refactor, to 1e-5).

- [ ] **Step 5: `scripts/track_route.py`**

Loop over the route in `captured_at` order, every `--stride` frames (default 1):
1. Prior: first frame initialised at the proxy pose + N(0, 10 m / 5°). Motion between consecutive frames: `src_in_query_se2(en_prev, up_prev, en_cur, up_cur)` from the proxy **plus** N(0, `pf.sigma_xy`, `pf.sigma_yaw`) noise, applied through `pf.predict` (this stands in for wheel/visual odometry; say so in the output json).
2. Reference crop centred on the current PF estimate, up-bearing = PF yaw estimate, size `cfg.grid.n * cfg.reference.scale`, year `--year`.
3. Query via `build_query` from `--ckpt`, decoder output `gm_cls`, `heat = vote_heatmap(gm, cert, matchable)[0]` (K×K, cell = 16 px = 4 m).
4. Particle log-likelihood: convert each particle's EN to crop pixels with `inv(ref_o.px_to_world)`, to cell coordinates `/16`, bilinear lookup in `heat` (outside the crop → `heat.min()`); `pf.update(loglik * cfg.pf.temperature)` with `temperature` default 1.0; yaw observation from the compass proxy with σ = `cfg.pf.sigma_yaw_obs` (5°): `pf.update(-0.5 * ((particles_yaw - yaw_obs) / σ)²)`.
5. Log `pf_err_m` (estimate vs proxy), `ransac_err_m` (per-frame `SatRoMa.match_encoded` pose on the same crop, `pose_from_homography` from `bevloc.baselines.common`), `dr_err_m` (dead reckoning = the noisy motion integrated alone).
6. Save json + `track_<route>.jpg`: trajectory overlay on the orthophoto (proxy green, PF red, RANSAC blue dots) with a 100 m scale bar, plus an error-vs-frame plot.

Config `pf: {n: 512, sigma_xy: 1.0, sigma_yaw: 1.0, sigma_yaw_obs: 5.0, temperature: 1.0, init_sigma_xy: 10.0, init_sigma_yaw: 5.0}`. Makefile `track-route ROUTE=Fixtor/irAsBUKtGCfhPHuMbmOcLd CKPT=... YEAR=2025 TAG=...`.

- [ ] **Step 6: Local smoke** on 30 frames of IcRzj with `--stride 5`: PF error must not explode; the plot must be readable.

- [ ] **Step 7: Eagle**: full validation and test routes, 2025 and 2024, for the best checkpoint from Tasks 1–3 and the best solver from Task 4. Report median/95th percentile of `pf_err_m`, `ransac_err_m`, `dr_err_m`, and the fraction of frames the PF is within 5 m, into `experiments/07_track/REPORT.md`.

**Gate:** PF median ≤ 5 m on the test route with a realistic prior is the H6-style claim; between 5 and 10 m is "drift-bounding only" and the paper says so; a PF that is worse than per-frame RANSAC means the heatmap is over-confident and `pf.temperature` < 1 is tried once (0.5) before concluding.

---

## Out of scope for this plan (needs its own brainstorm after Task 3)

- **H5, orientation-aware hard negatives.** The contrastive term is the claimed method contribution and it sharpens the posterior of whatever query wins Task 3; writing it against today's starved query would measure noise. Inputs it will need are produced here: the winning query, the manifest evaluation, and (from the ideas below) an edge-orientation raster for the negatives.
- **FG²/BevSplat transfer (task 02)** remains blocked on VIGOR imagery, the BevSplat OneDrive checkpoints and an `mmcv` environment; nothing here changes that.

---

## Ideas explored: semantics and OpenStreetMap (recommendation, not tasks)

State on disk: no OSM data (the `osm/water` folder next to the Poznań tiles is empty), no `osmnx`/`pyrosm`/`geopandas` in the local env, `transformers` and `timm` available. Poznań OSM would come from the Geofabrik `wielkopolskie` extract (~150 MB) rasterised with `rasterio.features.rasterize` at 0.25 m, or from a bounded Overpass query per tile. This is a one-day job.

Ranked by expected payoff against the failure modes seen in this repo:

1. **Semantic segmentation of the panorama as the camera-only contact line (fits Task 3 directly).** A frozen segmenter trained on Mapillary Vistas or Cityscapes (SegFormer/Mask2Former via `transformers`) gives per-column building/road/sidewalk/vegetation/vehicle masks on the ERP. The lowest building pixel per column *is* the wall–ground contact row `v_c(u)` of kick-off §3.1, with no LiDAR and no Durham. Facade features above it collapse onto the contact cell exactly as designed; vehicles and pedestrians are masked before lifting (they are absent from the orthophoto); road/sidewalk define where IPM is exact. This tests H2 on Fixtor without waiting for the geometry head, and the LiDAR-taught head then becomes the *refinement* of a semantic initialisation rather than the only path. Cheapest, most on-story option.
2. **OSM building footprints as free labels for the geometry head (H3 without LiDAR).** Project OSM building polygons at the proxy pose into the ERP: the polygon's near edge gives the contact row per column, the edge azimuth gives the wall normal `(cos 2α, sin 2α)`. Labels are noisy (footprints ~1–2 m, proxy pose a few m) but plentiful over all of Poznań. Together with 1 this removes Durham from the training loop entirely and makes the VIGOR transfer claim cleaner. Also gives the paper an "OSM-supervised vs LiDAR-supervised head" row.
3. **OSM as an auxiliary reference channel.** Rasterise roads (by class), buildings, water, vegetation at 0.25 m, embed with a small CNN and add to the `sat493m` reference tokens before the decoder. This is the OrienterNet/MapLocNet reference modality; MapLocNet reaches roughly 78 % R@5 m on nuScenes from OSM alone according to the sat_roma research note. It makes the reference partly appearance-invariant, which is the cross-year story, and roads/footprints are precisely what a ground BEV sees. Keep the orthophoto primary; add OSM as an ablation row "ortho / OSM / both". Risk: the decoder was trained on `sat493m` token statistics; the added channel must be a residual with a small learned gain.
4. **OSM road orientation as the hard-negative source for H5.** The 20 m same-orientation negatives of kick-off §3.4 need edge orientation per reference cell; the road centreline azimuth from OSM is cleaner than a Sobel on the orthophoto and is available for every cell on a road. Prepare it during 3.
5. **Depth foundation models as the training-time teacher instead of LiDAR.** Depth Any Camera (kick-off ref. 13) on the ERP gives metric-ish depth with the height known; it could supervise the depth head or the contact line where no LiDAR exists. Lower priority: 1 and 2 give placement without a dense depth network at all, which is the paper's premise.

Recommendation: add 1 as a Task 3b (`bevloc/bev/semantic_contact.py`: segmenter wrapper, contact row per column, dynamic mask; a `hybrid` option `contact: semantic`), then 3+4 as one "OSM raster" task feeding both the reference channel and the H5 negatives, then 2. Each is a config-switchable variant of the existing query or reference, so the manifest evaluation from Task 1 scores it without new tooling.

---

## Task 6 (added 2026-09-23 after reading Loc², `docs/related/architecture_report_roma_losses.md` §6): ERP-token query, placement after matching

**Why:** on the test manifest the four lifted-BEV checkpoints sit at 14.5–17.8 m median against a
centre-guess of 18.3 m; the lift starves the query. Loc² (ICLR 2026) matches the panorama's own tokens
to the aerial tokens and lifts only the matched points afterwards, and its App. A.2 shows the BEV-first
variant of the same pipeline is substantially worse. Verified this evening: the released decoder accepts
a 28×56 query grid (`gm_cls` (1, 3136, 28, 56)), and the package exposes the per-mode correspondences
(`sat_roma.ransac.correspondence.find_gaussians` → `pts_A` as (col, row) query-patch indices, `means_B`,
`peaks_B`, `covs_B`) so query points can be re-placed before consensus.

**Files:**
- Create: `src/bevloc/model/erp_query.py` (`ErpQuery`, `erp_placement`), `tests/test_erp_query.py`
- Modify: `src/bevloc/model/coarse.py` (`coarse_targets(..., query_xy=None)`), `src/bevloc/model/query.py` (mode `erp`),
  `src/bevloc/match/satroma.py` (`match_placed`), `scripts/train_lift_splat.py` and `scripts/eval_pose.py`
  (use `query.placement(batch)` when present), `configs/default.yaml` (`erp:` section), `Makefile` (`erp-train`)

**Interfaces:**
- `erp_placement(h, w, erp_hw, R_w2c (B,3,3), height_m, n, cell_m, max_range_m) -> (xy (B,h,w,2) float, valid (B,h,w) bool)`:
  for every ERP token centre ray, the ground intersection at the camera height expressed as **virtual BEV
  pixel coordinates** (the 224 px / 0.25 m grid every other query uses: `u = n/2 − y/cell − 0.5`,
  `v = n/2 − x/cell − 0.5`, x forward, y left). Tokens above the horizon or beyond `max_range_m` are invalid.
- `ErpQuery(cfg).forward(batch, matcher) -> (f_q (B,1024,h,w), patch_frac (B,h,w))` with
  `f_q = encoder(erp)[16]` untouched and `patch_frac = valid.float()`; `ErpQuery.placement(batch) -> (xy, valid)`.
- `coarse_targets(H, patch_valid, ref_valid=None, ..., query_xy=None)`: when `query_xy` is given it replaces
  the patch-centre grid, so the GT cell of an ERP token is where its placed ground point lands.
- `SatRoMa.match_placed(f_q, f_s, xy (h,w,2), valid (h,w), scale_factor, H_gt=None) -> Match`: decoder →
  zero invalid patches → `find_gaussians` → replace each mode's `pts_A` by the placed point in grid units
  `k = (xy − (8 − 0.5)) / 16` → RANSAC (`ransac_init` similarity for `srt`, `se2_ransac` for `se2`) →
  `convert_to_pixel_homography(H, in_patch_dim=14, out_patch_dim=56, crop_res=(224,224), map_res=(896,896))`.
  The returned H is query-BEV-px → reference-px exactly like every other mode, so `pose_errors` applies.

- [ ] **Step 1: failing tests** `tests/test_erp_query.py`: (a) the token looking at the ground 10 m ahead
  (R = cam z north) places at virtual BEV px (u = n/2 − 0.5, v = n/2 − 40 − 0.5) at 0.25 m cells; tokens above
  the horizon are invalid; (b) `coarse_targets` with `query_xy` and an identity-scale H returns the cell
  containing that point; (c) `match_placed` on a synthetic `gm_cls` whose every valid token votes for the
  cell its placed point maps to under an injected similarity recovers that transform within 0.5 cells.
- [ ] **Step 2: implement**; the ERP validity mask is elevation-only (Mapillary has no ego mask).
- [ ] **Step 3: trainer/eval wiring**: `--query erp` (T must be 1); `step()` passes `query_xy` to
  `coarse_targets`; `eval_pose` and `track_route` call `match_placed` for this mode.
- [ ] **Step 4: local smoke** `--query erp --overfit 4 --steps 40`: CE must fall; `n` supervised tokens
  should be several hundred (against ~150–200 for the BEV queries).
- [ ] **Step 5: Eagle**: `erp_train` 3000 steps with the standard recipe (cross-year, hinge, pose NLL 0.5, decoder
  fine-tuned), evaluations on both manifests with `srt` and `se2` solvers.

**Gate:** ERP query beats the best lifted query on the **test** manifest with non-overlapping median CIs, and its
`>30 m` fraction drops. If it only matches, the hypothesis is not refuted but the front-end is not the lever
either, and the next suspect is the reference side (map GSD / cross-year appearance).
