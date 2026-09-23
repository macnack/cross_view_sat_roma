# Eagle cluster setup for the fusion track

Scaffolding for running `scripts/train_fusion.py` on PCSS Eagle (proxima H100),
mirroring the pattern already proven in `~/Github/sat_roma`
(`docs/eagle_hpc_slurm_ssh_guide.md` there is the source for everything below).

**Status: UNTESTED.** This machine has no SSH key registered for
`eagle.man.poznan.pl` (`Permission denied (publickey,gssapi-keyex,gssapi-with-mic)`
when checked 2026-09-21), so none of this has actually been submitted. It is
built from the documented, verified pattern of a working sibling project, not
from a live run of this repo's job.

## One-time setup (needs Eagle access — do this yourself)

1. Build and push the image from a machine with Docker:
   ```
   docker build -t <dockerhub_user>/cross_view_sat_roma:latest .
   docker push <dockerhub_user>/cross_view_sat_roma:latest
   ```
   Fill that user name into `slurm/cross_view_sat_roma.def`'s `From:` line.
2. On Eagle, redirect the Singularity cache off the 1 GB home quota, then build the `.sif`:
   ```
   mkdir -p ~/<grant>/project_data/containers_$USER
   ln -s ~/<grant>/project_data/containers_$USER ~/.singularity
   srun --account=<grant> -p proxima-cpu --pty bash
   ./slurm/build_container.sh
   ```
3. Copy the Dur360BEV subset and a reference GeoTIFF/VRT to
   `~/<grant>/project_data/` or `scratch/` (via `rclone`, from an interactive
   job, not the login node — see the guide's Storage layout section). Data and
   checkpoints are never committed to this repo (`.gitignore`).

## Submitting a run

```
DATA_ROOT=~/<grant>/project_data/dur360bev \
ORTHO=~/<grant>/project_data/ortho/durham_2021_nlp_intensity_1m_uint8.tif \
sbatch slurm/train_fusion.sbatch --steps 20000
```
`configs/eagle.yaml` is `configs/default.yaml` with only `data.root` changed
to the in-container bind path — keep the two in sync by hand.

## Open questions to resolve before the first real run

- **Torch pin.** `environment.yml` pins `pytorch=2.4.*` / `cuda=12.1` for a
  conda-based setup, predating the knowledge (now in the sat_roma guide) that
  Eagle's GPU target is H100 (sm_90) on `proxima`. sat_roma's own lock floats
  torch forward and resolves 2.13.0+cu130 there; this repo's `Dockerfile`
  leaves torch **unpinned** for the same reason. Confirm the unpinned wheel
  actually gives good H100 throughput (or re-pin explicitly) before a long run
  — this is a real decision, not resolved here.
- **HF Hub access from a compute node.** `scripts/train_fusion.py` downloads
  the `0t1q66hy` checkpoint and the frozen DINOv3 backbones from Hugging Face
  Hub at start-up (`sat_roma_infer/weights.py`, `timm`'s `pretrained=True`).
  If proxima compute nodes don't reach the internet, pre-download into
  `$HF_HOME` on `project_data` and bind-mount + `export HF_HOME=...` in the
  job script, or pre-cache into the container image at build time.
- **Resource sizing.** `train_fusion.sbatch`'s `--mem`/`--time` are a starting
  guess (this repo's model is much smaller than sat_roma's own H100 jobs —
  7.6M trainable BEV params + an 84M-param fine-tuned decoder), not measured
  on real hardware. Check `seff <job_id>` after the first run and tighten.
- **Data staging path.** The bind mounts in `train_fusion.sbatch` assume the
  Dur360BEV subset and the reference raster already sit on `project_data`/
  `scratch` as flat paths; adjust to wherever they actually land in step 3
  above.
