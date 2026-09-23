# Eagle (PCSS, proxima H100) for this repo

Verified live on 2026-09-23 (`ssh eagle`, user `krupka.maciej`, grants `pl0467-01` and `pl1269-01`).
The generic pattern comes from `~/Github/sat_roma/docs/eagle_hpc_slurm_ssh_guide.md`; what is
different here: one single-GPU job script that takes a `CMD` string, and sat_roma's container plus
a pip overlay instead of an image of our own.

## Where things live (all on `storage_6` NFS project_data — see "storage_5" below)

| Item | Path |
|---|---|
| Checkout (branch `task03/eval-density-solver-pf`) | `/mnt/storage_6/project_data/pl1269-01/krupka_maciej/cross_view_sat_roma` |
| Poznań / Lantmäteriet tiles (`SAT_DATA_DIR`) | `/mnt/storage_6/project_data/pl1269-01/krupka_maciej/sat_data` (all years, pre-existing copy of `~/Github/sat_data`) |
| Mapillary Fixtor panoramas | `<checkout>/data/mapillary/Fixtor` (rsync'ed by `make eagle-sync`, 13 GB) |
| HF cache (`sat493m` + `mackop102/sat_roma` decoder `0t1q66hy`) | `/mnt/storage_6/project_data/pl1269-01/krupka_maciej/hf_cache/hub` |
| Container | `/mnt/storage_6/project_data/pl0467-01/container_mackop/sat_roma_2026-09-02.sif` (py 3.12, torch 2.13.0+cu130, timm 1.0.28, opencv 4.13, einops, kornia; **no rasterio/pyproj**) |
| pip overlay (rasterio, pyproj, pyyaml, einops, matplotlib, pytest) | `<checkout>/.pydeps` (`PYTHONPATH=src:.pydeps`) |
| Logs | `<checkout>/slurm/logs/<job>_<id>.log` |

`third_party/sat_roma_infer` and `third_party/Dur360BEV` are plain clones at the pinned commits, not
submodules: `git submodule update` failed on the cluster ("could not get a repository handle"), and
`git clone` needs `--template=/tmp/empty_tpl` to avoid copying hook samples onto the NFS mount.

## Submitting

From the checkout on the login node:

```bash
ssh eagle
cd /mnt/storage_6/project_data/pl1269-01/krupka_maciej/cross_view_sat_roma && git pull
make eagle-submit JOB=lift_A CMD="scripts/train_lift_splat.py --out experiments/05_lift_splat/fixtor_pose_nll \
   --ckpt checkpoints/05_lift_splat_fixtor_aug_best.pt --years 2025,2024,2021 --val-year 2025 --steps 2000 \
   --neighbour-radius 4 --neighbour-weight 0.5 --pose-nll-weight 0.5"
squeue -u $USER
tail -f slurm/logs/lift_A_*.log
```

`slurm/run.sbatch` sets the account (`pl1269-01`), partition (`proxima`), one H100, 8 CPUs, 64 GB,
12 h, and exports `SAT_DATA_DIR`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `HF_HUB_OFFLINE=1`,
`PYTHONPATH=src:.pydeps`, `SATROMA_INFER_DIR`. Override any of them with `sbatch --export=ALL,CMD=...,SB=...`
style environment, or edit the `#SBATCH` lines for a longer run. Jobs are independent: submit all
evaluations of a sweep at once.

From the laptop, `make eagle-sync` pushes `data/mapillary/Fixtor`, the manifests and the
`05_lift_splat_fixtor_*_best.pt` checkpoints to `EAGLE_DIR`. Pull results back with
`rsync -avP eagle:<checkout>/experiments/05_lift_splat/eval/ experiments/05_lift_splat/eval/`.

## One-time setup (done 2026-09-23; repeat only for a fresh checkout)

```bash
P=/mnt/storage_6/project_data/pl1269-01/krupka_maciej
cd $P && mkdir -p /tmp/empty_tpl && git clone --template=/tmp/empty_tpl git@github.com:macnack/cross_view_sat_roma.git
cd cross_view_sat_roma && git checkout task03/eval-density-solver-pf
git clone --template=/tmp/empty_tpl git@github.com:macnack/sat_roma_infer.git third_party/sat_roma_infer
(cd third_party/sat_roma_infer && git checkout a08af4c56f643d9b9911ca319a1ff4e43f5746c4)
git clone --template=/tmp/empty_tpl https://github.com/Tom-E-Durham/Dur360BEV.git third_party/Dur360BEV
(cd third_party/Dur360BEV && git checkout 29e5dfb12d98983949cd7b85ba353b601cdc5248)
# pip overlay: the container's venv has no pip, so bootstrap it into the overlay first (interactive node)
srun --account=pl1269-01 -p interactive --time=0:40:00 --ntasks=1 --cpus-per-task=4 --mem=16G --pty bash
SIF=/mnt/storage_6/project_data/pl0467-01/container_mackop/sat_roma_2026-09-02.sif
X="singularity exec --bind /mnt/storage_5,/mnt/storage_6 $SIF"
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && $X python /tmp/get-pip.py --target .pydeps
PYTHONPATH=.pydeps $X python -m pip install --no-cache-dir --target .pydeps rasterio pyproj pyyaml einops matplotlib pytest
rm -rf .pydeps/numpy .pydeps/numpy-*.dist-info .pydeps/numpy.libs     # keep the container's numpy
export HF_HOME=$P/hf_home HUGGINGFACE_HUB_CACHE=$P/hf_cache/hub       # internet works on interactive nodes
$X python -c "from huggingface_hub import hf_hub_download as d; [d('mackop102/sat_roma', f) for f in ('sat_roma_0t1q66hy_decoder.safetensors','sat_roma_0t1q66hy_config.json')]"
$X python -c "import timm; timm.create_model('vit_large_patch16_dinov3.sat493m', pretrained=True)"
PYTHONPATH=src:.pydeps SATROMA_INFER_DIR=$PWD/third_party/sat_roma_infer $X python -m pytest tests -q   # 61 passed
```

## Pitfall: commas in CMD

`sbatch --export=ALL,CMD="..."` splits the `--export` list on commas, so a CMD containing
`--years 2025,2024,2021` or `--seq-dists 0,2,5` is silently truncated at the first comma and the
job runs with defaults. `make eagle-submit` therefore passes `CMD` through the environment
(`CMD=... sbatch --export=ALL`). Always check the first log lines (`pose_nll_w`, `neigh`, `ortho
years`) against what you meant to submit.

## storage_5 (Lustre scratch): do not use it for this repo

On 2026-09-23 `/mnt/storage_5/scratch/pl0467-01/mackop` gave `Input/output error` while cloning,
`Disk quota exceeded` on a 350 MB Hugging Face download although `lfs quota` showed 9 GB of 20 TB
used, an rsync of 13 GB that silently left 284 KB on disk, and finally `Cannot send after transport
endpoint shutdown` on reads (`lfs quota` also reports "some devices may be not working or
deactivated"). Small writes (50 MB) succeeded, so the failures look like deactivated OSTs, not a
real quota. The NFS project_data on `storage_6` passed a 500 MB write test and holds everything now.

## Measured

Fill in after the first jobs: seconds per training step (batch 1, 896×448 ERP) and seconds per
evaluated manifest entry on an H100, from `slurm/logs/`.

## Older files

`train_fusion.sbatch`, `cross_view_sat_roma.def` and `build_container.sh` are the earlier
Durham-fusion container plan (own Docker image, never built). They are superseded by `run.sbatch`
for the Mapillary tracks and are kept only until the fusion track needs a job of its own.
