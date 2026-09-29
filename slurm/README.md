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
`PYTHONPATH=src:.pydeps`, `SATROMA_INFER_DIR`. Override any of them by exporting the variable before
`make eagle-submit` (they are read with `${VAR:-default}`), or edit the `#SBATCH` lines for a longer run. Jobs are independent: submit all
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

## Following jobs from the laptop: `make eagle-watch`

`slurm/watch_list.txt` lists the jobs of the current campaign (`<jobid> <jobname> <grep pattern>`); `make
eagle-watch` prints the queue and then follows all of them at once (`slurm/watch_all.sh`, one
`slurm/watch_eval.sh` per job, lines prefixed with the job name), polling every two minutes, and exits when
the last one leaves the queue. Come back after hours away and run it again: finished jobs print their result
lines once, running ones continue. Update the list when submitting a new batch (the job ids are printed by
`make eagle-submit`). A single job: `bash slurm/watch_eval.sh <jobid> <jobname> [pattern]`.

## Syncing the checkout: `make eagle-pull`, not `git pull`

Submit at least a minute after `make eagle-pull`: on 2026-09-24 a job submitted 5 s after the pull ran the
*previous* version of `scripts/train_vigor.py` (the compute node's NFS attribute cache still held the old
file) and failed on a flag that no longer existed. Resubmitting a few minutes later ran the new file.

Jobs write result files (`experiments/**/*.json`) that are later committed from the laptop; a plain
`git pull` then refuses ("untracked working tree files would be overwritten"). `make eagle-pull` does
`git fetch` + `git reset --hard origin/<branch>`, which is safe because the cluster checkout never
carries local commits. Always run it before submitting, and check `git log -1` matches the laptop.
`bash slurm/watch.sh <name>_<jobid> ...` prints one line per finished job (exit status, errors,
evaluation summary) and the number still queued; poll it from a monitor loop.

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

## Measured (H100, 2026-09-23)

- Training, batch 1, 896×448 ERP, single frame: 20 steps in about 1 s of wall time between prints
  (roughly 0.05 s/step; the 48-frame validation every 200 steps dominates a 2000-step run).
- `eval_pose.py` on the 400-entry validation manifest (200 frames × 2 years): about 4 min per checkpoint.
- Job start-up (container, encoder load): about 40 s.

## Long runs: probe, epochs, resume (2026-09-29, PanoRoMa 100 epochs)

`make eagle-probe` (slurm/probe_vigor.sbatch, 32 CPUs, 30 min) runs `train_vigor.py --profile N` over
`PROBE_BATCHES` x `PROBE_WORKERS` x `PROBE_PREFETCH` x `PROBE_PIN`; `grep PROFILE` the log. Measured on one H100
(coarse erp_depth + head, cell 0.125; raw lines in experiments/13_panoroma_long/*.jsonl):

- GPU-bound at every batch: 7.8-8.1 samples/s from batch 16 up (float32), 13.7 with `--tf32`. Batch 48 leaves 10 %
  memory headroom, 64 OOMs; batch 32 = 54 GiB reserved (42 % headroom).
- Batch 32 step (3.96 s, float32): frozen DINOv3 on the 896 px reference 2.47 s, on the panorama 1.0 s (together
  88 %), decoder + head forward ~0.3 s, backward 0.22 s, AdamW 4 ms.
- Data: `--pin-memory` takes the data wait from 0.077 s (2 %) to 0.0003 s; 2 workers already keep up at float32.
  Per-sample CPU cost ~45-55 ms (depth PNG 19 ms, panorama JPEG 9 ms, reference 5 ms, float conversion the rest).

`--epochs E --save-every-epochs K --resume auto` makes a run resumable: `checkpoints/vigor_<tag>_resume.pt` (790 MiB:
weights + AdamW + RNG) is refreshed after every validation (= every epoch), `_ep{e:03d}.pt` are lean (263 MiB: head +
decoder, no encoder). A job that hits the 7-day `proxima` MaxTime is continued by the next job in an `afterany` chain
running the same command; a job that finds the run finished exits at once. Verified with smoke_a/smoke_b (2 + 1
epochs, experiments/13_panoroma_long/smoke_*.log).

Chaining (review 2026-09-29): `run.sbatch` requests 8 CPUs; pass `--cpus-per-task=16` in SBATCH_ARGS for 8 workers
(the probe had 32; 2 workers already kept up). `afterany` continues a segment that hit the time limit OR crashed
(a crash that repeats crashes the next segment too); the fine run waits with `afterok` on the coarse chain's LAST
segment, so it starts only if that segment exits 0 (finished, or found finished). A coarse run that needs more than
two segments times out the second one and the fine job never starts (DependencyNeverSatisfied): resubmit it. On the
fine run's later segments `--resume auto` wins over `--ckpt` (the warm start is loaded, then overwritten by the resume
file). `_last.pt`, `_best.pt`, `_ep*.pt` all live in `checkpoints/` of the repo, whatever `--out` is; `--out` only
holds the CSV log and `config.yaml` (+ a per-tag `config_<tag>.yaml`, since both runs share the folder).

## Multi-GPU: 4 x H100 on one node (DDP, 2026-09-29)

`make eagle-submit-ddp JOB= CMD= SBATCH_ARGS=` (slurm/run_ddp.sbatch: proxima, `--gpus-per-node=h100:4`, 64 CPUs,
480 GB, 7 days, `torchrun --standalone --nproc_per_node 4` inside the same container) runs a DDP-aware script;
`make eagle-probe-ddp PROBE_RUNS=... PROBE_ARGS=...` (slurm/probe_ddp.sbatch, 30 min) runs short probes. Only
`scripts/train_vigor.py` is DDP-aware (`bevloc.model.ddp`); without torchrun it runs exactly as before.

- `--batch` is the GLOBAL batch; each of the 4 ranks draws batch / 4 (printed: `DDP: global batch 32 = 4 ranks x 8
  per rank`). Steps per epoch, lr and epochs are those of one GPU at that batch: global 32 = the reviewed plan
  (1,315 steps per epoch, 131,500 for 100 epochs); global 128 = 32,800 steps (another optimisation, not the plan).
- The epoch permutation is sharded (rank r takes positions r::4); every loss term is reweighted by the rank's share
  of the GLOBAL token / sample count (review 2026-09-29: `Dist.global_loss`, one small all-reduce before backward), so
  the averaged gradient IS the single-GPU gradient of the global batch; gradients of the head + decoder are averaged by one
  all-reduce per step (the frozen encoder is a plain module on each rank), a non-finite step on one rank is skipped on
  all, rank 0 validates (the same 400 frames, `--val-batch` default min(batch, 32)) and writes the CSV, the config
  snapshot and every checkpoint (same keys as single-GPU ones). The resume file also stores the world size, the
  per-rank batch and all ranks' RNG; a resume with another GPU count is refused. A single-GPU resume file cannot be
  continued under DDP (refused) and vice versa.
- H100 switches (default off = every earlier run): `--tf32`, `--encoder-dtype bfloat16` (the frozen encoder under
  bfloat16 autocast, outputs back to float32; validation unchanged within noise), `--compile encoder`. Measured, with
  accuracy checks and the DDP-vs-1-GPU loss comparison: experiments/13_panoroma_long/PROFILE.md ("4 x H100").

| 4 x H100, global 32 | Samples/s | 100 epochs coarse | fine |
|---|---|---|---|
| float32 | 29.6 | ~41 h | ~41 h |
| `--tf32` | 45.8 | ~27 h | ~27 h |
| `--tf32 --encoder-dtype bfloat16 --compile encoder` (recommended) | 92.2 (fine 86.7) | ~14 h | ~15 h |

PanoRoMa 100-epoch runs (recommended setting; the fine run starts when the coarse job exits 0):

```bash
make eagle-submit-ddp JOB=pano_coarse_e100 SBATCH_ARGS="--time=2-00:00:00" CMD="scripts/train_vigor.py \
  --config configs/vigor_cell0125.yaml --query erp_depth --head --pose-nll-weight 0.5 --split samearea \
  --val-frac 0.2 --val-samples 400 --batch 32 --epochs 100 --save-every-epochs 10 --resume auto \
  --workers 8 --pin-memory --tf32 --encoder-dtype bfloat16 --compile encoder \
  --tag samearea_4city_erp_depth_cell0125_e100 --out experiments/13_panoroma_long"
make eagle-submit-ddp JOB=pano_fine_e100 SBATCH_ARGS="--time=2-00:00:00 --dependency=afterok:<coarse job id>" \
  CMD="scripts/train_vigor.py --config configs/vigor_cell00625_fine.yaml \
  --ckpt checkpoints/vigor_samearea_4city_erp_depth_cell0125_e100_last.pt --query erp_depth --head \
  --pose-nll-weight 0.5 --split samearea --val-frac 0.2 --val-samples 400 --batch 32 --epochs 100 \
  --save-every-epochs 10 --resume auto --workers 8 --pin-memory --tf32 --encoder-dtype bfloat16 --compile encoder \
  --tag samearea_4city_fine00625_erp_depth_e100 --out experiments/13_panoroma_long"
```

Precision of the earlier runs instead: drop `--tf32 --encoder-dtype bfloat16 --compile encoder` (float32, ~41 h per
run) or keep only `--tf32` (~27 h); use the default 7-day `--time` then. A segment that times out or crashes continues
with the same command (`--resume auto`); resubmit it with the same GPU count.

## Older files

`train_fusion.sbatch`, `cross_view_sat_roma.def` and `build_container.sh` are the earlier
Durham-fusion container plan (own Docker image, never built). They are superseded by `run.sbatch`
for the Mapillary tracks and are kept only until the fusion track needs a job of its own.
