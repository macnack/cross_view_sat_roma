# PanoRoMa long run: throughput profile on one Eagle H100

Status 2026-09-29. Goal (Maciej): train PanoRoMa for a Loc²-like budget (100 epochs), after (1) finding the largest
batch that fits, (2) checking whether the GPU waits on the CPU and tuning workers / prefetch. Raw measurements:
`probe_batch_8855575.jsonl` (one `PROFILE` line per batch size). Probe code: branch `panoroma/long-train`
(`scripts/train_vigor.py --profile`), clone on Eagle `/mnt/storage_6/project_data/pl1269-01/krupka_maciej/probe_panoroma`.

## Batch size (job 8855575; coarse config `configs/vigor_cell0125.yaml`, erp_depth + head, 10 steps each)

DataLoader 16 workers, prefetch 2, no pinned memory, 32 CPUs.

| Batch | Peak reserved memory | Step time | Data wait | Samples/s | GPU util |
|---|---|---|---|---|---|
| 4 | 8.6 GiB | 0.60 s | 2.1 % | 6.7 | 96 % |
| 8 | 14.9 GiB | 1.09 s | 1.8 % | 7.3 | 96 % |
| 16 | 28.2 GiB | 2.06 s | 1.9 % | 7.8 | 97 % |
| 32 | 54.1 GiB | 4.09 s | 2.3 % | 7.8 | 96 % |
| 48 | 83.7 GiB (10 % headroom) | 6.18 s | 1.8 % | 7.8 | 97 % |
| 64 | out of memory (93.1 GiB card) | — | — | — | — |

Per-sample CPU cost in a worker: ≈ 0.045 s (panorama decode 0.009, resize 0.002, depth PNG 0.019, reference tile 0.005).

## Readings

- **The GPU does not wait on the CPU**: data wait ≈ 2 % of the step and GPU utilisation 96–97 % at every batch size with
  16 workers / prefetch 2. More workers or prefetch cannot speed training up; 16 workers supply ≈ 350 samples/s of CPU
  capacity against 7.8 consumed.
- **Throughput is compute-bound and flat from batch 16**: ≈ 7.8 samples/s. A larger batch does not train faster; it
  only changes the optimisation (fewer, larger steps).
- **Batch 32** is the working choice: 54 GiB, 42 % headroom (batch 48 leaves only 10 %).
- **Wall time**: 100 epochs of the ≈ 42 k-sample four-city training list = 4.2 M samples ≈ 150 h on one H100 for the
  coarse run alone (the fine run is similar). Options under study: DDP over the 4 H100s of one proxima node, and caching
  the frozen DINOv3 encoder's features (the encoder runs on every sample although it never changes).

## Pending probes (submitted 2026-09-29 by the profiling agent)

- 8855805 `probe_pipe`: workers / prefetch / pinned-memory sweep at the chosen batch.
- 8855806 `probe_tf32`: TF32 matmuls.
- 8855807 `probe_fine`: second-pass config (`configs/vigor_cell00625_fine.yaml`) at the chosen batch.

Results are added here when they land.

## 4 x H100 on one node (DDP) and H100 settings (2026-09-29, branch `panoroma/ddp`)

`torchrun --standalone --nproc_per_node 4 scripts/train_vigor.py ... --batch B` (`make eagle-submit-ddp`,
`slurm/run_ddp.sbatch`; `bevloc.model.ddp`). `--batch` is the GLOBAL batch; each rank draws B / 4. Probe jobs
8856513, 8856805 (one proxima node, 64 CPUs, 8 DataLoader workers per rank, pinned memory; coarse config
`configs/vigor_cell0125.yaml`, erp_depth + head unless noted): `probe_ddp_<job>.jsonl` (one line per rank + one
`"aggregate": true` line per setting). Accuracy checks: `check_encoder.jsonl` (`train_vigor.py --check-encoder`).
Smoke / resume / loss-comparison logs: `ddp_check_smoke_8856566.log`, `ddp_fast_smoke_8856805.log`, CSVs in `ddp_cmp/`.

### H100 settings (samples/s per node; 1 GPU float32 = 7.8)

| Setting | Global batch (per rank) | Samples/s (node) | Speed-up vs 1 GPU fp32 | Step (s) | Accuracy check |
|---|---|---|---|---|---|
| 4 GPU, float32 | 32 (8) | 29.6 | 3.8x | 1.08 | same maths as 1 GPU (loss comparison below) |
| 4 GPU, `--tf32` | 32 (8) | 45.8 | 5.9x | 0.70 | top-1 12.64 vs 12.66 %, VCE pose 3.166 vs 3.166 m |
| 4 GPU, `--tf32 --encoder-dtype bfloat16` | 32 (8) | 76.2 | 9.8x | 0.42 | top-1 12.67 vs 12.66 %, VCE pose 3.163 vs 3.166 m |
| **4 GPU, `--tf32 --encoder-dtype bfloat16 --compile encoder`** (recommended) | **32 (8)** | **92.2** | **11.8x** | 0.35 | compile: same keys, same numerics path |
| + `--compile both` (decoder too) | 32 (8) | 96.0 | 12.3x | 0.33 | +4 %: not adopted |
| recommended, fine config `vigor_cell00625_fine.yaml` | 32 (8) | 86.7 | 11.1x | 0.37 | fine ckpt: top-1 7.06 vs 7.05 %, VCE pose 1.728 vs 1.730 m |
| recommended | 64 (16) | 111.4 | 14.3x | 0.57 | other optimisation (half the steps) |
| recommended | 128 (32) | 125.2 | 16.0x | 1.02 | other optimisation (32,800 steps / 100 epochs) |
| `--tf32 --encoder-dtype bfloat16` | 128 (32) | 104.1 | 13.3x | 1.23 | |
| `--tf32` | 128 (32) | 54.4 | 7.0x | 2.35 | |
| float32 | 128 (32) | 33.0 | 4.2x | 3.87 | |

- **Scaling**: 4 ranks at 8 per rank give 7.41 samples/s each against 7.3 for one GPU at batch 8 (job 8855575): no loss
  from the data parallelism; the gradient all-reduce (one flat 69 M-float buffer) takes 5-50 ms of a 0.35-1.1 s step.
- **Data**: data wait < 0.07 % of the step on every rank in every setting (8 workers per rank, pinned memory, 64 CPUs
  requested = the whole node; each process sees 32 CPUs in its affinity mask). Not a bottleneck even at 125 samples/s
  (~45 ms CPU per sample = ~6 CPU-s/s for the node).
- **Memory**: 15 GiB per GPU at 8 per rank, 54 GiB at 32 per rank (bfloat16 encoder does not change the peak: the
  decoder's activations dominate).
- **Where the time goes now** (recommended, per rank, 8 samples): frozen encoder 0.094 s (reference 0.064 +
  panorama 0.030; was 0.81 s in float32), decoder forward + head + losses 0.174 s (50 % of the step; compiling the decoder
  saves only 0.012 s), backward 0.067 s, all-reduce 0.006 s, AdamW 0.004 s. Larger per-rank batches amortise the decoder
  part (0.33 s at 32 per rank = 4x the samples for 1.9x the time): that is the global-128 gain.

### Accuracy of the bfloat16 encoder (`--check-encoder`, 200 held-out validation frames, one GPU)

The frozen DINOv3 forward under `torch.autocast(bfloat16)` (outputs cast back to float32; the decoder keeps its own
float16 autocast), against the float32 encoder, same frames, same fixed VCE draw:

| Checkpoint | Encoder | CE | top-1 | top-5 | VCE pose (m) | heat-map argmax (m) |
|---|---|---|---|---|---|---|
| coarse `vigor_samearea_4city_erp_depth_cell0125_last` | float32 | 3.5411 | 12.656 % | 43.72 % | 3.1665 | 3.619 |
| | float32 + TF32 | 3.5412 | 12.640 % | 43.70 % | 3.1660 | 3.622 |
| | bfloat16 | 3.5403 | 12.666 % | 43.77 % | 3.1641 | 3.508 |
| | bfloat16 + TF32 | 3.5404 | 12.674 % | 43.76 % | 3.1633 | 3.497 |
| fine `vigor_samearea_4city_fine00625_erp_depth_last` | float32 + TF32 | 3.9608 | 7.054 % | 29.09 % | 1.7302 | 1.920 |
| | bfloat16 + TF32 | 3.9596 | 7.060 % | 29.09 % | 1.7281 | 1.927 |

Encoder tokens (first validation batch of 16): per-token cosine similarity to float32 0.99998 on average (minimum 0.998
reference, 0.994 panorama), mean absolute difference 1 % of the mean |token| (the max |difference| 28-98 sits on
DINOv3's large-magnitude outlier channels). Every metric moves by less than its last digit except the heat-map argmax
distance (+-0.1 m of 3.6 m on the coarse checkpoint, in both directions across checkpoints; it is an argmax over 3136
cells and the selection metric is the VCE pose). Verdict: unchanged within noise; adopted.

Attention kernels: the encoder is timm's `Eva` (DINOv3) with `fused_attn=True` in all 24 blocks, i.e.
`torch.nn.functional.scaled_dot_product_attention`; the Sat-RoMa decoder's transformer calls
`F.scaled_dot_product_attention` directly. In float32 SDPA can only use the memory-efficient / math kernels; under the
bfloat16 autocast it takes the flash kernel, which is part of the 3.1x encoder gain. Nothing hand-written to replace.

`torch.compile`: `nn.Module.compile(dynamic=False)` in place, so no `_orig_mod.` enters a state dict (checked: the
DDP + compile checkpoints have the same 90 keys and shapes as a single-GPU checkpoint). Encoder: +21 % (76.2 -> 92.2),
compiles in ~1 min per process, recompiles once for the validation batch shape; adopted. Decoder: +4 % more; left
off (not a clear gain, and it would compile the training graph with the RefinerTap hooks of `--refine-weight` runs).

### DDP vs one GPU: same seed, same global batch 32, first 12 steps (`ddp_cmp/cmp/`)

Training CE per step (CSV `ce`, the global-batch value); VCE off (`--vce-weight 0`) so that no random draw differs:

| Step | 1 GPU | 1 GPU again | 4 GPU | 4 GPU again | 4 GPU TF32 + bf16 encoder + compile |
|---|---|---|---|---|---|
| 1 | 7.8426 | 7.8426 | 7.8426 | 7.8426 | 7.8477 |
| 2 | 6.9614 | 6.9614 | 7.0256 | 7.0256 | 7.0267 |
| 3 | 6.7850 | 6.7852 | 6.8211 | 6.8210 | 6.8233 |
| 6 | 6.0073 | 6.0073 | 6.0852 | 6.0850 | 6.0874 |
| 9 | 6.3485 | 6.3485 | 6.3103 | 6.3102 | 6.3068 |
| 12 | 6.3206 | 6.3206 | 6.3083 | 6.3082 | 6.3110 |
| val @12 | 6.1721 | 6.1718 | 6.1508 | 6.1509 | 6.1514 |

- Step 1 is identical: the four rank batches together are the single-GPU batch (sampler sharding) and the logged
  statistics combine to the single-GPU value.
- Each mode repeats itself to 1e-4; 4 GPU vs 1 GPU differs from step 2 on, by up to 0.08 in CE, in both directions
  (4 GPU ends 0.02 lower on validation). The cause is the loss normalisation, as in any DDP: the CE and neighbour hinge
  are means over a rank's matchable TOKENS (their number differs between the ranks' 8 samples), so averaging 4 per-rank
  means weights tokens slightly differently from one mean over the 32 samples; the per-sample terms (pose NLL, VCE) and
  the certainty BCE (all tokens) are exact. AdamW's first steps are nearly sign(g) x lr per parameter, which turns the
  small gradient difference into visibly different early losses; the curves then run together. Making it exact would
  mean weighting each rank's token terms by n_rank x W / n_global (a second all-reduce before backward); not done.
- The fast setting (TF32 + bfloat16 encoder + compiled encoder) tracks the float32 4-GPU run to <= 0.005 over the 12 steps.

With VCE on (`cmp_*.csv`), step 1 CE is again identical (7.8426) and the VCE terms differ from step 1 (each rank draws
its own VCE samples; rank 0 keeps the single-GPU seed).

**Review 2026-09-29 (branch `review/panoroma-ddp`): the cause above was wrong, and the gradient is now the 1-GPU one.**
Step-1 gradient dumps (head + decoder, same command as above with `--steps 1`, probe jobs 8858140 / 8858387, debug
patch not committed; `ddp_cmp/review/gradcmp.txt`), relative L2 distance to the 1-GPU gradient:

| Step-1 gradient | rel. L2 vs 1 GPU | entries with flipped sign |
|---|---|---|
| 1 GPU again | 4.2e-4 | 0.001 % |
| 4 GPU, per-rank means averaged (implementer's version) | 0.43 | 10.3 % |
| 4 GPU, global-count normalisation, n_r·W/N + mean (first review commit) | 0.42 | 10.2 % |
| **4 GPU, each rank backpropagates its share n_r/N of the global loss, gradients summed (final)** | **5.5e-3** | **0.26 %** |
| 1 GPU with the loss x 1024 before backward and the gradient / 1024 after (diagnostic) | 0.70 | 14.7 % |

The token-count normalisation was a small part of the 0.42: the Sat-RoMa decoder runs under **float16 autocast on
CUDA** (`sat_roma.utils.get_autocast_params` enables it on any CUDA device) and training has **no GradScaler**, so the
backward flushes the tiny per-logit gradients ((p - y) / 37,000 tokens ~ 1e-8) to zero. A rank whose loss is its own
mean backpropagates W = 4 times larger values and underflows less: a different (arguably better) gradient, but not the
1-GPU one. With the share n_r / N the float16 backward sees exactly the single-GPU values, and the sum of the ranks'
gradients matches the 1-GPU gradient to float noise. 12 steps, same command as the table above (`ddp_cmp/review/`):

| Step | 1 GPU | 4 GPU (final) | 4 GPU (final) TF32 + bf16 encoder + compile |
|---|---|---|---|
| 1 | 7.8426 | 7.8426 | 7.8435 |
| 2 | 6.9614 | 6.9618 | 6.9618 |
| 3 | 6.7852 | 6.7848 | 6.7837 |
| 6 | 6.0074 | 6.0065 | 6.0087 |
| 9 | 6.3482 | 6.3484 | 6.3472 |
| 12 | 6.3204 | 6.3213 | 6.3245 |
| val @12 | 6.1718 | 6.1717 | 6.1719 |

4 GPU vs 1 GPU: at most 0.0009 CE over the 12 steps (was 0.08), validation equal to 1e-4. The 1 GPU x 1024 row says
the single-GPU plan's own gradient loses most of its small components to float16 underflow (70 % of its L2 changes
when nothing underflows). That is a property of the plan, not of DDP, and left as it is (the run of record = the
single-GPU optimisation); a static loss scale / GradScaler or a bfloat16 decoder autocast is an open decision.

Review smoke (job 8857863, `ddp_cmp/review/review_ddp_8857863_filtered.log`, `train_rsmoke.csv`): fast setting,
`--train-limit 128 --epochs 2` then `--epochs 3` with `--resume auto` and `--val-samples 48` (validation batches of
32 and 16: 6 compiled encoder graphs, within the raised recompile limit of 16, no error): resumed at step 8, wrote
`_ep003.pt`, CSV 1-12 without duplicates.

### Smoke + resume under DDP (`ddp_check_smoke_8856566.log`, `ddp_fast_smoke_8856805.log`)

`--train-limit 128 --batch 32` (4 steps per epoch), `--epochs 2 --save-every-epochs 1 --resume auto`, then the same
command with `--epochs 3`: the second segment printed `resumed ...: step 8 (epoch 2.00), best 13.663 at step 8`, trained
steps 9-12, validated at 12, wrote `_ep003.pt`; the CSV runs 1-12 without duplicates (`ddp_cmp/smoke/`). The same
command with 2 processes was refused: `resume refused: world size 4 != 2; per-rank batch 8 != 16`. Same smoke with the
fast setting (TF32, bfloat16 encoder, compiled encoder): identical behaviour. Checkpoint keys of the DDP checkpoints
(`_last.pt`, `_ep003.pt`, float32 and fast): 90 keys, no `module.` / `_orig_mod.` prefix, same keys and shapes as a
single-GPU checkpoint; the train dict records `world_size 4, batch_per_rank 8` (+ `encoder_dtype`, `compile`, `tf32`).

### Wall time for 100 epochs (42,087 training samples -> 1,315 steps per epoch at global 32, 131,500 steps)

Per epoch: 42,080 samples / node rate + rank-0 validation of 400 frames (~15-35 s) + checkpoint writes (resume 790 MiB
+ lean 263 MiB files, ~10-20 s on NFS).

| Setting (4 x H100, global 32) | Coarse | Fine | Both, chained |
|---|---|---|---|
| float32 | 29.6 /s -> ~24.7 min/epoch -> ~41 h | ~41 h | ~3.4 days |
| `--tf32` | 45.8 /s -> ~16 min/epoch -> ~27 h | ~27 h | ~2.3 days |
| **`--tf32 --encoder-dtype bfloat16 --compile encoder`** | **92.2 /s -> ~8.2 min/epoch -> ~14 h** | **86.7 /s -> ~8.7 min -> ~15 h** | **~1.2 days** |
| same, global 128 (32,800 steps; a different optimisation) | 125.2 /s -> ~6.2 min/epoch -> ~10.5 h | ~11 h | ~0.9 day |
| (1 GPU float32, the reviewed plan) | 7.8 /s -> ~150 h | ~150 h | ~12.5 days |

Every 4-GPU setting fits one 7-day `proxima` segment per run; `--resume auto` stays in the commands for requeues.

## Decoder precision: `--decoder-dtype bfloat16` (2026-09-29, branch `panoroma/bf16-decoder`)

Mechanism: the Sat-RoMa package opens `torch.autocast(device, dtype=self.amp_dtype)` in `Decoder.forward` (proj),
`TransformerDecoder.forward` (coarse classifier) and `ConvRefiner.forward` (refiner), and
`sat_roma.utils.get_autocast_params` forces that autocast ON on any CUDA device (OFF on CPU) with `amp_dtype = float16`
(`build_sat_roma`). `bevloc.model.speed.set_decoder_dtype` sets `amp_dtype` on those three modules: the package's
own attribute, read at every forward (no forward replaced, nothing in third_party edited; an outer autocast of ours
could not win, the package's inner context re-enables its dtype). `--decoder-dtype float16|bfloat16|float32` in
`scripts/train_vigor.py` (train dict, config snapshot, resume counters; a resume with another dtype warns),
`scripts/eval_vigor.py` and `scripts/eval_panoroma_poznan.py` (default float16 = every published row). A trained
refiner's `RefinerTap(precision="float32")` still wins for the refiner and restores the decoder dtype on exit.
Tools: `train_vigor.py --dump-grads PATH` (one step, the (DDP: summed) gradient of head + decoder) +
`scripts/grad_compare.py`; `train_vigor.py --check-decoder` (validation with float16 vs `--decoder-dtype`).
Raw: `bf16/grad/gradcmp_8858913.txt`, `bf16/grad/cmp_ref_*.json`, `bf16/loss/*.csv`, `bf16/check_decoder_8858915.jsonl`,
`bf16/probe_speed_8858916.jsonl` (jobs 8858913-8858916).

### Step-1 gradient vs a float32 decoder (job 8858913)

Coarse config, erp_depth + head, global batch 32, VCE off, encoder float32, no TF32 (only the decoder dtype differs),
same batch; relative L2 distance of the head + decoder gradient (68.9 M entries) to the float32-decoder gradient:

| Step-1 gradient | rel. L2 vs fp32 decoder | sign differs | exact zeros | cosine | norm / fp32 norm |
|---|---|---|---|---|---|
| float32 decoder, 4 GPU again (noise floor) | 7.3e-6 | 0.0000 % | 2.86 % | 1.000000 | 1.0000 |
| float32 decoder, 1 GPU vs 4 GPU | 1.3e-5 | 0.0001 % | 2.86 % | 1.000000 | 1.0000 |
| **float16 decoder (every run so far), 1 GPU** | **0.533** | **16.4 %** | 5.88 % | 0.851 | 0.763 |
| float16 decoder, 4 GPU | 0.534 | 16.4 % | 5.95 % | 0.850 | 0.763 |
| **bfloat16 decoder, 1 GPU** | **7.3e-3** | **0.36 %** | 2.86 % | 0.99997 | 1.0002 |
| bfloat16 decoder, 4 GPU | 7.2e-3 | 0.37 % | 2.88 % | 0.99997 | 1.0002 |

Per group (1 GPU) rel. L2, float16 / bfloat16: head 0.55 / 7.2e-3, transformer 0.46 / 7.1e-3, GP 0.25 / 8.4e-3, proj
0.57 / 7.3e-3. The float16 decoder loses a quarter of the gradient norm (3 % more entries are exactly zero: underflow)
and 16 % of the signs; bfloat16 is 70x closer to float32 (its 8-bit mantissa rounding, no underflow). DDP vs 1 GPU
with the same decoder dtype: bfloat16 1.4e-3 (0.08 % signs), float16 5.5e-3 (as in the review).

### First 12 steps, same seed, global batch 32, VCE off (job 8858914, `bf16/loss/`)

Training CE per step (`ce`, global batch). "fast" = `--tf32 --encoder-dtype bfloat16 --compile encoder` on 4 GPUs;
the 1-GPU float16 column is the review's `ddp_cmp/review/train_rcmp0_single.csv` (plain float32 encoder):

| Step | fast, fp32 decoder (reference) | **fast, bf16 decoder** | fast, fp16 decoder | 4 GPU bf16 (plain) | 1 GPU bf16 (plain) | 1 GPU fp16 (review) |
|---|---|---|---|---|---|---|
| 1 | 7.8441 | 7.8457 | 7.8485 | 7.8420 | 7.8419 | 7.8426 |
| 2 | 7.0450 | 7.0461 | 6.9602 | 7.0456 | 7.0457 | 6.9614 |
| 3 | 6.8412 | 6.8411 | 6.7840 | 6.8401 | 6.8399 | 6.7852 |
| 4 | 6.6514 | 6.6523 | 6.6748 | 6.6547 | 6.6546 | 6.6759 |
| 6 | 6.1074 | 6.1075 | 6.0098 | 6.1046 | 6.1045 | 6.0074 |
| 8 | 6.2920 | 6.2933 | 6.2494 | 6.2944 | 6.2945 | 6.2538 |
| 9 | 6.3372 | 6.3386 | 6.3458 | 6.3422 | 6.3425 | 6.3482 |
| 11 | 6.1663 | 6.1665 | 6.2111 | 6.1686 | 6.1686 | 6.2129 |
| 12 | 6.3064 | 6.3075 | 6.3228 | 6.3041 | 6.3042 | 6.3204 |
| val @12 | 6.1528 | 6.1532 | 6.1724 | 6.1532 | 6.1532 | 6.1718 |

- bf16 decoder tracks the fp32-decoder reference to <= 0.0016 CE over the 12 steps (validation 4e-4); the fp16
  decoder departs by up to 0.10 (step 6) and ends 0.02 higher on validation: the underflowed gradient is a different
  optimisation, not noise.
- DDP vs 1 GPU still holds with bf16: 4 GPU vs 1 GPU <= 0.0003 CE, validation equal to 1e-4.

### Existing (float16-trained) checkpoints evaluated with another decoder dtype (job 8858915)

`--check-decoder`, 400 held-out validation frames (`--val-frac 0.2`), encoder float32, no TF32, fixed VCE draw:

| Checkpoint | Decoder | CE | top-1 | top-5 | VCE pose (m) | heat-map argmax (m) |
|---|---|---|---|---|---|---|
| coarse `vigor_samearea_4city_erp_depth_cell0125_last` | float16 (trained) | 3.6867 | 11.99 % | 41.30 % | 3.651 | 4.124 |
| | bfloat16 | 3.6885 | 12.02 % | 41.33 % | 3.654 | 4.152 |
| | float32 | 3.6866 | 11.98 % | 41.27 % | 3.652 | 4.133 |
| fine `vigor_samearea_4city_fine00625_erp_depth_last` | float16 (trained) | 4.0502 | 6.74 % | 28.17 % | 1.824 | 2.054 |
| | bfloat16 | 4.0527 | 6.89 % | 28.28 % | 1.825 | 2.037 |
| | float32 | 4.0501 | 6.69 % | 28.15 % | 1.825 | 2.055 |

Evaluation dtype does not matter for the old checkpoints: VCE pose within 4 mm, CE within 0.003 (bf16 slightly higher,
float32 = float16 to 1e-4), top-1 within 0.15 pp (fine, both directions). Published rows (float16) stay as they are;
new checkpoints trained with bf16 record `decoder_dtype` in their train dict.

### Throughput (job 8858916; 4 x H100, global 32, `--tf32 --encoder-dtype bfloat16 --compile encoder`)

| Config | Decoder | Samples/s (node) | Step (s) | Peak reserved / GPU | GPU util |
|---|---|---|---|---|---|
| coarse `vigor_cell0125` | float16 | 91.6 | 0.349 | 14.9 GiB | 97 % |
| | **bfloat16** | **88.1** | 0.363 | 14.8 GiB | 95 % |
| fine `vigor_cell00625_fine` | float16 | 85.4 | 0.375 | 14.9 GiB | 96 % |
| | **bfloat16** | **84.6** | 0.378 | 14.8 GiB | 94 % |

Same within 1-4 % (per-phase breakdown equal to the ms: decoder forward 0.178 vs 0.177 s, backward 0.067 s); memory
unchanged. 100 epochs at global 32: coarse ~8.0 min/epoch -> ~14.5 h, fine ~8.3 min/epoch -> ~15 h.

Verdict: the bfloat16 decoder gives the float32 gradient (7e-3 vs 0.53 for float16) at float16 speed. Adopted for the
100-epoch runs (`--decoder-dtype bfloat16`); the default stays float16 so old commands and evaluations are unchanged.
