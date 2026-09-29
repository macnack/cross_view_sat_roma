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
