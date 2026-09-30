# Regularisation ablation, PanoRoMa coarse decoder, 30 epochs (2026-09-30)

Runs A–D (jobs 8868815–18, clone `cvsr_reg`, 4× H100 each, same flags as the 100-epoch run apart from the
regularisation) against the 100-epoch run's epoch-30 checkpoint (same sampler, constant lr). Evaluation at epoch 30:
VIGOR Chicago same-area 3000 (se2; two-pass with the OLD fine decoder, gate 6 m) and Poznań zero-shot (400 entries,
prior heading, 71 m imagery extent, same old fine decoder). All with `--decoder-dtype bfloat16`.

| Run | Train top-1 | Val top-1 | Val CE | Val pose (VCE) | VIGOR coarse median / mean | VIGOR two-pass median / mean | VIGOR two-pass ≤ 10 m | Poznań coarse | Poznań two-pass | Poznań ≤ 10 m |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (e100 ep030) | 33.9 % | 16.5 % | 3.89 | 3.40 m | 1.50 / 3.19 m | 1.23 / 3.06 m | 92 % | 4.96 m | 4.59 m | 74 % |
| A label smoothing 0.1 | 33.1 % | 16.2 % | 3.62 | 3.35 m | 1.48 / 3.18 m | 1.22 / 3.02 m | 92 % | 4.86 m | 4.26 m | 76 % |
| B photometric + rot90 ±10° + flip | 18.6 % | 16.5 % | **3.20** | 2.82 m | 1.52 / 2.96 m | **1.20** / 2.83 m | 94 % | 4.69 m | 4.33 m | 78 % |
| C weight decay (dec 0.3, head 0.05) | 32.8 % | 15.7 % | 3.81 | 3.30 m | 1.54 / 3.32 m | 1.22 / 3.17 m | 92 % | 4.57 m | 4.51 m | 78 % |
| **D = A + B + C** | 18.2 % | **16.7 %** | 3.27 | **2.79 m** | **1.46 / 2.78 m** | 1.22 / **2.69 m** | **94 %** | **4.28 m** | **4.01 m** | **82 %** |

Poznań two-pass CIs: baseline 4.14–5.21, A 3.76–5.00, B 3.79–4.83, C 3.81–5.02, D 3.65–4.62 m. VIGOR coarse medians:
baseline 1.45–1.57, D 1.42–1.51 m.

## Readings

- **Augmentation is what removes the overfitting.** With B and D train top-1 drops to 18 % against 16.5–16.7 % on
  validation (the baseline: 33.9 vs 16.5 %) and validation CE stays low (3.2–3.3 vs 3.9). Label smoothing (A) and weight
  decay (C) leave the gap as it was.
- **It pays where the model was weakest**: validation pose 3.40 → 2.79 m, VIGOR mean error (the tail) 3.19 → 2.78 m
  coarse and 3.06 → 2.69 m two-pass, and Poznań zero-shot 4.59 → 4.01 m two-pass with 82 % within 10 m (74 %). The VIGOR
  two-pass median does not move (1.20–1.23 m) because the old fine decoder limits it.
- **D is the best run on every localisation metric but the VIGOR two-pass median** (B 1.20 m, within noise). Its Poznań
  two-pass 4.01 m is now level with FG² (4.15 m) and 0.5 m behind Loc²-UniK3D (3.50 m), after 30 epochs, zero-shot.
- Next: D for 100 epochs (coarse), then a fine decoder trained with the fine-compatible part of D (rot90 + flip +
  photometric), and the full two-pass rows.
