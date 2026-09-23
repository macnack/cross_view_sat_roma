## Verdict (running notes; newest first)

**2026-09-23, validation manifest (IcRzj, 200 frames × 2025/2024), single-frame re-scores.**
The four Lift-Splat checkpoints (`multi`, `years`, `aug`, `seq`) are statistically indistinguishable:
peak medians 11.5–12.4 m at 2025 and 12.5–13.5 m at 2024, every 95 % interval overlapping every other,
R@5 between 0.13 and 0.19, R@10 between 0.37 and 0.47. All of them beat the centre-guess row
(16.9 m, R@5 0.05, R@10 0.17) by more than the interval width, so the matcher does localise, but
none of the training changes since the first multi-route run (cross-year references, augmentation,
multi-frame + pose NLL) moved the number beyond noise. The earlier "8.7 m, R@5 27 %" for `seq` came
from n = 48 frames selected on this same route; on 200 frames it reads 11.9 m, R@5 0.13.
Note: this table scores `seq` single-frame; the multi-frame (0/2/5 m) re-score is a separate row
(`seq` with `seq_dists 0,2,5`) once it lands. Implication: per plan Task 1 gate, no checkpoint
"wins"; Task 2 (IPM baseline) and Task 3 (dense hybrid query) proceed, and the paper's story cannot
rest on loss or augmentation variants of this query.
