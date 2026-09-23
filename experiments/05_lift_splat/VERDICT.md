## Verdict (running notes; newest first)

**2026-09-23, TEST manifest (irAsBUK, reserved route, 200 frames × 2025/2024).** The six lifted-BEV
checkpoints (`multi`, `years`, `aug`, `seq`, `seq_single`, `pose_nll`) score 14.9–17.5 m median at 2025
against a centre-guess of 18.3 m; R@5 0.11–0.17 against 0.04; R@10 0.29–0.38 against 0.15; the `>30 m`
tail is 0.17–0.22 against 0.01. They are better than chance but by only 1–3 m of median, and every
interval overlaps every other. The validation-route table below (11–13 m) was flattered by
checkpoint selection on that route: the same checkpoints lose 3–5 m when moved to a route no
training or selection ever saw. Multi-frame (`seq` vs `seq_single`) and pose NLL (`pose_nll`) change
nothing. Implication: the learned depth-bin lift does not generalise; the query, not the loss, is the
lever (plan Task 6, ERP-token query with placement after matching, is the response). Pending rows:
`ipm`, `hybrid`, `hybrid_warm`.

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
