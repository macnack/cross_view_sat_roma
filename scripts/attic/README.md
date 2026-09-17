# Attic — kept for the record, not part of the pipeline

| file | what it tried | why it was dropped |
|---|---|---|
| `fit_cam_lidar.py` | 6-DoF camera←LiDAR fit by maximising image gradient at LiDAR depth edges | Score rewards sliding points onto roof/sky boundaries; t_y, t_z ran to their bounds; result visibly worse than identity. It was also absorbing a lens-model error (203° equidistant vs 196/203 piecewise). |
| `fit_cam_lidar_parallax.py` | translation from the 1/r dependence of per-point edge offsets | Failed an inject-and-recover test: returns ≈0 whether ±0.27 m or 0.3 m is injected. No statistical power. |
| `check_lidar_erp.py` | first static LiDAR-on-ERP overlay | Superseded by `scripts/calib_overlay_viewer.py` and `scripts/calib_lens_check.py`. |

These scripts use the pre-refactor module paths and will not run as-is. The agreed extrinsic
(R = I, t = (0, 0, −0.27) m) was set visually; an ambient-image↔ERP PnP cross-check is still pending.
