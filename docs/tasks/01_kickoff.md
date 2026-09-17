# Task 01 — Kick-off experiment (tests H1, bounds H2)

Goal: does the Sat-RoMa matcher, unchanged, localize a ground BEV against an orthophoto?

## Inputs
- data/dur360bev/: one urban sequence subset with ERP image, LiDAR scan, OxTS pose per frame.
- data/ortho/durham/<year>/*.tif (EPSG:27700). Until available, code against a rasterio GeoTIFF interface with a placeholder.
- checkpoints/satroma_dinov3.pt (third_party/sat_roma inference code).

## Steps (stop and show output after each)
1. Loader + calibration check
   - Load N=20 frames with synchronized ERP, LiDAR, OxTS (lat, lon, alt, roll, pitch, yaw). Document timestamp pairing.
   - Convert lat/lon → EPSG:27700 (pyproj). Record heading convention after checking the RT3000 manual.
   - Project LiDAR into the ERP with the SI2BEV axis assumption; overlay on 5 frames; report edge alignment. Estimate camera height from the LiDAR ground plane.
   - Once ortho exists: keep LiDAR points 0.5–3 m above ground, transform with roll/pitch/yaw + position, overlay on the orthophoto for 5 frames; fit residual SE(2) (lever arm + yaw bias) and report with uncertainty. Abort if residual > ~1 m after fitting.
2. Oracle BEV — LiDAR points with RGB sampled from the ERP, gravity-aligned (roll/pitch only, NOT yaw), metric grid 224×224 cells at s=0.25 m (56 m extent, radius 28 m), ego-centred; validity mask; save 5 PNGs. Two variants: (a) LiDAR-only (no return → invalid), (b) LiDAR + IPM ground fill for cells without returns (wet-asphalt dropout). Report both.
3. IPM BEV — same grid; each cell placed by flat-ground IPM using roll/pitch and camera height; mask the ego-vehicle blind disc and the fisheye stitching seam in the validity mask (log their extents); save 5 PNGs side by side with the oracle.
4. Reference — orthophoto crop around the RTK pose at 4× BEV extent (896 px at 0.25 m = 224 m, for the 224 px / 56 m query), random offset ≤ 30 % so the footprint is not centred, random rotation within ±55° (matcher's trained range; 360° is a later task). Store the GT homography/SE(2).
5. Matching — run Sat-RoMa (multi-hypothesis refinement, not the ConvNet refiner) on (BEV, ortho) for both variants, no fine-tuning, N≥200 frames.
6. Report — experiments/01_kickoff/REPORT.md: recall@1/5/10 m, yaw error, corner error, retained GMM modes, inlier ratio; 10 overlays per variant; interpretation against H1 (oracle) and H2 (IPM vs oracle gap). Be explicit about failure modes seen.

## Not in scope
Geometry head, contrastive loss, Lift-Splat/SI2BEV baseline, particle filter, VIGOR.
