# data/

Nothing in this directory is committed.

## dur360bev/ — subset of Dur360BEV (ICRA 2025), recorded 2024-03-12 in Durham, UK

Source: https://huggingface.co/datasets/TomEeee/Dur360BEV — one gzip'd tar (170 GB) split into 80
parts and ordered **by modality** (all images, labels, metadata, LiDAR, OxTS), so no sequence-level
download exists. `scripts/fetch_dur360bev_subset.sh` (`make fetch`) streams it once and keeps:

- every 10th frame (name ends in 0) of every modality, over the whole drive (16 407 frames at 10 Hz),
- the dense clip 0000000000–0000000299,
- all `oxts/`, `timestamps.txt`, `dataformat.txt`, `metadata/`.

```
dur360bev/
  image/data/<frame>.png            1280x720 DUAL FISHEYE (not ERP), Ricoh Theta S
  image/timestamps.txt              one ISO timestamp per frame of the FULL dataset (line k = frame k)
  ouster_points/data/<frame>.bin    Ouster OS1-128 sweep, 128 x 2048 points, 36 bytes each
  oxts/data/<frame>.txt             lat lon alt roll pitch yaw        (NOT YET DOWNLOADED - last in the archive)
  labels/data/<frame>.txt           3D boxes (unused here)
  metadata/os1.json                 Ouster beam angles etc.; contains NO camera or INS extrinsics
```
Frames are synchronised to the LiDAR at 10 Hz with a 0.03 s tolerance (Dur360BEV paper), so image and
scan of one frame may be up to 30 ms apart (0.3 m at 10 m/s). Some frame numbers are missing.

### LiDAR `.bin` layout (verified on data, 2026-09-17)
9 four-byte slots per point, row-major on the native 128 x 2048 grid (row index = ring, row 0 = top beam):

| slot | bytes | type | content |
|---|---|---|---|
| 0–2 | 0–11 | float32 | x, y, z [m]; LiDAR frame x forward, y left, z up; (0,0,0) = no return |
| 3 | 12–15 | float32 | intensity |
| 4 | 16–19 | uint32 | t [ns] since sweep start; constant per column, 0 … 99.9 ms, monotonic → de-skew |
| 5 | 20–23 | float32 | reflectivity (0–255) |
| 6 | 24–27 | — | packed / not decodable; unused (ring = row index) |
| 7 | 28–31 | uint32 | ambient (near-IR) |
| 8 | 32–35 | uint32 | range [mm]; agrees with ‖xyz‖ to 5 mm |

Reading all 9 slots as float32 (as Dur360BEV's loader does) is only correct for slots 0–3 and 5.
Parser: `bevloc.data.dur360.read_scan`.

### Frames and calibration (see docs/decisions.md for how each was obtained)
- **Camera**: dual fisheye → ERP with Dur360BEV's piecewise lens model (196° inner / 203° outer, knee 1.0 rad).
  ERP: forward = image centre, azimuth increases to the right, zenith on top. fisheye_tools' 203° equidistant
  remap is ~6 px off at 57° off-axis and is not used.
- **Camera ← LiDAR**: R = I, t = (0, 0, −0.27) m (camera 0.27 m above the LiDAR origin). Set visually; provisional.
- **LiDAR height above road**: 1.57 m (6 near-level frames, σ 0.02 m) → camera height 1.84 m. Preliminary:
  wet asphalt returns almost nothing (median 21 % return rate in the fore/aft road sectors).
- **Ego vehicle** hides 30 % of the ERP; ground first visible ≈3.1 m ahead, 1.8 m to the sides, 7.1 m behind.
- **OxTS**: device convention per RT3000 manual is x fwd / y right / z down, heading clockwise from north.
  The STORED convention is UNVERIFIED: Dur360BEV's `map_api` treats yaw as radians CCW from east.
  To be settled by direction of travel once the files are on disk. INS→LiDAR lever arm unknown.

## ortho/durham/<year>/ — orthophoto reference (NOT YET AVAILABLE)
Environment Agency Vertical Aerial Photography (OGL v3), EPSG:27700. Build one VRT over the tiles
(`gdalbuildvrt`) and point `data.ortho` in `configs/default.yaml` at it; `bevloc.data.ortho.OrthoMap`
rejects any other CRS.
