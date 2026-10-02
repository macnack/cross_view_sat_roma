# Task 05 — Multi-year reference imagery for the VIGOR tiles (Esri World Imagery Wayback, 2025)

**Status (26 Sep 2026):** spec agreed with Maciej; not started. Depends on nothing running.

## Why

The paper's setting is localisation against public orthophotos whose capture date differs from the drive, and the
Poznań misses were year-consistent (decisions 2026-09-24). VIGOR ships one Google Maps Static tile per location
(captured 2020–2021, 640 px at 0.10–0.12 m/px). To test the same idea on the benchmark we need the same footprints
in other years: a **newer** reference (2025) for "map older/newer than the drive", and two or three dated versions
for **cross-year voting** (match against several years, RANSAC over the union of modes).

## Source

**Esri World Imagery Wayback** (https://livingatlas.arcgis.com/wayback/): every published release of World Imagery
since 2014 as a tiled WMTS/XYZ service, one layer per release date. Resolution in the four cities 0.15–0.3 m/px
(Maxar/Vexcel), so at zoom 19–20 a VIGOR footprint (≈ 71 m) is a 240–480 px window. Free; Esri's terms allow
research use with attribution ("Esri, Maxar, Earthstar Geographics"), no redistribution of the tiles: they stay
under `data/` (never committed), the paper cites the source and release dates.

Alternatives and why not first: Google Maps Static today (identical source and GSD to VIGOR, 2023–2025 in these
cities, ≈ 2 $/1000 tiles, but needs Maciej's API key and Google's terms restrict storing imagery — optional second
source if he chooses to); USGS NAIP (public domain, 0.6–1 m, latest 2023/2024, no 2025 yet) as the public-orthophoto
story later.

## Deliverables

1. `scripts/fetch_wayback_vigor.py` + `make wayback-fetch SPLIT= CITIES= LIMIT= YEARS="2025 2019"`: for every tile in
   the eval draw (the same `VigorPairs(limit, seed)` draw as `eval_vigor.py`), list Wayback releases covering the tile
   centre, pick the release closest to each requested year (record release date and version id), download the XYZ
   tiles covering the footprint at the finest zoom, mosaic and reproject from Web Mercator to the tile's local metric
   frame at 0.125 m/px, save `data/vigor/<City>/wayback_<year>/<sat_name>.png` (same size as the VIGOR tile at our
   GSD) plus a JSON sidecar (release id, date, zoom, source GSD, offset used). Resumable; polite rate limit.
2. **Georeferencing calibration**: Google's tile centres and Esri's tiles disagree by a few pixels. On 150 tiles per
   city, estimate the constant (dx, dy) offset between the VIGOR tile and the Wayback window by phase correlation on
   the *same-year* Wayback release (2020/2021) and apply it to all years; report the residual (target < 0.3 m). Same
   method as the row/column-sign calibration of 2026-09-24.
3. `VigorPairs` option `ref_source: vigor | wayback_<year>` (config `vigor.ref_source`) so every evaluator and trainer
   can take the alternative reference unchanged; `eval_vigor.py --ref-source` and a `--ref-sources a b` mode that runs
   the decoder against each and RANSACs over the union of modes (cross-year voting, like the Poznań years rows).
4. Rows (Chicago 3000 draw, then SF + Chicago 6000; the 2 m Task 04 checkpoint and the picture checkpoint):
   reference = VIGOR 2020/21 (today's rows) · Wayback closest to 2021 (source change only) · Wayback 2025
   (newer map) · Wayback 2017/2019 (older map) · union of two years (voting). Report the gap between "same year,
   other source" and "other year, same source" separately: the first is the price of the imagery, the second the
   price of time.

## Gates

- Calibration residual < 0.3 m per city, else the source cannot be used for sub-metre rows.
- Wayback same-year within 0.3 m median of the VIGOR row: the reference source change is benign.
- Cross-year voting must not be worse than the best single year on the median and must lower R@10 misses; otherwise
  the union hurts (a real finding for the paper).
