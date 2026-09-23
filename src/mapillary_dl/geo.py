"""Bbox tiling, path length, and per-sequence grouping. No network."""

import math
from collections import Counter, defaultdict

# Stay under the Graph API's 0.01 deg² limit. Binary splits land near this size.
MAX_BBOX_AREA = 0.004
# A tile that still returns the 2000-image cap below this area is reported, not split forever.
MIN_TILE_AREA = 2e-6
PAGE_CAP = 2000
# Steps longer than this are a gap (the drive left the rectangle), not motion.
MAX_STEP_M = 80.0
PANO_CAMERA_TYPES = {"spherical", "equirectangular"}

Bounds = tuple[float, float, float, float]


def haversine_m(lon1, lat1, lon2, lat2) -> float:
    radius = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def summarize(images: list[dict]) -> dict:
    coords = []
    for im in images:
        g = (im.get("computed_geometry") or im.get("geometry") or {}).get("coordinates")
        if g:
            coords.append(g)
    step = [haversine_m(a[0], a[1], b[0], b[1]) for a, b in zip(coords, coords[1:])]
    step_sorted = sorted(step)

    def pct(p):
        if not step_sorted:
            return None
        return step_sorted[min(len(step_sorted) - 1, int(p * (len(step_sorted) - 1)))]

    alts = [im["altitude"] for im in images if im.get("altitude") is not None]
    calts = [im["computed_altitude"] for im in images if im.get("computed_altitude") is not None]
    t0 = images[0].get("captured_at")
    t1 = images[-1].get("captured_at")
    return {
        "n": len(images),
        "camera_type": images[0].get("camera_type"),
        "make": images[0].get("make"),
        "model": images[0].get("model"),
        "width": images[0].get("width"),
        "height": images[0].get("height"),
        "is_pano": images[0].get("is_pano"),
        "username": (images[0].get("creator") or {}).get("username"),
        "sequence": images[0].get("sequence"),
        "t_start_ms": t0,
        "t_end_ms": t1,
        "duration_s": None if t0 is None or t1 is None else (t1 - t0) / 1000.0,
        "path_length_m": sum(step),
        "step_m": {"min": pct(0), "p50": pct(0.5), "p95": pct(0.95), "max": pct(1)},
        "gps_altitude_m": {"min": min(alts), "max": max(alts)} if alts else None,
        "computed_altitude_m": {"min": min(calts), "max": max(calts)} if calts else None,
        "bbox_lonlat": (
            [
                min(c[0] for c in coords),
                min(c[1] for c in coords),
                max(c[0] for c in coords),
                max(c[1] for c in coords),
            ]
            if coords
            else None
        ),
    }


def parse_bbox(text: str) -> Bounds:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise SystemExit("--bbox needs west,south,east,north (four comma-separated degrees).")
    west, south, east, north = (float(p) for p in parts)
    if not (west < east and south < north):
        raise SystemExit(f"--bbox is empty or reversed: {text}")
    return west, south, east, north


def tile_area(bounds: Bounds) -> float:
    west, south, east, north = bounds
    return (east - west) * (north - south)


def halve_bbox(bounds: Bounds) -> list[Bounds]:
    west, south, east, north = bounds
    if (east - west) >= (north - south):
        mid = (west + east) / 2
        return [(west, south, mid, north), (mid, south, east, north)]
    mid = (south + north) / 2
    return [(west, south, east, mid), (west, mid, east, north)]


def split_until(bounds: Bounds, max_area: float = MAX_BBOX_AREA):
    """Binary-split until every tile is within the Graph API area limit."""
    if tile_area(bounds) <= max_area:
        return [bounds]
    out = []
    for child in halve_bbox(bounds):
        out.extend(split_until(child, max_area))
    return out


def group_sequences(images: dict[str, dict]) -> tuple[list[dict], dict, int]:
    dropped: Counter = Counter()
    by_seq: dict[str, list] = defaultdict(list)
    n_360 = 0
    for im in images.values():
        if im.get("camera_type") not in PANO_CAMERA_TYPES:
            dropped[im.get("camera_type") or "unknown"] += 1
            continue
        n_360 += 1
        by_seq[im.get("sequence") or ""].append(im)

    rows = []
    for sid, ims in by_seq.items():
        ims.sort(key=lambda im: (im.get("captured_at") or 0, int(im["id"])))
        coords = []
        for im in ims:
            g = (im.get("computed_geometry") or {}).get("coordinates")
            if g:
                coords.append(g)
        length = 0.0
        for a, b in zip(coords, coords[1:]):
            step = haversine_m(a[0], a[1], b[0], b[1])
            if step <= MAX_STEP_M:
                length += step
        creator = ims[0].get("creator") or {}
        t0, t1 = ims[0].get("captured_at"), ims[-1].get("captured_at")
        rows.append(
            {
                "sequence": sid,
                "username": creator.get("username"),
                "creator_id": creator.get("id"),
                "n_in_bbox": len(ims),
                "camera_type": ims[0].get("camera_type"),
                "make": ims[0].get("make"),
                "model": ims[0].get("model"),
                "t_start_ms": t0,
                "t_end_ms": t1,
                "path_length_m": length,
                "bbox_lonlat": (
                    [
                        min(c[0] for c in coords),
                        min(c[1] for c in coords),
                        max(c[0] for c in coords),
                        max(c[1] for c in coords),
                    ]
                    if coords
                    else None
                ),
            }
        )
    rows.sort(key=lambda r: (-r["n_in_bbox"], r["sequence"] or ""))
    return rows, dict(dropped), n_360


def even_sample(items: list, k: int) -> list:
    """Pick ``k`` items spread from the first to the last, endpoints included."""
    n = len(items)
    if k < 1:
        raise SystemExit("--frames must be at least 1.")
    if n <= k:
        return list(items)
    idxs = []
    for i in range(k):
        j = round(i * (n - 1) / (k - 1))
        if not idxs or idxs[-1] != j:
            idxs.append(j)
    return [items[j] for j in idxs]


def longest_sequence_per_user(sequences: list[dict]) -> list[dict]:
    by_user: dict[str, list] = defaultdict(list)
    for seq in sequences:
        by_user[seq.get("username") or "unknown"].append(seq)
    picks = []
    for seqs in by_user.values():
        best = max(seqs, key=lambda s: (s.get("n_in_bbox") or 0, s.get("path_length_m") or 0))
        picks.append(best)
    picks.sort(key=lambda s: -(s.get("n_in_bbox") or 0))
    return picks
