"""List 360 sequences inside a lon/lat rectangle. Does not download pixels."""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .client import GRAPH, SCAN_FIELDS, USER_AGENT, token
from .geo import (
    MIN_TILE_AREA,
    PAGE_CAP,
    PANO_CAMERA_TYPES,
    group_sequences,
    halve_bbox,
    split_until,
    tile_area,
)


class BBoxRejected(Exception):
    """The tile is too large or the response was truncated. Split and retry."""


def _fetch_bbox_once(access_token: str, bounds: tuple[float, float, float, float]) -> list[dict]:
    west, south, east, north = bounds
    url = f"{GRAPH}/images?" + urllib.parse.urlencode(
        {
            "fields": SCAN_FIELDS,
            "bbox": f"{west:.7f},{south:.7f},{east:.7f},{north:.7f}",
            "is_pano": "true",
            "limit": str(PAGE_CAP),
            "access_token": access_token,
        }
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        # Do not include the request URL: it carries the access token.
        if exc.code == 429:
            raise TimeoutError("Mapillary rate limit") from None
        if exc.code >= 500 or "too large" in body.lower():
            raise BBoxRejected(f"HTTP {exc.code}") from None
        raise SystemExit(f"Mapillary HTTP {exc.code}: {body[:240]}") from None
    if isinstance(data, dict) and data.get("error"):
        raise BBoxRejected(str(data["error"])[:240])
    images = data.get("data") or []
    if len(images) >= PAGE_CAP:
        raise BBoxRejected(f"truncated at {PAGE_CAP}")
    return images


def fetch_bbox(access_token: str, bounds: tuple[float, float, float, float]) -> list[dict]:
    last = None
    for attempt in range(4):
        try:
            return _fetch_bbox_once(access_token, bounds)
        except TimeoutError as exc:
            last = exc
            time.sleep(2.0 * (attempt + 1))
        except urllib.error.URLError as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise SystemExit(f"Mapillary request failed after retries: {last}")


def scan_bbox(access_token: str, bounds: tuple[float, float, float, float], workers: int = 6):
    """Return unique image records inside ``bounds``, plus tile stats."""
    pending = split_until(bounds)
    images: dict[str, dict] = {}
    truncated = []
    n_requests = 0
    while pending:
        wave, pending = pending, []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(fetch_bbox, access_token, tile): tile for tile in wave}
            for fut in as_completed(futs):
                tile = futs[fut]
                n_requests += 1
                try:
                    batch = fut.result()
                except BBoxRejected:
                    if tile_area(tile) <= MIN_TILE_AREA:
                        truncated.append([round(v, 7) for v in tile])
                        print(f"tile {n_requests} truncated {tile}", flush=True)
                    else:
                        pending.extend(halve_bbox(tile))
                        print(
                            f"tile {n_requests} split area={tile_area(tile):.6f} "
                            f"queued {len(pending)}",
                            flush=True,
                        )
                    continue
                for im in batch:
                    images[im["id"]] = im
                print(f"tile {n_requests} +{len(batch)} unique {len(images)}", flush=True)
    return images, {"n_requests": n_requests, "truncated_tiles": truncated}


def write_scan_plot(path: Path, images: dict[str, dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; skipped coverage plot")
        return

    from collections import defaultdict

    by_user: dict[str, list] = defaultdict(list)
    for im in images.values():
        if im.get("camera_type") not in PANO_CAMERA_TYPES:
            continue
        g = (im.get("computed_geometry") or {}).get("coordinates")
        if not g:
            continue
        user = (im.get("creator") or {}).get("username") or "unknown"
        by_user[user].append(g)
    if not by_user:
        return
    ranked = sorted(by_user, key=lambda u: -len(by_user[u]))
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    colors = plt.cm.tab10.colors
    for i, user in enumerate(ranked[:8]):
        pts = by_user[user]
        ax.scatter(
            [p[0] for p in pts],
            [p[1] for p in pts],
            s=3,
            c=[colors[i % len(colors)]],
            label=f"{user} ({len(pts)})",
            linewidths=0,
        )
    rest = [p for user in ranked[8:] for p in by_user[user]]
    if rest:
        ax.scatter(
            [p[0] for p in rest],
            [p[1] for p in rest],
            s=3,
            c="0.6",
            label=f"other ({len(rest)})",
            linewidths=0,
        )
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Mapillary 360 coverage")
    ax.legend(loc="best", fontsize=7, markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_scan(bounds, name: str, out_dir: Path, workers: int) -> Path:
    access_token = token()
    images, stats = scan_bbox(access_token, bounds, workers=workers)
    sequences, dropped, n_360 = group_sequences(images)
    report = {
        "bbox": list(bounds),
        "n_requests": stats["n_requests"],
        "truncated_tiles": stats["truncated_tiles"],
        "n_images_api": len(images),
        "n_images_360": n_360,
        "dropped_camera_types": dropped,
        "n_sequences": len(sequences),
        "path_length_m": sum(s["path_length_m"] for s in sequences),
        "sequences": sequences,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name or "bbox"
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(report, indent=2))
    write_scan_plot(out_dir / f"{stem}.png", images)
    print(
        f"{report['n_sequences']} sequences, {n_360} 360 frames, "
        f"{report['path_length_m'] / 1000:.1f} km inside the box "
        f"({len(images)} is_pano hits, dropped {dropped or 'none'})"
    )
    print(f"{'n':>6}  {'km':>7}  {'user':<22}  {'model':<16}  sequence")
    for seq in sequences[:30]:
        print(
            f"{seq['n_in_bbox']:6d}  {seq['path_length_m'] / 1000:7.2f}  "
            f"{(seq['username'] or '?'):<22}  {(seq['model'] or '?'):<16}  {seq['sequence']}"
        )
    if len(sequences) > 30:
        print(f"... {len(sequences) - 30} more sequences in {json_path}")
    print(f"wrote {json_path}")
    return json_path
