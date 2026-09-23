"""Mapillary Graph API client. Signed image URLs are not stored."""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

GRAPH = "https://graph.mapillary.com"
USER_AGENT = "mapillary-dl/0.1"
FIELDS = ",".join(
    [
        "id",
        "captured_at",
        "compass_angle",
        "computed_compass_angle",
        "geometry",
        "computed_geometry",
        "altitude",
        "computed_altitude",
        "computed_rotation",
        "camera_type",
        "camera_parameters",
        "width",
        "height",
        "sequence",
        "creator",
        "is_pano",
        "make",
        "model",
        "exif_orientation",
        "thumb_original_url",
        "thumb_2048_url",
    ]
)
URL_FIELDS = ("thumb_original_url", "thumb_2048_url")
SCAN_FIELDS = ",".join(
    [
        "id",
        "sequence",
        "camera_type",
        "is_pano",
        "captured_at",
        "computed_geometry",
        "creator",
        "make",
        "model",
    ]
)


def token() -> str:
    value = os.environ.get("MAPILLARY_TOKEN", "").strip()
    if not value:
        raise SystemExit("Set MAPILLARY_TOKEN (it is not stored in the repo).")
    return value


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def image_info(image_id: str, access_token: str) -> dict:
    url = f"{GRAPH}/{image_id}?" + urllib.parse.urlencode(
        {"fields": FIELDS, "access_token": access_token}
    )
    return get_json(url)


def sequence_images(sequence_id: str, access_token: str) -> list[dict]:
    # sequence_ids returns the whole sequence in one response (limit is ignored).
    url = f"{GRAPH}/images?" + urllib.parse.urlencode(
        {
            "fields": FIELDS,
            "sequence_ids": sequence_id,
            "limit": "2000",
            "access_token": access_token,
        }
    )
    images = get_json(url).get("data") or []
    if not images:
        raise SystemExit(f"No images for sequence {sequence_id}.")
    images.sort(key=lambda im: (im.get("captured_at") or 0, int(im["id"])))
    return images


def download_file(url: str, path: Path) -> int:
    if path.exists() and path.stat().st_size > 10_000:
        return path.stat().st_size
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as r:
        blob = r.read()
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(blob)
    tmp.replace(path)
    return len(blob)


def public_record(im: dict) -> dict:
    return {k: v for k, v in im.items() if k not in URL_FIELDS}
