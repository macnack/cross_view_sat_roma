"""Download one sequence, or a short sample from each user in a scan."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .client import download_file, image_info, public_record, sequence_images, token
from .geo import even_sample, longest_sequence_per_user, summarize


def download_sequence(
    *,
    image_id: Optional[str],
    sequence_id: Optional[str],
    out_root: Path,
    quality: str,
    workers: int,
) -> Path:
    access_token = token()
    seed = None
    if image_id:
        seed = image_info(image_id, access_token)
        sequence_id = seed["sequence"]
    images = sequence_images(sequence_id, access_token)
    user = (images[0].get("creator") or {}).get("username") or "unknown"
    dest = out_root / user / sequence_id
    img_dir = dest / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    summary, _, _ = _fetch_jpegs(
        images, img_dir, quality, workers, dest, summarize_all=images, seed=seed, image_id=image_id
    )
    print(json.dumps(summary, indent=2))
    return dest


def _fetch_jpegs(chosen, img_dir, quality, workers, dest, summarize_all, seed, image_id, extra=None):
    url_key = "thumb_original_url" if quality == "original" else "thumb_2048_url"
    jobs = []
    for im in chosen:
        url = im.get(url_key)
        if not url:
            raise SystemExit(f"Image {im['id']} has no {url_key}.")
        jobs.append((im["id"], url, img_dir / f"{im['id']}.jpg"))
    nbytes = 0
    failed = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(download_file, url, path): iid for iid, url, path in jobs}
        for fut in as_completed(futs):
            iid = futs[fut]
            try:
                nbytes += fut.result()
            except Exception as exc:  # noqa: BLE001 — one frame should not abort the rest
                failed.append({"id": iid, "error": str(exc)})
            done += 1
            if len(jobs) > 50 and (done % 50 == 0 or done == len(jobs)):
                print(f"{done}/{len(jobs)}  {nbytes / 1e6:.1f} MB", flush=True)
    summary = summarize(summarize_all)
    summary["quality"] = quality
    summary["bytes"] = nbytes
    summary["n_failed"] = len(failed)
    if extra:
        summary.update(extra)
    else:
        summary["seed_image"] = image_id
        if seed is not None:
            summary["seed_in_sequence"] = any(im["id"] == seed["id"] for im in summarize_all)
    records = chosen if extra else summarize_all
    (dest / "images.json").write_text(json.dumps([public_record(im) for im in records]))
    (dest / "summary.json").write_text(json.dumps(summary, indent=2))
    if failed:
        (dest / "failed.json").write_text(json.dumps(failed, indent=2))
    return summary, nbytes, failed


def write_user_montage(path: Path, rows: list[tuple[str, Path]]) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow missing; skipped user montage")
        return
    tiles = []
    for label, img_path in rows:
        im = Image.open(img_path).convert("RGB")
        im = im.resize((1024, 512), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (1024, 512 + 28), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 4), label, fill=(0, 0, 0))
        canvas.paste(im, (0, 28))
        tiles.append(canvas)
    sheet = Image.new("RGB", (1024, sum(t.height for t in tiles)))
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y))
        y += tile.height
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=85)


def sample_users(scan_path: Path, out_root: Path, n_frames: int, quality: str, workers: int) -> None:
    report = json.loads(scan_path.read_text())
    picks = longest_sequence_per_user(report["sequences"])
    if not picks:
        raise SystemExit(f"No sequences in {scan_path}.")
    access_token = token()
    montage = []
    manifest = []
    for pick in picks:
        images = sequence_images(pick["sequence"], access_token)
        chosen = even_sample(images, n_frames)
        user = (images[0].get("creator") or {}).get("username") or pick.get("username") or "unknown"
        dest = out_root / user / pick["sequence"]
        img_dir = dest / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        summary, nbytes, failed = _fetch_jpegs(
            chosen,
            img_dir,
            quality,
            workers,
            dest,
            summarize_all=images,
            seed=None,
            image_id=None,
            extra={
                "sample_frames": len(chosen),
                "sample_ids": [im["id"] for im in chosen],
                "picked_because": "longest sequence of this user inside the scan bbox",
            },
        )
        mid = chosen[len(chosen) // 2]
        preview = img_dir / f"{mid['id']}.jpg"
        label = (
            f"{user}   {summary.get('model') or summary.get('camera_type')}   "
            f"{summary['n']} frames, sample {len(chosen)}"
        )
        if preview.exists():
            montage.append((label, preview))
        manifest.append(
            {
                "username": user,
                "sequence": pick["sequence"],
                "model": summary.get("model"),
                "n_sequence": summary["n"],
                "n_sample": len(chosen) - len(failed),
                "path_length_m": summary["path_length_m"],
                "dir": str(dest),
            }
        )
        print(
            f"{user}: {len(chosen) - len(failed)}/{summary['n']} frames, "
            f"{nbytes / 1e6:.1f} MB, {pick['sequence']}",
            flush=True,
        )
    montage_path = out_root / "samples" / f"{scan_path.stem}_users.jpg"
    write_user_montage(montage_path, montage)
    manifest_path = out_root / "samples" / f"{scan_path.stem}_users.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")
    print(f"wrote {montage_path}")
