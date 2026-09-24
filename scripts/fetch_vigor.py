"""Download the VIGOR release from the shared Google Drive folder with gdown (resumable).

  make eagle-fetch-vigor            # on Eagle: submits slurm/fetch_vigor.sbatch (CPU node, internet)

Layout written under --out (default $VIGOR_DIR):
  <out>/VIGOR_files/<City>/{panorama,satellite}.tar.gz, <out>/VIGOR_files/splits.zip   (raw downloads)
  <out>/<City>/{panorama,satellite}/...                                                (extracted)
  <out>/splits/...
Each file is downloaded on its own, skipped when already complete (size matches Drive's), so a rerun
after a quota error or a timeout continues where it stopped. Google Drive limits anonymous downloads
per file per day; a "Too many users have viewed or downloaded this file" error means: wait and rerun.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

FOLDER = "https://drive.google.com/drive/folders/1W2KvoeFKh9zW4f5drGamT-RpGDDELZw_"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.environ.get("VIGOR_DIR", "data/vigor"))
    ap.add_argument("--folder", default=FOLDER)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--only-extract", action="store_true", help="skip all downloads; extract completed files")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    listing = []
    if not a.only_extract:
        import gdown
        listing = gdown.download_folder(url=a.folder, skip_download=True, quiet=True)
        if not listing:
            sys.exit("could not list the Drive folder (private, or Google blocked this address)")
        print(f"{len(listing)} files in the folder", flush=True)
    for f in listing:
        dest = out / f.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        done = dest.with_suffix(dest.suffix + ".done")
        if done.exists() and dest.exists():
            print(f"skip   {f.path}  ({dest.stat().st_size / 1e9:.2f} GB, already complete)", flush=True)
            continue
        for attempt in range(1, a.retries + 1):
            t0 = time.time()
            print(f"fetch  {f.path}  (attempt {attempt})", flush=True)
            try:
                got = gdown.download(id=f.id, output=str(dest), quiet=False, resume=True)
            except Exception as e:                                   # quota, network, ...
                got = None
                print(f"  error: {type(e).__name__}: {str(e)[:300]}", flush=True)
            if got and dest.exists() and dest.stat().st_size > 0:
                done.write_text(f"{dest.stat().st_size}\n")
                print(f"  ok {dest.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)
                break
            time.sleep(30 * attempt)
        else:
            failed.append(f.path)
    if failed:
        print("FAILED (rerun later, Drive quota resets after ~24 h):", *failed, sep="\n  ", flush=True)
    if a.no_extract:
        sys.exit(2 if failed else 0)
    for tar in sorted((out / "VIGOR_files").glob("*/*.tar.gz")):
        if not tar.with_suffix(tar.suffix + ".done").exists():        # incomplete or failed download: never extract
            print(f"skip extract {tar.relative_to(out)} (download not complete)", flush=True)
            continue
        target = out / tar.parent.name
        marker = target / f".{tar.stem}.extracted"
        if marker.exists():
            print(f"skip extract {tar.relative_to(out)}", flush=True)
            continue
        target.mkdir(parents=True, exist_ok=True)
        print(f"extract {tar.relative_to(out)} -> {target}", flush=True)
        subprocess.run(["tar", "-xzf", str(tar), "-C", str(target)], check=True)
        marker.write_text("ok\n")
    z = out / "VIGOR_files" / "splits.zip"
    if z.exists() and not (out / "splits").exists():
        import zipfile                                    # the container has no `unzip`
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out / "splits")
    print("done" + (f" with {len(failed)} file(s) still missing" if failed else ""), flush=True)
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
