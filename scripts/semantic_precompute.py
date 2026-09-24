"""Precompute SegFormer (Cityscapes) class-id maps for every frame of the given Mapillary sequences.

  make semantic-precompute SEQS="Fixtor/iHfmEq03Tc6752Y4Ke8wlC Fixtor/IcRzj0wTLZX874qitxVsQa ..."

Writes data/mapillary/<seq>/semantic/<id>.png (uint8 class ids at 1792x896, Cityscapes train ids;
255 = not computed). MapillaryPairs uses them for the ipm query's validity mask (dynamic objects,
optional ground-only) when present; frames without a map fall back to the geometric mask alone.
One H100 does ~20 frames/s; the whole Fixtor set (10k frames) takes ~10 min.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from bevloc import config as C
from bevloc.data.mapillary import MAP_ROOT

MODEL = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
SEM_HW = (896, 1792)


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--seqs", nargs="+", required=True, help="sequence folders under data/mapillary")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = SegformerImageProcessor.from_pretrained(MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL).to(dev).eval()
    if dev == "cuda":
        model = model.half()
    for seq in a.seqs:
        d = MAP_ROOT / seq
        frames = json.loads((d / "images.json").read_text())
        out = d / "semantic"
        out.mkdir(exist_ok=True)
        todo = [fr for fr in frames if a.overwrite or not (out / f"{fr['id']}.png").exists()]
        print(f"{seq}: {len(todo)}/{len(frames)} frames to segment", flush=True)
        for k in range(0, len(todo), a.batch):
            chunk = todo[k:k + a.batch]
            imgs = []
            for fr in chunk:
                im = cv2.cvtColor(cv2.imread(str(d / "images" / f"{fr['id']}.jpg")), cv2.COLOR_BGR2RGB)
                imgs.append(cv2.resize(im, (1024, 512), interpolation=cv2.INTER_AREA))
            inputs = proc(images=imgs, return_tensors="pt").to(dev)
            if dev == "cuda":
                inputs = {kk: (v.half() if v.dtype == torch.float32 else v) for kk, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits.float()
                logits = torch.nn.functional.interpolate(logits, size=SEM_HW, mode="bilinear", align_corners=False)
                ids = logits.argmax(1).to(torch.uint8).cpu().numpy()
            for fr, m in zip(chunk, ids):
                cv2.imwrite(str(out / f"{fr['id']}.png"), m)
            if (k // a.batch) % 25 == 0:
                print(f"  {k + len(chunk)}/{len(todo)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
