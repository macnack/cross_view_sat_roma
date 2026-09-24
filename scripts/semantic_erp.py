"""Semantic classes of a Mapillary panorama from a frozen segmenter, and the camera-only contact line.

  make semantic-smoke        # one frame: class fractions, per-column contact row, a labelled sheet

The contact line v_c(u) is the lowest 'building'/'wall'/'fence' pixel per ERP column (kick-off §3.1,
decision 2026-09-17 on the LiDAR version); here it comes from a Cityscapes-trained SegFormer instead
of LiDAR, so it is available on Mapillary. Dynamic classes (car, truck, bus, person, rider, bicycle,
motorcycle, train) give the mask that removes moving objects from the IPM picture. This script is the
feasibility probe for the `ipm_cl` query on Mapillary (docs/decisions.md, 2026-09-24 next steps).
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
# Cityscapes train ids
CITYSCAPES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
              "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
              "motorcycle", "bicycle"]
GROUND = {"road", "sidewalk", "terrain"}
WALL = {"building", "wall", "fence"}
DYNAMIC = {"person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"}
PALETTE = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
                    [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]], np.uint8)


def segment(erp_rgb, model, proc, device, width=1024):
    """Class-id map (H, W) at the ERP's own resolution. The model saw pinhole crops; the ERP's lower
    half is what we need, so we feed the whole ERP resized to the model width and upsample the logits."""
    h, w = erp_rgb.shape[:2]
    small = cv2.resize(erp_rgb, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
    inputs = proc(images=small, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits                                  # (1, 19, h/4, w/4)
    logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def contact_line(seg):
    """Per-column lowest wall pixel row (or -1): the wall–ground contact seen from the camera."""
    wall = np.isin(seg, [CITYSCAPES.index(c) for c in WALL])
    h, w = seg.shape
    vc = np.full(w, -1, int)
    rows = np.arange(h)[:, None]
    lowest = np.where(wall, rows, -1).max(0)
    vc[:] = lowest
    return vc


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--seq", default="Fixtor/IcRzj0wTLZX874qitxVsQa")
    ap.add_argument("--index", type=int, default=400)
    ap.add_argument("--out", default="experiments/08_semantic")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    proc = SegformerImageProcessor.from_pretrained(MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL).to(a.device).eval()
    frames = json.loads((MAP_ROOT / a.seq / "images.json").read_text())
    fr = frames[a.index]
    erp = cv2.cvtColor(cv2.imread(str(MAP_ROOT / a.seq / "images" / f"{fr['id']}.jpg")), cv2.COLOR_BGR2RGB)
    erp = cv2.resize(erp, (1792, 896), interpolation=cv2.INTER_AREA)
    seg = segment(erp, model, proc, a.device)
    frac = {c: float((seg == i).mean()) for i, c in enumerate(CITYSCAPES)}
    vc = contact_line(seg)
    print(json.dumps({k: round(v, 3) for k, v in sorted(frac.items(), key=lambda kv: -kv[1]) if v > 0.005}))
    print(f"contact line defined on {float((vc >= 0).mean()):.2f} of columns; median row {np.median(vc[vc >= 0]) if (vc >= 0).any() else -1:.0f} of {seg.shape[0]}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    colour = PALETTE[seg]
    over = cv2.addWeighted(erp, 0.55, colour, 0.45, 0)
    for u in range(0, seg.shape[1], 4):
        if vc[u] >= 0:
            cv2.circle(over, (u, int(vc[u])), 1, (255, 255, 0), -1)
    sheet = np.vstack([erp, over])
    cv2.imwrite(str(out / f"semantic_{fr['id']}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {out / f'semantic_{fr['id']}.jpg'}")


if __name__ == "__main__":
    main()
