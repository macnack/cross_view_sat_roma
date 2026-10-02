"""Compare step-1 gradient dumps of scripts/train_vigor.py --dump-grads against a reference dump.

  python scripts/grad_compare.py REF.pt OTHER.pt [OTHER2.pt ...] [--json out.json]

Per dump: relative L2 distance ||g - g_ref|| / ||g_ref|| over all trainable tensors present in both (concatenated),
the share of entries whose torch.sign differs from the reference's (a zero where the reference is non-zero counts:
that is an underflowed component), the share of exactly-zero entries (both dumps), the cosine similarity, and the
relative L2 per parameter group (query.head, decoder.embedding_decoder, decoder.gps, decoder.proj, decoder.conv_refiner)
and the three worst tensors. One `GRADCMP` line per dump (the format of experiments/13_panoroma_long/ddp_cmp/review).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["grads"], d.get("meta", {})


def group_of(name):
    parts = name.split(".")
    return ".".join(parts[:2])


def compare(ref, other):
    keys = [k for k in ref if k in other]
    missing = sorted(set(ref) ^ set(other))
    r = torch.cat([ref[k].reshape(-1).double() for k in keys])
    o = torch.cat([other[k].reshape(-1).double() for k in keys])
    d = o - r
    out = dict(rel_l2=float(d.norm() / r.norm().clamp_min(1e-300)),
               sign_flip_pct=float((torch.sign(o) != torch.sign(r)).double().mean() * 100),
               zero_pct=float((o == 0).double().mean() * 100), zero_pct_ref=float((r == 0).double().mean() * 100),
               cosine=float(torch.dot(o, r) / (o.norm() * r.norm()).clamp_min(1e-300)),
               norm_ratio=float(o.norm() / r.norm().clamp_min(1e-300)),
               n_entries=int(r.numel()), n_tensors=len(keys), mismatched_keys=missing[:10])
    groups = {}
    for k in keys:
        g = groups.setdefault(group_of(k), [0.0, 0.0])
        g[0] += float((other[k].double() - ref[k].double()).pow(2).sum())
        g[1] += float(ref[k].double().pow(2).sum())
    out["per_group_rel_l2"] = {g: (a / b) ** 0.5 if b > 0 else float("nan") for g, (a, b) in groups.items()}
    worst = sorted(((float((other[k].double() - ref[k].double()).norm() / ref[k].double().norm().clamp_min(1e-300)), k)
                    for k in keys), reverse=True)[:3]
    out["worst"] = worst
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("others", nargs="+")
    ap.add_argument("--json", default=None, help="also write all comparisons to this json")
    a = ap.parse_args()
    ref, meta_ref = load(a.ref)
    print(f"reference {a.ref}: {meta_ref}", flush=True)
    rows = {}
    for p in a.others:
        g, meta = load(p)
        c = compare(ref, g)
        c["meta"] = meta
        rows[p] = c
        print(f"GRADCMP {Path(p).stem} vs {Path(a.ref).stem}: rel L2 {c['rel_l2']:.3e}  sign flips "
              f"{c['sign_flip_pct']:.4f}%  zeros {c['zero_pct']:.4f}% (ref {c['zero_pct_ref']:.4f}%)  cos "
              f"{c['cosine']:.6f}  |g|/|ref| {c['norm_ratio']:.4f}  worst {c['worst']}", flush=True)
        print("   per group rel L2: " + str({k: f"{v:.2e}" for k, v in c["per_group_rel_l2"].items()}), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(dict(ref=a.ref, meta_ref=meta_ref, rows=rows), indent=2))


if __name__ == "__main__":
    main()
