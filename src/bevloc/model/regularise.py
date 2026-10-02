"""Regularisation options of scripts/train_vigor.py (2026-09-30): AdamW parameter groups with decoupled weight decay
on the weights only, the `_best.pt` selection rule, and the settings record kept in checkpoints / resume files.

Before this module every run used ONE AdamW weight decay, cfg.train.weight_decay (0.01), on every trainable tensor of
both groups (query head and decoder), biases and normalisation weights included. `param_groups` keeps exactly that
(same groups, same order, same tensors) unless a per-module decay is given; then that module's group is split into
{weights: decay w} and {biases, norm weights, other 1-D tensors: decay 0}, same learning rate.
"""
from __future__ import annotations

import math

import torch.nn as nn

NORM_TYPES = (nn.modules.batchnorm._NormBase, nn.LayerNorm, nn.GroupNorm, nn.LocalResponseNorm)
SELECT_RULES = ("pose", "loss")


def no_decay_names(module: nn.Module) -> set:
    """Names of the parameters that get no weight decay: every bias, every parameter of a normalisation layer
    (BatchNorm / LayerNorm / GroupNorm / InstanceNorm), and any other tensor with fewer than 2 dimensions
    (LayerScale gammas, temperatures)."""
    out = set()
    for mname, m in module.named_modules():
        for pname, p in m.named_parameters(recurse=False):
            full = f"{mname}.{pname}" if mname else pname
            if isinstance(m, NORM_TYPES) or pname.endswith("bias") or p.ndim < 2:
                out.add(full)
    return out


def split_decay(module: nn.Module):
    """(decay params, no-decay params) among the module's trainable parameters, in named_parameters order."""
    skip = no_decay_names(module)
    dec, nod = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        (nod if name in skip else dec).append(p)
    return dec, nod


def module_groups(module: nn.Module, lr: float, weight_decay=None, name=""):
    """AdamW groups of one module: [{"params": all trainable, "lr": lr}] (the optimiser's default decay) when
    weight_decay is None; else the decay / no-decay split with explicit decays."""
    params = [p for p in module.parameters() if p.requires_grad]
    if not params:
        return []
    if weight_decay is None:
        return [{"params": params, "lr": lr}]
    dec, nod = split_decay(module)
    gs = []
    if dec:
        gs.append({"params": dec, "lr": lr, "weight_decay": float(weight_decay), "name": f"{name}_decay"})
    if nod:
        gs.append({"params": nod, "lr": lr, "weight_decay": 0.0, "name": f"{name}_no_decay"})
    return gs


def groups_report(opt) -> str:
    return "  ".join(f"{g.get('name', f'group{i}')}: {sum(p.numel() for p in g['params']) / 1e6:.2f} M "
                     f"(lr {g['lr']:g}, wd {g['weight_decay']:g})" for i, g in enumerate(opt.param_groups))


def select_score(rule: str, v: dict, pose_m: float, vce_on: bool) -> float:
    """The number `_best.pt` minimises. "pose" (default = the rule of every earlier run): the validation VCE
    Procrustes pose error (vce_pose_m) when VCE is on and finite, else the heat-map pose error pose_m (metres);
    "loss": the validation cell cross-entropy (plain, unsmoothed)."""
    if rule == "pose":
        return v["vce_pose_m"] if vce_on and v["vce_pose_m"] == v["vce_pose_m"] else pose_m
    if rule == "loss":
        return float(v["ce"])
    raise ValueError(f"select_by must be one of {SELECT_RULES}, got {rule!r}")


DEFAULTS = dict(label_smoothing=0.0, weight_decay_decoder=None, weight_decay_head=None, feat_dropout=0.0,
                select_by="pose", aug=None)


def resolve(cfg, a) -> dict:
    """The regularisation settings of a run: CLI flag if given, else cfg.train.<key>, else the default (off)."""
    T = getattr(cfg, "train", None)
    out = {}
    for k in ("label_smoothing", "weight_decay_decoder", "weight_decay_head", "feat_dropout", "select_by"):
        v = getattr(a, k, None)
        if v is None:
            v = getattr(T, k, DEFAULTS[k]) if T is not None else DEFAULTS[k]
        out[k] = v
    out["label_smoothing"] = float(out["label_smoothing"] or 0.0)
    out["feat_dropout"] = float(out["feat_dropout"] or 0.0)
    for k in ("weight_decay_decoder", "weight_decay_head"):
        out[k] = None if out[k] is None else float(out[k])
    out["select_by"] = str(out["select_by"] or "pose")
    if out["select_by"] not in SELECT_RULES:
        raise SystemExit(f"--select-by must be one of {SELECT_RULES}")
    if not 0.0 <= out["label_smoothing"] < 1.0 or not 0.0 <= out["feat_dropout"] < 1.0:
        raise SystemExit("--label-smoothing and --feat-dropout must be in [0, 1)")
    return out


def mismatch(saved: dict | None, now: dict) -> list:
    """Differences between a resume file's regularisation record (None = written before 2026-09-30 = all defaults)
    and this segment's settings."""
    s = dict(DEFAULTS)
    s.update(saved or {})
    bad = []
    for k in sorted(set(DEFAULTS) | set(now)):
        a, b = s.get(k), now.get(k)
        same = (a == b) or (isinstance(a, float) and isinstance(b, float) and math.isclose(a, b))
        if not same:
            bad.append(f"{k}: resume file {a!r} != now {b!r}")
    return bad
