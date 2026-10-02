"""Run control for the long VIGOR training (scripts/train_vigor.py): epochs -> steps, a resumable epoch-seeded
sampler, periodic lean checkpoints, the resume state (optimizer + RNG + counters) and the --profile probe.

Kept free of the model so the arithmetic, the checkpoint contents and resume can be tested on CPU with a stub.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


# ---- epochs / names ---------------------------------------------------------------------------------------------

def steps_per_epoch(n_samples: int, batch: int) -> int:
    """Optimizer steps per epoch with drop_last=True."""
    if batch <= 0:
        raise ValueError("batch must be positive")
    spe = int(n_samples) // int(batch)
    if spe <= 0:
        raise ValueError(f"{n_samples} samples cannot fill one batch of {batch}")
    return spe


def epoch_steps(n_samples: int, batch: int, epochs: int) -> int:
    """Total steps of `epochs` passes over n_samples at `batch` with drop_last=True."""
    return int(epochs) * steps_per_epoch(n_samples, batch)


def epoch_ckpt_path(ckpt_dir, tag: str, epoch: int) -> Path:
    return Path(ckpt_dir) / f"vigor_{tag}_ep{int(epoch):03d}.pt"


def resume_path(ckpt_dir, tag: str) -> Path:
    return Path(ckpt_dir) / f"vigor_{tag}_resume.pt"


def is_epoch_save(step: int, spe: int, every_epochs: int) -> int:
    """The epoch number when `step` closes an epoch that is a multiple of every_epochs, else 0."""
    if every_epochs <= 0 or spe <= 0 or step % spe:
        return 0
    e = step // spe
    return e if e % every_epochs == 0 else 0


class EpochSampler(Sampler):
    """A permutation of range(n) seeded by (seed, epoch), optionally skipping the first `skip` indices.

    Iterated in the main process every time the DataLoader starts an epoch (persistent workers included), so a
    resumed run sees exactly the sample order it would have seen: set_epoch(e, skip=samples already used).

    Data parallel (world > 1, bevloc.model.ddp): the permutation is the single-GPU one, truncated to a multiple of the
    GLOBAL batch (what drop_last drops on one GPU), and rank r takes its positions r, r + W, r + 2W, ... after `skip`
    (a multiple of the global batch). Rank r's j-th batch of b = B / W is then positions r::W of the single-GPU j-th
    batch of B: the ranks' batches at one step are disjoint and together ARE the single-GPU global batch, every
    sample is drawn once per epoch in total, and a resume (skip = steps done x B) continues every shard."""

    def __init__(self, n: int, seed: int = 0, rank: int = 0, world: int = 1, global_batch: int | None = None):
        self.n, self.seed = int(n), int(seed)
        self.rank, self.world = int(rank), int(world)
        self.global_batch = int(global_batch) if global_batch else None
        if self.world > 1:
            if not self.global_batch or self.global_batch % self.world:
                raise ValueError("a sharded EpochSampler needs a global batch that is a multiple of the world size")
            if not 0 <= self.rank < self.world:
                raise ValueError(f"rank {rank} outside world {world}")
        self.epoch, self.skip = 0, 0

    def set_epoch(self, epoch: int, skip: int = 0):
        if self.world > 1 and int(skip) % self.global_batch:
            raise ValueError(f"skip {skip} is not a whole number of global batches ({self.global_batch})")
        self.epoch, self.skip = int(epoch), int(skip)

    def order(self, epoch: int):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + int(epoch))
        return torch.randperm(self.n, generator=g)

    def _used(self):
        return (self.n // self.global_batch) * self.global_batch if self.world > 1 else self.n

    def __iter__(self):
        o = self.order(self.epoch)
        if self.world > 1:
            return iter(o[:self._used()][self.skip:][self.rank::self.world].tolist())
        return iter(o[self.skip:].tolist())

    def __len__(self):
        m = max(0, self._used() - self.skip)
        return m // self.world if self.world > 1 else m


# ---- checkpoint contents ----------------------------------------------------------------------------------------

def param_groups_report(named: dict) -> dict:
    """{group: (n_params, bytes)} of state dicts {group: state_dict}; tensors only."""
    out = {}
    for g, sd in named.items():
        n = sum(int(v.numel()) for v in sd.values() if torch.is_tensor(v))
        b = sum(int(v.numel() * v.element_size()) for v in sd.values() if torch.is_tensor(v))
        out[g] = (n, b)
    return out


def assert_no_frozen_encoder(state: dict, encoder: torch.nn.Module | None = None):
    """Raise when a checkpoint dict carries the frozen encoder: by key name ('encoder.' / the timm backbone) and, if
    `encoder` is given, by tensor identity with its parameters."""
    bad = [f"{grp}/{k}" for grp in ("query", "decoder") for k in (state.get(grp) or {})
           if k.startswith("encoder.") or ".encoder." in k or "backbone" in k]
    if encoder is not None:
        ptrs = {p.data_ptr() for p in encoder.parameters()}
        bad += [f"{grp}/{k}" for grp in ("query", "decoder") for k, v in (state.get(grp) or {}).items()
                if torch.is_tensor(v) and v.data_ptr() in ptrs]
    if bad:
        raise AssertionError(f"checkpoint carries frozen-encoder tensors: {bad[:5]} (+{max(0, len(bad) - 5)})")


def _distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def rng_state() -> dict:
    """Global RNG states. Under torchrun (a process group is up) only this rank's CUDA device: touching the other
    GPUs would open a CUDA context on each of them from every rank."""
    s = dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state() if _distributed() else torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s: dict):
    torch.set_rng_state(s["torch"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])
    if "cuda" in s and torch.cuda.is_available():
        if torch.is_tensor(s["cuda"]):             # one device (written under torchrun)
            torch.cuda.set_rng_state(s["cuda"])
        elif _distributed():                       # a single-GPU file read under torchrun: refused later anyway
            torch.cuda.set_rng_state(s["cuda"][0])
        else:
            torch.cuda.set_rng_state_all(s["cuda"])


def atomic_save(obj, path):
    """torch.save to path via a temporary file + fsync + rename, so a job killed mid-write (SLURM time limit, node
    failure) leaves the previous file intact; a stale `<name>.tmp` from such a kill is simply overwritten next time."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def check_resume(counters: dict, batch: int, spe: int, n_train: int, tag: str | None = None):
    """Raise SystemExit unless a resume file's counters match this run: same batch, steps per epoch and training-set
    size (the epoch permutation is over range(n_train), so a different set would silently reorder every epoch) and,
    when both are known, the same tag. Files written before n_train was stored skip that check."""
    bad = []
    if int(counters["batch"]) != int(batch):
        bad.append(f"batch {counters['batch']} != {batch}")
    if int(counters["steps_per_epoch"]) != int(spe):
        bad.append(f"steps per epoch {counters['steps_per_epoch']} != {spe}")
    if counters.get("n_train") is not None and int(counters["n_train"]) != int(n_train):
        bad.append(f"training samples {counters['n_train']} != {n_train}")
    if tag is not None and counters.get("tag") not in (None, tag):
        bad.append(f"tag {counters['tag']!r} != {tag!r}")
    if bad:
        raise SystemExit("resume refused: " + "; ".join(bad) + " (resume needs the same batch and training set)")


def truncate_log(path, k0: int) -> int:
    """Drop the CSV rows (after the header) whose step is > k0: the steps a killed segment ran after its last resume
    save are re-run by the resumed job and would otherwise appear twice. Returns the number of rows dropped."""
    path = Path(path)
    if not path.is_file():
        return 0
    lines = path.read_text().splitlines(keepends=True)
    keep, drop = lines[:1], 0
    for ln in lines[1:]:
        try:
            ok = int(ln.split(",", 1)[0]) <= k0 and ln.endswith("\n")
        except ValueError:                         # a torn last line from the kill
            ok = False
        if ok:
            keep.append(ln)
        else:
            drop += 1
    if drop:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(keep))
        os.replace(tmp, path)
    return drop


def make_resume(ckpt: dict, opt, counters: dict, loader_gen: torch.Generator | None = None) -> dict:
    """The lean checkpoint dict + optimizer state + RNG (global, and the DataLoader's own generator) + loop counters
    (step, best, best_step, ...)."""
    rng = rng_state()
    if loader_gen is not None:
        rng["loader"] = loader_gen.get_state()
    return dict(ckpt, optimizer=opt.state_dict(), rng=rng, counters=dict(counters))


def load_resume(path, query, decoder, opt, loader_gen: torch.Generator | None = None):
    """Restore query / decoder / optimizer / RNG from a resume file; returns its counters dict. Loaded on the CPU
    (the RNG states must stay CPU tensors); load_state_dict copies weights and optimizer moments to their devices."""
    r = torch.load(path, map_location="cpu", weights_only=False)
    query.load_state_dict(r["query"])
    decoder.load_state_dict(r["decoder"], strict=False)
    opt.load_state_dict(r["optimizer"])
    set_rng_state(r["rng"])
    if loader_gen is not None and "loader" in r["rng"]:
        loader_gen.set_state(r["rng"]["loader"])
    return dict(r["counters"])


# ---- --profile ----------------------------------------------------------------------------------------------

class GpuUtil:
    """Samples `nvidia-smi` GPU utilisation / memory every `ms` while active; mean() is None when unavailable."""

    def __init__(self, ms=250, index=None):
        self.ms, self.samples, self.proc, self.th = ms, [], None, None
        self.index = index                         # this process's GPU among CUDA_VISIBLE_DEVICES (torchrun rank)

    def __enter__(self):
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        i = int(self.index or 0)
        idx = (vis[i] if i < len(vis) and vis[i] else str(i)) if vis != [""] else str(i)
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "-i", idx, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits",
                 f"-lms={self.ms}"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except (OSError, ValueError):
            self.proc = None
            return self

        def read():
            for line in self.proc.stdout:
                try:
                    u, m = (float(x) for x in line.strip().split(","))
                    self.samples.append((u, m))
                except ValueError:
                    pass
        self.th = threading.Thread(target=read, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        return False

    def mean(self):
        if not self.samples:
            return None, None
        a = np.array(self.samples)
        return float(a[:, 0].mean()), float(a[:, 1].max())


def profile_summary(batch, t_data, t_h2d, t_compute, peak_alloc, peak_reserved, total_mem, util=None, extra=None):
    """The --profile report: per-step means (s) over the timed steps, data-wait share of the step, samples/s."""
    t_data, t_h2d, t_compute = (np.asarray(x, float) for x in (t_data, t_h2d, t_compute))
    step = t_data + t_h2d + t_compute
    gib = 1024 ** 3
    out = dict(batch=int(batch), steps=int(len(step)),
               step_s=float(step.mean()) if len(step) else float("nan"),
               data_wait_s=float(t_data.mean()) if len(step) else float("nan"),
               h2d_s=float(t_h2d.mean()) if len(step) else float("nan"),
               compute_s=float(t_compute.mean()) if len(step) else float("nan"),
               data_wait_frac=float(t_data.sum() / step.sum()) if len(step) and step.sum() > 0 else float("nan"),
               data_wait_p90_s=float(np.percentile(t_data, 90)) if len(step) else float("nan"),
               samples_per_s=float(batch * len(step) / step.sum()) if len(step) and step.sum() > 0 else float("nan"),
               peak_alloc_gib=peak_alloc / gib, peak_reserved_gib=peak_reserved / gib,
               total_gib=total_mem / gib if total_mem else None,
               headroom_frac=(1.0 - peak_reserved / total_mem) if total_mem else None,
               gpu_util_pct=util[0] if util else None, gpu_mem_used_mib=util[1] if util else None)
    out.update(extra or {})
    return out


def aggregate_profiles(rows, global_batch):
    """The node-level PROFILE line of a torchrun --profile: global batch x timed steps / the mean per-rank wall time
    (the ranks are synchronised by the per-step all-reduce), next to the sum and spread of the per-rank rates."""
    ok = [r for r in rows if not r.get("oom") and r.get("steps")]
    if not ok:
        return dict(aggregate=True, global_batch=int(global_batch), world_size=len(rows), oom=True)
    step_s = float(np.mean([r["step_s"] for r in ok]))
    per = [r["samples_per_s"] for r in ok]
    return dict(aggregate=True, global_batch=int(global_batch), world_size=len(rows), steps=int(ok[0]["steps"]),
                step_s=step_s, samples_per_s=float(global_batch / step_s), samples_per_s_sum_ranks=float(sum(per)),
                samples_per_s_rank_min=float(min(per)), samples_per_s_rank_max=float(max(per)),
                data_wait_frac_max=float(max(r["data_wait_frac"] for r in ok)),
                peak_reserved_gib_max=float(max(r["peak_reserved_gib"] for r in ok)),
                gpu_util_pct_mean=(float(np.mean([r["gpu_util_pct"] for r in ok if r.get("gpu_util_pct") is not None]))
                                   if any(r.get("gpu_util_pct") is not None for r in ok) else None),
                breakdown_s_rank0=ok[0].get("breakdown_s"), tf32=ok[0].get("tf32"),
                encoder_dtype=ok[0].get("encoder_dtype"), compile=ok[0].get("compile"), oom=False)


def sample_cost(ds, n=16, seed=0):
    """Per-sample CPU cost of a VigorPairs item in this process: total ds[i] and its parts (panorama JPEG decode,
    panorama resize, depth PNG read + resize, reference tile read + resample). Seconds, means over n random items."""
    import cv2
    from bevloc.data.vigor import depth_png_path, read_depth_png
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    parts = {k: [] for k in ("total", "total_warm", "pano_decode", "pano_resize", "depth", "reference")}
    for i in idx:                                  # "total" = first (cold file cache) read; the parts are warm
        lab = ds.labels[int(i)]
        t = time.perf_counter()
        ds[int(i)]
        parts["total"].append(time.perf_counter() - t)
    for i in idx:
        lab = ds.labels[int(i)]
        t = time.perf_counter()
        ds[int(i)]
        parts["total_warm"].append(time.perf_counter() - t)
        t = time.perf_counter()
        pano = cv2.imread(str(ds.root / lab["city"] / "panorama" / lab["pano"]), cv2.IMREAD_COLOR)
        parts["pano_decode"].append(time.perf_counter() - t)
        t = time.perf_counter()
        cv2.resize(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB), (ds.erp_w, ds.erp_h), interpolation=cv2.INTER_AREA)
        parts["pano_resize"].append(time.perf_counter() - t)
        if ds.depth:
            t = time.perf_counter()
            cv2.resize(read_depth_png(depth_png_path(ds.root, lab["city"], lab["pano"])), (ds.erp_w, ds.erp_h),
                       interpolation=cv2.INTER_NEAREST)
            parts["depth"].append(time.perf_counter() - t)
        t = time.perf_counter()
        if ds.ref_window_m is None:
            ds.reference(lab["city"], lab["sat"])
        else:
            ds.window_reference(lab["city"], lab["sat"], np.zeros(2))
        parts["reference"].append(time.perf_counter() - t)
    return {k: (float(np.mean(v)) if v else None) for k, v in parts.items()}


def print_profile(d: dict):
    print("PROFILE " + json.dumps(d, sort_keys=False), flush=True)
