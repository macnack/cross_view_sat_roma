"""Single-node multi-GPU data parallelism for scripts/train_vigor.py (torchrun), kept free of the model.

Launch: `torchrun --standalone --nproc_per_node W scripts/train_vigor.py ... --batch B` (slurm/run_ddp.sbatch). Without
torchrun (no WORLD_SIZE in the environment) `Dist` is the single-process no-op and the script is unchanged.

--batch is ALWAYS the global (optimizer) batch B; each of the W ranks draws b = B / W samples per step, so the
optimisation (steps per epoch = len(train) // B, lr, epochs) is that of a single-GPU run at batch B.

Why the gradient average is done here and not by torch.nn.parallel.DistributedDataParallel: DDP only reduces
gradients of a forward that goes through its wrapper, and the trainable decoder is called from inside the matcher
(`matcher.model.decoder(...)`, with the RefinerTap hooks on its submodules), while `query.placement` and
`decoder_state(matcher.model.decoder)` expect the bare modules. Wrapping would mean rewriting those call sites and the
checkpoint code; instead, after backward, `Dist.reduce_step` all-reduces ONE flat fp32 buffer holding every trainable
gradient (mean over ranks = DDP's average), the per-rank "no update" flags and the logged statistics. Same maths as
DDP without the backward overlap, which costs ~0.3 GB over NVLink per step against a ~1 s step. The modules stay
unwrapped, so every state dict keeps its keys (no `module.` prefix) and the evaluators load the checkpoints unchanged.
"""
from __future__ import annotations

import datetime
import math
import os

import torch


def env_world() -> tuple[int, int, int]:
    """(rank, world size, local rank) from torchrun's environment; (0, 1, 0) when not launched by torchrun."""
    w = int(os.environ.get("WORLD_SIZE", "1") or 1)
    if w <= 1:
        return 0, 1, 0
    return int(os.environ["RANK"]), w, int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))


def per_rank_batch(global_batch: int, world: int) -> int:
    """b = B / W; refuses a global batch that the ranks cannot split evenly (unequal shards would bias the mean)."""
    if global_batch <= 0 or world <= 0:
        raise ValueError("batch and world size must be positive")
    if global_batch % world:
        raise SystemExit(f"--batch {global_batch} is the GLOBAL batch and must be a multiple of the {world} ranks")
    return global_batch // world


# the per-step statistics of train_lift_splat.step() that are logged; n-weighted means (like validate's avg), plain
# means over the ranks that have a finite value, and sums
STAT_WMEAN = ("ce", "acc", "cell_err", "pose_nll", "pose_err")
STAT_MEAN = ("vce_m", "vce_pose_m", "fine_epe_px", "fine_epe_in_px")
STAT_SUM = ("n", "fine_skipped")


class Dist:
    """The process group of a torchrun launch (world > 1) or the single-process no-op (world == 1)."""

    def __init__(self, rank=0, world=1, local_rank=0, backend=None, timeout_min=60, init=True):
        self.rank, self.world, self.local_rank = int(rank), int(world), int(local_rank)
        self.on = self.world > 1
        if self.on and init and not torch.distributed.is_initialized():
            if backend is None:
                backend = "nccl" if torch.cuda.is_available() else "gloo"
            if backend == "nccl":
                torch.cuda.set_device(self.local_rank)
            # a long timeout: ranks 1..W-1 wait at a barrier while rank 0 validates and writes checkpoints
            kw = dict(rank=self.rank, world_size=self.world)
            dk = dict(device_id=torch.device("cuda", self.local_rank)) if backend == "nccl" else {}
            try:
                torch.distributed.init_process_group(backend, timeout=datetime.timedelta(minutes=timeout_min), **kw, **dk)
            except TypeError:                          # older torch without device_id
                torch.distributed.init_process_group(backend, timeout=datetime.timedelta(minutes=timeout_min), **kw)

    @classmethod
    def from_env(cls, **kw):
        return cls(*env_world(), **kw)

    @property
    def main(self) -> bool:
        return self.rank == 0

    def device(self) -> str:
        if not torch.cuda.is_available():
            return "cpu"
        return f"cuda:{self.local_rank}" if self.on else "cuda"

    def save(self, obj, path):
        """Checkpoint writes happen on rank 0 only (bevloc.model.trainrun.atomic_save); a no-op on the other ranks."""
        if self.main:
            from bevloc.model.trainrun import atomic_save
            atomic_save(obj, path)
            return True
        return False

    def barrier(self):
        if self.on:
            torch.distributed.barrier()

    def close(self):
        if self.on and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    # ---- start-up / resume -------------------------------------------------------------------------------------

    def broadcast_params(self, modules):
        """Rank 0's parameters and buffers to every rank (what DDP does at construction)."""
        if not self.on:
            return
        with torch.no_grad():
            for m in modules:
                for t in list(m.parameters()) + list(m.buffers()):
                    torch.distributed.broadcast(t.data, src=0)

    def gather_object(self, obj):
        """[obj of rank 0, ..., obj of rank W-1] on every rank (per-rank RNG states for the resume file)."""
        if not self.on:
            return [obj]
        out = [None] * self.world
        torch.distributed.all_gather_object(out, obj)
        return out

    def all_min(self, x: float) -> float:
        return self._all(x, torch.distributed.ReduceOp.MIN)

    def all_max(self, x: float) -> float:
        return self._all(x, torch.distributed.ReduceOp.MAX)

    def _all(self, x, op):
        if not self.on:
            return float(x)
        t = torch.tensor([float(x)], dtype=torch.float64, device=self._comm_device())
        torch.distributed.all_reduce(t, op=op)
        return float(t.item())

    def _comm_device(self):
        return torch.device("cuda", self.local_rank) if torch.distributed.get_backend() == "nccl" else torch.device("cpu")

    # ---- the training step ---------------------------------------------------------------------------------------

    def reduce_step(self, params, bad: bool, st: dict):
        """After a (possibly skipped) backward on every rank: ONE all-reduce of [grads | bad | stats].

        Gradients become the mean over ranks (a parameter whose grad is None on every rank stays None, as it would
        on one GPU; one missing on some ranks only counts as zero there). Returns (bad_any, st_global): the step is
        skipped on EVERY rank when any rank's loss was non-finite, and st_global holds the logged statistics of the
        global batch (n-weighted means of the heat-map terms, means of the per-batch metre terms, summed counts), so
        the non-finite guard and the CSV see the same values on all ranks. World 1: returns the inputs unchanged."""
        if not self.on:
            return bool(bad), st
        dev = self._comm_device()
        params = [p for p in params if p.requires_grad]
        sizes = [p.numel() for p in params]
        n_g = sum(sizes)
        n_p = len(params)
        stats = []
        n_loc = float(st.get("n", 0) or 0)
        for k in STAT_WMEAN:
            v = float(st.get(k, float("nan")))
            ok = math.isfinite(v) and n_loc > 0
            stats += [v * n_loc if ok else 0.0, n_loc if ok else 0.0]
        for k in STAT_MEAN:
            v = float(st.get(k, float("nan")))
            ok = math.isfinite(v)
            stats += [v if ok else 0.0, 1.0 if ok else 0.0]
        for k in STAT_SUM:
            stats.append(float(st.get(k, 0) or 0))
        buf = torch.zeros(n_g + n_p + 1 + len(stats), dtype=torch.float32, device=dev)
        have = [p.grad is not None and not bad for p in params]
        o = 0
        for p, s, h in zip(params, sizes, have):
            if h:
                buf[o:o + s].copy_(p.grad.detach().reshape(-1))
            o += s
        buf[n_g:n_g + n_p] = torch.tensor([1.0 if h else 0.0 for h in have], dtype=torch.float32, device=dev)
        buf[n_g + n_p] = 1.0 if bad else 0.0
        buf[n_g + n_p + 1:] = torch.tensor(stats, dtype=torch.float32, device=dev)
        torch.distributed.all_reduce(buf)
        tail = buf[n_g:].double().cpu()                  # one device->host copy for the flags and statistics
        bad_any = bool(tail[n_p].item() > 0)
        if not bad_any:
            o = 0
            for i, (p, s) in enumerate(zip(params, sizes)):
                if tail[i].item() > 0:
                    g = (buf[o:o + s] / self.world).view_as(p).to(p.dtype)
                    if p.grad is None:
                        p.grad = g
                    else:
                        p.grad.copy_(g)
                else:
                    p.grad = None
                o += s
        r = tail[n_p + 1:].tolist()
        out = dict(st)
        j = 0
        for k in STAT_WMEAN:
            out[k] = r[j] / r[j + 1] if r[j + 1] > 0 else float("nan")
            j += 2
        for k in STAT_MEAN:
            out[k] = r[j] / r[j + 1] if r[j + 1] > 0 else float("nan")
            j += 2
        for k in STAT_SUM:
            out[k] = int(round(r[j]))
            j += 1
        return bad_any, out


def check_resume_world(counters: dict, world: int, batch_per_rank: int):
    """Raise SystemExit unless the resume file was written with this world size and per-rank batch (files from a
    single-GPU run carry neither: they count as world 1 at the full batch). The global batch is check_resume's."""
    w0 = int(counters.get("world_size", 1) or 1)
    b0 = counters.get("batch_per_rank")
    b0 = int(counters["batch"]) if b0 is None else int(b0)
    bad = []
    if w0 != int(world):
        bad.append(f"world size {w0} != {world}")
    if b0 != int(batch_per_rank):
        bad.append(f"per-rank batch {b0} != {batch_per_rank}")
    if bad:
        raise SystemExit("resume refused: " + "; ".join(bad) + " (a run resumes with the same number of GPUs and "
                         "batch split: the per-rank sample shards and RNG streams depend on both)")


def assert_unwrapped(state: dict):
    """Raise when a checkpoint dict's query / decoder keys carry a DDP `module.` prefix (evaluators load bare keys)."""
    bad = [f"{g}/{k}" for g in ("query", "decoder") for k in (state.get(g) or {}) if k.startswith("module.")]
    if bad:
        raise AssertionError(f"checkpoint keys carry a DDP 'module.' prefix: {bad[:5]}")
