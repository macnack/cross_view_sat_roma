"""Stream a training CSV from Eagle into a Weights & Biases run, from the laptop (no change to the running job).

scripts/train_vigor.py writes experiments/<run>/train_<tag>.csv on Eagle (one row per logged train step and one per
validation, column `split`). This script polls that file over `ssh eagle`, logs every new row to one W&B run per tag
(train rows as train/<metric>, validation rows as val/<metric>, x = `step`), and resumes the same W&B run when it is
restarted (run id = the tag). Authentication is the laptop's own W&B login (~/.netrc, as in ~/Github/sat_roma); no key
is read, printed or copied by this script. Stops when the SLURM job has left the queue and the file has stopped growing.

  make wandb-sync TAG=samearea_4city_erp_depth_cell0125_e100 JOBID=8859503 [OUT=experiments/13_panoroma_long]
"""
from __future__ import annotations

import argparse
import csv
import io
import subprocess
import time

REMOTE = "/mnt/storage_6/project_data/pl1269-01/krupka_maciej/cross_view_sat_roma"


def ssh(cmd: str, timeout=120) -> str:
    return subprocess.run(["ssh", "eagle", cmd], capture_output=True, text=True, timeout=timeout).stdout


def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None                      # drop NaN / empty


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--jobid", default=None, help="SLURM job id; the sync stops once it has left the queue")
    ap.add_argument("--out", default="experiments/13_panoroma_long")
    ap.add_argument("--project", default="cross_view_sat_roma")
    ap.add_argument("--every", type=int, default=120, help="poll period (s)")
    a = ap.parse_args()
    import wandb

    path = f"{REMOTE}/{a.out}/train_{a.tag}.csv"
    cfg_path = f"{REMOTE}/{a.out}/config_{a.tag}.yaml"
    run = wandb.init(project=a.project, name=a.tag, id=a.tag.replace("/", "_"), resume="allow",
                     config={"tag": a.tag, "slurm_job": a.jobid, "csv": path})
    cfg = ssh(f"cat {cfg_path} 2>/dev/null")
    if cfg:
        run.config.update({"config_yaml": cfg}, allow_val_change=True)
    done = int(run.summary.get("_csv_rows", 0) or 0)   # rows already logged (the CSV only grows; resume-safe)
    last_step, idle = 0, 0
    while True:
        text = ssh(f"cat {path} 2>/dev/null")
        rows = [r for r in csv.DictReader(io.StringIO(text))] if text else []
        rows = [r for r in rows if num(r.get("step")) is not None]
        new = rows[done:]
        for r in new:
            step = int(float(r["step"]))
            if step < last_step:                        # a truncated-and-rewritten CSV (resume): skip the overlap
                continue
            split = r.get("split") or "train"
            m = {f"{split}/{k}": num(v) for k, v in r.items() if k not in ("step", "split") and num(v) is not None}
            m["epoch"] = step / 1315.0
            wandb.log(m, step=step)
            last_step = step
        if new:
            done = len(rows)
            run.summary["_csv_rows"] = done
            print(f"logged {len(new)} rows, last step {last_step}", flush=True)
            idle = 0
        else:
            idle += 1
        running = bool(ssh(f"squeue -h -j {a.jobid} -o %t 2>/dev/null").strip()) if a.jobid else True
        if not running and idle >= 2:
            break
        time.sleep(a.every)
    run.finish()


if __name__ == "__main__":
    main()
