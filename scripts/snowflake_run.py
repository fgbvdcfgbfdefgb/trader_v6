#!/usr/bin/env python3
"""Snowflake Notebook / Container Runtime entrypoint.

Designed for the constraint: the workspace is created *from this git repo*, pip can
install packages, but there is no pull/push and no outbound data download. Everything
the run needs (market data included) is already in the repo.

Paste this into a Snowflake notebook cell:

    !pip install -r requirements.txt
    %run scripts/snowflake_run.py --epochs 500 --mode futures

or from a Python cell:

    import snowflake_run; snowflake_run.train(epochs=500, mode="futures")

Outputs (charts, checkpoints, metrics) go to --out, which defaults to a writable
scratch directory because the workspace checkout itself may be read-only.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _writable_out() -> str:
    """Pick somewhere we can actually write, preferring a Snowflake stage mount."""
    for c in ("/mnt/stage/trader_v6_runs", "/tmp/trader_v6_runs",
              os.path.join(ROOT, "runs")):
        try:
            os.makedirs(c, exist_ok=True)
            t = os.path.join(c, ".probe")
            with open(t, "w") as f:
                f.write("ok")
            os.remove(t)
            return c
        except OSError:
            continue
    return tempfile.mkdtemp(prefix="trader_v6_")


def _cache_dir() -> str:
    """/dev/shm is usually large and fast in Snowflake containers."""
    try:
        st = os.statvfs("/dev/shm")
        if st.f_bavail * st.f_frsize > 2 * 1024 ** 3:
            d = "/dev/shm/trader_v6_cache"
            os.makedirs(d, exist_ok=True)
            return d
    except OSError:
        pass
    return ""


def train(epochs: int = 500, mode: str = "futures", out: str = "", size: str = "auto",
          leverage: float = 10.0, max_hours: float = 0.0, resume: str = "",
          plot_every: int = 10, eval_every: int = 100, gpus: int = -1,
          start_cash: float = 20.0, target: float = 30.0, name: str = "",
          archive: bool = True):
    from trader.config import Config
    from trader.resources import build_plan
    from trader.train.parallel import launch

    out = out or _writable_out()
    plan = build_plan(force_gpus=gpus, force_size=size, reserve_cores=0)
    print(plan.describe(), flush=True)

    cfg = Config()
    cfg.apply_size(plan.size_preset)
    cfg.data.cache_dir = _cache_dir() or os.path.join(out, "cache")
    cfg.env.mode = mode
    cfg.env.start_cash = start_cash
    cfg.env.target_equity = target
    cfg.env.max_leverage = leverage if mode == "futures" else 1.0
    cfg.train.epochs = epochs
    cfg.train.out_dir = out
    cfg.train.run_name = name or time.strftime("%Y%m%d-%H%M%S") + f"-{mode}"
    cfg.train.max_hours = max_hours
    cfg.train.resume = resume
    cfg.train.plot_every = plot_every
    cfg.train.eval_every = eval_every
    cfg.train.cpu_threads = plan.cpu_threads

    run_dir = os.path.join(out, cfg.train.run_name)
    print(f"[snowflake] repo   : {ROOT}")
    print(f"[snowflake] output : {run_dir}")
    print(f"[snowflake] data   : {os.path.join(ROOT, cfg.data.data_dir)} (offline, in-repo)")
    launch(cfg, plan, root=ROOT)

    if archive:
        zpath = shutil.make_archive(run_dir + "_artifacts", "zip", run_dir)
        print(f"[snowflake] artifacts zipped -> {zpath}\n"
              f"            download it, or PUT it to a stage:\n"
              f"            session.file.put('{zpath}', '@my_stage', auto_compress=False)")
    return run_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser("trader_v6 on Snowflake")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--mode", choices=["futures", "spot"], default="futures")
    p.add_argument("--out", default="")
    p.add_argument("--size", default="auto")
    p.add_argument("--leverage", type=float, default=10.0)
    p.add_argument("--max-hours", type=float, default=0.0)
    p.add_argument("--resume", default="")
    p.add_argument("--plot-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--gpus", type=int, default=-1)
    p.add_argument("--start-cash", type=float, default=20.0)
    p.add_argument("--target", type=float, default=30.0)
    p.add_argument("--name", default="")
    p.add_argument("--no-archive", action="store_true")
    a = p.parse_args()
    train(epochs=a.epochs, mode=a.mode, out=a.out, size=a.size, leverage=a.leverage,
          max_hours=a.max_hours, resume=a.resume, plot_every=a.plot_every,
          eval_every=a.eval_every, gpus=a.gpus, start_cash=a.start_cash,
          target=a.target, name=a.name, archive=not a.no_archive)
