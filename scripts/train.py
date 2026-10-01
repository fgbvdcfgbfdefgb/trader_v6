#!/usr/bin/env python3
"""trader_v6 training entrypoint - fully offline.

    python scripts/train.py --epochs 2000 --mode futures

Inspects the machine first, prints the plan, then trains the predictor, the market
analyst and the trade maker in parallel (one rank per GPU) on randomly sampled
trading days. Writes every chart and checkpoint under --out.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# hard-offline: no library may phone home
for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "no_proxy"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from trader.config import Config          # noqa: E402
from trader.resources import build_plan   # noqa: E402
from trader.train.parallel import launch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse():
    p = argparse.ArgumentParser("trader_v6", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("run")
    g.add_argument("--epochs", type=int, default=2000, help="one epoch = one random day")
    g.add_argument("--out", default=os.path.join(ROOT, "runs"))
    g.add_argument("--name", default="", help="run name (default: timestamp)")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--resume", default="", help="path to ckpt/last.pt")
    g.add_argument("--max-hours", type=float, default=0.0, help="0 = no limit")

    g = p.add_argument_group("environment")
    g.add_argument("--mode", choices=["futures", "spot"], default="futures")
    g.add_argument("--start-cash", type=float, default=20.0)
    g.add_argument("--target", type=float, default=30.0)
    g.add_argument("--leverage", type=float, default=10.0, help="futures gross cap")
    g.add_argument("--n-envs", type=int, default=0, help="0 = from size preset")
    g.add_argument("--no-risk-gate", action="store_true",
                   help="stop the analyst's risk budget from scaling trade size")

    g = p.add_argument_group("resources")
    g.add_argument("--gpus", type=int, default=-1, help="-1 = use all visible")
    g.add_argument("--size", default="auto", choices=["auto", "tiny", "small", "base", "large"])
    g.add_argument("--cpu-threads", type=int, default=0, help="0 = auto")
    g.add_argument("--reserve-cores", type=int, default=1,
                   help="cores left for other workloads on the box (e.g. a miner)")
    g.add_argument("--no-amp", action="store_true")

    g = p.add_argument_group("data")
    g.add_argument("--data-dir", default="data/klines")
    g.add_argument("--cache-dir", default="", help="default <out>/cache; use /dev/shm on small disks")
    g.add_argument("--train-end", default="2025-06-30", help="later days are held out")
    g.add_argument("--symbols", default="BTCUSDT,ETHUSDT,LTCUSDT")

    g = p.add_argument_group("logging")
    g.add_argument("--plot-every", type=int, default=10)
    g.add_argument("--dashboard-every", type=int, default=25)
    g.add_argument("--ckpt-every", type=int, default=50)
    g.add_argument("--eval-every", type=int, default=100)
    g.add_argument("--eval-days", type=int, default=16)
    g.add_argument("--no-curriculum", action="store_true")
    g.add_argument("--no-e2e", action="store_true",
                   help="disable the trader->analyst policy-gradient path")
    g.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return p.parse_args()


def main():
    a = parse()
    plan = build_plan(force_gpus=a.gpus, force_size=a.size,
                      cpu_threads=a.cpu_threads, reserve_cores=a.reserve_cores)
    print(plan.describe(), flush=True)
    if a.dry_run:
        return

    cfg = Config()
    cfg.apply_size(plan.size_preset)
    cfg.data.data_dir = a.data_dir
    cfg.data.symbols = [s.strip() for s in a.symbols.split(",") if s.strip()]
    cfg.data.train_end = a.train_end
    cfg.data.cache_dir = a.cache_dir or os.path.join(a.out, "cache")

    cfg.env.mode = a.mode
    cfg.env.start_cash = a.start_cash
    cfg.env.target_equity = a.target
    cfg.env.max_leverage = a.leverage if a.mode == "futures" else 1.0
    cfg.env.use_risk_gate = not a.no_risk_gate
    if a.n_envs > 0:
        cfg.env.n_envs = a.n_envs

    cfg.train.epochs = a.epochs
    cfg.train.seed = a.seed
    cfg.train.out_dir = a.out
    cfg.train.run_name = a.name or time.strftime("%Y%m%d-%H%M%S") + f"-{a.mode}"
    cfg.train.resume = a.resume
    cfg.train.max_hours = a.max_hours
    cfg.train.plot_every = a.plot_every
    cfg.train.dashboard_every = a.dashboard_every
    cfg.train.ckpt_every = a.ckpt_every
    cfg.train.eval_every = a.eval_every
    cfg.train.eval_days = a.eval_days
    cfg.train.curriculum = not a.no_curriculum
    cfg.train.e2e_grad = not a.no_e2e
    cfg.train.amp = not a.no_amp
    cfg.train.cpu_threads = plan.cpu_threads

    os.makedirs(os.path.join(cfg.train.out_dir, cfg.train.run_name), exist_ok=True)
    print(f"[run  ] {os.path.join(cfg.train.out_dir, cfg.train.run_name)}", flush=True)
    launch(cfg, plan, root=ROOT)


if __name__ == "__main__":
    main()
