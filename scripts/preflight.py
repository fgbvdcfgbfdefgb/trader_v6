#!/usr/bin/env python3
"""Offline readiness check. Run this first on any new machine (incl. Snowflake).

Verifies: python/torch/GPU, that the in-repo dataset is complete and readable, that
the cache can be built, that one full epoch runs end to end, and that charts are
written - all without touching the network.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OK, BAD = "  [ok]  ", "  [FAIL]"


def main():
    fails = 0
    print("=" * 68); print("  trader_v6 preflight"); print("=" * 68)

    print(f"{OK} python {sys.version.split()[0]}")
    try:
        import torch
        print(f"{OK} torch {torch.__version__}  cuda={torch.cuda.is_available()} "
              f"devices={torch.cuda.device_count()}")
    except Exception as e:
        print(f"{BAD} torch import: {e}"); return 1
    for mod in ("numpy", "pandas", "pyarrow", "matplotlib"):
        try:
            m = __import__(mod)
            print(f"{OK} {mod} {getattr(m, '__version__', '?')}")
        except Exception as e:
            print(f"{BAD} {mod}: {e}"); fails += 1

    from trader.resources import build_plan
    plan = build_plan()
    print(plan.describe())

    mpath = os.path.join(ROOT, "data", "manifest.json")
    if not os.path.exists(mpath):
        print(f"{BAD} data/manifest.json missing - is this a full checkout?"); return 1
    man = json.load(open(mpath))
    ddir = os.path.join(ROOT, "data", "klines")
    missing = [f["path"] for f in man["files"]
               if not os.path.exists(os.path.join(ROOT, f["path"]))]
    size = sum(os.path.getsize(os.path.join(ddir, f)) for f in os.listdir(ddir))
    present = len(man["files"]) - len(missing)
    if missing and present == 0:
        print(f"{BAD} no parquet files found - the checkout has no data at all")
        print("        if you used git-lfs or a shallow clone, re-clone fully")
        fails += 1
    elif missing:
        years = sorted({os.path.basename(f).split("_")[1][:4]
                        for f in os.listdir(ddir) if f.endswith(".parquet")})
        print(f"  [warn] partial checkout: {present}/{len(man['files'])} parquet files "
              f"present ({size/1e6:.0f} MB, years {years[0]}-{years[-1]})")
        print("         fine for a small disk - training just sees fewer days")
    else:
        print(f"{OK} dataset complete: {len(man['files'])} files, {size/1e6:.0f} MB")
        for s, v in man["symbols"].items():
            print(f"         {s}: {v['rows']:,} bars  {v['start']} -> {v['end']}")

    print("\n  building cache + running one epoch ...")
    t0 = time.time()
    from trader.config import Config
    from trader.train.loop import Trainer
    cfg = Config()
    cfg.apply_size("tiny")
    cfg.train.epochs = 1
    cfg.train.out_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "trader_v6_preflight")
    cfg.train.run_name = "preflight"
    cfg.data.cache_dir = os.path.join(cfg.train.out_dir, "cache")
    cfg.env.n_envs = 4
    try:
        plan.world_size = 1
        plan.roles = [["predictor", "analyst", "trader"]]
        plan.devices = [plan.devices[0]]
        tr = Trainer(cfg, plan, 0, root=ROOT)
        out = tr.train_epoch(0)
        tr.log(0, out["metrics"])
        tr.make_plots(0, out, force=True)
        pdir = os.path.join(tr.run_dir, "plots", "epoch_000000")
        pngs = sorted(os.listdir(pdir)) if os.path.isdir(pdir) else []
        print(f"{OK} epoch ran in {time.time()-t0:.1f}s on {plan.devices[0]}  "
              f"day={out['day'].date}  final=${out['summary']['equity_mean']:.2f}")
        print(f"{OK} {len(pngs)} charts written: {', '.join(pngs)}")
        if len(pngs) < 8:
            print(f"{BAD} expected 8 charts"); fails += 1
    except Exception as e:
        import traceback
        print(f"{BAD} epoch failed:\n{traceback.format_exc()}"); fails += 1

    print("=" * 68)
    print("  PREFLIGHT PASSED - ready to train offline" if not fails
          else f"  PREFLIGHT FAILED ({fails} problem(s))")
    print("=" * 68)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
