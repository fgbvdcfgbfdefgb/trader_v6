"""Fully-offline dataset: parquet -> memory-mapped minute grid -> per-day epochs.

Nothing here touches the network. On first run it builds a cache of
memory-mapped .npy arrays (one aligned minute grid shared by all symbols) so that
N training ranks share the same OS page cache instead of each holding a copy.

An *epoch* is one randomly chosen calendar day; every rank gets the full
[3 assets x 1440 minutes] slab for that day plus warmup history.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from collections import OrderedDict

from .features import N_FEATURES, RAW_COLS, day_features

MIN_MS = 60_000
DAY_MS = 86_400_000
DAY_MIN = 1440


def _ts_to_date(ts_ms: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts_ms / 1000))


def _date_to_ts(d: str) -> int:
    return int(time.mktime(time.strptime(d, "%Y-%m-%d")) // 86400) * DAY_MS


@dataclass
class DayBatch:
    """One epoch's worth of market data."""
    day_index: int
    date: str
    symbols: List[str]
    raw: np.ndarray        # [A, 1440, len(RAW_COLS)]
    feats: np.ndarray      # [A, 1440, N_FEATURES] normalised
    close: np.ndarray      # [A, 1440] float64
    volume: np.ndarray     # [A, 1440] float64 (base volume, for slippage)
    day_vol: float         # realised 1m vol averaged over assets (difficulty proxy)
    day_range: float       # mean high/low range of the day
    split: str = "train"


class MarketData:
    def __init__(self, cfg, root: str = ".", cache_dir: str = "", verbose: bool = True):
        self.cfg = cfg
        self.root = root
        self.symbols = list(cfg.symbols)
        self.verbose = verbose
        self.cache_dir = cache_dir or os.path.join(root, ".cache_trader_v6")
        os.makedirs(self.cache_dir, exist_ok=True)
        # Feature engineering is pandas-bound and identical every time a day is
        # resampled, so memoise it. On a 4-core box shared with another workload this
        # is the difference between starving the machine and barely touching the CPU.
        self.feat_dir = os.path.join(self.cache_dir, "feat")
        self.mem_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.mem_cache_max = int(os.environ.get("TRADER_FEAT_MEM_DAYS", "48"))
        self.disk_feat_cache = os.environ.get("TRADER_FEAT_DISK", "1") == "1"
        self._build_or_load()

    # ------------------------------------------------------------------ cache
    def _paths(self) -> Dict[str, str]:
        tag = "_".join(s.replace("USDT", "") for s in self.symbols)
        return {
            "grid": os.path.join(self.cache_dir, f"grid_{tag}.npy"),
            "meta": os.path.join(self.cache_dir, f"meta_{tag}.json"),
            "stats": os.path.join(self.cache_dir, f"featstats_{tag}.npz"),
            "lock": os.path.join(self.cache_dir, f"build_{tag}.lock"),
        }

    def _ensure_dir(self):
        """Re-create the cache dir before every write: on shared boxes /dev/shm and
        /tmp can be swept out from under a long run."""
        os.makedirs(self.cache_dir, exist_ok=True)

    def _log(self, *a):
        if self.verbose:
            print("[data]", *a, flush=True)

    def _build_or_load(self):
        p = self._paths()
        if not (os.path.exists(p["grid"]) and os.path.exists(p["meta"])):
            self._acquire_and_build(p)
        if not os.path.exists(p["grid"]):
            self._log("cache vanished after build (shared /tmp or /dev/shm swept?) "
                      "- rebuilding")
            self._acquire_and_build(p)
        with open(p["meta"]) as f:
            self.meta = json.load(f)
        self.t0 = int(self.meta["t0"])
        self.n_minutes = int(self.meta["n_minutes"])
        self.n_days = int(self.meta["n_days"])
        A, C = len(self.symbols), len(RAW_COLS)
        self.grid = np.load(p["grid"], mmap_mode="r").reshape(A, self.n_minutes, C)
        self.day_valid = np.asarray(self.meta["day_valid"], dtype=bool)
        self.day_vol = np.asarray(self.meta["day_vol"], dtype=np.float32)
        self.day_range = np.asarray(self.meta["day_range"], dtype=np.float32)
        st = np.load(p["stats"])
        self.feat_mean = st["mean"].astype(np.float32)
        self.feat_std = np.maximum(st["std"].astype(np.float32), 1e-3)
        self._make_splits()
        self._log(f"{self.n_days} days on grid, {int(self.day_valid.sum())} usable "
                  f"({len(self.train_days)} train / {len(self.eval_days)} eval), "
                  f"{self.n_minutes:,} minutes/symbol")

    def _acquire_and_build(self, p):
        """Only one process builds; the others wait for the lock to clear."""
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            fd = os.open(p["lock"], os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
        except FileExistsError:
            self._log("another rank is building the cache, waiting...")
            for _ in range(3600):
                if os.path.exists(p["meta"]) and not os.path.exists(p["lock"]):
                    return
                time.sleep(2)
            raise RuntimeError("timed out waiting for dataset cache build")
        try:
            self._build(p)
        finally:
            if os.path.exists(p["lock"]):
                os.remove(p["lock"])

    def _build(self, p):
        import pandas as pd
        ddir = os.path.join(self.root, self.cfg.data_dir)
        if not os.path.isdir(ddir):
            raise FileNotFoundError(
                f"dataset directory '{ddir}' missing. This repo ships the parquet files "
                "under data/klines - run from the repo root (offline, no download needed).")
        self._log(f"building memory-mapped cache in {self.cache_dir} (first run only)")

        # Pass 1: just the time span of each symbol (reads only the ts column).
        files_by_sym, spans = {}, {}
        for sym in self.symbols:
            files = sorted(f for f in os.listdir(ddir)
                           if f.startswith(sym + "_") and f.endswith(".parquet"))
            if not files:
                raise FileNotFoundError(f"no parquet files for {sym} in {ddir}")
            files_by_sym[sym] = [os.path.join(ddir, f) for f in files]
            first = pd.read_parquet(files_by_sym[sym][0], columns=["ts"])
            last = pd.read_parquet(files_by_sym[sym][-1], columns=["ts"])
            spans[sym] = (int(first.ts.min()), int(last.ts.max()))
            self._log(f"  {sym}: {_ts_to_date(spans[sym][0])} -> {_ts_to_date(spans[sym][1])}")
            del first, last

        # Common span, snapped to whole UTC days.
        lo = max(v[0] for v in spans.values())
        hi = min(v[1] for v in spans.values())
        t0 = ((lo + DAY_MS - 1) // DAY_MS) * DAY_MS
        t1 = (hi // DAY_MS) * DAY_MS
        n_minutes = int((t1 - t0) // MIN_MS)
        n_days = n_minutes // DAY_MIN
        n_minutes = n_days * DAY_MIN
        A, C = len(self.symbols), len(RAW_COLS)

        self._ensure_dir()
        grid = np.lib.format.open_memmap(p["grid"], mode="w+", dtype=np.float32,
                                         shape=(A, n_minutes, C))
        # Pass 2: stream each parquet straight into its slice of the memmap.
        for ai, sym in enumerate(self.symbols):
            seen = np.zeros(n_minutes, dtype=bool)
            for fp in files_by_sym[sym]:
                df = pd.read_parquet(fp)
                idx = (df.ts.values.astype(np.int64) - t0) // MIN_MS
                keep = (idx >= 0) & (idx < n_minutes)
                if not keep.any():
                    del df; continue
                idx = idx[keep]
                order = np.argsort(idx, kind="stable")
                idx = idx[order]
                uniq = np.ones(len(idx), dtype=bool)
                uniq[1:] = idx[1:] != idx[:-1]
                rows = np.empty((len(idx), C), dtype=np.float32)
                for ci, col in enumerate(RAW_COLS):
                    rows[:, ci] = df[col].values[keep][order].astype(np.float32)
                grid[ai, idx[uniq]] = rows[uniq]
                seen[idx] = True
                del df, rows, idx, order, uniq
            # minutes we never saw -> carry the last close forward, flag invalid
            if not seen.all():
                cl = np.asarray(grid[ai, :, RAW_COLS.index("close")])
                fill = np.maximum.accumulate(np.where(cl > 0, np.arange(n_minutes), 0))
                cl2 = cl[fill]
                miss = ~seen
                for col in ("open", "high", "low", "close"):
                    grid[ai, miss, RAW_COLS.index(col)] = cl2[miss]
                grid[ai, miss, RAW_COLS.index("valid")] = 0
                del cl, cl2, fill, miss
            self._log(f"  {sym}: wrote {int(seen.sum()):,}/{n_minutes:,} minutes")
            del seen
        grid.flush()

        # Per-day quality + difficulty statistics, computed in day-chunks so peak
        # RSS stays in the tens of MB even for a decade of minute bars.
        vi, ci = RAW_COLS.index("valid"), RAW_COLS.index("close")
        hi_i, lo_i = RAW_COLS.index("high"), RAW_COLS.index("low")
        day_ok = np.zeros(n_days, dtype=bool)
        day_vol = np.zeros(n_days, dtype=np.float64)
        day_range = np.zeros(n_days, dtype=np.float64)
        CH = 256
        for d0 in range(0, n_days, CH):
            d1 = min(n_days, d0 + CH)
            sl = np.asarray(grid[:, d0 * DAY_MIN:d1 * DAY_MIN, :]) \
                .reshape(A, d1 - d0, DAY_MIN, C)
            valid_frac = sl[:, :, :, vi].mean(axis=2).min(axis=0)
            closes = sl[:, :, :, ci].astype(np.float64)
            ok = valid_frac >= self.cfg.min_valid_frac
            ok &= (closes > 0).all(axis=2).all(axis=0)
            lr = np.diff(np.log(np.where(closes > 0, closes, 1.0)), axis=2)
            day_ok[d0:d1] = ok
            day_vol[d0:d1] = lr.std(axis=2).mean(axis=0)
            day_range[d0:d1] = ((sl[:, :, :, hi_i].max(axis=2)
                                 - sl[:, :, :, lo_i].min(axis=2))
                                / np.maximum(closes[:, :, 0], 1e-9)).mean(axis=0)
            del sl, closes, lr
        day_ok[:1] = False                     # first day has no warmup history

        meta = dict(t0=int(t0), n_minutes=int(n_minutes), n_days=int(n_days),
                    symbols=self.symbols, raw_cols=RAW_COLS,
                    start=_ts_to_date(t0), end=_ts_to_date(t0 + n_minutes * MIN_MS),
                    day_valid=day_ok.tolist(),
                    day_vol=np.nan_to_num(day_vol).astype(float).tolist(),
                    day_range=np.nan_to_num(day_range).astype(float).tolist())
        self._ensure_dir()
        with open(p["meta"], "w") as f:
            json.dump(meta, f)

        # Feature normaliser from a random sample of days.
        self.meta, self.t0, self.n_minutes, self.n_days = meta, t0, n_minutes, n_days
        self.grid = np.load(p["grid"], mmap_mode="r").reshape(A, n_minutes, C)
        self.feat_mean = np.zeros(N_FEATURES, dtype=np.float32)
        self.feat_std = np.ones(N_FEATURES, dtype=np.float32)
        ok_days = np.flatnonzero(day_ok)
        rng = np.random.default_rng(0)
        sample = rng.choice(ok_days, size=min(96, len(ok_days)), replace=False)
        acc = []
        for d in sample:
            acc.append(self._raw_features(int(d)).reshape(-1, N_FEATURES))
        allf = np.concatenate(acc, axis=0)
        mean = allf.mean(axis=0)
        std = allf.std(axis=0)
        self._ensure_dir()
        np.savez(p["stats"], mean=mean, std=std)
        self._log(f"cache built: {n_days} days, grid {grid.nbytes/1e6:.0f} MB")

    # ------------------------------------------------------------------ split
    def _make_splits(self):
        ok = np.flatnonzero(self.day_valid)
        cutoff = self.cfg.train_end
        dates = {int(d): _ts_to_date(self.t0 + int(d) * DAY_MS) for d in ok}
        train = np.array([d for d in ok if dates[int(d)] <= cutoff], dtype=np.int64)
        evald = np.array([d for d in ok if dates[int(d)] > cutoff], dtype=np.int64)
        if self.cfg.eval_frac_days > 0 and len(train):
            rng = np.random.default_rng(1234)
            n = int(len(train) * self.cfg.eval_frac_days)
            pick = rng.choice(len(train), size=n, replace=False)
            mask = np.ones(len(train), bool); mask[pick] = False
            evald = np.concatenate([evald, train[~mask]])
            train = train[mask]
        self.train_days, self.eval_days = train, np.sort(evald)
        # curriculum weights: favour volatile (opportunity-rich) days early on
        v = self.day_vol[self.train_days]
        r = v.argsort().argsort() / max(1, len(v) - 1)
        self.train_rank = r.astype(np.float32)

    # --------------------------------------------------------------- slicing
    def _window(self, day_index: int) -> Tuple[np.ndarray, np.ndarray]:
        w = self.cfg.warmup_bars
        start = day_index * DAY_MIN
        s = max(0, start - w)
        block = np.asarray(self.grid[:, s:start + DAY_MIN, :], dtype=np.float32)
        if block.shape[1] < w + DAY_MIN:                 # pad at the very beginning
            pad = w + DAY_MIN - block.shape[1]
            block = np.concatenate([np.repeat(block[:, :1], pad, axis=1), block], axis=1)
        mod = (np.arange(block.shape[1]) - (block.shape[1] - DAY_MIN)) % DAY_MIN
        return block, mod

    def _raw_features(self, day_index: int) -> np.ndarray:
        block, mod = self._window(day_index)
        return day_features(block, mod, DAY_MIN)

    def _features_for(self, day_index: int, block, mod) -> np.ndarray:
        """Normalised features for one day, memoised in RAM and on disk."""
        hit = self.mem_cache.get(day_index)
        if hit is not None:
            self.mem_cache.move_to_end(day_index)
            return hit

        path = os.path.join(self.feat_dir, f"d{day_index:06d}.npy")
        feats = None
        if self.disk_feat_cache and os.path.exists(path):
            try:
                feats = np.load(path).astype(np.float32)
            except (OSError, ValueError):
                feats = None
        if feats is None:
            feats = day_features(block, mod, DAY_MIN)
            feats = (feats - self.feat_mean) / self.feat_std
            feats = np.clip(feats, -8.0, 8.0).astype(np.float32)
            if self.disk_feat_cache:
                try:
                    os.makedirs(self.feat_dir, exist_ok=True)
                    tmp = path + f".{os.getpid()}.tmp"
                    np.save(tmp, feats.astype(np.float16))
                    os.replace(tmp, path)
                except OSError:
                    self.disk_feat_cache = False      # read-only or full disk: skip

        self.mem_cache[day_index] = feats
        while len(self.mem_cache) > self.mem_cache_max:
            self.mem_cache.popitem(last=False)
        return feats

    def get_day(self, day_index: int, split: str = "train") -> DayBatch:
        block, mod = self._window(day_index)
        feats = self._features_for(day_index, block, mod)
        day = block[:, -DAY_MIN:, :]
        ci, vi = RAW_COLS.index("close"), RAW_COLS.index("volume")
        return DayBatch(
            day_index=day_index, date=_ts_to_date(self.t0 + day_index * DAY_MS),
            symbols=self.symbols, raw=day, feats=feats,
            close=day[:, :, ci].astype(np.float64),
            volume=day[:, :, vi].astype(np.float64),
            day_vol=float(self.day_vol[day_index]),
            day_range=float(self.day_range[day_index]), split=split)

    # --------------------------------------------------------------- sampling
    def sample_day(self, rng: np.random.Generator, progress: float = 1.0,
                   curriculum: bool = True) -> int:
        """Uniform over training days, optionally tilted toward volatile days early."""
        days = self.train_days
        if not curriculum:
            return int(days[rng.integers(len(days))])
        # progress 0 -> strong tilt to high-vol days, progress 1 -> uniform
        temp = max(0.0, 1.0 - progress) * 2.0
        w = np.exp(temp * (self.train_rank - 0.5))
        w /= w.sum()
        return int(days[rng.choice(len(days), p=w)])

    def eval_day_list(self, n: int, seed: int = 0) -> List[int]:
        if len(self.eval_days) == 0:
            return []
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.eval_days), size=min(n, len(self.eval_days)),
                         replace=False)
        return [int(self.eval_days[i]) for i in np.sort(idx)]
