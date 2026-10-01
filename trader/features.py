"""Feature engineering. Pure numpy/pandas, no lookahead.

A day's observation tensor is built from [warmup + 1440] raw bars so every rolling
indicator is already warm at 00:00. Only the final 1440 rows are returned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# column order produced by data.build_cache()
RAW_COLS = ["open", "high", "low", "close", "volume", "quote_volume",
            "trades", "taker_base", "valid"]
C = {c: i for i, c in enumerate(RAW_COLS)}

FEATURE_NAMES = [
    "ret1", "ret5", "ret15", "ret60",
    "vol15", "vol60", "vol240",
    "ema15_z", "ema60_z", "ema240_z",
    "rsi14", "macd", "macd_sig", "bb_pctb", "atr14",
    "range_hl", "upper_wick", "lower_wick",
    "vol_z60", "quote_z60", "trades_z60", "taker_ratio",
    "vwap_dev", "accel", "tod_sin", "tod_cos",
    "xs_ret_spread", "xs_vol_ratio",
]
N_FEATURES = len(FEATURE_NAMES)


def _ewm(a: pd.Series, span: int) -> pd.Series:
    return a.ewm(span=span, adjust=False, min_periods=1).mean()


def _zdiv(a, b, eps=1e-8):
    return a / (b + eps)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=1).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=1).mean()
    rs = _zdiv(up, dn)
    return (100 - 100 / (1 + rs)) / 50.0 - 1.0          # -> [-1, 1]


def asset_features(raw: np.ndarray, minute_of_day: np.ndarray) -> np.ndarray:
    """raw: [T, len(RAW_COLS)] float32 for ONE symbol (warmup included)."""
    o = pd.Series(raw[:, C["open"]].astype(np.float64))
    h = pd.Series(raw[:, C["high"]].astype(np.float64))
    lo = pd.Series(raw[:, C["low"]].astype(np.float64))
    c = pd.Series(raw[:, C["close"]].astype(np.float64))
    v = pd.Series(raw[:, C["volume"]].astype(np.float64))
    qv = pd.Series(raw[:, C["quote_volume"]].astype(np.float64))
    tr = pd.Series(raw[:, C["trades"]].astype(np.float64))
    tb = pd.Series(raw[:, C["taker_base"]].astype(np.float64))

    lc = np.log(c.clip(lower=1e-12))
    r1 = lc.diff().fillna(0.0)

    vol60 = r1.rolling(60, min_periods=5).std().bfill().fillna(1e-4)
    scale = (vol60 * np.sqrt(60.0)).clip(lower=1e-5)

    f = {}
    f["ret1"] = _zdiv(r1, vol60.clip(lower=1e-6))
    f["ret5"] = _zdiv(lc.diff(5).fillna(0.0), scale)
    f["ret15"] = _zdiv(lc.diff(15).fillna(0.0), scale * np.sqrt(15 / 60))
    f["ret60"] = _zdiv(lc.diff(60).fillna(0.0), scale)

    v15 = r1.rolling(15, min_periods=5).std().bfill()
    v240 = r1.rolling(240, min_periods=20).std().bfill()
    med = vol60.rolling(240, min_periods=20).median().bfill().clip(lower=1e-6)
    f["vol15"] = np.log(_zdiv(v15, med).clip(lower=1e-3))
    f["vol60"] = np.log(_zdiv(vol60, med).clip(lower=1e-3))
    f["vol240"] = np.log(_zdiv(v240, med).clip(lower=1e-3))

    for sp in (15, 60, 240):
        f[f"ema{sp}_z"] = _zdiv(np.log(_zdiv(c, _ewm(c, sp))), scale)

    f["rsi14"] = _rsi(c)
    macd = _ewm(c, 12) - _ewm(c, 26)
    sig = _ewm(macd, 9)
    f["macd"] = _zdiv(macd, c * scale)
    f["macd_sig"] = _zdiv(macd - sig, c * scale)

    m20 = c.rolling(20, min_periods=5).mean()
    s20 = c.rolling(20, min_periods=5).std().fillna(0.0)
    f["bb_pctb"] = (_zdiv(c - m20, 2 * s20)).clip(-3, 3)

    prev_c = c.shift(1).bfill()
    tr_rng = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    f["atr14"] = _zdiv(tr_rng.ewm(alpha=1 / 14, adjust=False, min_periods=1).mean(), c * scale)

    f["range_hl"] = _zdiv(np.log(_zdiv(h, lo).clip(lower=1.0)), scale)
    body_hi = pd.concat([o, c], axis=1).max(axis=1)
    body_lo = pd.concat([o, c], axis=1).min(axis=1)
    f["upper_wick"] = _zdiv(_zdiv(h - body_hi, c), scale)
    f["lower_wick"] = _zdiv(_zdiv(body_lo - lo, c), scale)

    def zs(s, n=60):
        m = s.rolling(n, min_periods=10).mean()
        sd = s.rolling(n, min_periods=10).std().fillna(0.0)
        return _zdiv(s - m, sd).fillna(0.0)

    f["vol_z60"] = zs(np.log1p(v))
    f["quote_z60"] = zs(np.log1p(qv))
    f["trades_z60"] = zs(np.log1p(tr))
    f["taker_ratio"] = (_zdiv(tb, v) * 2 - 1).fillna(0.0)

    vwap = _zdiv(qv.rolling(60, min_periods=5).sum(), v.rolling(60, min_periods=5).sum())
    f["vwap_dev"] = _zdiv(np.log(_zdiv(c, vwap.replace(0, np.nan)).fillna(1.0)), scale)
    f["accel"] = _zdiv(r1 - r1.shift(1).fillna(0.0), vol60.clip(lower=1e-6))

    ang = 2 * np.pi * minute_of_day / 1440.0
    f["tod_sin"] = pd.Series(np.sin(ang))
    f["tod_cos"] = pd.Series(np.cos(ang))

    # cross-asset slots are filled in by day_features()
    f["xs_ret_spread"] = pd.Series(np.zeros(len(c)))
    f["xs_vol_ratio"] = pd.Series(np.zeros(len(c)))

    out = np.stack([np.asarray(f[k], dtype=np.float64) for k in FEATURE_NAMES], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def day_features(raw: np.ndarray, minute_of_day: np.ndarray, n_keep: int) -> np.ndarray:
    """raw: [A, T, RAW] -> features [A, n_keep, N_FEATURES] (last n_keep rows)."""
    A = raw.shape[0]
    feats = np.stack([asset_features(raw[a], minute_of_day) for a in range(A)], axis=0)

    i_ret, i_vol = FEATURE_NAMES.index("ret60"), FEATURE_NAMES.index("vol60")
    mean_ret = feats[:, :, i_ret].mean(axis=0, keepdims=True)
    mean_vol = feats[:, :, i_vol].mean(axis=0, keepdims=True)
    feats[:, :, FEATURE_NAMES.index("xs_ret_spread")] = feats[:, :, i_ret] - mean_ret
    feats[:, :, FEATURE_NAMES.index("xs_vol_ratio")] = feats[:, :, i_vol] - mean_vol

    return np.clip(feats[:, -n_keep:, :], -10.0, 10.0)
