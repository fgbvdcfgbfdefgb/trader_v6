"""Turns the analyst's tensors into a written verdict on the day.

This is the "advisor response" that gets saved next to every epoch's charts, both
as a PNG card (viz.plot_advisor) and as JSON for later analysis.
"""
from __future__ import annotations

import numpy as np

from .models.analyst import REGIME_NAMES

SESSIONS = [("Asia   00-08", 0, 480), ("Europe 08-16", 480, 960), ("US     16-24", 960, 1440)]


def _dominant(reg: np.ndarray, lo: int, hi: int):
    p = reg[lo:hi].mean(0)
    i = int(p.argmax())
    return REGIME_NAMES[i], float(p[i])


def build_advice(day, an: dict, summary: dict, env_cfg, bh_returns: np.ndarray) -> dict:
    """an: numpy dict with regime [T,R], vol [A,T], exposure [A,T], risk [T], conf [A,T]."""
    reg = an["regime"]                    # [T, R] market-averaged posterior
    risk = an["risk_budget"]
    conf = an["confidence"].mean(0)
    expo = an["exposure"]                 # [A, T]
    T = reg.shape[0]

    overall, p_overall = _dominant(reg, 0, T)
    mean_risk = float(risk.mean())
    mean_conf = float(conf.mean())
    net_bias = float(expo.mean())

    if mean_risk > 0.6 and net_bias > 0.08:
        verdict = "RISK-ON"
    elif mean_risk < 0.4 or net_bias < -0.08:
        verdict = "RISK-OFF"
    else:
        verdict = "NEUTRAL"

    lines = []
    lines.append(f"REGIME     {overall} (p={p_overall:.2f})   "
                 f"risk budget {mean_risk:.2f}   confidence {mean_conf:.2f}")
    lines.append(f"DAY VOL    {day.day_vol*100:.3f}%/min   intraday range "
                 f"{day.day_range*100:.2f}%")
    lines.append("")
    lines.append("PER-ASSET READ")
    for a, sym in enumerate(day.symbols):
        e = float(expo[a].mean())
        stance = "LONG " if e > 0.08 else ("SHORT" if e < -0.08 else "FLAT ")
        lines.append(f"  {sym:<8} {stance} avg-exposure {e:+.2f}  "
                     f"fwd-vol {float(an['vol'][a].mean()):+.2f}  "
                     f"buy&hold {bh_returns[a]*100:+.2f}%")
    lines.append("")
    lines.append("SESSION BREAKDOWN")
    for name, lo, hi in SESSIONS:
        r, p = _dominant(reg, lo, min(hi, T))
        lines.append(f"  {name}  {r:<14} p={p:.2f}  risk={float(risk[lo:min(hi,T)].mean()):.2f}")
    lines.append("")
    lines.append("TRADE MAKER RESULT")
    lines.append(f"  equity   ${env_cfg.start_cash:.2f} -> ${summary['equity_mean']:.2f} "
                 f"(median ${summary['equity_median']:.2f}, best ${summary['equity_best']:.2f})")
    lines.append(f"  target   ${env_cfg.target_equity:.0f} reached by "
                 f"{100*summary['hit_target']:.0f}% of trajectories")
    lines.append(f"  activity {summary['n_trades']:.0f} rebalances, turnover "
                 f"{summary['turnover']:.1f}x, fees ${summary['fees']:.3f}")
    if summary["bankrupt"] > 0:
        lines.append(f"  WARNING  {100*summary['bankrupt']:.0f}% of trajectories liquidated")

    note = _coaching(verdict, summary, env_cfg, net_bias, bh_returns)
    lines.append("")
    lines.append("ADVISOR NOTE")
    for l in note:
        lines.append(f"  {l}")

    return {"verdict": verdict, "text": "\n".join(lines), "regime": overall,
            "risk_budget": mean_risk, "confidence": mean_conf, "net_bias": net_bias}


def _coaching(verdict, summary, env_cfg, net_bias, bh):
    out = []
    ret = summary["equity_mean"] / env_cfg.start_cash - 1
    best_bh = float(np.max(np.abs(bh)))
    if summary["bankrupt"] > 0.1:
        out.append("Position sizing too aggressive for this tape - liquidations hit.")
    if ret > 0 and ret > best_bh:
        out.append("Agent beat the best single-asset buy&hold on this day.")
    elif ret > 0:
        out.append("Positive day, but passive exposure to the strongest asset did better.")
    else:
        out.append("Negative day - the policy paid more in fees/noise than it captured.")
    if summary["turnover"] > 40:
        out.append("Very high turnover; fee drag is material at this trade rate.")
    if abs(net_bias) < 0.05:
        out.append("Analyst saw no directional edge; a flat book was the honest call.")
    gap = env_cfg.target_equity - summary["equity_mean"]
    if gap > 0:
        need = 100 * gap / max(1e-9, summary["equity_mean"])
        out.append(f"Still ${gap:.2f} ({need:.0f}%) short of the ${env_cfg.target_equity:.0f} "
                   f"target for this epoch.")
    else:
        out.append("Target met for this epoch.")
    return out
