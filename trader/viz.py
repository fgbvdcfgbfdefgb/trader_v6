"""Every chart the run produces. Pure matplotlib (Agg), writes PNG only.

Per epoch (cadence `plot_every`):
    01_equity.png        equity paths, target line, drawdown, cumulative reward
    02_market.png        price of each asset with the median agent's position overlaid
    03_positions.png     weight heat-map over the day + gross/net exposure
    04_predictor.png     forecast vs realised, per-horizon IC and direction accuracy
    05_analyst.png       regime posterior, vol forecast, suggested exposure, risk budget
    06_ppo.png           policy/value loss, KL, clip fraction, entropy, value calibration
    07_advisor.png       the analyst's written verdict on that day
    08_distribution.png  final-equity distribution across the parallel trajectories
Run level (cadence `dashboard_every`):
    dashboard.png, agents.png, eval.png
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap

from .models.analyst import REGIME_NAMES

plt.rcParams.update({
    "figure.facecolor": "#0e1117", "axes.facecolor": "#0e1117",
    "savefig.facecolor": "#0e1117", "text.color": "#e6e6e6",
    "axes.labelcolor": "#e6e6e6", "xtick.color": "#9aa0a6", "ytick.color": "#9aa0a6",
    "axes.edgecolor": "#2a2f3a", "grid.color": "#232936", "axes.grid": True,
    "grid.alpha": 0.6, "font.size": 9, "figure.dpi": 110,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.framealpha": 0.2,
    "text.parse_math": False,          # '$' is currency here, never math
})
ACCENT = ["#4cc9f0", "#f72585", "#ffd166", "#06d6a0", "#b388ff", "#ff8a65"]
HEAT = LinearSegmentedColormap.from_list("pnl", ["#f72585", "#0e1117", "#06d6a0"])


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _mins(T):
    return np.arange(T) / 60.0


# --------------------------------------------------------------------- epoch
def plot_equity(d: Dict, path: str):
    eq = d["equity"]                                  # [T+1, N]
    T = eq.shape[0]
    x = _mins(T)
    fig = plt.figure(figsize=(11, 7))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.2, 1, 1], hspace=0.32)

    ax = fig.add_subplot(gs[0])
    for i in range(eq.shape[1]):
        ax.plot(x, eq[:, i], color="#4cc9f0", alpha=0.18, lw=0.8)
    med = np.median(eq, axis=1)
    ax.plot(x, med, color="#4cc9f0", lw=2.2, label="median trajectory")
    ax.plot(x, eq.max(1), color="#06d6a0", lw=1.2, ls="--", label="best")
    ax.plot(x, eq.min(1), color="#f72585", lw=1.2, ls="--", label="worst")
    ax.axhline(d["start_cash"], color="#9aa0a6", lw=1, ls=":")
    ax.axhline(d["target"], color="#ffd166", lw=1.4, ls="--",
               label=f"target ${d['target']:.0f}")
    ax.set_title(f"epoch {d['epoch']} - {d['date']} - equity  "
                 f"(final median ${med[-1]:.2f}, best ${eq[-1].max():.2f}, "
                 f"hit-target {100*d['hit_target']:.0f}%)")
    ax.set_ylabel("equity  $"); ax.legend(loc="upper left", ncol=4, fontsize=8)

    ax2 = fig.add_subplot(gs[1], sharex=ax)
    peak = np.maximum.accumulate(eq, axis=0)
    dd = 100 * (1 - eq / np.maximum(peak, 1e-9))
    ax2.fill_between(x, np.median(dd, axis=1), color="#f72585", alpha=0.45)
    ax2.plot(x, dd.max(1), color="#f72585", lw=0.9, alpha=0.8, label="worst-path DD")
    ax2.set_ylabel("drawdown %"); ax2.invert_yaxis(); ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[2], sharex=ax)
    cr = np.cumsum(d["rewards"], axis=0)
    ax3.plot(_mins(cr.shape[0]), cr.mean(1), color="#ffd166", lw=1.6)
    ax3.fill_between(_mins(cr.shape[0]), cr.min(1), cr.max(1),
                     color="#ffd166", alpha=0.18)
    ax3.set_ylabel("cum. reward"); ax3.set_xlabel("hour of day (UTC)")
    return _save(fig, path)


def plot_market(d: Dict, path: str):
    close, w = d["close"], d["weights_med"]           # [A,T], [T,A]
    A, T = close.shape
    fig, axes = plt.subplots(A, 1, figsize=(11, 2.6 * A), sharex=True)
    axes = np.atleast_1d(axes)
    x = _mins(T)
    xw = _mins(w.shape[0])
    for a in range(A):
        ax = axes[a]
        ax.plot(x, close[a], color=ACCENT[a], lw=1.3)
        ax.set_ylabel(f"{d['symbols'][a]}\nprice", color=ACCENT[a])
        pos = w[:, a]
        twin = ax.twinx(); twin.grid(False)
        twin.fill_between(xw, 0, pos, color="#06d6a0", alpha=0.18,
                          where=pos >= 0, step="mid")
        twin.fill_between(xw, 0, pos, color="#f72585", alpha=0.18,
                          where=pos < 0, step="mid")
        twin.set_ylabel("weight", color="#9aa0a6")
        lim = max(0.05, float(np.abs(pos).max()) * 1.15)
        twin.set_ylim(-lim, lim)
        dpos = np.diff(pos, prepend=pos[0])
        buys = np.flatnonzero(dpos > 0.05 * lim)
        sells = np.flatnonzero(dpos < -0.05 * lim)
        ax.scatter(xw[buys], close[a][buys], s=7, marker="^", color="#06d6a0",
                   alpha=0.55, zorder=5)
        ax.scatter(xw[sells], close[a][sells], s=7, marker="v", color="#f72585",
                   alpha=0.55, zorder=5)
        chg = 100 * (close[a][-1] / close[a][0] - 1)
        ax.set_title(f"{d['symbols'][a]}  buy-and-hold {chg:+.2f}%  "
                     f"(agent exposure shaded)", loc="left")
    axes[-1].set_xlabel("hour of day (UTC)")
    fig.suptitle(f"epoch {d['epoch']} - {d['date']} - market & executed positions",
                 y=1.002)
    return _save(fig, path)


def plot_positions(d: Dict, path: str):
    w = d["weights_med"]
    fig = plt.figure(figsize=(11, 6))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.5, 1, 1], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    lim = max(0.1, float(np.abs(w).max()))
    im = ax.imshow(w.T, aspect="auto", cmap=HEAT, vmin=-lim, vmax=lim,
                   extent=[0, w.shape[0] / 60, len(d["symbols"]) - 0.5, -0.5],
                   interpolation="nearest")
    ax.set_yticks(range(len(d["symbols"]))); ax.set_yticklabels(d["symbols"])
    ax.set_title("target weight per asset over the day (median trajectory)")
    fig.colorbar(im, ax=ax, pad=0.01, label="weight")

    ax2 = fig.add_subplot(gs[1])
    x = _mins(w.shape[0])
    ax2.plot(x, np.abs(w).sum(1), color="#4cc9f0", lw=1.3, label="gross")
    ax2.plot(x, w.sum(1), color="#ffd166", lw=1.3, label="net")
    ax2.axhline(0, color="#9aa0a6", lw=0.7)
    ax2.set_ylabel("exposure"); ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[2])
    ax3.hist(d["actions"].reshape(-1, d["actions"].shape[-1]), bins=60,
             label=d["symbols"], color=ACCENT[:len(d["symbols"])], histtype="stepfilled",
             alpha=0.55)
    ax3.set_title("raw action distribution"); ax3.legend(fontsize=8)
    ax3.set_xlabel("pre-squash action")
    return _save(fig, path)


def plot_predictor(d: Dict, path: str):
    p, y = d["pred_mu"], d["pred_y"]                 # [A,T,H]
    H = p.shape[-1]
    hz = d["horizons"]
    fig, axes = plt.subplots(2, H, figsize=(3.1 * H, 6.4))
    axes = np.atleast_2d(axes)
    for h in range(H):
        ax = axes[0, h]
        pp, yy = p[..., h].ravel(), y[..., h].ravel()
        k = min(4000, pp.size)
        idx = np.random.default_rng(0).choice(pp.size, k, replace=False)
        ax.scatter(pp[idx], yy[idx], s=3, alpha=0.25, color=ACCENT[h % len(ACCENT)])
        lim = np.percentile(np.abs(np.concatenate([pp, yy])), 99) + 1e-6
        ax.plot([-lim, lim], [-lim, lim], color="#9aa0a6", lw=0.8, ls="--")
        ic = np.corrcoef(pp, yy)[0, 1] if pp.std() > 0 and yy.std() > 0 else 0.0
        ax.set_title(f"h={hz[h]}m  IC={ic:.3f}")
        ax.set_xlabel("predicted z"); ax.set_ylim(-lim, lim); ax.set_xlim(-lim, lim)
        if h == 0:
            ax.set_ylabel("realised z")

        ax = axes[1, h]
        T = p.shape[1]
        ax.plot(_mins(T), p[0, :, h], color="#4cc9f0", lw=0.9, label="pred")
        ax.plot(_mins(T), y[0, :, h], color="#ffd166", lw=0.7, alpha=0.65, label="real")
        ax.set_xlabel("hour"); ax.set_title(f"{d['symbols'][0]} h={hz[h]}m", fontsize=8)
        if h == 0:
            ax.set_ylabel("z-score"); ax.legend(fontsize=7)
    fig.suptitle(f"epoch {d['epoch']} - price predictor  "
                 f"(nll {d['m'].get('pred_nll', 0):.3f}, "
                 f"dir-acc {100*d['m'].get('pred_dir_acc', 0):.1f}%, "
                 f"IC {d['m'].get('pred_ic', 0):.3f})", y=1.0)
    return _save(fig, path)


def plot_analyst(d: Dict, path: str):
    reg = d["regime"]                                # [T, R] market-mean posterior
    T = reg.shape[0]
    x = _mins(T)
    fig = plt.figure(figsize=(11, 8))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1.3, 1, 1, 1], hspace=0.42)

    ax = fig.add_subplot(gs[0])
    ax.stackplot(x, reg.T, labels=REGIME_NAMES[:reg.shape[1]],
                 colors=ACCENT[:reg.shape[1]], alpha=0.85)
    ax.set_ylim(0, 1); ax.set_ylabel("P(regime)")
    ax.legend(loc="upper center", ncol=6, fontsize=7)
    ax.set_title("market analyser - regime posterior through the day")

    ax2 = fig.add_subplot(gs[1], sharex=ax)
    ax2.plot(x, d["vol_pred"], color="#4cc9f0", lw=1.2, label="forecast vol (log-ratio)")
    ax2.plot(x, d["vol_real"], color="#ffd166", lw=1.0, alpha=0.8, label="realised fwd vol")
    ax2.set_ylabel("vol"); ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[2], sharex=ax)
    for a in range(d["exposure"].shape[0]):
        ax3.plot(x, d["exposure"][a], color=ACCENT[a], lw=1.0, label=d["symbols"][a])
    ax3.axhline(0, color="#9aa0a6", lw=0.7)
    ax3.set_ylabel("suggested\nexposure"); ax3.set_ylim(-1.05, 1.05); ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[3], sharex=ax)
    ax4.fill_between(x, d["risk_budget"], color="#06d6a0", alpha=0.35)
    ax4.plot(x, d["risk_budget"], color="#06d6a0", lw=1.2, label="risk budget")
    conf = np.asarray(d["confidence"])
    ax4.plot(x, conf.mean(0) if conf.ndim == 2 else conf, color="#b388ff", lw=1.0,
             label="confidence")
    ax4.set_ylim(0, 1); ax4.set_ylabel("0-1"); ax4.set_xlabel("hour of day (UTC)")
    ax4.legend(fontsize=8)
    return _save(fig, path)


def plot_ppo(hist: Dict[str, List[float]], d: Dict, path: str):
    keys = [("pg_loss", "policy loss"), ("v_loss", "value loss"), ("entropy", "entropy"),
            ("kl", "approx KL"), ("clipfrac", "clip fraction"), ("grad_norm", "grad norm")]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.ravel()
    for i, (k, lab) in enumerate(keys):
        ax = axes[i]
        v = hist.get(k, [])
        ax.plot(v, color=ACCENT[i % len(ACCENT)], lw=1.1)
        if len(v) > 20:
            kernel = np.ones(11) / 11
            ax.plot(np.arange(10, len(v)), np.convolve(v, kernel, "valid"),
                    color="#e6e6e6", lw=1.0, alpha=0.7)
        ax.set_title(lab); ax.set_xlabel("epoch")
    ax = axes[6]
    ax.scatter(d["values"].ravel()[::7], d["returns"].ravel()[::7], s=3, alpha=0.2,
               color="#4cc9f0")
    lo = float(min(d["values"].min(), d["returns"].min()))
    hi = float(max(d["values"].max(), d["returns"].max()))
    ax.plot([lo, hi], [lo, hi], color="#9aa0a6", ls="--", lw=0.8)
    ax.set_title("value calibration"); ax.set_xlabel("V(s)"); ax.set_ylabel("return")
    ax = axes[7]
    ax.hist(d["rewards"].ravel(), bins=80, color="#ffd166", alpha=0.8)
    ax.set_yscale("log"); ax.set_title("per-step reward histogram")
    fig.suptitle(f"epoch {d['epoch']} - trade maker (PPO) diagnostics", y=1.01)
    return _save(fig, path)


def plot_distribution(d: Dict, path: str):
    final = d["equity"][-1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    ax = axes[0]
    ax.hist(final, bins=max(6, len(final) // 2), color="#4cc9f0", alpha=0.85)
    ax.axvline(d["start_cash"], color="#9aa0a6", ls=":", label="start $20")
    ax.axvline(d["target"], color="#ffd166", ls="--", label=f"target ${d['target']:.0f}")
    ax.axvline(float(np.median(final)), color="#f72585", lw=1.6, label="median")
    ax.set_title("final equity across trajectories"); ax.set_xlabel("$"); ax.legend(fontsize=7)

    ax = axes[1]
    rets = 100 * (np.diff(d["equity"], axis=0) / np.maximum(d["equity"][:-1], 1e-9))
    ax.hist(rets.ravel(), bins=100, color="#06d6a0", alpha=0.8)
    ax.set_yscale("log"); ax.set_title("per-minute return %"); ax.set_xlabel("%")

    ax = axes[2]
    labels = ["turnover", "fees $", "trades", "bankrupt %"]
    vals = [d["summary"]["turnover"], d["summary"]["fees"],
            d["summary"]["n_trades"], 100 * d["summary"]["bankrupt"]]
    ax.barh(labels, vals, color=ACCENT[:4], alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3g}", va="center", fontsize=8)
    ax.set_title("execution stats"); ax.grid(axis="y", alpha=0)
    return _save(fig, path)


def plot_advisor(text: str, d: Dict, path: str):
    fig = plt.figure(figsize=(11, 6.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                               facecolor="#131821", edgecolor="#2a2f3a", lw=1.5))
    verdict = d.get("verdict", "")
    col = {"RISK-ON": "#06d6a0", "RISK-OFF": "#f72585",
           "NEUTRAL": "#ffd166"}.get(verdict.split()[0] if verdict else "", "#4cc9f0")
    ax.text(0.05, 0.93, f"MARKET ADVISOR  -  {d['date']}", fontsize=15, weight="bold",
            color="#e6e6e6", transform=ax.transAxes)
    ax.text(0.05, 0.875, f"epoch {d['epoch']}   |   mode {d['mode']}   |   "
                         f"result ${d['equity'][-1].mean():.2f} from ${d['start_cash']:.0f}",
            fontsize=9.5, color="#9aa0a6", transform=ax.transAxes)
    ax.text(0.95, 0.925, verdict, fontsize=16, weight="bold", color=col,
            ha="right", transform=ax.transAxes)
    ax.plot([0.05, 0.95], [0.85, 0.85], color="#2a2f3a", lw=1.2,
            transform=ax.transAxes)
    ax.text(0.05, 0.82, text, fontsize=10, color="#cfd3dc", va="top", ha="left",
            transform=ax.transAxes, family="monospace", linespacing=1.6)
    return _save(fig, path)


# ----------------------------------------------------------------- run level
def _roll(v, k=25):
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return v
    k = max(1, min(k, len(v) // 2))
    return np.convolve(v, np.ones(k) / k, mode="valid")


def plot_dashboard(hist: Dict[str, List[float]], cfg, path: str, title_extra: str = ""):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]
    eq = hist.get("equity_mean", [])
    ax.plot(eq, color="#4cc9f0", lw=0.7, alpha=0.4)
    if len(eq) > 4:
        r = _roll(eq); ax.plot(np.arange(len(eq) - len(r), len(eq)), r,
                               color="#4cc9f0", lw=2, label="rolling mean")
    ax.plot(hist.get("equity_best", []), color="#06d6a0", lw=0.6, alpha=0.5, label="best path")
    ax.axhline(cfg.env.start_cash, color="#9aa0a6", ls=":")
    ax.axhline(cfg.env.target_equity, color="#ffd166", ls="--", label="target")
    ax.set_title("final equity per epoch ($20 start)"); ax.set_xlabel("epoch")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ht = hist.get("hit_target", [])
    ax.plot(100 * np.asarray(ht), color="#06d6a0", lw=0.7, alpha=0.4)
    if len(ht) > 4:
        r = _roll(100 * np.asarray(ht))
        ax.plot(np.arange(len(ht) - len(r), len(ht)), r, color="#06d6a0", lw=2)
    ax.set_title(f"% trajectories reaching ${cfg.env.target_equity:.0f}")
    ax.set_xlabel("epoch"); ax.set_ylim(-2, 102)

    ax = axes[0, 2]
    for k, c, lab in (("return_pct", "#ffd166", "mean return %"),
                      ("bankrupt", "#f72585", "liquidation rate %")):
        v = np.asarray(hist.get(k, []), dtype=float)
        if k == "bankrupt":
            v = 100 * v
        if len(v) > 4:
            r = _roll(v); ax.plot(np.arange(len(v) - len(r), len(v)), r, color=c, lw=1.6,
                                  label=lab)
    ax.axhline(0, color="#9aa0a6", lw=0.7); ax.set_title("daily return vs blow-ups")
    ax.set_xlabel("epoch"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    for k, c in (("pred_nll", "#4cc9f0"), ("pred_ce", "#b388ff")):
        v = hist.get(k, [])
        if v:
            ax.plot(v, color=c, lw=0.9, label=k)
    ax.set_title("predictor losses"); ax.set_xlabel("epoch"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    for k, c in (("an_ce", "#4cc9f0"), ("an_vol", "#ffd166"), ("an_align", "#06d6a0")):
        v = hist.get(k, [])
        if v:
            ax.plot(v, color=c, lw=0.9, label=k)
    ax.set_title("analyst losses"); ax.set_xlabel("epoch"); ax.legend(fontsize=8)

    ax = axes[1, 2]
    for k, c in (("pred_dir_acc", "#4cc9f0"), ("an_acc", "#06d6a0"), ("pred_ic", "#ffd166")):
        v = np.asarray(hist.get(k, []), dtype=float)
        if len(v) > 4:
            r = _roll(v); ax.plot(np.arange(len(v) - len(r), len(v)), r, color=c, lw=1.5,
                                  label=k)
    ax.set_title("forecast quality"); ax.set_xlabel("epoch"); ax.legend(fontsize=8)

    fig.suptitle(f"trader_v6 - training dashboard - {len(eq)} epochs {title_extra}", y=1.0)
    return _save(fig, path)


def plot_agents(hist: Dict[str, List[float]], path: str):
    keys = [k for k in ("pred_nll", "pred_ce", "pred_dir_acc", "pred_ic",
                        "an_ce", "an_vol", "an_align", "an_acc", "an_conf", "an_risk",
                        "pg_loss", "v_loss", "entropy", "kl", "clipfrac", "grad_norm",
                        "equity_mean", "equity_best", "hit_target", "turnover",
                        "fees", "n_trades", "day_vol", "epoch_sec")
            if hist.get(k)]
    n = len(keys)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, k in enumerate(keys):
        ax = axes[i]
        v = np.asarray(hist[k], dtype=float)
        ax.plot(v, color=ACCENT[i % len(ACCENT)], lw=0.7, alpha=0.45)
        if len(v) > 8:
            r = _roll(v, 15)
            ax.plot(np.arange(len(v) - len(r), len(v)), r, color="#e6e6e6", lw=1.3)
        ax.set_title(k, fontsize=9)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("trader_v6 - every tracked metric", y=1.001)
    return _save(fig, path)


def plot_eval(ev: Dict, path: str):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    ax = axes[0]
    ax.bar(range(len(ev["finals"])), ev["finals"], color="#4cc9f0", alpha=0.85)
    ax.axhline(ev["start_cash"], color="#9aa0a6", ls=":")
    ax.axhline(ev["target"], color="#ffd166", ls="--")
    ax.set_xticks(range(len(ev["dates"])))
    ax.set_xticklabels(ev["dates"], rotation=75, fontsize=6)
    ax.set_title(f"held-out days: final equity (mean ${np.mean(ev['finals']):.2f})")
    ax.set_ylabel("$")

    ax = axes[1]
    for c in ev["curves"]:
        ax.plot(np.asarray(c) , color="#4cc9f0", alpha=0.35, lw=0.9)
    ax.axhline(ev["target"], color="#ffd166", ls="--")
    ax.set_title("held-out equity curves"); ax.set_xlabel("minute")

    ax = axes[2]
    ax.hist(100 * (np.asarray(ev["finals"]) / ev["start_cash"] - 1), bins=20,
            color="#06d6a0", alpha=0.85)
    ax.axvline(0, color="#9aa0a6", ls=":")
    ax.set_title("held-out daily return %")
    fig.suptitle(f"trader_v6 - out-of-sample evaluation @ epoch {ev['epoch']}", y=1.02)
    return _save(fig, path)
