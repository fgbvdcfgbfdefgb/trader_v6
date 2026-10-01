"""Vectorised, GPU-resident trading environment for one calendar day.

N independent trajectories run over the *same* day (different sampled actions), which
is what gives PPO a usable batch while keeping "one epoch == one random day".

Two modes:
  spot    - long only, gross exposure <= 1, 10 bps taker fee
  futures - perpetual-style, long AND short, up to `max_leverage` gross,
            4 bps taker fee, 8-hourly funding, liquidation at maintenance margin

Every episode starts with `start_cash` (default $20) and is scored against
`target_equity` (default $30).
"""
from __future__ import annotations

import torch


class MarketEnv:
    def __init__(self, close: torch.Tensor, volume: torch.Tensor, cfg, n_envs: int,
                 device, dtype=torch.float32):
        """close/volume: [A, T] tensors for the chosen day."""
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.close = close.to(device=device, dtype=torch.float32)
        self.volume = volume.to(device=device, dtype=torch.float32)
        self.A, self.T = self.close.shape
        self.N = n_envs
        self.fee = cfg.spot_fee if cfg.mode == "spot" else cfg.taker_fee
        self.max_lev = 1.0 if cfg.mode == "spot" else cfg.max_leverage
        self.rets = torch.zeros_like(self.close)
        self.rets[:, :-1] = self.close[:, 1:] / self.close[:, :-1].clamp_min(1e-12) - 1.0
        self.rets = self.rets.clamp(-0.5, 0.5)
        # quote volume per minute, used for a (tiny but non-zero) market-impact term
        self.qvol = (self.volume * self.close).clamp_min(1.0)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        N, A, dev = self.N, self.A, self.device
        self.t = 0
        self.equity = torch.full((N,), self.cfg.start_cash, device=dev, dtype=torch.float32)
        self.w = torch.zeros(N, A, device=dev, dtype=torch.float32)
        self.peak = self.equity.clone()
        self.alive = torch.ones(N, device=dev, dtype=torch.bool)
        self.turnover_total = torch.zeros(N, device=dev)
        self.fees_total = torch.zeros(N, device=dev)
        self.n_trades = torch.zeros(N, device=dev)
        return self.port_state()

    def port_state(self) -> torch.Tensor:
        """Per-env portfolio observation: [N, A*2 + 6]."""
        cfg = self.cfg
        eq = self.equity.clamp_min(1e-6)
        dd = 1.0 - eq / self.peak.clamp_min(1e-6)
        prog = (eq - cfg.start_cash) / max(1e-6, cfg.target_equity - cfg.start_cash)
        tleft = 1.0 - self.t / max(1, self.T - 1)
        gross = self.w.abs().sum(-1)
        net = self.w.sum(-1)
        per_asset = torch.cat([self.w, self.w.abs()], dim=-1)
        scal = torch.stack([
            torch.log(eq / cfg.start_cash),
            prog.clamp(-2, 3),
            dd.clamp(0, 1),
            torch.full_like(eq, tleft),
            gross / max(1e-6, self.max_lev),
            net / max(1e-6, self.max_lev),
        ], dim=-1)
        return torch.cat([per_asset, scal], dim=-1)

    # ------------------------------------------------------------------- step
    def step(self, w_target: torch.Tensor):
        cfg = self.cfg
        t = self.t
        eq0 = self.equity
        alive_f = self.alive.float()
        w_target = w_target * alive_f.unsqueeze(-1)

        dw = w_target - self.w
        small = dw.abs() < cfg.min_trade_frac
        dw = torch.where(small, torch.zeros_like(dw), dw)
        w_exec = self.w + dw

        turnover = dw.abs().sum(-1)                               # fraction of equity
        notional = turnover * eq0
        impact = (notional.unsqueeze(-1) / self.qvol[:, t].unsqueeze(0)).clamp(0, 0.05)
        slip = (cfg.slippage_bps / 1e4) * (1.0 + 10.0 * impact.mean(-1))
        cost = notional * (self.fee + slip)

        r = self.rets[:, t].unsqueeze(0)                          # [1, A]
        pnl = eq0 * (w_exec * r).sum(-1)

        funding = torch.zeros_like(eq0)
        if t > 0 and t % 480 == 0:
            funding = cfg.funding_rate_8h * w_exec.sum(-1) * eq0   # longs pay

        eq1 = eq0 + pnl - cost - funding
        gross_notional = w_exec.abs().sum(-1) * eq0
        liq = (eq1 <= cfg.maint_margin * gross_notional) & self.alive
        eq1 = torch.where(liq, torch.zeros_like(eq1), eq1).clamp_min(0.0)

        value = w_exec * eq0.unsqueeze(-1) * (1.0 + r)
        w1 = torch.where(eq1.unsqueeze(-1) > 1e-6,
                         value / eq1.clamp_min(1e-6).unsqueeze(-1),
                         torch.zeros_like(value))
        w1 = torch.where((~self.alive | liq).unsqueeze(-1), torch.zeros_like(w1), w1)
        w1 = w1.clamp(-3 * self.max_lev, 3 * self.max_lev)

        dd0 = 1.0 - eq0 / self.peak.clamp_min(1e-6)
        peak1 = torch.maximum(self.peak, eq1)
        dd1 = 1.0 - eq1 / peak1.clamp_min(1e-6)

        log_growth = torch.log(eq1.clamp_min(1e-4) / eq0.clamp_min(1e-4))
        reward = cfg.reward_scale * log_growth
        reward = reward - cfg.turnover_penalty * turnover * cfg.reward_scale
        reward = reward - cfg.drawdown_penalty * (dd1 - dd0).clamp_min(0.0) * cfg.reward_scale
        reward = reward * alive_f
        reward = reward - cfg.bankrupt_penalty * liq.float()

        self.equity, self.w, self.peak = eq1, w1, peak1
        self.alive = self.alive & (~liq)
        self.turnover_total += turnover * alive_f
        self.fees_total += (cost + funding.abs()) * alive_f
        self.n_trades += (turnover > cfg.min_trade_frac).float() * alive_f
        self.t += 1

        done = self.t >= self.T - 1
        if done:
            reward = reward + self.terminal_bonus()
        return self.port_state(), reward, done, {"liquidated": liq, "turnover": turnover}

    def terminal_bonus(self) -> torch.Tensor:
        cfg = self.cfg
        span = max(1e-6, cfg.target_equity - cfg.start_cash)
        progress = (self.equity - cfg.start_cash) / span
        bonus = cfg.target_bonus * progress.clamp(-1.0, 1.0)
        bonus = bonus + 0.5 * cfg.target_bonus * (self.equity >= cfg.target_equity).float()
        return bonus

    # ----------------------------------------------------------------- scoring
    def summary(self) -> dict:
        eq = self.equity
        return {
            "equity_mean": eq.mean().item(),
            "equity_median": eq.median().item(),
            "equity_best": eq.max().item(),
            "equity_worst": eq.min().item(),
            "hit_target": (eq >= self.cfg.target_equity).float().mean().item(),
            "bankrupt": (~self.alive).float().mean().item(),
            "return_pct": ((eq.mean() / self.cfg.start_cash - 1) * 100).item(),
            "turnover": self.turnover_total.mean().item(),
            "fees": self.fees_total.mean().item(),
            "n_trades": self.n_trades.mean().item(),
        }
