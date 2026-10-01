"""Vectorised, GPU-resident trading environment for one calendar day.

N independent trajectories run over the *same* day (different sampled actions), which
is what gives PPO a usable batch while keeping "one epoch == one random day".

Accounting is true mark-to-market: the state is signed asset **quantities** plus cash,
not weights. That makes it exact for both longs and shorts, and it lets the agent
decide every `step_minutes` while the portfolio is still marked and liquidation-checked
on every single minute in between (one matmul per interval instead of a Python loop).

Two modes:
  spot    - long only, gross <= 1, 10 bps taker fee
  futures - perpetual-style, long AND short, up to `max_leverage` gross,
            4 bps taker fee, 8-hourly funding, liquidation at maintenance margin

Every episode starts with `start_cash` (default $20) and is scored against
`target_equity` (default $30).
"""
from __future__ import annotations

import torch


class MarketEnv:
    def __init__(self, close: torch.Tensor, volume: torch.Tensor, cfg, n_envs: int,
                 device, dtype=torch.float32, step_minutes: int = 1):
        """close/volume: [A, T] tensors for the chosen day."""
        self.cfg = cfg
        self.device = device
        self.close = close.to(device=device, dtype=torch.float32).clamp_min(1e-9)
        self.volume = volume.to(device=device, dtype=torch.float32)
        self.A, self.T = self.close.shape
        self.N = n_envs
        self.K = max(1, int(step_minutes))
        self.n_steps = max(1, (self.T - 1) // self.K)
        self.fee = cfg.spot_fee if cfg.mode == "spot" else cfg.taker_fee
        self.max_lev = 1.0 if cfg.mode == "spot" else cfg.max_leverage
        # quote volume per decision interval, for a (small but non-zero) impact term
        self.qvol = (self.volume * self.close).clamp_min(1.0)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        N, A, dev = self.N, self.A, self.device
        self.t = 0
        self.cash = torch.full((N,), self.cfg.start_cash, device=dev)
        self.q = torch.zeros(N, A, device=dev)
        self.equity = self.cash.clone()
        self.peak = self.equity.clone()
        self.alive = torch.ones(N, device=dev, dtype=torch.bool)
        self.turnover_total = torch.zeros(N, device=dev)
        self.fees_total = torch.zeros(N, device=dev)
        self.n_trades = torch.zeros(N, device=dev)
        return self.port_state()

    def _prices(self, minute: int) -> torch.Tensor:
        return self.close[:, min(minute, self.T - 1)]

    def _weights(self) -> torch.Tensor:
        p = self._prices(self.t * self.K)
        return self.q * p.unsqueeze(0) / self.equity.clamp_min(1e-6).unsqueeze(-1)

    def port_state(self) -> torch.Tensor:
        """Per-env portfolio observation: [N, A*2 + 6]."""
        cfg = self.cfg
        eq = self.equity.clamp_min(1e-6)
        w = self._weights()
        dd = 1.0 - eq / self.peak.clamp_min(1e-6)
        prog = (eq - cfg.start_cash) / max(1e-6, cfg.target_equity - cfg.start_cash)
        tleft = 1.0 - self.t / max(1, self.n_steps)
        scal = torch.stack([
            torch.log(eq / cfg.start_cash),
            prog.clamp(-2, 3),
            dd.clamp(0, 1),
            torch.full_like(eq, tleft),
            w.abs().sum(-1) / max(1e-6, self.max_lev),
            w.sum(-1) / max(1e-6, self.max_lev),
        ], dim=-1)
        return torch.cat([w, w.abs(), scal], dim=-1)

    # ------------------------------------------------------------------- step
    def step(self, w_target: torch.Tensor):
        cfg = self.cfg
        t0 = self.t * self.K
        t1 = min(t0 + self.K, self.T - 1)
        alive_f = self.alive.float()
        w_target = w_target * alive_f.unsqueeze(-1)

        p0 = self._prices(t0)                                    # [A]
        eq0 = (self.cash + (self.q * p0.unsqueeze(0)).sum(-1)).clamp_min(0.0)

        # ------------------------------------------------- rebalance at t0
        q_target = w_target * eq0.unsqueeze(-1) / p0.unsqueeze(0)
        dq = q_target - self.q
        notional_i = dq.abs() * p0.unsqueeze(0)
        small = notional_i < cfg.min_trade_frac * eq0.clamp_min(1e-6).unsqueeze(-1)
        dq = torch.where(small, torch.zeros_like(dq), dq)
        notional_i = dq.abs() * p0.unsqueeze(0)
        notional = notional_i.sum(-1)

        impact = (notional_i / self.qvol[:, t0:t1].sum(-1).clamp_min(1.0).unsqueeze(0)
                  ).clamp(0, 0.05)
        slip = (cfg.slippage_bps / 1e4) * (1.0 + 10.0 * impact.mean(-1))
        cost = notional * (self.fee + slip)

        q = self.q + dq
        cash = self.cash - (dq * p0.unsqueeze(0)).sum(-1) - cost

        # ------------------------------- mark to market on every minute in (t0, t1]
        path_px = self.close[:, t0 + 1:t1 + 1]                   # [A, k]
        if path_px.shape[1] == 0:
            path_px = self.close[:, t1:t1 + 1]
        eq_path = cash.unsqueeze(-1) + q @ path_px                # [N, k]
        gross_path = q.abs() @ path_px                            # [N, k]

        funding = torch.zeros_like(eq0)
        n_fund = sum(1 for m in range(t0 + 1, t1 + 1) if m % 480 == 0)
        if n_fund:
            funding = cfg.funding_rate_8h * n_fund * (q * path_px[:, -1]).sum(-1)

        eq1 = eq_path[:, -1] - funding
        liq = ((eq_path <= cfg.maint_margin * gross_path).any(-1)) & self.alive
        liq = liq | ((eq1 <= 0) & self.alive)
        eq1 = torch.where(liq, torch.zeros_like(eq1), eq1).clamp_min(0.0)

        dead = (~self.alive) | liq
        q = torch.where(dead.unsqueeze(-1), torch.zeros_like(q), q)
        cash = torch.where(dead, torch.zeros_like(cash), eq1 - (q * path_px[:, -1]).sum(-1))

        # ----------------------------------------------------------- reward
        dd0 = 1.0 - eq0 / self.peak.clamp_min(1e-6)
        peak1 = torch.maximum(self.peak, eq1)
        dd1 = 1.0 - eq1 / peak1.clamp_min(1e-6)
        turnover = notional / eq0.clamp_min(1e-6)

        log_growth = torch.log(eq1.clamp_min(1e-4) / eq0.clamp_min(1e-4))
        reward = cfg.reward_scale * log_growth
        reward = reward - cfg.turnover_penalty * turnover * cfg.reward_scale
        reward = reward - cfg.drawdown_penalty * (dd1 - dd0).clamp_min(0.0) * cfg.reward_scale
        reward = reward * alive_f
        reward = reward - cfg.bankrupt_penalty * liq.float()

        self.q, self.cash, self.equity, self.peak = q, cash, eq1, peak1
        self.alive = self.alive & (~liq)
        self.turnover_total += turnover * alive_f
        self.fees_total += (cost + funding.abs()) * alive_f
        self.n_trades += (turnover > cfg.min_trade_frac).float() * alive_f
        self.t += 1

        done = self.t >= self.n_steps
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
