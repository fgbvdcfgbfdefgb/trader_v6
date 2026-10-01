"""Agent 3 - the trade maker.

A recurrent actor-critic. At each minute it sees:
  * the raw market features
  * the predictor's latent + its forecast mean/sigma
  * the analyst's latent, regime posterior, suggested exposure, risk budget
  * its own portfolio state (weights, equity vs target, drawdown, time left)
and outputs target portfolio weights for BTC / ETH / LTC.

The context (everything except portfolio state) is precomputed once per day for all
1440 minutes, so the rollout only pays for a GRU cell per step.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PORT_STATE_DIM_BASE = 6          # per-env scalars that are not per-asset


class TradeMaker(nn.Module):
    def __init__(self, n_assets: int, ctx_dim: int, cfg, env_cfg):
        super().__init__()
        h = cfg.policy_hidden
        self.n_assets = n_assets
        self.port_dim = n_assets * 2 + PORT_STATE_DIM_BASE
        self.enc = nn.Sequential(
            nn.Linear(ctx_dim + self.port_dim, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU())
        self.cell = nn.GRUCell(h, h)
        self.body = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU())
        self.mu = nn.Linear(h, n_assets)
        self.v = nn.Linear(h, 1)
        self.log_std = nn.Parameter(torch.full((n_assets,), -0.7))
        self.hidden = h
        nn.init.zeros_(self.mu.bias)
        self.mu.weight.data.mul_(0.1)

    def init_state(self, n: int, device, dtype=torch.float32):
        return torch.zeros(n, self.hidden, device=device, dtype=dtype)

    def step(self, ctx_t: torch.Tensor, port_t: torch.Tensor, h: torch.Tensor):
        """ctx_t: [N, ctx_dim]  port_t: [N, port_dim]  h: [N, H]."""
        x = self.enc(torch.cat([ctx_t, port_t], dim=-1))
        h = self.cell(x, h)
        z = self.body(h)
        return self.mu(z), self.v(z).squeeze(-1), h

    def dist(self, mu: torch.Tensor):
        std = self.log_std.clamp(-3.0, 0.7).exp().expand_as(mu)
        return torch.distributions.Normal(mu, std)

    @staticmethod
    def to_weights(action: torch.Tensor, mode: str, max_leverage: float,
                   risk_budget: torch.Tensor | None = None):
        """Squash raw actions into legal portfolio weights."""
        if mode == "spot":
            w = torch.sigmoid(action)                     # long-only
            s = w.sum(-1, keepdim=True).clamp_min(1e-6)
            w = w * torch.clamp(1.0 / s, max=1.0)         # gross <= 1 (no leverage)
        else:
            w = torch.tanh(action) * max_leverage
            gross = w.abs().sum(-1, keepdim=True).clamp_min(1e-6)
            w = w * torch.clamp(max_leverage / gross, max=1.0)
        if risk_budget is not None:
            w = w * risk_budget.unsqueeze(-1)
        return w


def build_context(feats: torch.Tensor, pred: dict, an: dict) -> torch.Tensor:
    """Flatten per-asset streams into a single per-minute context vector.

    feats [A,T,F], predictor/analyst outputs -> ctx [T, ctx_dim]
    """
    A, T, _ = feats.shape
    parts = [
        feats.permute(1, 0, 2).reshape(T, -1),
        pred["latent"].permute(1, 0, 2).reshape(T, -1),
        pred["mu"].permute(1, 0, 2).reshape(T, -1),
        pred["logvar"].mul(0.5).exp().permute(1, 0, 2).reshape(T, -1),
        pred["dir"].softmax(-1).permute(1, 0, 2, 3).reshape(T, -1),
        an["latent"].permute(1, 0, 2).reshape(T, -1),
        an["regime"].softmax(-1).permute(1, 0, 2).reshape(T, -1),
        an["vol"].permute(1, 0).reshape(T, -1),
        an["exposure"].permute(1, 0).reshape(T, -1),
        an["confidence"].permute(1, 0).reshape(T, -1),
        an["risk_budget"].reshape(T, 1),
    ]
    return torch.cat(parts, dim=-1)


def context_dim(n_assets: int, n_features: int, model_cfg) -> int:
    H = len(model_cfg.horizons)
    d = model_cfg.d_model
    per_asset = n_features + d + H + H + 3 * H + d + model_cfg.n_regimes + 1 + 1 + 1
    return n_assets * per_asset + 1
