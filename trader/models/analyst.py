"""Agent 2 - the market analyser / advisor.

Consumes raw features *and* the predictor's latent, runs a causal temporal GRU per
asset plus cross-asset attention at each minute, and emits:
  * regime probabilities      (6 classes, from trend x volatility)
  * a volatility forecast     (next 60m realised vol, log-scale)
  * a suggested exposure      per asset in [-1, 1]
  * a market-wide risk budget in [0, 1]  (how much gross leverage is sane now)
  * confidence                in [0, 1]
These become both the trade maker's context and the human-readable advisory that
gets rendered to PNG each epoch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

REGIME_NAMES = ["calm-up", "calm-down", "choppy", "volatile-up", "volatile-down", "crash/panic"]


class CrossAssetAttention(nn.Module):
    """Tiny self-attention across the A assets, applied independently at each minute."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        heads = max(1, min(heads, d // 16))
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):                        # x: [A, T, D]
        A, T, D = x.shape
        h = x.permute(1, 0, 2)                   # [T, A, D] -> T as batch
        o, _ = self.att(h, h, h, need_weights=False)
        return self.norm(x + o.permute(1, 0, 2))


class MarketAnalyst(nn.Module):
    def __init__(self, n_features: int, pred_dim: int, cfg):
        super().__init__()
        d = cfg.d_model
        self.inp = nn.Linear(n_features + pred_dim + 2 * len(cfg.horizons), d)
        self.gru = nn.GRU(d, d, num_layers=max(1, cfg.n_layers // 2),
                          batch_first=True, dropout=cfg.dropout if cfg.n_layers > 2 else 0.0)
        self.xatt = CrossAssetAttention(d, cfg.n_heads)
        self.norm = nn.LayerNorm(d)
        self.head_regime = nn.Linear(d, cfg.n_regimes)
        self.head_vol = nn.Linear(d, 1)
        self.head_expo = nn.Linear(d, 1)
        self.head_conf = nn.Linear(d, 1)
        self.head_risk = nn.Linear(d, 1)
        self.out_dim = d
        self.n_regimes = cfg.n_regimes

    def forward(self, feats: torch.Tensor, pred: dict):
        x = torch.cat([feats, pred["latent"], pred["mu"],
                       pred["logvar"].mul(0.5).exp()], dim=-1)
        h, _ = self.gru(F.gelu(self.inp(x)))
        h = self.norm(self.xatt(h))                                  # [A, T, D]
        mkt = h.mean(dim=0, keepdim=True)                            # [1, T, D]
        return {
            "latent": h,
            "regime": self.head_regime(h),                           # [A, T, R]
            "vol": self.head_vol(h).squeeze(-1),                     # [A, T]
            "exposure": torch.tanh(self.head_expo(h)).squeeze(-1),   # [A, T]
            # logits kept alongside the probability: BCE-with-logits is the only
            # autocast-safe form, and AMP is on by default on GPU.
            "conf_logit": self.head_conf(h).squeeze(-1),
            "confidence": torch.sigmoid(self.head_conf(h)).squeeze(-1),
            "risk_budget": torch.sigmoid(self.head_risk(mkt)).squeeze(-1).squeeze(0),  # [T]
        }

    # ------------------------------------------------------------------ loss
    @staticmethod
    def regime_targets(close: torch.Tensor, vol: torch.Tensor, win: int = 60):
        """Label each minute by the *forward* 60m trend and volatility regime."""
        logc = torch.log(close.clamp_min(1e-12))
        T = close.shape[1]
        fwd = torch.zeros_like(logc)
        fwd[:, :T - win] = logc[:, win:] - logc[:, :T - win]
        mask = torch.zeros_like(fwd); mask[:, :T - win] = 1.0
        fvol = torch.zeros_like(logc)
        d1 = torch.zeros_like(logc)
        d1[:, 1:] = logc[:, 1:] - logc[:, :-1]
        cs = torch.cumsum(d1 ** 2, dim=1)
        fvol[:, :T - win] = ((cs[:, win:] - cs[:, :T - win]) / win).sqrt()

        ref = vol.clamp_min(1e-8)
        z = fwd / (ref * win ** 0.5).clamp_min(1e-8)
        hot = fvol > 1.6 * ref
        lab = torch.full_like(z, 2, dtype=torch.long)                # choppy
        lab = torch.where((~hot) & (z > 0.75), torch.zeros_like(lab), lab)
        lab = torch.where((~hot) & (z < -0.75), torch.ones_like(lab), lab)
        lab = torch.where(hot & (z > 0.75), torch.full_like(lab, 3), lab)
        lab = torch.where(hot & (z < -0.75), torch.full_like(lab, 4), lab)
        lab = torch.where(hot & (fvol > 3.0 * ref) & (z < -0.3),
                          torch.full_like(lab, 5), lab)
        return lab, fvol, fwd, mask

    def loss(self, out, close, vol, win: int = 60):
        lab, fvol, fwd, mask = self.regime_targets(close, vol, win)
        m = mask.reshape(-1)
        ce = F.cross_entropy(out["regime"].reshape(-1, self.n_regimes),
                             lab.reshape(-1), reduction="none")
        ce = (ce * m).sum() / m.sum().clamp_min(1.0)

        tgt_vol = torch.log(fvol.clamp_min(1e-8) / vol.clamp_min(1e-8))
        vl = (F.smooth_l1_loss(out["vol"], tgt_vol.clamp(-3, 3), reduction="none")
              * mask).sum() / mask.sum().clamp_min(1.0)

        # exposure head: maximise risk-adjusted alignment with the forward move
        z = (fwd / (vol * win ** 0.5).clamp_min(1e-8)).clamp(-5, 5)
        align = -(out["exposure"] * z * mask).sum() / mask.sum().clamp_min(1.0)
        reg = 1e-3 * (out["exposure"] ** 2).mean()

        # risk budget should shrink when forward vol spikes
        rb_tgt = torch.sigmoid(-2.0 * (fvol / vol.clamp_min(1e-8) - 1.0).mean(0)).detach()
        rl = F.mse_loss(out["risk_budget"], rb_tgt)

        # confidence is calibrated against regime correctness
        with torch.no_grad():
            correct = (out["regime"].argmax(-1) == lab).float()
        cl = (F.binary_cross_entropy_with_logits(out["conf_logit"], correct,
                                                 reduction="none") * mask).sum() \
            / mask.sum().clamp_min(1.0)

        total = ce + 0.5 * vl + 0.3 * align + reg + 0.2 * rl + 0.1 * cl
        with torch.no_grad():
            acc = ((out["regime"].argmax(-1) == lab).float() * mask).sum() \
                / mask.sum().clamp_min(1.0)
        return total, {"an_ce": ce.detach(), "an_vol": vl.detach(),
                       "an_align": align.detach(), "an_risk": rl.detach(),
                       "an_acc": acc, "an_conf": out["confidence"].mean().detach()}
