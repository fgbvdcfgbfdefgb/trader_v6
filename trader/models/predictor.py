"""Agent 1 - the price predictor.

A causal dilated TCN that, at every minute t, emits a forecast of the forward
log-return at several horizons (mean + log-variance + direction logits) for each
asset, using only information available at or before t. Its per-timestep latent
is consumed by the analyst and the trade maker.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelNorm(nn.Module):
    """LayerNorm over channels only, applied independently at each timestep.

    GroupNorm/BatchNorm over a [B, C, T] conv activation would normalise across the
    time axis and leak future information into the past - that breaks causality and
    silently inflates backtest performance. This does not.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):                       # [B, C, T]
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class CausalBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float, kernel: int = 3):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.c1 = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.c2 = nn.Conv1d(ch, ch, 1)
        self.n1 = ChannelNorm(ch)
        self.n2 = ChannelNorm(ch)
        self.do = nn.Dropout(dropout)

    def forward(self, x):                       # x: [B, C, T]
        h = F.pad(x, (self.pad, 0))
        h = self.do(F.gelu(self.n1(self.c1(h))))
        h = self.do(F.gelu(self.n2(self.c2(h))))
        return x + h


class PricePredictor(nn.Module):
    def __init__(self, n_features: int, cfg):
        super().__init__()
        ch, H = cfg.tcn_channels, len(cfg.horizons)
        self.horizons = list(cfg.horizons)
        self.inp = nn.Conv1d(n_features, ch, 1)
        dil = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        n = max(4, cfg.n_layers * 2)
        self.blocks = nn.ModuleList([CausalBlock(ch, dil[i % len(dil)], cfg.dropout)
                                     for i in range(n)])
        self.norm = ChannelNorm(ch)
        self.proj = nn.Linear(ch, cfg.d_model)
        self.head_mu = nn.Linear(cfg.d_model, H)
        self.head_logvar = nn.Linear(cfg.d_model, H)
        self.head_dir = nn.Linear(cfg.d_model, H * 3)
        self.out_dim = cfg.d_model
        self.n_h = H

    def forward(self, feats: torch.Tensor):
        """feats: [A, T, F] -> dict of [A, T, ...] tensors."""
        x = self.inp(feats.transpose(1, 2))
        for b in self.blocks:
            x = b(x)
        z = self.proj(F.gelu(self.norm(x)).transpose(1, 2))      # [A, T, D]
        mu = self.head_mu(z)
        logvar = self.head_logvar(z).clamp(-12.0, 4.0)
        dirs = self.head_dir(z).view(*z.shape[:2], self.n_h, 3)
        return {"latent": z, "mu": mu, "logvar": logvar, "dir": dirs}

    # ------------------------------------------------------------------ loss
    @staticmethod
    def targets(close: torch.Tensor, horizons, vol: torch.Tensor):
        """close: [A, T] -> forward log-returns scaled by local vol, + direction labels."""
        logc = torch.log(close.clamp_min(1e-12))
        outs, labels, masks = [], [], []
        T = close.shape[1]
        for h in horizons:
            fwd = torch.zeros_like(logc)
            fwd[:, :T - h] = logc[:, h:] - logc[:, :T - h]
            m = torch.zeros_like(fwd)
            m[:, :T - h] = 1.0
            scaled = fwd / (vol * (h ** 0.5)).clamp_min(1e-6)
            outs.append(scaled.clamp(-10, 10))
            lab = torch.ones_like(scaled, dtype=torch.long)       # 1 = flat
            lab = torch.where(scaled > 0.5, torch.full_like(lab, 2), lab)
            lab = torch.where(scaled < -0.5, torch.zeros_like(lab), lab)
            labels.append(lab)
            masks.append(m)
        return (torch.stack(outs, -1), torch.stack(labels, -1), torch.stack(masks, -1))

    def loss(self, out, close, vol):
        y, lab, mask = self.targets(close, self.horizons, vol)
        mu, logvar = out["mu"], out["logvar"]
        nll = 0.5 * (logvar + (y - mu) ** 2 / logvar.exp().clamp_min(1e-8))
        nll = (nll * mask).sum() / mask.sum().clamp_min(1.0)
        ce = F.cross_entropy(out["dir"].reshape(-1, 3), lab.reshape(-1), reduction="none")
        ce = (ce * mask.reshape(-1)).sum() / mask.sum().clamp_min(1.0)
        with torch.no_grad():
            pred_dir = out["dir"].argmax(-1)
            acc = ((pred_dir == lab).float() * mask).sum() / mask.sum().clamp_min(1.0)
            yv, mv = y[mask > 0], mu[mask > 0]
            ic = torch.nan_to_num(
                ((yv - yv.mean()) * (mv - mv.mean())).mean()
                / (yv.std().clamp_min(1e-8) * mv.std().clamp_min(1e-8)))
        return nll + 0.5 * ce, {"pred_nll": nll.detach(), "pred_ce": ce.detach(),
                                "pred_dir_acc": acc, "pred_ic": ic}
