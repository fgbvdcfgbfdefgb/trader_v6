"""Recurrent PPO with truncated-BPTT segments.

Rollout is sequential over the day's 1440 minutes (N parallel trajectories).
The update replays the GRU over fixed-length segments whose initial hidden states
were cached during the rollout, so all segments can be processed as one batch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from ..models.trade_maker import TradeMaker


@dataclass
class Rollout:
    ctx: torch.Tensor          # [T, ctx_dim]  (shared across envs)
    port: torch.Tensor         # [T, N, P]
    actions: torch.Tensor      # [T, N, A]
    logp: torch.Tensor         # [T, N]
    values: torch.Tensor       # [T, N]
    rewards: torch.Tensor      # [T, N]
    alive: torch.Tensor        # [T, N]
    h0: torch.Tensor           # [S, N, H] segment-initial hidden states
    seg_len: int
    equity: torch.Tensor       # [T+1, N] for plotting
    weights: torch.Tensor      # [T, N, A] executed weights
    last_value: torch.Tensor   # [N]


@torch.no_grad()
def collect(agent: TradeMaker, env, ctx: torch.Tensor, risk_budget: torch.Tensor,
            cfg_env, cfg_ppo, seg_len: int, deterministic: bool = False) -> Rollout:
    """ctx is minute-resolution [T_minutes, D]; the agent acts every env.K minutes."""
    device = ctx.device
    T = env.n_steps
    K = env.K
    N, A = env.N, env.A
    port = env.reset()
    h = agent.init_state(N, device)
    n_seg = (T + seg_len - 1) // seg_len

    P = port.shape[-1]
    buf = dict(
        port=torch.zeros(T, N, P, device=device),
        actions=torch.zeros(T, N, A, device=device),
        logp=torch.zeros(T, N, device=device),
        values=torch.zeros(T, N, device=device),
        rewards=torch.zeros(T, N, device=device),
        alive=torch.zeros(T, N, device=device),
        weights=torch.zeros(T, N, A, device=device),
    )
    h0 = torch.zeros(n_seg, N, agent.hidden, device=device)
    equity = torch.zeros(T + 1, N, device=device)
    equity[0] = env.equity

    for t in range(T):
        if t % seg_len == 0:
            h0[t // seg_len] = h.detach()
        buf["port"][t] = port
        buf["alive"][t] = env.alive.float()
        mu, v, h = agent.step(ctx[t * K].unsqueeze(0).expand(N, -1), port, h)
        dist = agent.dist(mu)
        a = mu if deterministic else dist.sample()
        lp = dist.log_prob(a).sum(-1)
        w = TradeMaker.to_weights(a, cfg_env.mode, env.max_lev,
                                  risk_budget[t * K].expand(N)
                                  if risk_budget is not None else None)
        port, r, done, _ = env.step(w)
        buf["actions"][t] = a
        buf["logp"][t] = lp
        buf["values"][t] = v
        buf["rewards"][t] = r
        buf["weights"][t] = w
        equity[t + 1] = env.equity

    _, last_v, _ = agent.step(ctx[(T - 1) * K].unsqueeze(0).expand(N, -1), port, h)
    return Rollout(ctx=ctx, h0=h0, seg_len=seg_len, equity=equity,
                   last_value=last_v * 0.0, **buf)


def compute_gae(roll: Rollout, gamma: float, lam: float):
    T, N = roll.rewards.shape
    adv = torch.zeros_like(roll.rewards)
    last = torch.zeros(N, device=roll.rewards.device)
    next_v = roll.last_value
    for t in reversed(range(T)):
        nonterm = roll.alive[t]
        delta = roll.rewards[t] + gamma * next_v * nonterm - roll.values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
        next_v = roll.values[t]
    ret = adv + roll.values
    return adv, ret


def _segment(x: torch.Tensor, seg_len: int, T: int):
    """[T, N, ...] -> [S*N, L, ...] with right padding on the final segment."""
    N = x.shape[1]
    rest = x.shape[2:]
    S = (T + seg_len - 1) // seg_len
    pad = S * seg_len - T
    if pad:
        x = torch.cat([x, x[-1:].expand(pad, *x.shape[1:])], dim=0)
    x = x.view(S, seg_len, N, *rest).permute(0, 2, 1, *range(3, 3 + len(rest)))
    return x.reshape(S * N, seg_len, *rest), S, pad


def ppo_update(agent: TradeMaker, roll: Rollout, adv: torch.Tensor, ret: torch.Tensor,
               cfg_ppo, cfg_env, entropy_coef: float, optimizer, scaler=None,
               ctx_grad: torch.Tensor | None = None, decision_every: int = 1,
               extra_backward=None) -> Dict[str, float]:
    """One PPO phase. If `ctx_grad` is given it replaces the detached rollout context,
    which lets the policy-gradient flow back into the analyst/predictor."""
    T, N = roll.rewards.shape
    L = roll.seg_len
    device = roll.rewards.device
    ctx = ctx_grad if ctx_grad is not None else roll.ctx
    ctx = ctx[::decision_every]

    port_s, S, pad = _segment(roll.port, L, T)
    act_s, _, _ = _segment(roll.actions, L, T)
    lp_s, _, _ = _segment(roll.logp.unsqueeze(-1), L, T)
    adv_s, _, _ = _segment(adv.unsqueeze(-1), L, T)
    ret_s, _, _ = _segment(ret.unsqueeze(-1), L, T)
    alive_s, _, _ = _segment(roll.alive.unsqueeze(-1), L, T)
    h0 = roll.h0.reshape(S * N, -1)

    ctx_pad = ctx[:T]
    if pad:
        ctx_pad = torch.cat([ctx_pad, ctx_pad[-1:].expand(pad, -1)], dim=0)
    ctx_s = ctx_pad.view(S, L, -1).unsqueeze(1).expand(S, N, L, -1).reshape(S * N, L, -1)

    adv_m = adv_s.mean()
    adv_std = adv_s.std().clamp_min(1e-6)
    adv_n = (adv_s - adv_m) / adv_std

    stats: Dict[str, float] = {}
    for ep in range(cfg_ppo.update_epochs):
        h = h0.clone()
        new_lp, new_v, ent = [], [], []
        for i in range(L):
            mu, v, h = agent.step(ctx_s[:, i], port_s[:, i], h)
            d = agent.dist(mu)
            new_lp.append(d.log_prob(act_s[:, i]).sum(-1))
            ent.append(d.entropy().sum(-1))
            new_v.append(v)
        new_lp = torch.stack(new_lp, 1).unsqueeze(-1)
        new_v = torch.stack(new_v, 1).unsqueeze(-1)
        ent = torch.stack(ent, 1).unsqueeze(-1)

        m = alive_s
        msum = m.sum().clamp_min(1.0)
        ratio = (new_lp - lp_s).clamp(-20, 20).exp()
        s1 = ratio * adv_n
        s2 = ratio.clamp(1 - cfg_ppo.clip_eps, 1 + cfg_ppo.clip_eps) * adv_n
        pg = -(torch.min(s1, s2) * m).sum() / msum
        vl = (((new_v - ret_s) ** 2) * m).sum() / msum
        el = (ent * m).sum() / msum
        loss = pg + cfg_ppo.value_coef * vl - entropy_coef * el

        if extra_backward is not None and ep == 0:
            loss = loss + extra_backward()

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g["params"]],
            cfg_ppo.max_grad_norm)
        if scaler is not None:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()

        with torch.no_grad():
            kl = (((lp_s - new_lp).exp() - 1) - (lp_s - new_lp)).mul(m).sum() / msum
            clipfrac = (((ratio - 1).abs() > cfg_ppo.clip_eps).float() * m).sum() / msum
        stats = {"pg_loss": pg.item(), "v_loss": vl.item(), "entropy": el.item(),
                 "kl": kl.item(), "clipfrac": clipfrac.item(), "grad_norm": float(gn),
                 "ppo_epochs_run": ep + 1}
        if kl.item() > cfg_ppo.target_kl * 1.5:
            break
        ctx_grad = None          # only the first inner epoch carries e2e gradients
        if ctx_grad is None and ctx is not roll.ctx:
            ctx_s = ctx_s.detach()
    return stats
