"""The training loop that runs inside every rank.

Each rank holds a full copy of all three agents but only *owns* (optimises) the ones
assigned to it by resources.assign_roles(). Ranks that own the same agent all-reduce
their gradients; every `sync_every` epochs the owner broadcasts fresh weights to
everyone, so each agent always consumes near-current versions of the other two.

One epoch = one randomly sampled trading day, drawn independently per rank, so with
4 ranks the system sees 4 different days per epoch step.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from .. import viz
from ..advisor import build_advice
from ..data import MarketData
from ..envs.market_env import MarketEnv
from ..features import N_FEATURES
from ..models.analyst import MarketAnalyst
from ..models.predictor import PricePredictor
from ..models.trade_maker import TradeMaker, build_context, context_dim
from .ppo import collect, compute_gae, ppo_update

AGENTS = ["predictor", "analyst", "trader"]


class Trainer:
    def __init__(self, cfg, plan, rank: int, root: str = "."):
        self.cfg, self.plan, self.rank, self.root = cfg, plan, rank, root
        self.world = plan.world_size
        self.is_main = rank == 0
        self.device = torch.device(plan.devices[rank])
        self.owned = set(plan.roles[rank])
        self.dev_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.amp = bool(cfg.train.amp and self.dev_type == "cuda")

        torch.manual_seed(cfg.train.seed + rank)
        np.random.seed(cfg.train.seed + rank)
        self.rng = np.random.default_rng(cfg.train.seed * 1000 + rank)

        self.run_dir = os.path.join(cfg.train.out_dir, cfg.train.run_name)
        if self.is_main:
            for sub in ("plots", "ckpt", "advice"):
                os.makedirs(os.path.join(self.run_dir, sub), exist_ok=True)
            cfg.save(os.path.join(self.run_dir, "config.json"))
            with open(os.path.join(self.run_dir, "resources.json"), "w") as f:
                json.dump(plan.to_dict(), f, indent=2)

        self.data = MarketData(cfg.data, root=root,
                               cache_dir=cfg.data.cache_dir, verbose=self.is_main)
        self.A = len(cfg.data.symbols)
        self._build_models()
        self.hist: Dict[str, List[float]] = {}
        self.start_epoch = 0
        if cfg.train.resume:
            self._load_ckpt(cfg.train.resume)
        self.t_start = time.time()

    # ------------------------------------------------------------------ setup
    def _build_models(self):
        cfg = self.cfg
        m = cfg.model
        self.predictor = PricePredictor(N_FEATURES, m).to(self.device)
        self.analyst = MarketAnalyst(N_FEATURES, m.d_model, m).to(self.device)
        self.ctx_dim = context_dim(self.A, N_FEATURES, m)
        self.trader = TradeMaker(self.A, self.ctx_dim, m, cfg.env).to(self.device)
        self.models = {"predictor": self.predictor, "analyst": self.analyst,
                       "trader": self.trader}

        self.opt, self.scaler = {}, {}
        lrs = {"predictor": cfg.train.lr_predictor, "analyst": cfg.train.lr_analyst,
               "trader": cfg.ppo.lr}
        for a in self.owned:
            self.opt[a] = torch.optim.AdamW(self.models[a].parameters(), lr=lrs[a],
                                            weight_decay=cfg.train.weight_decay)
            self.scaler[a] = torch.amp.GradScaler(self.dev_type, enabled=self.amp)

        self.groups, self.src = {}, {}
        for a in AGENTS:
            owners = self.plan.owners_of(a)
            self.src[a] = owners[0] if owners else 0
            if self.world > 1 and dist.is_initialized():
                self.groups[a] = dist.new_group(ranks=owners) if len(owners) > 1 else None

        if self.is_main:
            n = {k: sum(p.numel() for p in v.parameters()) for k, v in self.models.items()}
            print(f"[model] params: " + "  ".join(f"{k}={v/1e6:.2f}M" for k, v in n.items())
                  + f"   total={sum(n.values())/1e6:.2f}M   ctx_dim={self.ctx_dim}",
                  flush=True)

    # ------------------------------------------------------------- dist utils
    def _sync_weights(self):
        if self.world < 2 or not dist.is_initialized():
            return
        for a in AGENTS:
            for p in self.models[a].parameters():
                dist.broadcast(p.data, src=self.src[a])

    def _reduce_grads(self, agent: str):
        g = self.groups.get(agent)
        if g is None:
            return
        n = float(len(self.plan.owners_of(agent)))
        for p in self.models[agent].parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=g)
                p.grad /= n

    def _reduce_metrics(self, m: Dict[str, float]) -> Dict[str, float]:
        if self.world < 2 or not dist.is_initialized():
            return m
        keys = sorted(m.keys())
        t = torch.tensor([float(m[k]) for k in keys], device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= self.world
        return {k: float(v) for k, v in zip(keys, t.tolist())}

    # ------------------------------------------------------------------ epoch
    def _forward_context(self, feats: torch.Tensor, need_grad: bool):
        amp_ctx = torch.amp.autocast(self.dev_type, dtype=torch.float16, enabled=self.amp)
        grad_ctx = torch.enable_grad() if need_grad else torch.no_grad()
        with grad_ctx, amp_ctx:
            pred = self.predictor(feats)
            an = self.analyst(feats, pred if "analyst" in self.owned
                              else {k: v.detach() for k, v in pred.items()})
            ctx = build_context(feats, pred, an)
        return pred, an, ctx.float()

    def train_epoch(self, epoch: int) -> Dict:
        cfg = self.cfg
        t0 = time.time()
        progress = min(1.0, epoch / max(1, cfg.train.epochs * 0.6))
        di = self.data.sample_day(self.rng, progress, cfg.train.curriculum)
        day = self.data.get_day(di)

        feats = torch.from_numpy(day.feats).to(self.device)
        close = torch.from_numpy(day.close).to(self.device, torch.float32)
        volume = torch.from_numpy(day.volume).to(self.device, torch.float32)
        logc = torch.log(close.clamp_min(1e-12))
        vol_ref = torch.diff(logc, dim=1).std(dim=1, keepdim=True).clamp_min(1e-6)

        need_grad = bool({"predictor", "analyst"} & self.owned)
        pred, an, ctx = self._forward_context(feats, need_grad)

        metrics: Dict[str, float] = {}

        # ---------------------------------------------------- supervised heads
        sup_loss = None
        if "predictor" in self.owned:
            with torch.amp.autocast(self.dev_type, dtype=torch.float16, enabled=self.amp):
                lp, mp = self.predictor.loss(pred, close, vol_ref)
            sup_loss = lp if sup_loss is None else sup_loss + lp
            metrics.update({k: float(v) for k, v in mp.items()})
        if "analyst" in self.owned:
            with torch.amp.autocast(self.dev_type, dtype=torch.float16, enabled=self.amp):
                la, ma = self.analyst.loss(an, close, vol_ref)
            sup_loss = la if sup_loss is None else sup_loss + la
            metrics.update({k: float(v) for k, v in ma.items()})

        # ------------------------------------------------------------ rollout
        rb = an["risk_budget"].detach().float()
        gate = (0.5 + 0.5 * rb) if cfg.env.use_risk_gate else None
        env_cfg = self._env_cfg(progress)
        env = MarketEnv(close, volume, env_cfg, cfg.env.n_envs, self.device,
                        step_minutes=cfg.env.decision_every)
        seg = min(cfg.ppo.minibatch_steps, env.n_steps)
        ctx_det = ctx.detach()
        roll = collect(self.trader, env, ctx_det, gate, env_cfg, cfg.ppo, seg)
        adv, ret = compute_gae(roll, cfg.ppo.gamma, cfg.ppo.gae_lambda)
        summary = env.summary()
        metrics.update(summary)

        # ------------------------------------------------------- agent updates
        ent_coef = cfg.ppo.entropy_coef + (cfg.ppo.entropy_final - cfg.ppo.entropy_coef) * progress
        if "trader" in self.owned:
            st = ppo_update(self.trader, roll, adv, ret, cfg.ppo, env_cfg, ent_coef,
                            self.opt["trader"], None,
                            decision_every=cfg.env.decision_every)
            self._reduce_grads("trader")
            metrics.update(st)

        if sup_loss is not None:
            total = sup_loss
            if cfg.train.e2e_grad and "analyst" in self.owned:
                total = total + self._policy_gradient_term(roll, adv, ctx)
            for a in ("predictor", "analyst"):
                if a in self.owned:
                    self.opt[a].zero_grad(set_to_none=True)
            self.scaler["analyst" if "analyst" in self.owned else "predictor"] \
                .scale(total).backward()
            for a in ("predictor", "analyst"):
                if a in self.owned:
                    sc = self.scaler[a]
                    sc.unscale_(self.opt[a])
                    torch.nn.utils.clip_grad_norm_(self.models[a].parameters(), 1.0)
                    self._reduce_grads(a)
                    sc.step(self.opt[a])
                    sc.update()

        if (epoch + 1) % cfg.train.sync_every == 0:
            self._sync_weights()

        metrics["day_vol"] = day.day_vol
        metrics["leverage"] = env.max_lev
        metrics["epoch_sec"] = time.time() - t0
        return {"metrics": metrics, "day": day, "roll": roll, "env": env,
                "pred": pred, "an": an, "adv": adv, "ret": ret,
                "summary": summary, "close": close, "vol_ref": vol_ref}

    def _env_cfg(self, progress: float):
        """Leverage curriculum: a random policy at 10x liquidates in minutes, so the
        gross cap ramps up only as the policy becomes competent."""
        import dataclasses
        c = self.cfg.env
        if c.mode == "spot" or c.lev_warmup_frac <= 0:
            return c
        f = min(1.0, progress / max(1e-6, c.lev_warmup_frac))
        lev = c.lev_start + (c.max_leverage - c.lev_start) * f
        return dataclasses.replace(c, max_leverage=float(lev))

    def _policy_gradient_term(self, roll, adv, ctx_grad):
        """REINFORCE-style surrogate so the trader's advantage shapes the analyst."""
        from .ppo import _segment
        T, N = roll.rewards.shape
        L = roll.seg_len
        port_s, S, pad = _segment(roll.port, L, T)
        act_s, _, _ = _segment(roll.actions, L, T)
        adv_s, _, _ = _segment(adv.unsqueeze(-1), L, T)
        alive_s, _, _ = _segment(roll.alive.unsqueeze(-1), L, T)
        c = ctx_grad[::self.cfg.env.decision_every][:T]
        if pad:
            c = torch.cat([c, c[-1:].expand(pad, -1)], dim=0)
        ctx_s = c.view(S, L, -1).unsqueeze(1).expand(S, N, L, -1).reshape(S * N, L, -1)
        adv_n = (adv_s - adv_s.mean()) / adv_s.std().clamp_min(1e-6)

        h = roll.h0.reshape(S * N, -1)
        for p in self.trader.parameters():
            p.requires_grad_(False)
        lps = []
        stride = max(1, L // 64)                      # subsample steps: this is a hint,
        for i in range(0, L, stride):                 # not the trader's own objective
            mu, _, h = self.trader.step(ctx_s[:, i], port_s[:, i], h)
            lps.append((self.trader.dist(mu).log_prob(act_s[:, i]).sum(-1)
                        * adv_n[:, i, 0] * alive_s[:, i, 0]))
            h = h.detach()
        for p in self.trader.parameters():
            p.requires_grad_(True)
        return -0.05 * torch.stack(lps).mean()

    # ------------------------------------------------------------------ plots
    def make_plots(self, epoch: int, out: Dict, force: bool = False):
        cfg = self.cfg
        if not self.is_main:
            return
        if not force and (epoch % cfg.train.plot_every != 0):
            return
        day, roll, env = out["day"], out["roll"], out["env"]
        eq = roll.equity.detach().cpu().numpy()
        wts = roll.weights.detach().cpu().numpy()
        med_env = int(np.argsort(eq[-1])[len(eq[-1]) // 2])
        pred, an = out["pred"], out["an"]
        with torch.no_grad():
            y, _, _ = PricePredictor.targets(out["close"], cfg.model.horizons, out["vol_ref"])
            _, fvol, _, _ = MarketAnalyst.regime_targets(out["close"], out["vol_ref"])
            vol_real = torch.log((fvol / out["vol_ref"].clamp_min(1e-8)).clamp_min(1e-3))

        d = {
            "epoch": epoch, "date": day.date, "symbols": day.symbols,
            "mode": cfg.env.mode, "start_cash": cfg.env.start_cash,
            "dstep": cfg.env.decision_every,
            "target": cfg.env.target_equity, "equity": eq,
            "rewards": roll.rewards.detach().cpu().numpy(),
            "weights_med": wts[:, med_env, :],
            "actions": roll.actions.detach().cpu().numpy(),
            "values": roll.values.detach().cpu().numpy(),
            "returns": out["ret"].detach().cpu().numpy(),
            "close": day.close, "hit_target": out["summary"]["hit_target"],
            "summary": out["summary"], "m": out["metrics"],
            "horizons": cfg.model.horizons,
            "pred_mu": pred["mu"].detach().float().cpu().numpy(),
            "pred_y": y.detach().float().cpu().numpy(),
            "regime": an["regime"].detach().float().softmax(-1).mean(0).cpu().numpy(),
            "vol_pred": an["vol"].detach().float().mean(0).cpu().numpy(),
            "vol_asset": an["vol"].detach().float().cpu().numpy(),
            "vol_real": vol_real.mean(0).cpu().numpy(),
            "exposure": an["exposure"].detach().float().cpu().numpy(),
            "risk_budget": an["risk_budget"].detach().float().cpu().numpy(),
            "confidence": an["confidence"].detach().float().cpu().numpy(),
        }
        bh = day.close[:, -1] / day.close[:, 0] - 1
        advice = build_advice(day, {"regime": d["regime"], "vol": d["vol_asset"],
                                    "exposure": d["exposure"],
                                    "risk_budget": d["risk_budget"],
                                    "confidence": d["confidence"]},
                              out["summary"], cfg.env, bh)
        d["verdict"] = advice["verdict"]

        pdir = os.path.join(self.run_dir, "plots", f"epoch_{epoch:06d}")
        viz.plot_equity(d, os.path.join(pdir, "01_equity.png"))
        viz.plot_market(d, os.path.join(pdir, "02_market.png"))
        viz.plot_positions(d, os.path.join(pdir, "03_positions.png"))
        viz.plot_predictor(d, os.path.join(pdir, "04_predictor.png"))
        viz.plot_analyst(d, os.path.join(pdir, "05_analyst.png"))
        viz.plot_ppo(self.hist, d, os.path.join(pdir, "06_ppo.png"))
        viz.plot_advisor(advice["text"], d, os.path.join(pdir, "07_advisor.png"))
        viz.plot_distribution(d, os.path.join(pdir, "08_distribution.png"))
        with open(os.path.join(self.run_dir, "advice", f"epoch_{epoch:06d}.json"), "w") as f:
            json.dump({"epoch": epoch, "date": day.date, **advice,
                       "summary": out["summary"]}, f, indent=2)

    def make_dashboard(self, epoch: int):
        if not self.is_main:
            return
        p = os.path.join(self.run_dir, "plots")
        viz.plot_dashboard(self.hist, self.cfg, os.path.join(p, "dashboard.png"),
                           f"- {self.cfg.env.mode} mode")
        viz.plot_agents(self.hist, os.path.join(p, "agents.png"))

    # ------------------------------------------------------------------- eval
    @torch.no_grad()
    def evaluate(self, epoch: int) -> Optional[Dict]:
        if not self.is_main:
            return None
        days = self.data.eval_day_list(self.cfg.train.eval_days, seed=epoch)
        if not days:
            return None
        finals, curves, dates, hits = [], [], [], []
        for di in days:
            day = self.data.get_day(di, split="eval")
            feats = torch.from_numpy(day.feats).to(self.device)
            close = torch.from_numpy(day.close).to(self.device, torch.float32)
            volume = torch.from_numpy(day.volume).to(self.device, torch.float32)
            _, an, ctx = self._forward_context(feats, need_grad=False)
            rb = an["risk_budget"].float()
            gate = (0.5 + 0.5 * rb) if self.cfg.env.use_risk_gate else None
            env = MarketEnv(close, volume, self.cfg.env,
                            max(4, self.cfg.env.n_envs // 2), self.device,
                            step_minutes=self.cfg.env.decision_every)
            seg = min(self.cfg.ppo.minibatch_steps, env.n_steps)
            roll = collect(self.trader, env, ctx, gate, self.cfg.env, self.cfg.ppo, seg,
                           deterministic=True)
            s = env.summary()
            finals.append(s["equity_mean"]); hits.append(s["hit_target"])
            curves.append(roll.equity.mean(1).cpu().numpy().tolist())
            dates.append(day.date)
        ev = {"epoch": epoch, "finals": finals, "curves": curves, "dates": dates,
              "start_cash": self.cfg.env.start_cash, "target": self.cfg.env.target_equity,
              "hit_rate": float(np.mean(hits)), "mean_final": float(np.mean(finals))}
        viz.plot_eval(ev, os.path.join(self.run_dir, "plots", "eval.png"))
        with open(os.path.join(self.run_dir, "eval.json"), "w") as f:
            json.dump({k: v for k, v in ev.items() if k != "curves"}, f, indent=2)
        print(f"[eval ] epoch {epoch}: mean final ${ev['mean_final']:.2f} over "
              f"{len(finals)} held-out days, hit-rate {100*ev['hit_rate']:.1f}%", flush=True)
        return ev

    # ------------------------------------------------------------ persistence
    def save_ckpt(self, epoch: int, tag: str = "last"):
        if not self.is_main:
            return
        path = os.path.join(self.run_dir, "ckpt", f"{tag}.pt")
        torch.save({"epoch": epoch, "cfg": self.cfg.to_dict(), "hist": self.hist,
                    "predictor": self.predictor.state_dict(),
                    "analyst": self.analyst.state_dict(),
                    "trader": self.trader.state_dict()}, path)

    def _load_ckpt(self, path: str):
        if not os.path.exists(path):
            print(f"[warn] resume checkpoint {path} not found, starting fresh", flush=True)
            return
        ck = torch.load(path, map_location=self.device, weights_only=False)
        for a in AGENTS:
            self.models[a].load_state_dict(ck[a])
        self.hist = ck.get("hist", {})
        self.start_epoch = int(ck.get("epoch", 0)) + 1
        if self.is_main:
            print(f"[ckpt ] resumed from {path} at epoch {self.start_epoch}", flush=True)

    def log(self, epoch: int, m: Dict[str, float]):
        for k, v in m.items():
            self.hist.setdefault(k, []).append(float(v))
        if not self.is_main:
            return
        with open(os.path.join(self.run_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps({"epoch": epoch, **{k: round(float(v), 6)
                                                   for k, v in m.items()}}) + "\n")

    # ------------------------------------------------------------------- main
    def run(self):
        cfg = self.cfg
        last = time.time()
        for epoch in range(self.start_epoch, cfg.train.epochs):
            out = self.train_epoch(epoch)
            m = self._reduce_metrics(out["metrics"])
            self.log(epoch, m)
            self.make_plots(epoch, out)
            if self.is_main and (epoch % cfg.train.dashboard_every == 0 or
                                 epoch == cfg.train.epochs - 1):
                self.make_dashboard(epoch)
            if cfg.train.eval_every and epoch > 0 and epoch % cfg.train.eval_every == 0:
                self.evaluate(epoch)
            if epoch % cfg.train.ckpt_every == 0 or epoch == cfg.train.epochs - 1:
                self.save_ckpt(epoch)
            if self.is_main and (time.time() - last > 5 or epoch < 5):
                last = time.time()
                print(f"[ep {epoch:>5}] {out['day'].date}  "
                      f"eq ${m['equity_mean']:.2f} (best ${m['equity_best']:.2f}, "
                      f"hit {100*m['hit_target']:4.0f}%)  "
                      f"ret {m['return_pct']:+6.2f}%  "
                      f"pred_ic {m.get('pred_ic', 0):+.3f}  "
                      f"an_acc {100*m.get('an_acc', 0):4.1f}%  "
                      f"kl {m.get('kl', 0):.4f}  {m['epoch_sec']:.2f}s", flush=True)
            if cfg.train.max_hours and (time.time() - self.t_start) / 3600 > cfg.train.max_hours:
                if self.is_main:
                    print(f"[stop ] wall-clock limit {cfg.train.max_hours}h reached", flush=True)
                break
        self.make_dashboard(cfg.train.epochs - 1)
        self.evaluate(cfg.train.epochs - 1)
        self.save_ckpt(cfg.train.epochs - 1, tag="final")
        if self.is_main:
            print(f"[done ] run dir: {self.run_dir}", flush=True)
