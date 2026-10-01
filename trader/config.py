"""Central configuration. Everything is plain dataclasses so a run config can be
serialised to JSON next to the checkpoints (offline reproducibility)."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import List, Optional

SYMBOLS = ["BTCUSDT", "ETHUSDT", "LTCUSDT"]

# ---------------------------------------------------------------- model sizes
# Picked automatically from the smallest GPU found (see resources.py).
SIZE_PRESETS = {
    "tiny":  dict(d_model=96,  n_layers=2, n_heads=4, tcn_channels=64,  policy_hidden=128, n_envs=16),
    "small": dict(d_model=160, n_layers=3, n_heads=4, tcn_channels=96,  policy_hidden=192, n_envs=32),
    "base":  dict(d_model=256, n_layers=4, n_heads=8, tcn_channels=128, policy_hidden=256, n_envs=64),
    "large": dict(d_model=384, n_layers=6, n_heads=8, tcn_channels=192, policy_hidden=384, n_envs=96),
}


@dataclass
class DataConfig:
    data_dir: str = "data/klines"
    manifest: str = "data/manifest.json"
    symbols: List[str] = field(default_factory=lambda: list(SYMBOLS))
    bars_per_day: int = 1440          # one epoch == one calendar day of 1m bars
    warmup_bars: int = 360            # extra history before 00:00 so indicators are warm
    min_valid_frac: float = 0.97      # reject days with too many non-traded minutes
    cache_dir: str = ""               # "" -> <out_dir>/cache ; use /dev/shm on small disks
    train_end: str = "2025-06-30"     # days after this are held out for evaluation
    eval_frac_days: float = 0.0       # optional extra random holdout from the train span


@dataclass
class ModelConfig:
    size: str = "auto"                # auto|tiny|small|base|large
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    tcn_channels: int = 128
    policy_hidden: int = 256
    dropout: float = 0.1
    horizons: List[int] = field(default_factory=lambda: [1, 5, 15, 60])
    n_regimes: int = 6
    attn_window: int = 240            # local causal attention span for the analyst


@dataclass
class EnvConfig:
    mode: str = "futures"             # spot | futures
    start_cash: float = 20.0
    target_equity: float = 30.0
    max_leverage: float = 10.0        # futures only; spot is capped at 1.0 gross
    lev_start: float = 1.0            # leverage curriculum: start here...
    lev_warmup_frac: float = 0.35     # ...reach max_leverage after this much of training
    taker_fee: float = 0.0004         # 4 bps futures taker
    spot_fee: float = 0.001           # 10 bps spot taker
    slippage_bps: float = 1.0         # base slippage, scaled by trade size / liquidity
    funding_rate_8h: float = 0.0001   # perpetual funding proxy, charged every 8h
    maint_margin: float = 0.005       # liquidation threshold on gross notional
    min_trade_frac: float = 0.002     # ignore rebalances smaller than this (no-op)
    use_risk_gate: bool = True        # analyst's risk budget scales the trader's size
    decision_every: int = 5           # minutes between rebalances (marking stays 1m)
    n_envs: int = 16                  # parallel sampled trajectories over the same day
    reward_scale: float = 10.0
    turnover_penalty: float = 0.0005
    drawdown_penalty: float = 0.25
    target_bonus: float = 2.0         # terminal bonus for finishing at/above target
    bankrupt_penalty: float = 3.0


@dataclass
class PPOConfig:
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_final: float = 0.001
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_steps: int = 96         # decision steps per truncated-BPTT segment
    lr: float = 3e-4
    target_kl: float = 0.03


@dataclass
class TrainConfig:
    epochs: int = 2000
    seed: int = 7
    lr_predictor: float = 3e-4
    lr_analyst: float = 3e-4
    weight_decay: float = 1e-5
    sync_every: int = 5               # epochs between cross-rank weight broadcasts
    plot_every: int = 10              # full per-epoch plot bundle cadence
    dashboard_every: int = 25
    ckpt_every: int = 50
    eval_every: int = 100
    eval_days: int = 16
    amp: bool = True
    e2e_grad: bool = True             # let the trader's PPO loss flow into the analyst
    curriculum: bool = True           # bias early epochs toward volatile days
    out_dir: str = "runs"
    run_name: str = ""
    resume: str = ""
    max_hours: float = 0.0            # 0 = unlimited; wall-clock stop for rented boxes
    cpu_threads: int = 0              # 0 = auto (leaves cores for a co-resident miner)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def apply_size(self, size: str) -> None:
        self.model.size = size
        for k, v in SIZE_PRESETS[size].items():
            if k == "n_envs":
                self.env.n_envs = v
            else:
                setattr(self.model, k, v)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path) as f:
            d = json.load(f)
        return Config(
            data=DataConfig(**d["data"]), model=ModelConfig(**d["model"]),
            env=EnvConfig(**d["env"]), ppo=PPOConfig(**d["ppo"]),
            train=TrainConfig(**d["train"]),
        )
