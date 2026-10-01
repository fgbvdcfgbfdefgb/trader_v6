# trader_v6

Offline, multi-agent reinforcement learning for intraday crypto trading on **BTC / ETH / LTC**
minute bars. Three agents train **in parallel** and consume each other's output:

| agent | what it is | what it learns | who uses it |
|---|---|---|---|
| **price predictor** | causal dilated TCN | forward log-return mean + variance + direction at 1 / 5 / 15 / 60 min | analyst, trade maker |
| **market analyst** | GRU + cross-asset attention | regime posterior, volatility forecast, suggested exposure, risk budget, confidence | trade maker (and it writes the daily advisory) |
| **trade maker** | recurrent actor-critic, PPO | target portfolio weights for the three assets | the P&L |

**One epoch = one calendar day picked at random.** The agent is handed **$20** at 00:00 UTC
and trades 1,440 one-minute bars; the scoring target is **$30+** by 23:59.

Everything needed to train is in this repository. No network access is required at any
point after `pip install -r requirements.txt` — which is what makes it runnable in a
Snowflake workspace that can only clone the repo and install pip packages.

---

## Quick start

```bash
pip install -r requirements.txt
python scripts/preflight.py              # verify machine + data + one full epoch
python scripts/train.py --epochs 2000    # train (auto-detects GPUs)
```

On Snowflake:

```python
!pip install -r requirements.txt
%run scripts/snowflake_run.py --epochs 500 --mode futures
```

`snowflake_run.py` writes to a writable scratch dir (`/mnt/stage/...`, else `/tmp/...`),
keeps the dataset cache in `/dev/shm`, and zips all artifacts at the end so you can
`PUT` them to a stage. Nothing is ever pushed back to git.

---

## The data

`data/klines/` holds 1-minute spot klines from the Binance public data archive,
re-gridded to a gap-free minute index and stored as zstd Parquet.

| symbol | bars | span |
|---|---|---|
| BTCUSDT | 4,797,840 | 2017-08-17 → 2026-09-30 |
| ETHUSDT | 4,797,840 | 2017-08-17 → 2026-09-30 |
| LTCUSDT | 4,627,948 | 2017-12-13 → 2026-09-30 |

3,212 days where all three symbols overlap → **2,721 training days / 456 held-out days**
(split at `--train-end`, default 2025-06-30). Columns: `ts, open, high, low, close,
volume, quote_volume, trades, taker_base, valid`. `valid=0` marks a minute with no
trades (price carried forward) so the loader can reject thin days.

Minutes that Binance never published are forward-filled and flagged, never interpolated.

On first run the parquet files are compiled into one memory-mapped minute grid
(`~500 MB`) under the cache dir. All ranks share it through the OS page cache, so
4 training processes do **not** cost 4× the RAM. Building it peaks at well under
200 MB RSS, so it works on small containers.

---

## How the three agents train in parallel

`trader/resources.py` inspects the box and assigns agents to ranks:

```
4 GPUs -> rank0 predictor | rank1 analyst | rank2 trader | rank3 trader (2nd owner)
3 GPUs -> one agent per rank
2 GPUs -> rank0 predictor+analyst | rank1 trader
1 GPU  -> rank0 owns all three, time-sliced, model preset shrunk to fit VRAM
0 GPU  -> same as 1 GPU on CPU, 'tiny' preset
```

Every rank holds a **full copy of all three agents** but only applies optimizer steps to
the ones it owns. Ranks sharing an agent all-reduce that agent's gradients (DDP-style);
every `--sync-every` epochs the owner broadcasts fresh weights to everyone. Because each
rank samples its **own** random day, a 4-GPU box consumes 4 different trading days per
epoch step.

Model size is chosen from the smallest GPU's free VRAM:

| free VRAM | preset | d_model | layers | parallel trajectories |
|---|---|---|---|---|
| ≥ 22 GB | large | 384 | 6 | 24 |
| ≥ 11 GB | base | 256 | 4 | 16 |
| ≥ 5.5 GB | small | 160 | 3 | 12 |
| < 5.5 GB | tiny | 96 | 2 | 8 |

If a GPU is already busy (e.g. a co-resident miner) it is shared, never evicted, and
`--reserve-cores` keeps CPU headroom for it.

### How they feed each other

```
features ─► predictor ─► latent + μ/σ/direction ─┬─► analyst ─► regime, vol, exposure,
                                                 │              risk budget, confidence
                                                 └─────────────────────┬────────────────
                                                                       ▼
                                           context[t] ──► trade maker (PPO) ──► weights
```

* the analyst conditions on the predictor's latent and forecasts;
* the trade maker's context concatenates **both** agents' outputs at every minute;
* the analyst's `risk_budget` directly scales the trade maker's position size
  (`--no-risk-gate` to disable);
* the trade maker's advantages flow **back** into the analyst through a REINFORCE-style
  surrogate (`--no-e2e` to disable), so the analyst is rewarded for producing context
  that leads to profitable trades.

---

## The environment

Two modes, same code path:

* `--mode spot` — long-only, gross ≤ 1.0, 10 bps taker fee.
* `--mode futures` *(default)* — perpetual-style, long **and** short, gross up to
  `--leverage` (default 10×), 4 bps taker, 8-hourly funding, and liquidation at
  maintenance margin.

`n_envs` trajectories run over the *same* day with independently sampled actions, which
is what gives PPO a batch while keeping "one epoch = one day".

**Leverage curriculum:** a random policy at 10× liquidates within minutes, so the gross
cap ramps from `lev_start` (1×) to `--leverage` over the first 35% of training.

Reward = scaled log-growth of equity − turnover penalty − incremental-drawdown penalty,
plus a terminal bonus for reaching the $30 target and a penalty for liquidation.

### On the $20 → $30 target

+50% in one day is **not** achievable on spot without leverage — a very good day on BTC is
+2–5%. The target is reachable only in `futures` mode on volatile days, and even there it
is a stretch goal that shapes the reward rather than a promise. Use `--mode spot` for an
honest benchmark; the dashboards always report true equity, the % of trajectories that
actually hit $30, **and** the liquidation rate, so a policy that reaches the target by
gambling is immediately visible.

---

## What gets saved

Per epoch (every `--plot-every`), in `runs/<name>/plots/epoch_XXXXXX/`:

| file | contents |
|---|---|
| `01_equity.png` | every trajectory's equity, median/best/worst, target line, drawdown, cumulative reward |
| `02_market.png` | each asset's price with the agent's position shaded and entry/exit markers |
| `03_positions.png` | weight heat-map over the day, gross/net exposure, raw action distribution |
| `04_predictor.png` | predicted vs realised return per horizon, information coefficient, time series |
| `05_analyst.png` | regime posterior stack, vol forecast vs realised, suggested exposure, risk budget |
| `06_ppo.png` | policy/value loss, KL, clip fraction, entropy, grad norm, value calibration, reward histogram |
| `07_advisor.png` | **the analyst's written verdict on that day's market** |
| `08_distribution.png` | final-equity distribution, per-minute returns, execution stats |

Run level: `dashboard.png` (equity per epoch, % hitting target, returns vs blow-ups, all
agent losses), `agents.png` (every tracked metric), `eval.png` (held-out days).
Also `metrics.jsonl`, `advice/epoch_*.json`, `config.json`, `resources.json`, and
checkpoints in `ckpt/` (`--resume runs/<name>/ckpt/last.pt`).

---

## Useful flags

```bash
python scripts/train.py --help

--mode spot|futures     --leverage 10      --target 30     --start-cash 20
--epochs 2000           --max-hours 6      --resume <ckpt/last.pt>
--gpus N                --size tiny|small|base|large       --reserve-cores 1
--plot-every 10         --eval-every 100   --cache-dir /dev/shm/tv6
--no-risk-gate          --no-e2e           --no-curriculum --no-amp
--dry-run               # print the resource plan and exit
```

## Layout

```
data/klines/*.parquet     the dataset (committed, no LFS, every file < 20 MB)
trader/resources.py       machine probe + rank/role/model-size plan
trader/data.py            offline loader, mmap cache, day sampling, train/eval split
trader/features.py        28 causal features per asset
trader/models/            predictor.py, analyst.py, trade_maker.py
trader/envs/market_env.py vectorised spot/futures simulator
trader/train/             ppo.py, loop.py (the trainer), parallel.py (spawn + NCCL)
trader/viz.py             every chart
trader/advisor.py         the written daily verdict
scripts/preflight.py      offline readiness check
scripts/train.py          main entrypoint
scripts/snowflake_run.py  Snowflake entrypoint
tests/                    unit tests, run with `python -m pytest tests -q`
```

## Honest notes

* Backtest only. Fills are assumed at the next minute's close with a fee + slippage
  model; no order book, no queue position, no partial fills, no exchange outages.
* Funding is a constant proxy (1 bp / 8 h), not historical funding.
* Results on held-out days after 2025-06-30 are the only ones worth quoting; training-day
  equity is contaminated by the curriculum that favours volatile days.
