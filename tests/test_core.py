"""Unit tests: `python -m pytest tests -q` (works offline, no GPU needed)."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from trader.config import Config                                    # noqa: E402
from trader.envs.market_env import MarketEnv                        # noqa: E402
from trader.features import N_FEATURES, RAW_COLS, day_features      # noqa: E402
from trader.models.analyst import MarketAnalyst                     # noqa: E402
from trader.models.predictor import PricePredictor                  # noqa: E402
from trader.models.trade_maker import (TradeMaker, build_context,   # noqa: E402
                                       context_dim)
from trader.resources import assign_roles, pick_preset              # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    c = Config()
    c.apply_size("tiny")
    return c


def _fake_day(A=3, T=300, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.001, (A, T)).cumsum(1)
    close = (100 * np.exp(steps)).astype(np.float32)
    raw = np.zeros((A, T, len(RAW_COLS)), dtype=np.float32)
    for n, i in ((c, i) for i, c in enumerate(RAW_COLS)):
        pass
    ci = {c: i for i, c in enumerate(RAW_COLS)}
    for k in ("open", "high", "low", "close"):
        raw[:, :, ci[k]] = close
    raw[:, :, ci["high"]] *= 1.0005
    raw[:, :, ci["low"]] *= 0.9995
    raw[:, :, ci["volume"]] = rng.uniform(1, 10, (A, T))
    raw[:, :, ci["quote_volume"]] = raw[:, :, ci["volume"]] * close
    raw[:, :, ci["trades"]] = rng.integers(1, 100, (A, T))
    raw[:, :, ci["taker_base"]] = raw[:, :, ci["volume"]] * 0.5
    raw[:, :, ci["valid"]] = 1
    return raw, close


# ---------------------------------------------------------------- features
def test_features_shape_and_finite():
    raw, _ = _fake_day()
    mod = np.arange(raw.shape[1]) % 1440
    f = day_features(raw, mod, 200)
    assert f.shape == (3, 200, N_FEATURES)
    assert np.isfinite(f).all()
    assert np.abs(f).max() <= 10.0


def test_features_are_causal():
    """Perturbing the future must not change a past feature row."""
    raw, _ = _fake_day()
    mod = np.arange(raw.shape[1]) % 1440
    a = day_features(raw, mod, 300)
    raw2 = raw.copy()
    raw2[:, 250:, :4] *= 1.05                      # change the future only
    b = day_features(raw2, mod, 300)
    assert np.allclose(a[:, :249], b[:, :249], atol=1e-5)


# -------------------------------------------------------------------- env
def test_env_conserves_cash_when_flat(cfg):
    _, close = _fake_day()
    c = torch.from_numpy(close)
    env = MarketEnv(c, torch.ones_like(c), cfg.env, 5, torch.device("cpu"),
                    step_minutes=5)
    for _ in range(50):
        env.step(torch.zeros(5, 3))
    assert torch.allclose(env.equity, torch.full((5,), cfg.env.start_cash), atol=1e-4)


def test_env_fees_make_churn_lose_money(cfg):
    _, close = _fake_day()
    c = torch.from_numpy(close)
    env = MarketEnv(c, torch.ones_like(c) * 1e6, cfg.env, 4, torch.device("cpu"),
                    step_minutes=1)
    flip = torch.zeros(4, 3); flip[:, 0] = 1.0
    for i in range(40):
        env.step(flip if i % 2 == 0 else -flip)
    assert (env.equity < cfg.env.start_cash).all()
    assert env.fees_total.mean() > 0


def test_spot_mode_is_long_only_and_unlevered(cfg):
    a = torch.randn(64, 3) * 5
    w = TradeMaker.to_weights(a, "spot", 1.0)
    assert (w >= 0).all()
    assert w.abs().sum(-1).max() <= 1.0 + 1e-5


def test_futures_respects_gross_cap(cfg):
    a = torch.randn(64, 3) * 5
    w = TradeMaker.to_weights(a, "futures", 10.0)
    assert w.abs().sum(-1).max() <= 10.0 + 1e-4
    assert (w < 0).any()                            # shorts are reachable


def test_mark_to_market_matches_buy_and_hold(cfg):
    """With one full long position and no further trades, equity must track price."""
    import dataclasses
    A, T = 3, 61
    close = np.ones((A, T), dtype=np.float32) * 100.0
    close[0] = np.linspace(100, 110, T)
    c = torch.from_numpy(close)
    ec = dataclasses.replace(cfg.env, mode="spot", spot_fee=0.0, slippage_bps=0.0,
                             turnover_penalty=0.0)
    env = MarketEnv(c, torch.ones_like(c) * 1e9, ec, 1, torch.device("cpu"),
                    step_minutes=10)
    w = torch.zeros(1, 3); w[0, 0] = 1.0
    for _ in range(6):
        env.step(w)
    # 100 -> 110 on the only held asset == +10% on $20
    assert abs(env.equity.item() - ec.start_cash * 1.1) < 2e-2


def test_decision_interval_preserves_minute_marking(cfg):
    """Liquidation inside an interval must be caught even though we act every K min."""
    import dataclasses
    A, T = 3, 31
    close = np.ones((A, T), dtype=np.float32) * 100.0
    close[0, 5] = 40.0                       # one-minute crash, recovers after
    c = torch.from_numpy(close)
    ec = dataclasses.replace(cfg.env, mode="futures", max_leverage=10.0)
    env = MarketEnv(c, torch.ones_like(c) * 1e9, ec, 2, torch.device("cpu"),
                    step_minutes=30)
    full = torch.zeros(2, 3); full[:, 0] = 10.0
    env.step(full)
    assert (~env.alive).all(), "intra-interval liquidation was missed"


def test_liquidation_zeroes_equity(cfg):
    import dataclasses
    A, T = 3, 20
    close = np.ones((A, T), dtype=np.float32) * 100
    close[0, 5:] = 50                               # -50% gap on asset 0
    c = torch.from_numpy(close)
    ec = dataclasses.replace(cfg.env, mode="futures", max_leverage=10.0)
    env = MarketEnv(c, torch.ones_like(c) * 1e9, ec, 2, torch.device("cpu"),
                    step_minutes=1)
    full = torch.zeros(2, 3); full[:, 0] = 10.0
    for _ in range(8):
        env.step(full)
    assert (env.equity == 0).all() and (~env.alive).all()


# ----------------------------------------------------------------- models
def test_predictor_is_causal(cfg):
    torch.manual_seed(0)
    m = PricePredictor(N_FEATURES, cfg.model).eval()
    x = torch.randn(3, 128, N_FEATURES)
    with torch.no_grad():
        a = m(x)["mu"]
        x2 = x.clone(); x2[:, 100:] = torch.randn_like(x2[:, 100:])
        b = m(x2)["mu"]
    assert torch.allclose(a[:, :99], b[:, :99], atol=1e-5)


def test_agents_chain_and_shapes(cfg):
    torch.manual_seed(0)
    A, T = 3, 64
    p = PricePredictor(N_FEATURES, cfg.model)
    an = MarketAnalyst(N_FEATURES, cfg.model.d_model, cfg.model)
    x = torch.randn(A, T, N_FEATURES)
    po = p(x)
    ao = an(x, po)
    assert po["mu"].shape == (A, T, len(cfg.model.horizons))
    assert ao["regime"].shape == (A, T, cfg.model.n_regimes)
    assert ao["risk_budget"].shape == (T,)
    ctx = build_context(x, po, ao)
    assert ctx.shape == (T, context_dim(A, N_FEATURES, cfg.model))
    tm = TradeMaker(A, ctx.shape[-1], cfg.model, cfg.env)
    h = tm.init_state(4, torch.device("cpu"))
    mu, v, h2 = tm.step(ctx[0].expand(4, -1), torch.zeros(4, tm.port_dim), h)
    assert mu.shape == (4, A) and v.shape == (4,) and h2.shape == h.shape


def test_gradients_flow_through_the_whole_chain(cfg):
    torch.manual_seed(0)
    p = PricePredictor(N_FEATURES, cfg.model)
    an = MarketAnalyst(N_FEATURES, cfg.model.d_model, cfg.model)
    x = torch.randn(3, 32, N_FEATURES)
    ctx = build_context(x, p(x), an(x, p(x)))
    ctx.sum().backward()
    assert any(q.grad is not None and q.grad.abs().sum() > 0 for q in p.parameters())
    assert any(q.grad is not None and q.grad.abs().sum() > 0 for q in an.parameters())


def test_losses_are_finite(cfg):
    torch.manual_seed(0)
    _, close = _fake_day(T=200)
    c = torch.from_numpy(close)
    vol = torch.diff(torch.log(c), dim=1).std(1, keepdim=True).clamp_min(1e-6)
    x = torch.randn(3, 200, N_FEATURES)
    p = PricePredictor(N_FEATURES, cfg.model)
    an = MarketAnalyst(N_FEATURES, cfg.model.d_model, cfg.model)
    po = p(x)
    lp, mp = p.loss(po, c, vol)
    la, ma = an.loss(an(x, po), c, vol)
    assert torch.isfinite(lp) and torch.isfinite(la)
    assert all(torch.isfinite(torch.as_tensor(v)) for v in {**mp, **ma}.values())


# -------------------------------------------------------------------- ppo
def test_gae_matches_reference(cfg):
    from trader.train.ppo import Rollout, compute_gae
    T, N = 6, 2
    r = Rollout(ctx=torch.zeros(T, 1), port=torch.zeros(T, N, 1),
                actions=torch.zeros(T, N, 1), logp=torch.zeros(T, N),
                values=torch.arange(T * N, dtype=torch.float32).reshape(T, N),
                rewards=torch.ones(T, N), alive=torch.ones(T, N),
                h0=torch.zeros(1, N, 1), seg_len=T, equity=torch.zeros(T + 1, N),
                weights=torch.zeros(T, N, 1), last_value=torch.zeros(N))
    adv, ret = compute_gae(r, 0.99, 0.95)
    exp = torch.zeros(T, N); last = torch.zeros(N); nv = torch.zeros(N)
    for t in reversed(range(T)):
        d = r.rewards[t] + 0.99 * nv - r.values[t]
        last = d + 0.99 * 0.95 * last
        exp[t] = last; nv = r.values[t]
    assert torch.allclose(adv, exp, atol=1e-5)
    assert torch.allclose(ret, exp + r.values, atol=1e-5)


def test_segment_roundtrip():
    from trader.train.ppo import _segment
    T, N, D = 10, 3, 2
    x = torch.arange(T * N * D, dtype=torch.float32).reshape(T, N, D)
    s, S, pad = _segment(x, 4, T)
    assert S == 3 and pad == 2 and s.shape == (S * N, 4, D)
    assert torch.allclose(s[0, 0], x[0, 0])
    assert torch.allclose(s[1, 0], x[0, 1])          # rank-major over envs


# -------------------------------------------------------------- resources
def test_role_assignment_covers_every_agent():
    for w in range(1, 9):
        roles = assign_roles(w)
        assert len(roles) == w
        flat = [a for r in roles for a in r]
        for agent in ("predictor", "analyst", "trader"):
            assert agent in flat, (w, roles)


def test_preset_monotonic():
    assert pick_preset(2) == "tiny"
    assert pick_preset(8) == "small"
    assert pick_preset(15) == "base"
    assert pick_preset(40) == "large"


# ------------------------------------------------------------------ offline
def test_no_network_imports():
    """The training path must not import networking libraries."""
    import trader.data, trader.train.loop, trader.viz        # noqa: F401
    banned = {"requests", "urllib3", "httpx", "aiohttp", "boto3", "snowflake"}
    assert not (banned & set(sys.modules))


def test_dataset_files_present():
    import json
    man = os.path.join(ROOT, "data", "manifest.json")
    assert os.path.exists(man), "dataset manifest missing"
    m = json.load(open(man))
    for f in m["files"]:
        assert os.path.exists(os.path.join(ROOT, f["path"])), f["path"]
