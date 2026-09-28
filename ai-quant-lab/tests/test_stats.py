"""Statistics tests - these are the numbers the whole pipeline's honesty rests on."""

import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from quantlab import stats


def test_norm_ppf_and_cdf_are_inverses():
    for p in (0.01, 0.25, 0.5, 0.75, 0.975, 0.999):
        assert abs(stats.norm_cdf(stats.norm_ppf(p)) - p) < 1e-4


def test_sharpe_of_constant_returns_is_zero_not_infinite():
    assert stats.sharpe_ratio(np.full(100, 0.001)) == 0.0


def test_sharpe_scales_with_annualisation():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 2000)
    daily = stats.sharpe_ratio(r, periods=1)
    annual = stats.sharpe_ratio(r, periods=252)
    assert abs(annual - daily * math.sqrt(252)) < 1e-9


def test_max_drawdown_is_negative_and_bounded():
    equity = np.array([1.0, 1.2, 0.6, 0.9])
    assert abs(stats.max_drawdown(equity) - (0.6 / 1.2 - 1)) < 1e-12


def test_trade_blocks_group_consecutive_positions():
    pos = np.array([0, 1, 1, 0, -1, -1, -1, 0])
    rets = np.array([0, 0.01, 0.02, 0, -0.01, 0.03, 0.01, 0])
    blocks = stats.trade_blocks(pos, rets)
    assert blocks.size == 2
    assert abs(blocks[0] - 0.03) < 1e-12
    assert abs(blocks[1] - 0.03) < 1e-12


def test_expected_max_sharpe_grows_with_number_of_trials():
    v = 1.0 / 1000
    assert stats.expected_max_sharpe(10, v) < stats.expected_max_sharpe(1_000, v) < stats.expected_max_sharpe(100_000, v)


def test_deflated_sharpe_falls_as_more_strategies_are_tried():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0006, 0.01, 2000)  # a genuinely decent-looking series
    few = stats.deflated_sharpe(r, n_trials=1)
    many = stats.deflated_sharpe(r, n_trials=100_000)
    assert few["dsr"] > many["dsr"]
    assert many["sr_benchmark"] > few["sr_benchmark"]


def test_deflated_sharpe_rejects_a_lucky_winner_out_of_many():
    """The best of 5,000 random strategies should not pass the 0.95 bar."""
    rng = np.random.default_rng(11)
    paths = rng.normal(0.0, 0.01, (1500, 5000))
    best = paths[:, int(np.argmax(paths.mean(axis=0) / paths.std(axis=0, ddof=1)))]
    assert stats.deflated_sharpe(best, n_trials=5000)["dsr"] < 0.95


def test_monte_carlo_interval_brackets_the_observed_sharpe():
    rng = np.random.default_rng(5)
    r = rng.normal(0.0005, 0.01, 1500)
    mc = stats.monte_carlo(r, n_sims=400)
    assert mc["sharpe_p05"] <= mc["sharpe_median"] <= mc["sharpe_p95"]
    assert 0.0 <= mc["prob_negative"] <= 1.0


def test_pbo_is_high_when_every_strategy_is_noise():
    rng = np.random.default_rng(9)
    m = rng.normal(0.0, 0.01, (1200, 40))
    out = stats.pbo_cscv(m)
    assert 0.0 <= out["pbo"] <= 1.0
    assert out["pbo"] > 0.25  # picking the in-sample winner from noise rarely holds up


def test_pbo_is_low_when_one_strategy_is_genuinely_better():
    rng = np.random.default_rng(13)
    m = rng.normal(0.0, 0.01, (1200, 20))
    m[:, 7] += 0.002  # a real, persistent edge
    assert stats.pbo_cscv(m)["pbo"] < 0.25
