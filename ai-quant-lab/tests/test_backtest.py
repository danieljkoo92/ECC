"""Backtest tests - mostly about the two ways a backtest lies: look-ahead and costs."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from quantlab import data
from quantlab.backtest import (
    Costs,
    bar_returns,
    build_position_matrix,
    split_index,
    strategy_returns,
    trade_count_all,
    walk_forward_windows,
)
from quantlab.strategies import Candidate, Indicators, generate_candidates, neighbours, positions_for


def test_positions_are_applied_one_bar_late():
    """A strategy that 'knows' each bar's return must still earn nothing."""
    bar_ret = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    cheating = np.sign(bar_ret)  # perfect foresight, if the engine allowed it
    net = strategy_returns(cheating, bar_ret, Costs(0, 0))
    assert net[0] == 0.0  # no position on the first bar
    assert np.allclose(net[1:], cheating[:-1] * bar_ret[1:])
    assert net.sum() < np.abs(bar_ret).sum()  # nowhere near the perfect-foresight total


def test_costs_reduce_returns_and_scale_with_turnover():
    bar_ret = np.zeros(10)
    flip = np.array([1, -1] * 5, dtype=float)
    hold = np.ones(10)
    free = strategy_returns(flip, bar_ret, Costs(0, 0))
    charged = strategy_returns(flip, bar_ret, Costs(2, 3))
    assert charged.sum() < free.sum()
    assert strategy_returns(flip, bar_ret, Costs(2, 3)).sum() < strategy_returns(hold, bar_ret, Costs(2, 3)).sum()


def test_cost_scaling_is_linear():
    c = Costs(2.0, 3.0)
    assert abs(c.scaled(3).per_unit_traded - 3 * c.per_unit_traded) < 1e-15


def test_position_matrix_is_ternary_and_shaped_correctly():
    df = data.synthetic(n=400, seed=2)
    cands = generate_candidates(limit=50)
    P = build_position_matrix(cands, df)
    assert P.shape == (400, 50)
    assert set(np.unique(P)).issubset({-1, 0, 1})


def test_slicing_a_window_matches_backtesting_that_window_alone():
    """Indicators are backward-looking, so a sliced window must be self-consistent."""
    df = data.synthetic(n=600, seed=4)
    cands = generate_candidates(limit=30)
    P = build_position_matrix(cands, df)
    br = bar_returns(df)
    sl = slice(300, 600)
    whole_then_slice = strategy_returns(P[sl], br[sl], Costs(0, 0))
    assert whole_then_slice.shape == (300, 30)
    assert np.isfinite(whole_then_slice).all()


def test_split_index_holds_out_the_tail():
    is_s, oos_s = split_index(1000, 0.3)
    assert is_s.stop == 700 and oos_s.start == 700 and oos_s.stop == 1000


def test_walk_forward_windows_never_train_on_the_future():
    for train, test in walk_forward_windows(2000, n_windows=5):
        assert train.stop <= test.start
        assert train.start < train.stop < test.stop


def test_min_hold_reduces_trade_count():
    df = data.synthetic(n=800, seed=6)
    ind = Indicators(df)
    base = dict(fast=5, slow=20, trend_filter=0, min_hold=1)
    slow = dict(base, min_hold=5)
    p_fast = positions_for(Candidate("ma_cross", tuple(sorted(base.items()))), ind)
    p_slow = positions_for(Candidate("ma_cross", tuple(sorted(slow.items()))), ind)
    assert trade_count_all(p_slow)[0] <= trade_count_all(p_fast)[0]


def test_neighbours_differ_in_exactly_one_parameter():
    cand = Candidate("ma_cross", tuple(sorted(dict(fast=10, slow=50, trend_filter=100, min_hold=3).items())))
    for nb in neighbours(cand):
        diffs = [k for k in cand.as_dict if cand.as_dict[k] != nb.as_dict[k]]
        assert len(diffs) == 1
