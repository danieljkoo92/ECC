"""Vectorised backtesting across the whole candidate set at once.

Two rules keep this honest:

1. A position decided on bar t is applied to bar t+1's return. No strategy ever
   trades on information it could not have had.
2. Every change in position pays a cost (commission + slippage, in basis points).
   A strategy that only works at zero cost is not a strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from .strategies import Candidate, Indicators, positions_for


@dataclass
class Costs:
    """Round-turn trading friction, expressed in basis points of notional."""

    commission_bps: float = 2.0
    slippage_bps: float = 3.0

    @property
    def per_unit_traded(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 10_000.0

    def scaled(self, factor: float) -> "Costs":
        return Costs(self.commission_bps * factor, self.slippage_bps * factor)


def build_position_matrix(candidates: List[Candidate], df: pd.DataFrame) -> np.ndarray:
    """Positions for every candidate over the full history, as int8 (-1/0/+1).

    Computed once on the full series and sliced afterwards. Every indicator is
    backward-looking, so slicing a later window cannot leak future information.
    """
    ind = Indicators(df)
    n = len(df)
    out = np.zeros((n, len(candidates)), dtype=np.int8)
    for j, cand in enumerate(candidates):
        out[:, j] = np.clip(np.rint(positions_for(cand, ind)), -1, 1).astype(np.int8)
    return out


def bar_returns(df: pd.DataFrame) -> np.ndarray:
    return df["close"].pct_change().fillna(0.0).to_numpy(dtype=float)


def strategy_returns(
    positions: np.ndarray,
    bar_ret: np.ndarray,
    costs: Costs = Costs(),
) -> np.ndarray:
    """Net per-bar returns for one or many strategies.

    positions may be 1-D (one strategy) or 2-D (bars x strategies).
    """
    pos = np.asarray(positions, dtype=float)
    single = pos.ndim == 1
    if single:
        pos = pos[:, None]
    applied = np.vstack([np.zeros((1, pos.shape[1])), pos[:-1]])  # lag by one bar
    gross = applied * bar_ret[:, None]
    turnover = np.abs(np.vstack([applied[:1], np.diff(applied, axis=0)]))
    net = gross - turnover * costs.per_unit_traded
    return net[:, 0] if single else net


def sharpe_all(net: np.ndarray, periods: int = 252) -> np.ndarray:
    """Annualised Sharpe for every column of a returns matrix."""
    sd = np.maximum(net.std(axis=0, ddof=1), 1e-12)
    return net.mean(axis=0) / sd * np.sqrt(periods)


def max_drawdown_all(net: np.ndarray) -> np.ndarray:
    equity = np.cumprod(1.0 + net, axis=0)
    peak = np.maximum.accumulate(equity, axis=0)
    return np.min(equity / peak - 1.0, axis=0)


def trade_count_all(positions: np.ndarray) -> np.ndarray:
    pos = np.asarray(positions, dtype=float)
    if pos.ndim == 1:
        pos = pos[:, None]
    changes = np.abs(np.diff(pos, axis=0)) > 0
    return changes.sum(axis=0)


def split_index(n: int, oos_fraction: float = 0.3) -> Tuple[slice, slice]:
    """In-sample / out-of-sample split. The OOS tail is never used for selection."""
    cut = int(n * (1 - oos_fraction))
    return slice(0, cut), slice(cut, n)


def walk_forward_windows(n: int, n_windows: int = 5, train_frac: float = 0.6) -> List[Tuple[slice, slice]]:
    """Rolling anchored train/test windows covering the back half of the history."""
    windows: List[Tuple[slice, slice]] = []
    block = n // (n_windows + 1)
    if block < 60:
        return windows
    for i in range(n_windows):
        train_end = block * (i + 1)
        test_end = min(block * (i + 2), n)
        train_start = max(0, int(train_end - block * (1 / max(train_frac, 0.1))))
        windows.append((slice(train_start, train_end), slice(train_end, test_end)))
    return windows
