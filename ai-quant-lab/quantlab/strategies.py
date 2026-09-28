"""Strategy families and the parameter grid that turns them into a search space.

Every family returns a position array in {-1, 0, +1} computed from information
available up to and including each bar. The backtester applies the position on
the *next* bar, which is what keeps the whole thing free of look-ahead.

Six families x their parameter grids x a trend filter x a minimum holding period
generates roughly ten thousand candidate strategies - the same "AI generated
tens of thousands of strategies" move, run honestly so the count can be fed into
the deflated Sharpe ratio later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Candidate:
    """One fully specified strategy: a family plus its parameters."""

    family: str
    params: Tuple[Tuple[str, object], ...]

    @property
    def as_dict(self) -> Dict[str, object]:
        return dict(self.params)

    @property
    def key(self) -> str:
        inner = ",".join(f"{k}={v}" for k, v in self.params)
        return f"{self.family}({inner})"

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.key


class Indicators:
    """Rolling indicators, memoised per (kind, window) so the search stays fast."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.close = df["close"].to_numpy(dtype=float)
        self.high = df["high"].to_numpy(dtype=float)
        self.low = df["low"].to_numpy(dtype=float)
        self._s_close = pd.Series(self.close)
        self._s_high = pd.Series(self.high)
        self._s_low = pd.Series(self.low)
        self._cache: Dict[Tuple[str, int], np.ndarray] = {}

    def _get(self, kind: str, window: int) -> np.ndarray:
        key = (kind, window)
        if key in self._cache:
            return self._cache[key]
        s = self._s_close
        if kind == "sma":
            out = s.rolling(window, min_periods=window).mean()
        elif kind == "std":
            out = s.rolling(window, min_periods=window).std(ddof=0)
        elif kind == "hh":
            out = self._s_high.rolling(window, min_periods=window).max()
        elif kind == "ll":
            out = self._s_low.rolling(window, min_periods=window).min()
        elif kind == "rsi":
            delta = s.diff()
            gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
            rs = gain / loss.replace(0.0, np.nan)
            out = 100 - 100 / (1 + rs)
            out = out.fillna(50.0)
        elif kind == "atr":
            prev_close = s.shift(1)
            tr = pd.concat(
                [
                    self._s_high - self._s_low,
                    (self._s_high - prev_close).abs(),
                    (self._s_low - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            out = tr.rolling(window, min_periods=window).mean()
        elif kind == "mom":
            out = s.pct_change(window)
        else:  # pragma: no cover - guard
            raise ValueError(f"unknown indicator {kind!r}")
        arr = out.to_numpy(dtype=float)
        self._cache[key] = arr
        return arr

    def sma(self, w: int) -> np.ndarray:
        return self._get("sma", w)

    def std(self, w: int) -> np.ndarray:
        return self._get("std", w)

    def highest(self, w: int) -> np.ndarray:
        return self._get("hh", w)

    def lowest(self, w: int) -> np.ndarray:
        return self._get("ll", w)

    def rsi(self, w: int) -> np.ndarray:
        return self._get("rsi", w)

    def atr(self, w: int) -> np.ndarray:
        return self._get("atr", w)

    def momentum(self, w: int) -> np.ndarray:
        return self._get("mom", w)


def _hold(raw: np.ndarray, min_hold: int) -> np.ndarray:
    """Force a position to be held for at least min_hold bars before it can change."""
    if min_hold <= 1:
        return raw
    out = raw.copy()
    last_change = -10**9
    current = 0.0
    for i in range(out.size):
        if raw[i] != current and (i - last_change) >= min_hold:
            current = raw[i]
            last_change = i
        out[i] = current
    return out


def _apply_trend_filter(pos: np.ndarray, ind: Indicators, window: int) -> np.ndarray:
    """Longs only above the trend average, shorts only below it. 0 disables."""
    if not window:
        return pos
    ma = ind.sma(window)
    up = ind.close > ma
    out = pos.copy()
    out[(out > 0) & ~up] = 0.0
    out[(out < 0) & up] = 0.0
    out[np.isnan(ma)] = 0.0
    return out


def _ma_cross(ind: Indicators, p: Dict[str, object]) -> np.ndarray:
    fast, slow = ind.sma(int(p["fast"])), ind.sma(int(p["slow"]))
    pos = np.where(fast > slow, 1.0, -1.0)
    pos[np.isnan(fast) | np.isnan(slow)] = 0.0
    return pos


def _donchian(ind: Indicators, p: Dict[str, object]) -> np.ndarray:
    entry, exit_ = int(p["entry"]), int(p["exit"])
    hh, ll = ind.highest(entry), ind.lowest(entry)
    hh_x, ll_x = ind.highest(exit_), ind.lowest(exit_)
    c = ind.close
    pos = np.zeros(c.size)
    state = 0.0
    for i in range(c.size):
        if np.isnan(hh[i]) or np.isnan(ll[i]):
            pos[i] = 0.0
            continue
        if state <= 0 and c[i] >= hh[i]:
            state = 1.0
        elif state >= 0 and c[i] <= ll[i]:
            state = -1.0
        elif state > 0 and not np.isnan(ll_x[i]) and c[i] <= ll_x[i]:
            state = 0.0
        elif state < 0 and not np.isnan(hh_x[i]) and c[i] >= hh_x[i]:
            state = 0.0
        pos[i] = state
    return pos


def _rsi_reversion(ind: Indicators, p: Dict[str, object]) -> np.ndarray:
    rsi = ind.rsi(int(p["period"]))
    low, high = float(p["low"]), float(p["high"])
    pos = np.zeros(rsi.size)
    state = 0.0
    for i in range(rsi.size):
        if rsi[i] <= low:
            state = 1.0
        elif rsi[i] >= high:
            state = -1.0
        elif (state > 0 and rsi[i] >= 50) or (state < 0 and rsi[i] <= 50):
            state = 0.0
        pos[i] = state
    return pos


def _bollinger(ind: Indicators, p: Dict[str, object]) -> np.ndarray:
    w, k = int(p["period"]), float(p["k"])
    ma, sd = ind.sma(w), ind.std(w)
    upper, lower = ma + k * sd, ma - k * sd
    c = ind.close
    pos = np.zeros(c.size)
    state = 0.0
    for i in range(c.size):
        if np.isnan(ma[i]):
            pos[i] = 0.0
            continue
        if c[i] <= lower[i]:
            state = 1.0
        elif c[i] >= upper[i]:
            state = -1.0
        elif (state > 0 and c[i] >= ma[i]) or (state < 0 and c[i] <= ma[i]):
            state = 0.0
        pos[i] = state
    return pos


def _momentum(ind: Indicators, p: Dict[str, object]) -> np.ndarray:
    mom = ind.momentum(int(p["lookback"]))
    thresh = float(p["threshold"])
    pos = np.where(mom > thresh, 1.0, np.where(mom < -thresh, -1.0, 0.0))
    pos[np.isnan(mom)] = 0.0
    return pos


def _vol_breakout(ind: Indicators, p: Dict[str, object]) -> np.ndarray:
    lb, atr_w, mult = int(p["lookback"]), int(p["atr"]), float(p["mult"])
    ma, atr = ind.sma(lb), ind.atr(atr_w)
    c = ind.close
    upper, lower = ma + mult * atr, ma - mult * atr
    pos = np.where(c > upper, 1.0, np.where(c < lower, -1.0, 0.0))
    pos[np.isnan(ma) | np.isnan(atr)] = 0.0
    return pos


FAMILIES = {
    "ma_cross": _ma_cross,
    "donchian": _donchian,
    "rsi_reversion": _rsi_reversion,
    "bollinger": _bollinger,
    "momentum": _momentum,
    "vol_breakout": _vol_breakout,
}

GRIDS: Dict[str, Dict[str, List]] = {
    "ma_cross": {"fast": [5, 8, 10, 13, 20, 25, 30, 40, 50], "slow": [20, 30, 40, 50, 60, 80, 100, 150, 200]},
    "donchian": {"entry": [10, 15, 20, 30, 40, 55, 80, 100], "exit": [5, 10, 15, 20, 30, 40]},
    "rsi_reversion": {"period": [2, 3, 5, 7, 10, 14, 21], "low": [10, 15, 20, 25, 30], "high": [70, 75, 80, 85, 90]},
    "bollinger": {"period": [10, 15, 20, 30, 50], "k": [1.0, 1.5, 2.0, 2.5, 3.0]},
    "momentum": {"lookback": [20, 40, 60, 90, 120, 180, 250], "threshold": [0.0, 0.02, 0.05, 0.10]},
    "vol_breakout": {"lookback": [10, 20, 30], "atr": [5, 10, 14, 20], "mult": [0.5, 1.0, 1.5, 2.0, 3.0]},
}

TREND_FILTERS = [0, 50, 100, 200]
MIN_HOLDS = [1, 3, 5]


def _valid(family: str, params: Dict[str, object]) -> bool:
    if family == "ma_cross":
        return int(params["fast"]) < int(params["slow"])
    if family == "donchian":
        return int(params["exit"]) <= int(params["entry"])
    if family == "rsi_reversion":
        return float(params["low"]) < float(params["high"])
    return True


def generate_candidates(
    families: List[str] | None = None,
    trend_filters: List[int] | None = None,
    min_holds: List[int] | None = None,
    limit: int | None = None,
) -> List[Candidate]:
    """Build the full search space - the honest equivalent of "72,000 strategies"."""
    families = families or list(FAMILIES)
    trend_filters = TREND_FILTERS if trend_filters is None else trend_filters
    min_holds = MIN_HOLDS if min_holds is None else min_holds
    out: List[Candidate] = []
    for fam in families:
        grid = GRIDS[fam]
        names = list(grid)
        for values in product(*(grid[n] for n in names)):
            base = dict(zip(names, values))
            if not _valid(fam, base):
                continue
            for tf, mh in product(trend_filters, min_holds):
                params = dict(base)
                params["trend_filter"] = tf
                params["min_hold"] = mh
                out.append(Candidate(fam, tuple(sorted(params.items()))))
                if limit and len(out) >= limit:
                    return out
    return out


def positions_for(candidate: Candidate, ind: Indicators) -> np.ndarray:
    """Position series for one candidate, using information up to each bar only."""
    params = candidate.as_dict
    raw = FAMILIES[candidate.family](ind, params)
    raw = _apply_trend_filter(raw, ind, int(params.get("trend_filter", 0)))
    raw = _hold(raw, int(params.get("min_hold", 1)))
    return np.nan_to_num(raw, nan=0.0)


def neighbours(candidate: Candidate) -> List[Candidate]:
    """Candidates one grid step away in any single parameter.

    Used for the plateau test: a real edge sits on a plateau, where neighbouring
    parameters work too. An overfit one sits on a spike, alone.
    """
    grid = dict(GRIDS[candidate.family])
    grid["trend_filter"] = TREND_FILTERS
    grid["min_hold"] = MIN_HOLDS
    base = candidate.as_dict
    out: List[Candidate] = []
    for name, values in grid.items():
        if name not in base:
            continue
        try:
            i = values.index(base[name])
        except ValueError:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(values):
                params = dict(base)
                params[name] = values[j]
                if not _valid(candidate.family, params):
                    continue
                out.append(Candidate(candidate.family, tuple(sorted(params.items()))))
    return out
