"""Performance statistics, plus the corrections that large strategy searches need.

The headline numbers (Sharpe, profit factor, win rate) are the easy part and the
part marketing videos show. The functions that matter here are the last three:

* deflated_sharpe   - how much of a Sharpe ratio survives once you admit how
                      many strategies you tried before picking the best one
                      (Bailey & Lopez de Prado, 2014).
* monte_carlo       - block bootstrap of the return stream, giving a confidence
                      interval instead of a single lucky path.
* pbo_cscv          - Probability of Backtest Overfitting via combinatorially
                      symmetric cross-validation: of all the ways to split the
                      history, how often does the in-sample winner land in the
                      bottom half out of sample?
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Dict, List, Sequence

import numpy as np

TRADING_DAYS = 252
EULER_MASCHERONI = 0.5772156649015329


def norm_cdf(x: float) -> float:
    import math

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    import math

    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


@dataclass
class Perf:
    """Headline performance of one return stream."""

    n_bars: int
    n_trades: int
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    exposure: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def sharpe_ratio(returns: np.ndarray, periods: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods))


def sortino_ratio(returns: np.ndarray, periods: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=float)
    downside = r[r < 0]
    if downside.size < 2:
        return 0.0
    dd = downside.std(ddof=1)
    if dd <= 1e-12:
        return 0.0
    return float(r.mean() / dd * np.sqrt(periods))


def max_drawdown(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq / peak - 1.0))


def trade_blocks(positions: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Group bar returns into trades - one block per unbroken position run."""
    pos = np.asarray(positions, dtype=float)
    rets = np.asarray(returns, dtype=float)
    out: List[float] = []
    i, n = 0, pos.size
    while i < n:
        if pos[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and pos[j + 1] == pos[i]:
            j += 1
        out.append(float(np.sum(rets[i : j + 1])))
        i = j + 1
    return np.array(out, dtype=float)


def performance(returns: np.ndarray, positions: np.ndarray, periods: int = TRADING_DAYS) -> Perf:
    r = np.asarray(returns, dtype=float)
    equity = np.cumprod(1.0 + r)
    trades = trade_blocks(positions, r)
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    gross_win = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(-losses.sum()) if losses.size else 0.0
    years = max(r.size / periods, 1e-9)
    total = float(equity[-1] - 1.0) if equity.size else 0.0
    return Perf(
        n_bars=int(r.size),
        n_trades=int(trades.size),
        total_return=total,
        cagr=float((equity[-1]) ** (1 / years) - 1.0) if equity.size and equity[-1] > 0 else -1.0,
        sharpe=sharpe_ratio(r, periods),
        sortino=sortino_ratio(r, periods),
        max_drawdown=max_drawdown(equity),
        profit_factor=float(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
        win_rate=float(wins.size / trades.size) if trades.size else 0.0,
        exposure=float(np.mean(np.asarray(positions) != 0)),
    )


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """E[max SR] over n independent trials with zero true skill."""
    if n_trials < 2:
        return 0.0
    sd = np.sqrt(max(sharpe_variance, 1e-12))
    g = EULER_MASCHERONI
    return float(sd * ((1 - g) * norm_ppf(1 - 1.0 / n_trials) + g * norm_ppf(1 - 1.0 / (n_trials * np.e))))


def deflated_sharpe(
    returns: np.ndarray,
    n_trials: int,
    sharpe_variance: float | None = None,
    periods: int = TRADING_DAYS,
) -> Dict[str, float]:
    """Probability the observed Sharpe is real, given how many strategies were tried.

    Returns the annualised Sharpe, the benchmark Sharpe that a lucky search would
    be expected to produce with no skill at all, and the deflated Sharpe ratio -
    a probability. Below ~0.95 the result is not distinguishable from luck.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 30:
        return {"sharpe": 0.0, "sr_benchmark": 0.0, "dsr": 0.0, "n_trials": float(n_trials)}
    sr_ann = sharpe_ratio(r, periods)
    sr = sr_ann / np.sqrt(periods)  # per-bar Sharpe, as the maths requires
    mu, sd = r.mean(), r.std(ddof=1)
    if sd <= 1e-12:
        return {"sharpe": 0.0, "sr_benchmark": 0.0, "dsr": 0.0, "n_trials": float(n_trials)}
    z = (r - mu) / sd
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    if sharpe_variance is None:
        sharpe_variance = (1 - skew * sr + (kurt - 1) / 4 * sr**2) / max(n - 1, 1)
    sr0 = expected_max_sharpe(n_trials, sharpe_variance)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    stat = (sr - sr0) * np.sqrt(max(n - 1, 1)) / denom
    return {
        "sharpe": float(sr_ann),
        "sr_benchmark": float(sr0 * np.sqrt(periods)),
        "dsr": float(norm_cdf(stat)),
        "n_trials": float(n_trials),
    }


def monte_carlo(
    returns: np.ndarray,
    n_sims: int = 1000,
    block: int = 10,
    seed: int = 7,
    periods: int = TRADING_DAYS,
) -> Dict[str, float]:
    """Stationary block bootstrap: resample the return stream, keep short-run structure."""
    r = np.asarray(returns, dtype=float)
    if r.size < block * 3:
        return {"sharpe_p05": 0.0, "sharpe_median": 0.0, "sharpe_p95": 0.0,
                "maxdd_p95": 0.0, "prob_negative": 1.0}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(r.size / block))
    starts = rng.integers(0, r.size - block, size=(n_sims, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_sims, -1)[:, : r.size]
    paths = r[idx]
    sharpes = paths.mean(axis=1) / np.maximum(paths.std(axis=1, ddof=1), 1e-12) * np.sqrt(periods)
    equity = np.cumprod(1.0 + paths, axis=1)
    peak = np.maximum.accumulate(equity, axis=1)
    dds = np.min(equity / peak - 1.0, axis=1)
    return {
        "sharpe_p05": float(np.percentile(sharpes, 5)),
        "sharpe_median": float(np.median(sharpes)),
        "sharpe_p95": float(np.percentile(sharpes, 95)),
        "maxdd_p95": float(np.percentile(dds, 5)),
        "prob_negative": float(np.mean(equity[:, -1] < 1.0)),
    }


def pbo_cscv(returns_matrix: np.ndarray, n_splits: int = 8, periods: int = TRADING_DAYS) -> Dict[str, float]:
    """Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado, Zhu).

    returns_matrix: shape (n_bars, n_strategies) of per-bar returns.
    Splits the history into n_splits blocks, takes every balanced combination as
    the in-sample half, and asks where the in-sample champion ranks out of sample.
    A PBO above 0.5 means the winner of your search is worse than a coin flip.
    """
    m = np.asarray(returns_matrix, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        return {"pbo": float("nan"), "n_combinations": 0.0, "median_oos_rank": float("nan")}
    n_bars, n_strats = m.shape
    block = n_bars // n_splits
    if block < 20:
        n_splits = max(2, n_bars // 40)
        block = n_bars // n_splits
        if block < 10:
            return {"pbo": float("nan"), "n_combinations": 0.0, "median_oos_rank": float("nan")}
    blocks = [m[i * block : (i + 1) * block] for i in range(n_splits)]

    def _sr(x: np.ndarray) -> np.ndarray:
        sd = np.maximum(x.std(axis=0, ddof=1), 1e-12)
        return x.mean(axis=0) / sd * np.sqrt(periods)

    logits: List[float] = []
    ranks: List[float] = []
    half = n_splits // 2
    for combo in combinations(range(n_splits), half):
        is_idx = list(combo)
        oos_idx = [i for i in range(n_splits) if i not in combo]
        is_r = np.concatenate([blocks[i] for i in is_idx], axis=0)
        oos_r = np.concatenate([blocks[i] for i in oos_idx], axis=0)
        best = int(np.argmax(_sr(is_r)))
        oos_sr = _sr(oos_r)
        rank = float((oos_sr < oos_sr[best]).sum()) / max(n_strats - 1, 1)  # 1.0 = best OOS
        ranks.append(rank)
        w = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(w / (1 - w))))
    logits_arr = np.array(logits)
    return {
        "pbo": float(np.mean(logits_arr <= 0.0)),
        "n_combinations": float(len(logits)),
        "median_oos_rank": float(np.median(ranks)),
    }


def regime_breakdown(returns: np.ndarray, closes: np.ndarray, periods: int = TRADING_DAYS) -> Dict[str, float]:
    """Sharpe inside four crude regimes: up/down trend, high/low volatility."""
    r = np.asarray(returns, dtype=float)
    c = np.asarray(closes, dtype=float)
    if r.size != c.size or r.size < periods:
        return {}
    import pandas as pd

    s = pd.Series(c)
    trend_up = (s > s.rolling(200, min_periods=50).mean()).to_numpy()
    bar_ret = s.pct_change().fillna(0.0)
    vol = bar_ret.rolling(20, min_periods=10).std()
    high_vol = (vol > vol.rolling(250, min_periods=60).median()).fillna(False).to_numpy()
    out: Dict[str, float] = {}
    for name, mask in (
        ("uptrend", trend_up),
        ("downtrend", ~trend_up),
        ("high_vol", high_vol),
        ("low_vol", ~high_vol),
    ):
        sel = r[mask]
        out[f"sharpe_{name}"] = sharpe_ratio(sel, periods) if sel.size > 30 else 0.0
        out[f"bars_{name}"] = float(sel.size)
    return out
