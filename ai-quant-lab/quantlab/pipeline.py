"""The full search-and-validation funnel.

Stage by stage, this is the same pipeline an "AI quant" video describes -
generate thousands of strategies, screen them, test out of sample, walk forward,
Monte Carlo, deflated Sharpe, regime test. The difference is that every stage
reports how many candidates it killed, and the final answer is allowed to be
"nothing survived", which is the honest outcome most of the time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import stats
from .backtest import (
    Costs,
    bar_returns,
    build_position_matrix,
    max_drawdown_all,
    sharpe_all,
    split_index,
    strategy_returns,
    trade_count_all,
    walk_forward_windows,
)
from .strategies import Candidate, generate_candidates, neighbours


@dataclass
class Thresholds:
    """Every gate the pipeline applies, in one place so nothing is hidden."""

    min_is_sharpe: float = 0.7
    min_trades: int = 30
    max_drawdown: float = -0.40
    min_exposure: float = 0.05
    plateau_ratio: float = 0.5
    cost_stress: float = 3.0
    min_stressed_sharpe: float = 0.3
    min_oos_sharpe: float = 0.5
    min_oos_is_ratio: float = 0.3
    min_wf_sharpe: float = 0.5
    min_mc_p05_sharpe: float = 0.0
    min_dsr: float = 0.95
    min_positive_regimes: int = 3


@dataclass
class FunnelStage:
    name: str
    description: str
    survivors: int
    killed: int


@dataclass
class Result:
    symbol: str
    bars: int
    start: str
    end: str
    n_candidates: int
    funnel: List[FunnelStage] = field(default_factory=list)
    survivors: List[Dict] = field(default_factory=list)
    pbo: Dict[str, float] = field(default_factory=dict)
    best_overall: Dict[str, float] = field(default_factory=dict)
    walk_forward: Dict[str, float] = field(default_factory=dict)
    buy_hold: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _stage(result: Result, name: str, description: str, before: int, after: int) -> None:
    result.funnel.append(FunnelStage(name, description, after, before - after))


def run_pipeline(
    df: pd.DataFrame,
    symbol: str,
    costs: Costs = Costs(),
    thresholds: Thresholds = Thresholds(),
    candidates: Optional[List[Candidate]] = None,
    oos_fraction: float = 0.3,
    periods: int = 252,
    max_survivors: int = 10,
    verbose: bool = True,
) -> Result:
    cands = candidates if candidates is not None else generate_candidates()
    n = len(df)
    result = Result(
        symbol=symbol,
        bars=n,
        start=str(df.index[0].date()),
        end=str(df.index[-1].date()),
        n_candidates=len(cands),
    )

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    log(f"[1/9] generating and backtesting {len(cands):,} candidates on {symbol} ({n:,} bars)")
    positions = build_position_matrix(cands, df)
    br = bar_returns(df)
    is_slice, oos_slice = split_index(n, oos_fraction)

    net_is = strategy_returns(positions[is_slice], br[is_slice], costs)
    sr_is = sharpe_all(net_is, periods)
    dd_is = max_drawdown_all(net_is)
    trades_is = trade_count_all(positions[is_slice])
    exposure_is = np.mean(positions[is_slice] != 0, axis=0)

    alive = np.arange(len(cands))
    _stage(result, "universe", f"all parameter combinations across {len(set(c.family for c in cands))} strategy families", len(cands), len(cands))

    # The single best-looking backtest in the whole search, kept aside whether or
    # not it survives. This is the number a marketing video would put on screen,
    # so it is the number worth comparing against the luck benchmark.
    best_i = int(np.argmax(sr_is))
    best_full = strategy_returns(positions[:, best_i], br, costs)
    best_oos = strategy_returns(positions[oos_slice][:, best_i], br[oos_slice], costs)
    best_dsr = stats.deflated_sharpe(best_full, n_trials=len(cands), periods=periods)
    result.best_overall = {
        "key": cands[best_i].key,
        "is_sharpe": float(sr_is[best_i]),
        "oos_sharpe": float(stats.sharpe_ratio(best_oos, periods)),
        "dsr": best_dsr["dsr"],
        "sr_benchmark": best_dsr["sr_benchmark"],
        "max_drawdown": float(dd_is[best_i]),
        "n_trades": float(trades_is[best_i]),
    }

    keep = (
        (sr_is[alive] >= thresholds.min_is_sharpe)
        & (trades_is[alive] >= thresholds.min_trades)
        & (dd_is[alive] >= thresholds.max_drawdown)
        & (exposure_is[alive] >= thresholds.min_exposure)
    )
    before, alive = len(alive), alive[keep]
    _stage(result, "in_sample_screen", f"in-sample Sharpe >= {thresholds.min_is_sharpe}, >= {thresholds.min_trades} trades, drawdown better than {thresholds.max_drawdown:.0%}", before, len(alive))
    log(f"[2/9] in-sample screen: {len(alive):,} survive")

    # Probability of backtest overfitting, measured on the screened pool.
    if len(alive) >= 2:
        pool = alive[np.argsort(-sr_is[alive])[:200]]
        result.pbo = stats.pbo_cscv(strategy_returns(positions[:, pool], br, costs), periods=periods)
        log(f"[3/9] PBO on the screened pool: {result.pbo.get('pbo', float('nan')):.2f}")

    # Plateau test - neighbours in parameter space must work too.
    index_of = {c.key: i for i, c in enumerate(cands)}
    plateau_keep: List[int] = []
    for i in alive:
        nb = [index_of[c.key] for c in neighbours(cands[i]) if c.key in index_of]
        if not nb:
            continue
        if float(np.median(sr_is[nb])) >= thresholds.plateau_ratio * float(sr_is[i]):
            plateau_keep.append(int(i))
    before, alive = len(alive), np.array(plateau_keep, dtype=int)
    _stage(result, "plateau", f"neighbouring parameters keep >= {thresholds.plateau_ratio:.0%} of the Sharpe (no lucky spikes)", before, len(alive))
    log(f"[4/9] plateau test: {len(alive):,} survive")

    before = len(alive)
    if len(alive):
        stressed = strategy_returns(positions[is_slice][:, alive], br[is_slice], costs.scaled(thresholds.cost_stress))
        alive = alive[sharpe_all(stressed, periods) >= thresholds.min_stressed_sharpe]
    _stage(result, "cost_stress", f"still works at {thresholds.cost_stress:g}x trading costs", before, len(alive))
    log(f"[5/9] cost stress: {len(alive):,} survive")

    sr_oos = np.zeros(len(cands))
    before = len(alive)
    if len(alive):
        net_oos = strategy_returns(positions[oos_slice][:, alive], br[oos_slice], costs)
        sr_oos_alive = sharpe_all(net_oos, periods)
        sr_oos[alive] = sr_oos_alive
        ratio = sr_oos_alive / np.maximum(sr_is[alive], 1e-9)
        alive = alive[(sr_oos_alive >= thresholds.min_oos_sharpe) & (ratio >= thresholds.min_oos_is_ratio)]
    _stage(result, "out_of_sample", f"held-out final {oos_fraction:.0%} of history: Sharpe >= {thresholds.min_oos_sharpe} and >= {thresholds.min_oos_is_ratio:.0%} of in-sample", before, len(alive))
    log(f"[6/9] out-of-sample: {len(alive):,} survive")

    # Walk-forward, applied to the selection process itself, not one strategy.
    windows = walk_forward_windows(n)
    if windows:
        stitched: List[np.ndarray] = []
        picks: List[str] = []
        for train, test in windows:
            tr = strategy_returns(positions[train], br[train], costs)
            best = int(np.argmax(sharpe_all(tr, periods)))
            te = strategy_returns(positions[test][:, best], br[test], costs)
            stitched.append(te)
            picks.append(cands[best].key)
        wf = np.concatenate(stitched)
        wf_perf = stats.performance(wf, np.ones(wf.size), periods)
        result.walk_forward = {
            **wf_perf.to_dict(),
            "n_windows": float(len(windows)),
            "picks": picks,
        }
        log(f"[7/9] walk-forward of the whole process: Sharpe {wf_perf.sharpe:.2f}, return {wf_perf.total_return:.1%}")

    bh = stats.performance(br, np.ones(br.size), periods)
    result.buy_hold = bh.to_dict()

    # Monte Carlo, deflated Sharpe and regime consistency on whatever is left.
    finalists: List[Dict] = []
    for i in alive:
        net_full = strategy_returns(positions[:, i], br, costs)
        net_oos_i = strategy_returns(positions[oos_slice][:, i], br[oos_slice], costs)
        mc = stats.monte_carlo(net_oos_i, periods=periods)
        dsr = stats.deflated_sharpe(net_full, n_trials=len(cands), periods=periods)
        regimes = stats.regime_breakdown(net_full, df["close"].to_numpy(), periods)
        positive_regimes = sum(1 for k, v in regimes.items() if k.startswith("sharpe_") and v > 0)
        perf = stats.performance(net_full, positions[:, i], periods)
        finalists.append(
            {
                "key": cands[i].key,
                "family": cands[i].family,
                "is_sharpe": float(sr_is[i]),
                "oos_sharpe": float(sr_oos[i]),
                "mc_sharpe_p05": mc["sharpe_p05"],
                "mc_prob_negative": mc["prob_negative"],
                "dsr": dsr["dsr"],
                "sr_benchmark": dsr["sr_benchmark"],
                "positive_regimes": positive_regimes,
                **{k: v for k, v in perf.to_dict().items() if k in ("cagr", "max_drawdown", "profit_factor", "win_rate", "n_trades")},
            }
        )

    before = len(finalists)
    finalists = [f for f in finalists if f["mc_sharpe_p05"] >= thresholds.min_mc_p05_sharpe]
    _stage(result, "monte_carlo", "5th-percentile bootstrap Sharpe still positive", before, len(finalists))
    log(f"[8/9] Monte Carlo: {len(finalists):,} survive")

    before = len(finalists)
    finalists = [f for f in finalists if f["dsr"] >= thresholds.min_dsr]
    _stage(result, "deflated_sharpe", f"deflated Sharpe >= {thresholds.min_dsr:.2f} after correcting for {len(cands):,} strategies tested", before, len(finalists))
    log(f"[9/9] deflated Sharpe: {len(finalists):,} survive")

    before = len(finalists)
    finalists = [f for f in finalists if f["positive_regimes"] >= thresholds.min_positive_regimes]
    _stage(result, "regime_consistency", f"positive Sharpe in >= {thresholds.min_positive_regimes} of 4 market regimes", before, len(finalists))

    finalists.sort(key=lambda f: (-f["dsr"], -f["oos_sharpe"]))
    result.survivors = finalists[:max_survivors]
    if not finalists:
        result.notes.append(
            "Nothing survived the full funnel. On this data, with these families and these costs, "
            "the best-looking backtests are not distinguishable from luck."
        )
    return result
