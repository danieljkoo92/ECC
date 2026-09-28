"""The validation funnel for betting systems.

The same stages as the market funnel, translated to bets: return on investment
per bet replaces return per bar, seasons replace calendar windows, and the cost
stress becomes a worse price (-115 instead of -110). A system has to clear the
bookmaker's margin in-sample, keep clearing it at worse prices and on seasons
it never saw, and do so by more than a search of this size would manage by luck.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import stats
from ..pipeline import FunnelStage
from .features import SIDES, Prices, build_features, payoffs
from .strategies import BetCandidate, generate_candidates, neighbours, side_and_mask


@dataclass
class BetThresholds:
    """Every gate the funnel applies, in one place."""

    min_is_bets: int = 150
    min_is_roi: float = 0.03
    plateau_ratio: float = 0.5
    min_stressed_roi: float = 0.0
    min_oos_bets: int = 50
    min_oos_roi: float = 0.0
    min_oos_is_ratio: float = 0.3
    min_mc_p05_roi: float = 0.0
    min_dsr: float = 0.95
    min_season_bets: int = 20
    min_positive_season_frac: float = 0.6
    wf_min_train_bets: int = 150
    wf_warmup_seasons: int = 3


@dataclass
class BetResult:
    label: str
    n_games: int
    seasons: List[str]
    is_seasons: List[str]
    oos_seasons: List[str]
    n_candidates: int
    funnel: List[FunnelStage] = field(default_factory=list)
    survivors: List[Dict] = field(default_factory=list)
    best_overall: Dict = field(default_factory=dict)
    walk_forward: Dict = field(default_factory=dict)
    baseline: Dict[str, float] = field(default_factory=dict)
    pbo: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def bootstrap_roi(bet_returns: np.ndarray, n_sims: int = 2000, seed: int = 11) -> Dict[str, float]:
    """Resample individual bets with replacement; report the spread of ROI outcomes."""
    r = np.asarray(bet_returns, dtype=float)
    if r.size < 10:
        return {"roi_p05": float("nan"), "roi_median": float("nan"), "roi_p95": float("nan"), "prob_negative": 1.0}
    rng = np.random.default_rng(seed)
    means = r[rng.integers(0, r.size, size=(n_sims, r.size))].mean(axis=1)
    return {
        "roi_p05": float(np.percentile(means, 5)),
        "roi_median": float(np.median(means)),
        "roi_p95": float(np.percentile(means, 95)),
        "prob_negative": float(np.mean(means < 0)),
    }


def _stage(result: BetResult, name: str, description: str, before: int, after: int) -> None:
    result.funnel.append(FunnelStage(name, description, after, before - after))


def _roi(pnl: np.ndarray, bets: np.ndarray) -> np.ndarray:
    n = bets.sum(axis=0)
    return np.where(n > 0, pnl.sum(axis=0) / np.maximum(n, 1), 0.0)


def run_bet_pipeline(
    games: pd.DataFrame,
    label: str = "NBA",
    prices: Prices = Prices(),
    thresholds: BetThresholds = BetThresholds(),
    candidates: Optional[List[BetCandidate]] = None,
    oos_fraction: float = 0.3,
    max_survivors: int = 10,
    verbose: bool = True,
) -> BetResult:
    t = thresholds
    cands = candidates if candidates is not None else generate_candidates()
    seasons = list(dict.fromkeys(games["season"].tolist()))
    n_oos = max(1, int(round(len(seasons) * oos_fraction)))
    is_seasons, oos_seasons = seasons[:-n_oos], seasons[-n_oos:]
    result = BetResult(label, len(games), seasons, is_seasons, oos_seasons, len(cands))

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    log(f"[1/9] building features and {len(cands):,} betting systems over {len(games):,} games")
    feats = build_features(games)
    pay = payoffs(games, prices)
    pay_stress = payoffs(games, prices.stressed())
    side_index = {s: i for i, s in enumerate(SIDES)}
    pay6 = np.stack([pay[s] for s in SIDES], axis=1)
    pay6_stress = np.stack([pay_stress[s] for s in SIDES], axis=1)
    bettable = ~np.isnan(pay6)

    sides = np.empty(len(cands), dtype=int)
    bets = np.zeros((len(games), len(cands)), dtype=bool)
    for j, cand in enumerate(cands):
        side, mask = side_and_mask(cand, feats)
        sides[j] = side_index[side]
        bets[:, j] = mask & bettable[:, sides[j]]
    pnl = (bets * np.nan_to_num(pay6)[:, sides]).astype(np.float32)

    season_of = games["season"].to_numpy()
    is_rows = np.isin(season_of, is_seasons)
    oos_rows = ~is_rows
    bets_is, bets_oos = bets[is_rows].sum(axis=0), bets[oos_rows].sum(axis=0)
    roi_is, roi_oos = _roi(pnl[is_rows], bets[is_rows]), _roi(pnl[oos_rows], bets[oos_rows])

    result.baseline = {s: float(np.nanmean(pay[s])) for s in SIDES}

    alive = np.arange(len(cands))
    _stage(result, "universe", f"betting systems across {len(set(c.family for c in cands))} families", len(cands), len(cands))

    eligible = np.where(bets_is >= t.min_is_bets)[0]
    if eligible.size:
        b = int(eligible[np.argmax(roi_is[eligible])])
        full = pnl[bets[:, b], b]
        dsr_b = stats.deflated_sharpe(full, n_trials=len(cands), periods=1)
        result.best_overall = {
            "key": cands[b].key, "is_roi": float(roi_is[b]), "oos_roi": float(roi_oos[b]),
            "is_bets": int(bets_is[b]), "oos_bets": int(bets_oos[b]),
            "dsr": dsr_b["dsr"], "sr_benchmark": dsr_b["sr_benchmark"],
        }

    keep = (bets_is >= t.min_is_bets) & (roi_is >= t.min_is_roi)
    before, alive = len(alive), alive[keep]
    _stage(result, "in_sample_screen", f"in-sample ROI >= {t.min_is_roi:+.0%} after the bookmaker's margin, over >= {t.min_is_bets} bets", before, len(alive))
    log(f"[2/9] in-sample screen: {len(alive):,} survive")

    if len(alive) >= 2:
        pool = alive[np.argsort(-roi_is[alive])[:200]]
        result.pbo = stats.pbo_cscv(pnl[:, pool], periods=1)
        log(f"[3/9] PBO on the screened pool: {result.pbo.get('pbo', float('nan')):.2f}")

    index_of = {c.key: i for i, c in enumerate(cands)}
    plateau: List[int] = []
    for i in alive:
        nb = [index_of[c.key] for c in neighbours(cands[i]) if c.key in index_of]
        if not nb or float(np.median(roi_is[nb])) >= t.plateau_ratio * float(roi_is[i]):
            plateau.append(int(i))
    before, alive = len(alive), np.array(plateau, dtype=int)
    _stage(result, "plateau", f"neighbouring parameters keep >= {t.plateau_ratio:.0%} of the ROI (no lucky spikes)", before, len(alive))
    log(f"[4/9] plateau: {len(alive):,} survive")

    before = len(alive)
    if len(alive):
        stressed = bets[is_rows][:, alive] * np.nan_to_num(pay6_stress[is_rows])[:, sides[alive]]
        alive = alive[_roi(stressed, bets[is_rows][:, alive]) >= t.min_stressed_roi]
    _stage(result, "price_stress", "still profitable at -115 and with 5% shaved off moneyline wins", before, len(alive))
    log(f"[5/9] price stress: {len(alive):,} survive")

    before = len(alive)
    if len(alive):
        ok = (bets_oos[alive] >= t.min_oos_bets) & (roi_oos[alive] >= t.min_oos_roi) & (
            roi_oos[alive] >= t.min_oos_is_ratio * roi_is[alive])
        alive = alive[ok]
    _stage(result, "out_of_sample", f"held-out seasons {oos_seasons[0]} to {oos_seasons[-1]}: profitable and >= {t.min_oos_is_ratio:.0%} of in-sample ROI", before, len(alive))
    log(f"[6/9] out-of-sample: {len(alive):,} survive")

    # Walk-forward of the selection process: each season, bet the system that
    # had the best record over every season before it.
    picks: List[Dict] = []
    units = n_bets = 0.0
    for k in range(t.wf_warmup_seasons, len(seasons)):
        train = np.isin(season_of, seasons[:k])
        test = season_of == seasons[k]
        tr_bets = bets[train].sum(axis=0)
        tr_roi = _roi(pnl[train], bets[train])
        ok = np.where(tr_bets >= t.wf_min_train_bets)[0]
        if not ok.size:
            continue
        j = int(ok[np.argmax(tr_roi[ok])])
        s_units, s_bets = float(pnl[test, j].sum()), int(bets[test, j].sum())
        units += s_units
        n_bets += s_bets
        picks.append({"season": seasons[k], "system": cands[j].key, "train_roi": float(tr_roi[j]),
                      "bets": s_bets, "units": s_units, "roi": s_units / s_bets if s_bets else 0.0})
    if picks:
        result.walk_forward = {"units": units, "bets": int(n_bets), "roi": units / n_bets if n_bets else 0.0,
                               "picks": picks}
        log(f"[7/9] walk-forward of the selection process: {units:+.1f} units over {int(n_bets):,} bets")

    finalists: List[Dict] = []
    for i in alive:
        col = bets[:, i]
        full = pnl[col, i]
        oos = pnl[col & oos_rows, i]
        mc = bootstrap_roi(oos)
        dsr = stats.deflated_sharpe(full, n_trials=len(cands), periods=1)
        season_rois = []
        for s in seasons:
            m = col & (season_of == s)
            if m.sum() >= t.min_season_bets:
                season_rois.append(float(pnl[m, i].mean()))
        finalists.append({
            "key": cands[i].key, "family": cands[i].family,
            "is_roi": float(roi_is[i]), "oos_roi": float(roi_oos[i]),
            "is_bets": int(bets_is[i]), "oos_bets": int(bets_oos[i]),
            "mc_roi_p05": mc["roi_p05"], "dsr": dsr["dsr"], "sr_benchmark": dsr["sr_benchmark"],
            "positive_season_frac": float(np.mean(np.array(season_rois) > 0)) if season_rois else 0.0,
        })

    before = len(finalists)
    finalists = [f for f in finalists if f["mc_roi_p05"] >= t.min_mc_p05_roi]
    _stage(result, "monte_carlo", "5th-percentile bootstrap ROI on held-out bets still positive", before, len(finalists))
    log(f"[8/9] Monte Carlo: {len(finalists):,} survive")

    before = len(finalists)
    finalists = [f for f in finalists if f["dsr"] >= t.min_dsr]
    _stage(result, "deflated_sharpe", f"deflated Sharpe >= {t.min_dsr:.2f} after correcting for {len(cands):,} systems tried", before, len(finalists))
    log(f"[9/9] deflated Sharpe: {len(finalists):,} survive")

    before = len(finalists)
    finalists = [f for f in finalists if f["positive_season_frac"] >= t.min_positive_season_frac]
    _stage(result, "season_consistency", f"profitable in >= {t.min_positive_season_frac:.0%} of seasons", before, len(finalists))

    finalists.sort(key=lambda f: (-f["dsr"], -f["oos_roi"]))
    result.survivors = finalists[:max_survivors]
    if not finalists:
        result.notes.append(
            "No betting system survived. After the bookmaker's margin, worse prices, unseen seasons and the "
            "correction for how many systems were tried, none is distinguishable from luck."
        )
    return result
