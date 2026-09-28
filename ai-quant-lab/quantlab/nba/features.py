"""Pre-game features and bet payoffs.

Every feature for a game is computed only from games that finished before it,
and every bet is settled at the closing price - the last price available before
tip-off. Betting systems that quietly use the final score of the game they bet
on are the sports version of look-ahead bias; the state here is updated only
after a game's features have been read.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict

import numpy as np
import pandas as pd

SIDES = ("home_ml", "away_ml", "home_ats", "away_ats", "over", "under")
MAX_REST = 7
FORM_WINDOWS = (5, 10)


@dataclass(frozen=True)
class Prices:
    """How bets are paid. Spread and total bets at -110 is the standard US price."""

    ats_american: float = -110.0
    ml_profit_factor: float = 1.0  # 0.95 = keep only 95% of every moneyline profit

    @property
    def ats_win(self) -> float:
        return 100.0 / abs(self.ats_american)

    def stressed(self) -> "Prices":
        """Worse prices: -115 on spreads and totals, 5% shaved off moneyline profits."""
        return Prices(ats_american=-115.0, ml_profit_factor=0.95)


def payoffs(games: pd.DataFrame, prices: Prices = Prices()) -> Dict[str, np.ndarray]:
    """Profit per one-unit bet on each side of each game. NaN where no line was posted.

    A win pays the price, a loss costs the stake, and a push on a spread or
    total returns the stake (profit 0).
    """
    margin = (games["home_score"] - games["away_score"]).to_numpy(float)
    points = (games["home_score"] + games["away_score"]).to_numpy(float)
    spread = games["home_spread_close"].to_numpy(float)
    total = games["total_close"].to_numpy(float)
    home_ml = games["home_ml"].to_numpy(float)
    away_ml = games["away_ml"].to_numpy(float)

    def settle(edge: np.ndarray) -> np.ndarray:
        return np.where(edge > 0, prices.ats_win, np.where(edge < 0, -1.0, 0.0))

    cover = margin + spread  # > 0 means the home side covered
    over = points - total
    out = {
        "home_ml": np.where(margin > 0, (home_ml - 1.0) * prices.ml_profit_factor, -1.0),
        "away_ml": np.where(margin < 0, (away_ml - 1.0) * prices.ml_profit_factor, -1.0),
        "home_ats": settle(cover),
        "away_ats": settle(-cover),
        "over": settle(over),
        "under": settle(-over),
    }
    out["home_ml"][np.isnan(home_ml)] = np.nan
    out["away_ml"][np.isnan(away_ml)] = np.nan
    for side in ("home_ats", "away_ats"):
        out[side][np.isnan(spread)] = np.nan
    for side in ("over", "under"):
        out[side][np.isnan(total)] = np.nan
    return out


def _devig_home_probability(home_ml: np.ndarray, away_ml: np.ndarray) -> np.ndarray:
    """Market-implied home win probability with the bookmaker's margin removed."""
    ph, pa = 1.0 / home_ml, 1.0 / away_ml
    return ph / (ph + pa)


def build_features(games: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Everything a betting system may look at before a game, as aligned arrays."""
    n = len(games)
    home_rest = np.full(n, float(MAX_REST))
    away_rest = np.full(n, float(MAX_REST))
    home_streak = np.zeros(n)
    away_streak = np.zeros(n)
    cover_rate = {(side, w): np.full(n, np.nan) for side in ("home", "away") for w in FORM_WINDOWS}

    last_played: Dict[str, pd.Timestamp] = {}
    streak: Dict[str, int] = defaultdict(int)
    covers: Dict[str, Deque[int]] = defaultdict(lambda: deque(maxlen=max(FORM_WINDOWS)))
    season_of: Dict[str, str] = {}

    dates = games["date"].to_numpy()
    seasons = games["season"].to_numpy()
    homes = games["home"].to_numpy()
    aways = games["away"].to_numpy()
    margins = (games["home_score"] - games["away_score"]).to_numpy(float)
    spreads = games["home_spread_close"].to_numpy(float)

    for i in range(n):
        for team, rest_arr, streak_arr, side in (
            (homes[i], home_rest, home_streak, "home"),
            (aways[i], away_rest, away_streak, "away"),
        ):
            if season_of.get(team) != seasons[i]:  # new season: form and rest start fresh
                season_of[team] = seasons[i]
                last_played.pop(team, None)
                streak[team] = 0
                covers[team].clear()
            if team in last_played:
                days = (pd.Timestamp(dates[i]) - last_played[team]).days
                rest_arr[i] = float(min(max(days, 1), MAX_REST))
            streak_arr[i] = float(streak[team])
            history = list(covers[team])
            for w in FORM_WINDOWS:
                if len(history) >= w:
                    cover_rate[(side, w)][i] = float(np.mean(history[-w:]))

        # The game is over - only now does its result enter the teams' history.
        margin, spread = margins[i], spreads[i]
        when = pd.Timestamp(dates[i])
        for team, won in ((homes[i], margin > 0), (aways[i], margin < 0)):
            last_played[team] = when
            s = streak[team]
            streak[team] = (s + 1 if s > 0 else 1) if won else (s - 1 if s < 0 else -1)
        if not np.isnan(spread):
            cover = margin + spread
            if cover != 0:  # pushes do not count towards form
                covers[homes[i]].append(int(cover > 0))
                covers[aways[i]].append(int(cover < 0))

    season_series = games["season"]
    date_series = games["date"]
    start = date_series.groupby(season_series).transform("min")
    end = date_series.groupby(season_series).transform("max")
    span = (end - start).dt.days.replace(0, 1)
    phase = ((date_series - start).dt.days / span).to_numpy(float)

    features: Dict[str, np.ndarray] = {
        "home_rest": home_rest,
        "away_rest": away_rest,
        "home_streak": home_streak,
        "away_streak": away_streak,
        "phase": phase,
        "home_spread": spreads,
        "total": games["total_close"].to_numpy(float),
        "spread_move": (games["home_spread_close"] - games["home_spread_open"]).to_numpy(float),
        "total_move": (games["total_close"] - games["total_open"]).to_numpy(float),
        "home_prob": _devig_home_probability(games["home_ml"].to_numpy(float), games["away_ml"].to_numpy(float)),
    }
    for (side, w), arr in cover_rate.items():
        features[f"{side}_cover_{w}"] = arr
    return features
