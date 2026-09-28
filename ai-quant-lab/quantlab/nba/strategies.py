"""Betting systems: a filter over games plus the side to bet.

The families are the angles betting services actually sell - fade the public
line move, back rested teams against tired ones, ride hot streaks, take home
underdogs. Each is expanded over its parameter grid and then over two modifiers
(season phase, and whether the team bet on is the favourite or the underdog),
which is how a handful of ideas becomes a couple of thousand "systems".
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Tuple

import numpy as np

Features = Dict[str, np.ndarray]

PHASES = ("all", "early", "mid", "late")
ROLES = ("any", "fav", "dog")

# Parameters whose values have a natural order. Only these are stepped by the
# plateau test; categorical choices (home/away, follow/fade) are not "nearby".
ORDINAL = {"band", "threshold", "k", "window", "level"}

SPREAD_BANDS = (0.0, 3.0, 6.0, 9.0, 12.0, 99.0)
TOTAL_BANDS = (0.0, 195.0, 205.0, 215.0, 225.0, 235.0, 999.0)
PROB_BANDS = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.01)

GRIDS: Dict[str, Dict[str, list]] = {
    "spread_band": {"team": ["home", "away"], "role": ["fav", "dog"], "band": list(range(len(SPREAD_BANDS) - 1))},
    "total_band": {"side": ["over", "under"], "band": list(range(len(TOTAL_BANDS) - 1))},
    "ml_band": {"team": ["home", "away"], "band": list(range(len(PROB_BANDS) - 1))},
    "line_move": {"market": ["ats", "total"], "direction": ["up", "down"], "action": ["follow", "fade"],
                  "threshold": [0.5, 1.0, 1.5, 2.0, 3.0]},
    "rest": {"scenario": ["home_b2b", "away_b2b", "both_b2b", "home_rest_adv", "away_rest_adv"],
             "bet": ["home_ats", "away_ats", "home_ml", "away_ml", "over", "under"]},
    "streak": {"team": ["home", "away"], "kind": ["won", "lost"], "k": [2, 3, 4, 5, 6],
               "bet": ["team_ats", "opp_ats", "team_ml", "opp_ml"]},
    "ats_form": {"team": ["home", "away"], "window": [5, 10], "state": ["hot", "cold"], "level": [0, 1],
                 "action": ["follow", "fade"]},
}

# Families whose own parameters already fix the bet team's role.
ROLE_FIXED = {"spread_band", "ml_band", "total_band"}


@dataclass(frozen=True)
class BetCandidate:
    family: str
    params: Tuple[Tuple[str, object], ...]

    @property
    def as_dict(self) -> Dict[str, object]:
        return dict(self.params)

    @property
    def key(self) -> str:
        return f"{self.family}(" + ",".join(f"{k}={v}" for k, v in self.params) + ")"


def _other(team: str) -> str:
    return "away" if team == "home" else "home"


def _team_spread(f: Features, team: str) -> np.ndarray:
    """The spread from the named team's side: negative when that team is favoured."""
    return f["home_spread"] if team == "home" else -f["home_spread"]


def _in_band(values: np.ndarray, edges: Tuple[float, ...], band: int) -> np.ndarray:
    return (values >= edges[band]) & (values < edges[band + 1])


def _spread_band(f: Features, p: Dict[str, object]) -> Tuple[str, np.ndarray]:
    team = str(p["team"])
    s = _team_spread(f, team)
    role = (s < 0) if p["role"] == "fav" else (s > 0)
    return f"{team}_ats", role & _in_band(np.abs(s), SPREAD_BANDS, int(p["band"]))


def _total_band(f: Features, p: Dict[str, object]) -> Tuple[str, np.ndarray]:
    return str(p["side"]), _in_band(f["total"], TOTAL_BANDS, int(p["band"]))


def _ml_band(f: Features, p: Dict[str, object]) -> Tuple[str, np.ndarray]:
    team = str(p["team"])
    prob = f["home_prob"] if team == "home" else 1.0 - f["home_prob"]
    return f"{team}_ml", _in_band(prob, PROB_BANDS, int(p["band"]))


def _line_move(f: Features, p: Dict[str, object]) -> Tuple[str, np.ndarray]:
    thr = float(p["threshold"])
    follow = p["action"] == "follow"
    if p["market"] == "total":
        moved = f["total_move"] >= thr if p["direction"] == "up" else f["total_move"] <= -thr
        side = ("over" if follow else "under") if p["direction"] == "up" else ("under" if follow else "over")
        return side, moved
    # Spread moving down means the home side became more favoured - money came in on home.
    moved = f["spread_move"] <= -thr if p["direction"] == "down" else f["spread_move"] >= thr
    money_on = "home" if p["direction"] == "down" else "away"
    team = money_on if follow else _other(money_on)
    return f"{team}_ats", moved


def _rest(f: Features, p: Dict[str, object]) -> Tuple[str, np.ndarray]:
    h, a = f["home_rest"], f["away_rest"]
    masks = {
        "home_b2b": (h == 1) & (a > 1),
        "away_b2b": (a == 1) & (h > 1),
        "both_b2b": (h == 1) & (a == 1),
        "home_rest_adv": (h - a) >= 2,
        "away_rest_adv": (a - h) >= 2,
    }
    return str(p["bet"]), masks[str(p["scenario"])]


def _team_bet(team: str, bet: str) -> str:
    who = team if bet.startswith("team") else _other(team)
    return f"{who}_{bet.split('_')[1]}"


def _streak(f: Features, p: Dict[str, object]) -> Tuple[str, np.ndarray]:
    team, k = str(p["team"]), int(p["k"])
    s = f[f"{team}_streak"]
    mask = s >= k if p["kind"] == "won" else s <= -k
    return _team_bet(team, str(p["bet"])), mask


def _ats_form(f: Features, p: Dict[str, object]) -> Tuple[str, np.ndarray]:
    team, window, level = str(p["team"]), int(p["window"]), int(p["level"])
    rate = f[f"{team}_cover_{window}"]
    hot = (0.7, 0.8)[level]
    cold = (0.3, 0.2)[level]
    with np.errstate(invalid="ignore"):
        mask = rate >= hot if p["state"] == "hot" else rate <= cold
    mask = mask & ~np.isnan(rate)
    backed = team if p["action"] == "follow" else _other(team)
    return f"{backed}_ats", mask


FAMILIES: Dict[str, Callable[[Features, Dict[str, object]], Tuple[str, np.ndarray]]] = {
    "spread_band": _spread_band,
    "total_band": _total_band,
    "ml_band": _ml_band,
    "line_move": _line_move,
    "rest": _rest,
    "streak": _streak,
    "ats_form": _ats_form,
}


def _base_side(family: str, params: Dict[str, object]) -> str:
    """The side a system bets, worked out from its parameters without any data."""
    if family == "total_band":
        return str(params["side"])
    if family == "rest":
        return str(params["bet"])
    if family == "line_move":
        if params["market"] == "total":
            up, follow = params["direction"] == "up", params["action"] == "follow"
            return "over" if up == follow else "under"
        money_on = "home" if params["direction"] == "down" else "away"
        return f"{money_on if params['action'] == 'follow' else _other(money_on)}_ats"
    if family == "streak":
        return _team_bet(str(params["team"]), str(params["bet"]))
    if family == "ats_form":
        team = str(params["team"])
        return f"{team if params['action'] == 'follow' else _other(team)}_ats"
    if family in ("spread_band",):
        return f"{params['team']}_ats"
    return f"{params['team']}_ml"


def generate_candidates(limit: int | None = None) -> List[BetCandidate]:
    """Every system in the search space."""
    out: List[BetCandidate] = []
    for family, grid in GRIDS.items():
        names = list(grid)
        for values in product(*(grid[n] for n in names)):
            base = dict(zip(names, values))
            side = _base_side(family, base)
            team_bet = not side.startswith(("over", "under"))
            roles = ("any",) if family in ROLE_FIXED or not team_bet else ROLES
            for phase, role in product(PHASES, roles):
                params = dict(base, phase=phase, bet_role=role)
                out.append(BetCandidate(family, tuple(sorted(params.items()))))
                if limit and len(out) >= limit:
                    return out
    return out


def side_and_mask(candidate: BetCandidate, f: Features) -> Tuple[str, np.ndarray]:
    """The side a system bets and the games it bets on, using pre-game information only."""
    params = candidate.as_dict
    side, mask = FAMILIES[candidate.family](f, params)
    phase = params.get("phase", "all")
    if phase == "early":
        mask = mask & (f["phase"] < 0.2)
    elif phase == "mid":
        mask = mask & (f["phase"] >= 0.2) & (f["phase"] <= 0.8)
    elif phase == "late":
        mask = mask & (f["phase"] > 0.8)
    role = params.get("bet_role", "any")
    if role != "any" and not side.startswith(("over", "under")):
        s = _team_spread(f, side.split("_")[0])
        mask = mask & ((s < 0) if role == "fav" else (s > 0))
    return side, np.nan_to_num(mask.astype(float), nan=0.0).astype(bool)


def neighbours(candidate: BetCandidate) -> List[BetCandidate]:
    """Systems one step away in a single ordered parameter (band, threshold, streak length...)."""
    grid = GRIDS[candidate.family]
    base = candidate.as_dict
    out: List[BetCandidate] = []
    for name in ORDINAL & set(grid):
        values = grid[name]
        i = values.index(base[name])
        for j in (i - 1, i + 1):
            if 0 <= j < len(values):
                params = dict(base)
                params[name] = values[j]
                out.append(BetCandidate(candidate.family, tuple(sorted(params.items()))))
    return out
