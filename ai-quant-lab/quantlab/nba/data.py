"""NBA scores and closing lines, 2007-08 onward, from a free public archive.

Source: sportsbookreviewsonline.com NBA odds archives. Each season page is one
table with two rows per game - visitor then home - carrying the final score,
the opening and closing line, and the closing moneyline.

The line columns hold the spread on the favourite's row and the total on the
other row, so which number is which has to be inferred per game: the total is
always the larger of the two.
"""

from __future__ import annotations

import io
import os
import re
import urllib.request
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ..data import CACHE_DIR

SEASONS: List[str] = [f"{y}-{str(y + 1)[2:]}" for y in range(2007, 2023)]
SOURCE_URL = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-{season}"

GAME_COLUMNS = [
    "date", "season", "home", "away", "home_score", "away_score",
    "home_spread_open", "home_spread_close", "total_open", "total_close",
    "home_ml", "away_ml",
]

_LEADING_NUMBER = re.compile(r"^[-+]?\d+(?:\.\d+)?")


def parse_number(value: object) -> float:
    """Parse one archive cell. 'pk' is a zero spread; 'NL', blanks and junk are missing.

    Some cells carry a price suffix after the line (for example '197.5u10'),
    so only the leading number is kept.
    """
    if value is None:
        return float("nan")
    if isinstance(value, float) and np.isnan(value):
        return float("nan")
    text = str(value).strip()
    if text.lower() == "pk":
        return 0.0
    match = _LEADING_NUMBER.match(text)
    return float(match.group(0)) if match else float("nan")


def american_to_decimal(american: float) -> float:
    """American moneyline to decimal odds (stake included). -200 -> 1.5, +150 -> 2.5."""
    if american is None or np.isnan(american) or american == 0:
        return float("nan")
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def _split_line(visitor_value: float, home_value: float) -> tuple[float, float]:
    """Return (home_spread, total) from the two raw line cells of one game.

    The larger number is the total; the smaller is the spread and sits on the
    favourite's row. The home spread is negative when the home team is favoured.
    """
    v, h = visitor_value, home_value
    if np.isnan(v) or np.isnan(h):
        return float("nan"), float("nan")
    if v > h:
        return -h, v
    return v, h


def parse_rows(rows: Sequence[Sequence[object]], season: str) -> pd.DataFrame:
    """Turn one season's raw table rows (header excluded) into one row per game.

    Dates in the archive are MMDD with no year. A season starts in the autumn
    of its first year; the year rolls over the first time the month drops (for
    example December to January). The 2019-20 season resumed in July-October
    2020, which this rule places correctly because the month only rises.
    """
    start_year = int(season[:4])
    year, prev_month = start_year, None
    games: List[dict] = []
    i = 0
    while i + 1 < len(rows):
        v_row, h_row = rows[i], rows[i + 1]
        if str(v_row[2]).strip().upper() != "V" or str(h_row[2]).strip().upper() != "H":
            i += 1  # out of step - resynchronise on the next row
            continue
        i += 2

        raw_date = str(h_row[0]).split(".")[0].strip()
        if not raw_date.isdigit() or len(raw_date) < 3:
            continue
        month, day = int(raw_date[:-2]), int(raw_date[-2:])
        if prev_month is not None and month < prev_month:
            year += 1
        prev_month = month
        try:
            date = pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            continue

        spread_open, total_open = _split_line(parse_number(v_row[9]), parse_number(h_row[9]))
        spread_close, total_close = _split_line(parse_number(v_row[10]), parse_number(h_row[10]))
        games.append(
            {
                "date": date,
                "season": season,
                "home": str(h_row[3]).strip(),
                "away": str(v_row[3]).strip(),
                "home_score": parse_number(h_row[8]),
                "away_score": parse_number(v_row[8]),
                "home_spread_open": spread_open,
                "home_spread_close": spread_close,
                "total_open": total_open,
                "total_close": total_close,
                "home_ml": american_to_decimal(parse_number(h_row[11])),
                "away_ml": american_to_decimal(parse_number(v_row[11])),
            }
        )
    df = pd.DataFrame(games, columns=GAME_COLUMNS)
    return df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)


def fetch_season(season: str, timeout: int = 30) -> pd.DataFrame:
    """Download and parse one season from the public archive."""
    request = urllib.request.Request(SOURCE_URL.format(season=season), headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(request, timeout=timeout).read().decode("utf-8", "replace")
    table = max(pd.read_html(io.StringIO(html), header=None), key=len)
    rows = table.iloc[1:].to_numpy(dtype=object).tolist()
    return parse_rows(rows, season)


def load_seasons(seasons: Optional[Sequence[str]] = None, use_cache: bool = True) -> pd.DataFrame:
    """All requested seasons as one chronologically sorted frame, cached as CSV."""
    frames: List[pd.DataFrame] = []
    for season in seasons or SEASONS:
        path = os.path.join(CACHE_DIR, f"nba-{season}.csv")
        if use_cache and os.path.exists(path):
            frame = pd.read_csv(path, parse_dates=["date"])
        else:
            frame = fetch_season(season)
            if use_cache:
                os.makedirs(CACHE_DIR, exist_ok=True)
                frame.to_csv(path, index=False)
        frames.append(frame)
    games = pd.concat(frames, ignore_index=True)
    return games.sort_values(["date", "home"], kind="stable").reset_index(drop=True)


MARGIN_SD = 12.0
MONEYLINE_OVERROUND = 1.045


def fair_moneylines(home_spread: np.ndarray, margin_sd: float = MARGIN_SD,
                    overround: float = MONEYLINE_OVERROUND) -> tuple[np.ndarray, np.ndarray]:
    """Decimal moneylines implied by the spread, plus a typical bookmaker margin.

    If the home margin is Normal(-spread, margin_sd), the home side wins with
    probability Phi(-spread / margin_sd). Pricing both sides from that and then
    shortening them by the overround gives moneylines that agree exactly with
    the spread market.
    """
    from math import erf, sqrt

    z = -np.nan_to_num(np.asarray(home_spread, dtype=float), nan=0.0) / margin_sd
    p_home = 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))
    p_home = np.clip(p_home, 0.005, 0.995)
    return 1.0 / (p_home * overround), 1.0 / ((1.0 - p_home) * overround)


def simulate_outcomes(games: pd.DataFrame, seed: int = 0, margin_sd: float = MARGIN_SD, total_sd: float = 18.0) -> pd.DataFrame:
    """Replace every score with one drawn from a world where the closing lines are exactly right.

    The home margin is centred on the closing spread and the total on the
    closing total, so every spread and total bet is a fair coin flip before the
    bookmaker's margin. Moneylines are re-priced from the spread with the same
    model, so all three markets agree and none can be beaten either. No betting
    system has an edge in this world - it is the control that shows what a
    strategy search finds when there is nothing to find.
    """
    rng = np.random.default_rng(seed)
    out = games.copy()
    out["home_ml"], out["away_ml"] = fair_moneylines(out["home_spread_close"].to_numpy(float), margin_sd)
    spread = out["home_spread_close"].fillna(0.0).to_numpy()
    total = out["total_close"].fillna(out["total_close"].median()).to_numpy()
    margin = rng.normal(-spread, margin_sd)
    margin = np.where(np.abs(margin) < 0.5, np.sign(margin + 1e-9) * 1.0, margin)  # no ties in the NBA
    points = np.maximum(rng.normal(total, total_sd), np.abs(margin) + 100.0)
    out["home_score"] = np.round((points + margin) / 2.0)
    out["away_score"] = np.round((points - margin) / 2.0)
    tied = out["home_score"] == out["away_score"]
    out.loc[tied, "home_score"] += np.where(margin[tied.to_numpy()] > 0, 1, -1)
    return out


def synthetic_games(n_seasons: int = 6, games_per_team: int = 82, n_teams: int = 30, seed: int = 0) -> pd.DataFrame:
    """A fake league with realistic lines and fair outcomes - used by the tests."""
    rng = np.random.default_rng(seed)
    teams = [f"Team{i:02d}" for i in range(n_teams)]
    per_day = n_teams // 2
    days = games_per_team * n_teams // (2 * per_day) + 1
    rows: List[dict] = []
    for s in range(n_seasons):
        start = pd.Timestamp(year=2007 + s, month=10, day=28)
        season = f"{2007 + s}-{str(2008 + s)[2:]}"
        strength = rng.normal(0, 4, n_teams)
        for d in range(days):
            order = rng.permutation(n_teams)
            date = start + pd.Timedelta(days=int(d * 1.9))
            for g in range(per_day):
                home, away = order[2 * g], order[2 * g + 1]
                spread = float(np.round((strength[away] - strength[home] - 3.0) * 2) / 2)
                total_line = float(np.round(rng.normal(210, 10) * 2) / 2)
                rows.append(
                    {
                        "date": date, "season": season,
                        "home": teams[home], "away": teams[away],
                        "home_score": np.nan, "away_score": np.nan,
                        "home_spread_open": spread + float(rng.choice([-1, -0.5, 0, 0, 0.5, 1])),
                        "home_spread_close": spread,
                        "total_open": total_line + float(rng.choice([-2, -1, 0, 0, 1, 2])),
                        "total_close": total_line,
                        "home_ml": np.nan,  # priced from the spread by simulate_outcomes
                        "away_ml": np.nan,
                    }
                )
    games = pd.DataFrame(rows, columns=GAME_COLUMNS)
    return simulate_outcomes(games, seed=seed + 1).sort_values(["date", "home"], kind="stable").reset_index(drop=True)
