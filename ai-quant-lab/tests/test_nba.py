"""NBA betting-system tests: parsing, settlement, look-ahead, and the no-edge control."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from quantlab.nba import data, features, strategies
from quantlab.nba.pipeline import bootstrap_roi, run_bet_pipeline
from quantlab.nba.report import to_markdown


def _row(date, vh, team, final, open_, close, ml):
    """One archive row: Date Rot VH Team 1st 2nd 3rd 4th Final Open Close ML 2H."""
    return [date, 0, vh, team, 0, 0, 0, 0, final, open_, close, ml, 0]


# --- parsing --------------------------------------------------------------------------------


def test_parse_number_handles_archive_quirks():
    assert data.parse_number("pk") == 0.0
    assert data.parse_number("PK") == 0.0
    assert np.isnan(data.parse_number("NL"))
    assert np.isnan(data.parse_number(""))
    assert np.isnan(data.parse_number(float("nan")))
    assert data.parse_number("+145") == 145.0
    assert data.parse_number("197.5u10") == 197.5
    assert data.parse_number(" -3.5 ") == -3.5


def test_american_to_decimal():
    assert abs(data.american_to_decimal(-200) - 1.5) < 1e-12
    assert abs(data.american_to_decimal(150) - 2.5) < 1e-12
    assert np.isnan(data.american_to_decimal(0))
    assert np.isnan(data.american_to_decimal(float("nan")))


def test_parse_rows_assigns_spread_to_the_favourite_and_total_to_the_other_row():
    rows = [
        _row("1016", "V", "Philadelphia", 87, "208.5", "211.5", "170"),
        _row("1016", "H", "Boston", 105, "5", "4.5", "-200"),       # home favoured by 4.5
        _row("1017", "V", "Milwaukee", 113, "1.5", "3", "-165"),     # visitor favoured by 3
        _row("1017", "H", "Indiana", 100, "220", "221.5", "+145"),
        _row("1018", "V", "Utah", 99, "pk", "pk", "-110"),           # pick'em
        _row("1018", "H", "Denver", 101, "205", "206", "-110"),
    ]
    g = data.parse_rows(rows, "2018-19")
    assert list(g["home"]) == ["Boston", "Indiana", "Denver"]
    assert g.loc[0, "home_spread_close"] == -4.5 and g.loc[0, "total_close"] == 211.5
    assert g.loc[1, "home_spread_close"] == 3.0 and g.loc[1, "total_close"] == 221.5
    assert g.loc[2, "home_spread_close"] == 0.0 and g.loc[2, "total_close"] == 206.0
    assert abs(g.loc[0, "home_ml"] - 1.5) < 1e-12
    assert abs(g.loc[1, "home_ml"] - 2.45) < 1e-12


def test_parse_rows_rolls_the_year_over_and_handles_the_2020_bubble():
    dates = ["1230", "0102", "0311", "0730", "1011"]
    rows = []
    for d in dates:
        rows += [_row(d, "V", "A", 100, "3", "3", "130"), _row(d, "H", "B", 101, "210", "210", "-150")]
    g = data.parse_rows(rows, "2019-20")
    assert [str(x.date()) for x in g["date"]] == ["2019-12-30", "2020-01-02", "2020-03-11", "2020-07-30", "2020-10-11"]


def test_parse_rows_resynchronises_after_a_broken_pair():
    rows = [
        _row("1101", "H", "Orphan", 90, "4", "4", "-170"),  # a home row with no visitor row before it
        _row("1102", "V", "A", 100, "210", "211", "+120"),
        _row("1102", "H", "B", 104, "2", "2.5", "-140"),
    ]
    g = data.parse_rows(rows, "2010-11")
    assert len(g) == 1 and g.loc[0, "home"] == "B"


# --- settlement -----------------------------------------------------------------------------


def _one_game(home_score, away_score, spread=-4.0, total=210.0, home_ml=1.5, away_ml=2.7):
    return pd.DataFrame([{
        "date": pd.Timestamp("2015-11-01"), "season": "2015-16", "home": "H", "away": "A",
        "home_score": home_score, "away_score": away_score,
        "home_spread_open": spread, "home_spread_close": spread,
        "total_open": total, "total_close": total, "home_ml": home_ml, "away_ml": away_ml,
    }])


def test_spread_bets_win_lose_and_push_correctly():
    win = features.payoffs(_one_game(110, 100))       # home wins by 10, laid 4
    assert abs(win["home_ats"][0] - 100 / 110) < 1e-12 and win["away_ats"][0] == -1.0
    push = features.payoffs(_one_game(104, 100))      # wins by exactly 4
    assert push["home_ats"][0] == 0.0 and push["away_ats"][0] == 0.0


def test_moneyline_and_totals_settle_at_the_closing_price():
    p = features.payoffs(_one_game(100, 105, total=200.0))
    assert p["home_ml"][0] == -1.0
    assert abs(p["away_ml"][0] - 1.7) < 1e-12
    assert abs(p["over"][0] - 100 / 110) < 1e-12 and p["under"][0] == -1.0


def test_worse_prices_pay_less():
    base = features.payoffs(_one_game(110, 100))
    worse = features.payoffs(_one_game(110, 100), features.Prices().stressed())
    assert worse["home_ats"][0] < base["home_ats"][0]
    assert worse["home_ml"][0] < base["home_ml"][0]


def test_missing_lines_cannot_be_bet():
    p = features.payoffs(_one_game(110, 100, spread=float("nan"), total=float("nan"), home_ml=float("nan")))
    assert np.isnan(p["home_ats"][0]) and np.isnan(p["over"][0]) and np.isnan(p["home_ml"][0])


# --- look-ahead -----------------------------------------------------------------------------


def _schedule(results):
    """Team H hosts A on consecutive days; results are home margins."""
    rows = []
    for i, margin in enumerate(results):
        rows.append({
            "date": pd.Timestamp("2016-11-01") + pd.Timedelta(days=i), "season": "2016-17",
            "home": "H", "away": "A", "home_score": 100 + margin, "away_score": 100,
            "home_spread_open": -2.0, "home_spread_close": -2.0, "total_open": 200.0, "total_close": 200.0,
            "home_ml": 1.8, "away_ml": 2.1,
        })
    return pd.DataFrame(rows)


def test_features_for_a_game_never_include_its_own_result():
    a = features.build_features(_schedule([5, 5, 5, 5]))
    b = features.build_features(_schedule([5, 5, 5, -5]))  # only the last result differs
    for name in a:
        assert np.allclose(a[name][:4], b[name][:4], equal_nan=True), name


def test_streak_and_rest_reflect_only_earlier_games():
    f = features.build_features(_schedule([5, 5, -5, 5]))
    assert list(f["home_streak"]) == [0, 1, 2, -1]
    assert list(f["away_streak"]) == [0, -1, -2, 1]
    assert f["home_rest"][0] == features.MAX_REST and f["home_rest"][1] == 1


def test_cover_form_needs_a_full_window():
    f = features.build_features(_schedule([5] * 7))
    assert np.isnan(f["home_cover_5"][:5]).all()
    assert f["home_cover_5"][5] == 1.0 and f["away_cover_5"][5] == 0.0


# --- systems --------------------------------------------------------------------------------


def test_search_space_is_large_and_every_system_has_a_valid_side():
    cands = strategies.generate_candidates()
    assert len(cands) > 2000
    f = features.build_features(data.synthetic_games(n_seasons=2, seed=1))
    for c in cands[::37]:
        side, mask = strategies.side_and_mask(c, f)
        assert side in features.SIDES and mask.dtype == bool


def test_neighbours_differ_in_exactly_one_ordered_parameter():
    cand = next(c for c in strategies.generate_candidates() if c.family == "line_move")
    for nb in strategies.neighbours(cand):
        diffs = [k for k in cand.as_dict if cand.as_dict[k] != nb.as_dict[k]]
        assert len(diffs) == 1 and diffs[0] in strategies.ORDINAL


# --- the control ----------------------------------------------------------------------------


def test_the_no_edge_world_costs_the_bookmakers_margin_on_every_side():
    p = features.payoffs(data.synthetic_games(n_seasons=16, seed=5))
    for side in features.SIDES:
        assert -0.08 < np.nanmean(p[side]) < -0.01, side


def test_bootstrap_interval_is_ordered():
    r = np.random.default_rng(0).choice([100 / 110, -1.0], size=500)
    out = bootstrap_roi(r)
    assert out["roi_p05"] <= out["roi_median"] <= out["roi_p95"]


def test_no_system_survives_when_no_edge_exists():
    """If this ever fails, the funnel is fooling itself."""
    for seed in (2, 7):
        res = run_bet_pipeline(data.synthetic_games(n_seasons=6, seed=seed), verbose=False)
        assert res.survivors == []
        counts = [s.survivors for s in res.funnel]
        assert counts == sorted(counts, reverse=True)
        assert res.best_overall and res.walk_forward


def test_report_renders_every_section():
    res = run_bet_pipeline(data.synthetic_games(n_seasons=5, seed=4), verbose=False)
    md = to_markdown(res, res)
    for heading in ("Bottom line", "The funnel", "What survived", "season by season", "Control"):
        assert heading in md
