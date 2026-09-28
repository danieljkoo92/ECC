"""End-to-end tests. The important one is the last: noise in, nothing out."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantlab import data, report
from quantlab.pipeline import Thresholds, run_pipeline
from quantlab.strategies import generate_candidates


def _small_run(seed: int, n: int = 1200):
    df = data.synthetic(n=n, seed=seed)
    cands = generate_candidates(limit=300)
    return run_pipeline(df, f"SIM{seed}", candidates=cands, verbose=False)


def test_pipeline_reports_every_stage_in_order():
    res = _small_run(1)
    names = [s.name for s in res.funnel]
    assert names[0] == "universe"
    assert names[-1] == "regime_consistency"
    assert "deflated_sharpe" in names


def test_funnel_survivor_counts_never_increase():
    res = _small_run(2)
    counts = [s.survivors for s in res.funnel]
    assert counts == sorted(counts, reverse=True)


def test_random_data_produces_no_survivors():
    """The control experiment. If this ever fails, the pipeline is fooling itself."""
    for seed in (3, 4, 5):
        assert _small_run(seed).survivors == []


def test_report_renders_without_survivors():
    md = report.to_markdown(_small_run(6))
    assert "# Strategy search report" in md
    assert "No strategy survived" in md


def test_buy_and_hold_is_always_reported_for_comparison():
    res = _small_run(7)
    assert res.buy_hold["n_bars"] > 0
    assert "sharpe" in res.buy_hold


def test_best_overall_is_always_reported_even_when_nothing_survives():
    """The headline backtest number must be visible next to its luck benchmark."""
    res = _small_run(8)
    best = res.best_overall
    assert best["key"]
    assert best["sr_benchmark"] > 0  # a search this size produces luck, by construction
    assert 0.0 <= best["dsr"] <= 1.0
    assert "Best-looking backtest" in report.to_markdown(res)


def test_every_stage_is_reported_even_after_the_funnel_empties():
    names = [s.name for s in _small_run(9).funnel]
    assert names == [
        "universe",
        "in_sample_screen",
        "plateau",
        "cost_stress",
        "out_of_sample",
        "monte_carlo",
        "deflated_sharpe",
        "regime_consistency",
    ]
