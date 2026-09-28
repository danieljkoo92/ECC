"""Turn a pipeline Result into a markdown report a non-quant can read."""

from __future__ import annotations

from typing import List, Optional

from .pipeline import Result


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _num(x: float) -> str:
    if x == float("inf"):
        return "inf"
    return f"{x:.2f}"


def funnel_table(result: Result) -> str:
    rows = ["| Stage | What it tests | Left | Killed |", "|---|---|---:|---:|"]
    for s in result.funnel:
        rows.append(f"| `{s.name}` | {s.description} | {s.survivors:,} | {s.killed:,} |")
    return "\n".join(rows)


def survivors_table(result: Result) -> str:
    if not result.survivors:
        return "_No strategy survived every stage._"
    head = (
        "| Strategy | IS Sharpe | OOS Sharpe | Deflated Sharpe (prob. real) | Luck benchmark | CAGR | Max DD | Trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    )
    rows: List[str] = list(head)
    for s in result.survivors:
        rows.append(
            f"| `{s['key']}` | {_num(s['is_sharpe'])} | {_num(s['oos_sharpe'])} | {s['dsr']:.3f} | "
            f"{_num(s['sr_benchmark'])} | {_pct(s['cagr'])} | {_pct(s['max_drawdown'])} | {int(s['n_trades'])} |"
        )
    return "\n".join(rows)


def verdict(result: Result, control: Optional[Result] = None) -> str:
    lines: List[str] = []
    wf = result.walk_forward
    bh = result.buy_hold
    if wf:
        lines.append(
            f"- Walk-forward of the **whole selection process** (pick the best strategy on past data, "
            f"trade it on the next unseen window, repeat {int(wf['n_windows'])}x): "
            f"**{_pct(wf['total_return'])} total return, Sharpe {_num(wf['sharpe'])}, "
            f"max drawdown {_pct(wf['max_drawdown'])}**."
        )
    if bh:
        lines.append(
            f"- Buy and hold over the same history: **{_pct(bh['total_return'])} total return, "
            f"Sharpe {_num(bh['sharpe'])}, max drawdown {_pct(bh['max_drawdown'])}**."
        )
    best = result.best_overall
    if best:
        lines.append(
            f"- Best-looking backtest in the entire search: `{best['key']}`, in-sample Sharpe "
            f"**{_num(best['is_sharpe'])}**, which falls to **{_num(best['oos_sharpe'])}** on the held-out data. "
            f"A search of {result.n_candidates:,} strategies is expected to throw up a Sharpe of "
            f"**{_num(best['sr_benchmark'])}** from pure luck, and its deflated Sharpe - the probability the "
            f"edge is real - is **{best['dsr']:.3f}**."
        )
    if result.pbo:
        pbo = result.pbo.get("pbo", float("nan"))
        lines.append(
            f"- Probability of backtest overfitting (PBO) across {int(result.pbo.get('n_combinations', 0))} "
            f"history splits: **{pbo:.0%}**. Above 50% means the in-sample winner is worse than a coin flip out of sample."
        )
    if control is not None:
        surv = len(control.survivors)
        cwf = control.walk_forward
        lines.append(
            f"- Control run on **pure random data** (a coin-flip price series with no edge to find): "
            f"the same funnel produced **{surv} survivor(s)**"
            + (f" and a walk-forward Sharpe of {_num(cwf['sharpe'])}" if cwf else "")
            + ". Anything the real run produces has to beat this to mean anything."
        )
    if not result.survivors:
        lines.append(
            "- **Nothing survived.** That is a real result, not a failure of the code: after paying costs and "
            "correcting for how many strategies were tried, none of the winners are distinguishable from luck."
        )
    else:
        best = result.survivors[0]
        lines.append(
            f"- Best surviving candidate: `{best['key']}`, deflated Sharpe {best['dsr']:.3f} "
            f"(the probability its edge is real, after correcting for {int(result.n_candidates):,} strategies tested)."
        )
    return "\n".join(lines)


def to_markdown(result: Result, control: Optional[Result] = None) -> str:
    parts = [
        f"# Strategy search report - {result.symbol}",
        "",
        f"History: **{result.start} to {result.end}** ({result.bars:,} bars). "
        f"Search space: **{result.n_candidates:,} strategies**.",
        "",
        "## Bottom line",
        "",
        verdict(result, control),
        "",
        "## The funnel",
        "",
        "Each stage is a test the previous stage's winners have to pass again.",
        "",
        funnel_table(result),
        "",
        "## What survived",
        "",
        survivors_table(result),
        "",
    ]
    if result.walk_forward.get("picks"):
        parts += [
            "## What the process picked, window by window",
            "",
            *[f"{i + 1}. `{p}`" for i, p in enumerate(result.walk_forward["picks"])],
            "",
            "_If these picks have nothing in common, the process is chasing noise: "
            "every window it re-learns a different 'edge'._",
            "",
        ]
    if control is not None:
        parts += [
            "## Control: the same funnel on random data",
            "",
            funnel_table(control),
            "",
        ]
    for note in result.notes:
        parts += [f"> {note}", ""]
    parts += [
        "## How to read this",
        "",
        "- **Sharpe ratio**: return per unit of risk. Above 1 is good, above 2 in a backtest usually means overfitting.",
        "- **Deflated Sharpe**: the probability the edge is real once you admit how many strategies you tried. "
        "Test 5,000 strategies and the best one will look brilliant by luck alone - this is the number that strips that out.",
        "- **PBO**: how often the in-sample winner underperforms out of sample across every way of splitting the history.",
        "- **Luck benchmark**: the Sharpe ratio a search this size is expected to produce from a strategy with no edge at all.",
        "",
    ]
    return "\n".join(parts)
