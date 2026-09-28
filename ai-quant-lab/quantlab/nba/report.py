"""Markdown report for a betting-system search."""

from __future__ import annotations

from typing import List, Optional

from ..report import funnel_table
from .pipeline import BetResult


def _pct(x: float) -> str:
    return f"{x:+.1%}"


def survivors_table(result: BetResult) -> str:
    if not result.survivors:
        return "_No betting system survived every stage._"
    rows: List[str] = [
        "| System | In-sample ROI | Held-out ROI | Bets (in / out) | Deflated Sharpe (prob. real) |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in result.survivors:
        rows.append(f"| `{s['key']}` | {_pct(s['is_roi'])} | {_pct(s['oos_roi'])} | "
                    f"{s['is_bets']:,} / {s['oos_bets']:,} | {s['dsr']:.3f} |")
    return "\n".join(rows)


def bottom_line(result: BetResult, control: Optional[BetResult] = None) -> str:
    lines: List[str] = []
    if result.baseline:
        worst = min(result.baseline.values())
        best = max(result.baseline.values())
        lines.append(
            f"- The starting line: betting every game on any one side returns between **{_pct(worst)}** and "
            f"**{_pct(best)}** per bet. That gap below zero is the bookmaker's margin every system has to beat first."
        )
    b = result.best_overall
    if b:
        lines.append(
            f"- Best-looking system in the whole search: `{b['key']}` — **{_pct(b['is_roi'])}** per bet on "
            f"{b['is_bets']:,} in-sample bets, then **{_pct(b['oos_roi'])}** on {b['oos_bets']:,} bets in seasons "
            f"it never saw. Probability its edge is real after correcting for {result.n_candidates:,} systems: "
            f"**{b['dsr']:.3f}**."
        )
    if control is not None and control.best_overall:
        lines.append(
            f"- Control: the same search on seasons where every line is exactly right, so no system can win. "
            f"Its best-looking system still showed **{_pct(control.best_overall['is_roi'])}** per bet in-sample. "
            f"That is what pure luck looks like at this search size."
        )
    wf = result.walk_forward
    if wf:
        lines.append(
            f"- Running the process for real — each season, bet the system with the best record so far: "
            f"**{wf['units']:+.1f} units** over {wf['bets']:,} bets (**{_pct(wf['roi'])}** per bet)."
        )
    if result.pbo:
        lines.append(f"- Probability of backtest overfitting: **{result.pbo.get('pbo', float('nan')):.0%}** "
                     f"(above 50% means the in-sample winner does worse than a coin flip on new data).")
    if not result.survivors:
        lines.append("- **Nothing survived.**")
    return "\n".join(lines)


def to_markdown(result: BetResult, control: Optional[BetResult] = None) -> str:
    parts = [
        f"# Betting system search - {result.label}",
        "",
        f"Seasons **{result.seasons[0]} to {result.seasons[-1]}** ({result.n_games:,} games). "
        f"Systems tested: **{result.n_candidates:,}**. In-sample {result.is_seasons[0]}–{result.is_seasons[-1]}, "
        f"held out {result.oos_seasons[0]}–{result.oos_seasons[-1]}. Every bet settled at the closing line; "
        "spreads and totals at -110.",
        "",
        "## Bottom line",
        "",
        bottom_line(result, control),
        "",
        "## The funnel",
        "",
        funnel_table(result),
        "",
        "## What survived",
        "",
        survivors_table(result),
        "",
    ]
    wf = result.walk_forward
    if wf.get("picks"):
        parts += ["## What the process bet, season by season", "",
                  "| Season | System chosen (best record before this season) | Record before | Bets | Units | ROI |",
                  "|---|---|---:|---:|---:|---:|"]
        for p in wf["picks"]:
            parts.append(f"| {p['season']} | `{p['system']}` | {_pct(p['train_roi'])} | {p['bets']} | "
                         f"{p['units']:+.1f} | {_pct(p['roi'])} |")
        parts += ["", "_A real edge gets picked again and again. A process chasing noise picks something new "
                      "every season and pays for it._", ""]
    if control is not None:
        parts += ["## Control: the same funnel where no edge exists", "", funnel_table(control), ""]
    for note in result.notes:
        parts += [f"> {note}", ""]
    parts += [
        "## How to read this",
        "",
        "- **ROI per bet**: profit per 1-unit bet. -4.5% is what betting blind at -110 costs you.",
        "- **Units**: total profit in bet-sized units. +10 units betting $100 a game is +$1,000.",
        "- **Deflated Sharpe**: probability the edge is real once you admit how many systems you tried. "
        "Below 0.95, treat it as luck.",
        "- **Held-out seasons**: never used to choose anything. The only honest test of a system is how it does there.",
        "",
        "Nothing here is betting advice.",
        "",
    ]
    return "\n".join(parts)
