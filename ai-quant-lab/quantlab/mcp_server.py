"""MCP server: run the market and NBA truth filters from a Claude chat.

Claude Desktop (or any MCP client) launches this over stdio. Three tools:

* test_trading_strategies   - the 4,752-strategy market search and funnel
* test_nba_betting_systems  - the 2,120-system NBA search and funnel
* get_report                - collect a run that was still going

A full run can take longer than a client is willing to wait on one call, so
every run executes in a background worker. A tool call waits up to
WAIT_SECONDS and returns the report if it is ready; otherwise it returns a job
id that get_report can be called with.

Command line:
    quantlab-mcp                         serve over stdio (what Claude runs)
    quantlab-mcp --selftest              quick offline check, then exit
    quantlab-mcp --install-claude-desktop --uvx-path PATH
                                         register this server with Claude Desktop
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import sys
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Literal, Optional

# Downloaded prices and NBA seasons are cached here, outside the install
# directory, so they survive reinstalls. Must be set before quantlab.data loads.
os.environ.setdefault("QUANTLAB_CACHE", str(Path.home() / ".ai-quant-lab" / "cache"))

from mcp.server import MCPServer  # noqa: E402

from . import __version__  # noqa: E402
from .claude_desktop import DEFAULT_SOURCE, ConfigError, install  # noqa: E402

WAIT_SECONDS = 40.0
_SYMBOL = re.compile(r"^[A-Za-z0-9=^./\-]{1,20}$")

INSTRUCTIONS = """\
ai-quant-lab tests whether trading strategies or NBA betting systems actually work.

It generates thousands of strategies or betting systems, backtests all of them
with realistic costs, then runs the winners through a validation funnel that
ends with the deflated Sharpe ratio - the correction for how many strategies
were tried. A control run on data with no edge shows what luck alone produces.

Use it when the user asks whether a trading bot, strategy, course, tipster or
betting system is real, or asks to test a market or NBA betting angles.

When presenting a report: lead with the verdict (how many survived), then the
best-looking strategy's in-sample result versus its held-out result, then the
luck benchmark or control, then what running the process for real would have
returned. Use plain language; the user may not know what a Sharpe ratio is.
Nothing here is financial or betting advice.

A first run downloads data and can take a minute or two. If a tool returns a
job id instead of a report, call get_report with it.
"""

server = MCPServer(name="ai-quant-lab", instructions=INSTRUCTIONS, version=__version__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="quantlab")
_jobs: Dict[str, dict] = {}


# --- the work ----------------------------------------------------------------------------------


def _market_report(symbol: str, start: str, source: str, include_control: bool) -> str:
    from . import data, report
    from .pipeline import run_pipeline
    from .strategies import generate_candidates

    df = data.load(symbol, source=source, start=start)
    if len(df) < 500:
        raise ValueError(f"only {len(df)} bars of data for {symbol}; need at least 500 (about two years)")
    candidates = generate_candidates()
    result = run_pipeline(df, symbol, candidates=candidates, verbose=False)
    control = None
    if include_control:
        control = run_pipeline(data.synthetic(n=len(df), seed=42), "RANDOM-CONTROL",
                               candidates=candidates, verbose=False)
    return report.to_markdown(result, control)


def season_range(first: str, last: str) -> List[str]:
    """Seasons from first to last inclusive, validated against the archive."""
    from .nba.data import SEASONS

    if first not in SEASONS or last not in SEASONS:
        raise ValueError(f"seasons must be between {SEASONS[0]} and {SEASONS[-1]}, written like 2015-16")
    i, j = SEASONS.index(first), SEASONS.index(last)
    if j - i < 3:
        raise ValueError("pick at least four seasons so there is something left over to test on")
    return SEASONS[i:j + 1]


def _nba_report(first_season: str, last_season: str, include_control: bool, demo: bool) -> str:
    from .nba import data as nba_data
    from .nba.pipeline import run_bet_pipeline
    from .nba.report import to_markdown

    if demo:
        games, label = nba_data.synthetic_games(n_seasons=8, seed=42), "NBA (demo league)"
    else:
        games, label = nba_data.load_seasons(season_range(first_season, last_season)), "NBA"
    result = run_bet_pipeline(games, label=label, verbose=False)
    control = None
    if include_control:
        control = run_bet_pipeline(nba_data.simulate_outcomes(games, seed=42), label="NBA control (no edge)",
                                   verbose=False)
    return to_markdown(result, control)


# --- jobs ---------------------------------------------------------------------------------------


def _start(kind: str, fn, *args) -> str:
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"kind": kind, "future": _executor.submit(fn, *args), "started": time.monotonic()}
    return job_id


async def _wait(job_id: str) -> str:
    job = _jobs.get(job_id)
    if job is None:
        return f"No run with job id '{job_id}'. Start a new one with test_trading_strategies or test_nba_betting_systems."
    future: Future = job["future"]
    try:
        await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), timeout=WAIT_SECONDS)
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - job["started"]
        return (f"Still running ({elapsed:.0f}s so far, {job['kind']}). "
                f"Call get_report with job_id '{job_id}' to collect the result.")
    except Exception:
        pass  # reported below
    error = future.exception()
    if error is not None:
        return f"The run failed: {error}"
    return future.result()


# --- tools --------------------------------------------------------------------------------------


@server.tool()
async def test_trading_strategies(
    symbol: str = "QQQ",
    start: str = "2005-01-01",
    include_control: bool = True,
    source: Literal["yfinance", "synthetic"] = "yfinance",
) -> str:
    """Search 4,752 trading strategies on one market and test whether any of them are real.

    Six strategy families (moving-average cross, breakout, RSI and Bollinger
    reversion, momentum, volatility breakout) are backtested with trading costs,
    then put through: in-sample screen, plateau test, 3x cost stress, held-out
    data, walk-forward of the selection process, Monte Carlo and the deflated
    Sharpe ratio. Daily data from Yahoo Finance.

    symbol: Yahoo Finance ticker. Examples: QQQ (Nasdaq 100), NQ=F (Nasdaq
        futures), SPY (S&P 500), ES=F (S&P futures), BTC-USD, ETH-USD,
        EURUSD=X, GC=F (gold), AAPL.
    start: first date of history, YYYY-MM-DD.
    include_control: also run the identical search on random data with no edge.
    source: "synthetic" runs an offline demo on fake data.
    """
    if not _SYMBOL.match(symbol):
        return "That doesn't look like a ticker. Use a Yahoo Finance symbol such as QQQ, NQ=F, BTC-USD or EURUSD=X."
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", start):
        return "start must be a date written YYYY-MM-DD, for example 2005-01-01."
    return await _wait(_start(f"market search on {symbol}", _market_report, symbol, start, source, include_control))


@server.tool()
async def test_nba_betting_systems(
    first_season: str = "2007-08",
    last_season: str = "2022-23",
    include_control: bool = True,
    demo: bool = False,
) -> str:
    """Search about 2,100 NBA betting systems and test whether any beat the bookmaker.

    Systems cover the angles betting services sell: follow or fade line moves,
    rested vs back-to-back teams, win and loss streaks, hot and cold records
    against the spread, spread/total/moneyline bands, each by season phase and
    favourite/underdog. Every bet is settled at the real closing line (spreads
    and totals at -110), using scores and lines from 2007-08 to 2022-23. The
    last seasons are held out; the control redraws every result from a world
    where the lines are exactly right.

    first_season, last_season: written like 2015-16; at least four seasons.
    include_control: also run the no-edge control.
    demo: run on a fake league instead of downloading real seasons.
    """
    return await _wait(_start("NBA betting-system search", _nba_report, first_season, last_season,
                              include_control, demo))


@server.tool()
async def get_report(job_id: str) -> str:
    """Collect the report from a run that was still going when its tool call returned."""
    return await _wait(job_id)


# --- command line -------------------------------------------------------------------------------


def _selftest() -> int:
    from . import data
    from .nba import data as nba_data
    from .nba.pipeline import run_bet_pipeline
    from .nba.strategies import generate_candidates as nba_candidates
    from .pipeline import run_pipeline
    from .strategies import generate_candidates

    market = run_pipeline(data.synthetic(n=800, seed=1), "SELFTEST", candidates=generate_candidates(limit=300),
                          verbose=False)
    nba = run_bet_pipeline(nba_data.synthetic_games(n_seasons=4, seed=1), candidates=nba_candidates(limit=300),
                           verbose=False)
    print(f"ai-quant-lab {__version__} selftest OK: market funnel {len(market.funnel)} stages, "
          f"NBA funnel {len(nba.funnel)} stages, cache at {os.environ['QUANTLAB_CACHE']}")
    return 0


def _install_claude_desktop(uvx_path: Optional[str], source: str) -> int:
    uvx = uvx_path or shutil.which("uvx")
    if not uvx:
        print("Could not find uvx. Pass --uvx-path with the full path to uvx.", file=sys.stderr)
        return 2
    try:
        results = install(uvx, source=source)
    except ConfigError as exc:
        print(f"Nothing was changed: {exc}", file=sys.stderr)
        return 2
    for r in results:
        verb = "Updated" if r.replaced_existing_entry else "Added"
        kept = ", ".join(r.other_servers) if r.other_servers else "none"
        print(f"{verb} ai-quant-lab in {r.path}")
        print(f"  other servers kept: {kept}")
        if r.backup:
            print(f"  backup of the previous file: {r.backup}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="quantlab-mcp", description="ai-quant-lab MCP server")
    ap.add_argument("--version", action="version", version=f"ai-quant-lab {__version__}")
    ap.add_argument("--selftest", action="store_true", help="run a quick offline check and exit")
    ap.add_argument("--install-claude-desktop", action="store_true", help="register this server with Claude Desktop")
    ap.add_argument("--uvx-path", default=None, help="full path to uvx, written into the Claude config")
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="where uvx installs ai-quant-lab from")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.install_claude_desktop:
        return _install_claude_desktop(args.uvx_path, args.source)
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
