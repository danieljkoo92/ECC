# ai-quant-lab

Generate thousands of trading strategies, then run the statistics that tell you
whether any of them are real.

This exists because of a specific claim, made in a lot of trading reels:

> "My AI reads 1,100 research papers, generates 72,000 strategies, backtests them
> with walk-forward, Monte Carlo, deflated Sharpe and regime tests, and picks the
> winners."

Every single one of those steps is a real quant technique. The claim is not
nonsense. But the same machinery that finds a winner is also the machinery that
manufactures one out of thin air: search 72,000 strategies and the best one will
look superb **even if none of them has any edge at all**. This repo runs that
exact pipeline, and adds the one number the videos never show - the
**deflated Sharpe ratio**, which asks how good the winner would have looked by
luck alone, given how many strategies were tried.

## What it does

1. **Generates** a search space of ~4,800 strategies: six families (moving average
   cross, Donchian breakout, RSI reversion, Bollinger reversion, momentum,
   volatility breakout) x their parameter grids x a trend filter x a minimum
   holding period.
2. **Backtests all of them at once** on real free data, with costs and slippage,
   applying every position one bar after the signal so nothing can peek ahead.
3. **Puts the winners through nine stages** of validation, and reports how many
   die at each one.
4. **Runs the identical funnel on a pure random walk** as a control. Whatever the
   real market produces has to beat what noise produces, or it means nothing.

## The funnel

| Stage | What it kills |
|---|---|
| In-sample screen | Strategies that never looked good in the first place |
| PBO (CSCV) | Measures, across every way of splitting history, how often the in-sample winner is below median out of sample |
| Plateau test | Lucky parameter spikes - a real edge still works when you nudge the parameters |
| Cost stress | Edges that vanish at 3x trading costs |
| Out-of-sample | The held-out final 30% of history, never used for selection |
| Walk-forward | Applied to the *selection process*: pick the best on past data, trade it on the next unseen window, repeat |
| Monte Carlo | Block-bootstraps the return stream; the 5th-percentile path must still be positive |
| Deflated Sharpe | Corrects the Sharpe ratio for the number of strategies tested. This is the stage that kills almost everything |
| Regime consistency | Edges that only exist in one market regime |

## Running it

Install dependencies once:

```bash
pip install -r requirements.txt
```

Then:

```bash
python run.py --symbol QQQ                      # Nasdaq 100 ETF - the NQ futures proxy
python run.py --symbol SPY --start 1995-01-01   # S&P 500, longer history
python run.py --symbol BTC/USDT --source ccxt   # crypto, free exchange data
python run.py --symbol SIM --source synthetic   # offline demo, no network needed
```

Reports are written to `reports/<symbol>.md` and `reports/<symbol>.json`.

No API keys, no paid data, no broker connection. Data comes from Yahoo Finance
(`yfinance`) and public exchange endpoints (`ccxt`).

## NBA betting systems

The same funnel, pointed at sports betting. `quantlab/nba/` generates about
2,100 betting systems - the angles betting services sell: fade or follow line
moves, back rested teams against tired ones, ride or fade winning streaks and
hot against-the-spread runs, home underdogs, unders on high totals - and
settles every bet at the real closing line, with spreads and totals at -110.

```bash
python run_nba.py              # every season 2007-08 to 2022-23, downloaded once and cached
python run_nba.py --synthetic  # offline demo on a fake league
```

The stages are the same ones, translated to bets: clear the bookmaker's margin
in-sample, survive a plateau test, stay profitable at -115, hold up on seasons
never used for selection, survive bootstrap and the deflated Sharpe correction,
and win in most seasons. The control run redraws every result from a world
where the closing lines are exactly right, so it shows what the search finds
when no edge exists at all.

Scores and closing lines come from the free public archive at
sportsbookreviewsonline.com. Every feature a system uses - rest days, streaks,
recent record against the spread, line movement - is built only from games
finished before tip-off.

## Use it from Claude Desktop

ai-quant-lab ships as an MCP server, so Claude can run both searches for you
from a normal chat. It adds three tools: `test_trading_strategies`,
`test_nba_betting_systems` and `get_report`.

**Windows**: open PowerShell (Start, type `PowerShell`, press Enter), paste
this line and press Enter:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/danieljkoo92/ECC/main/ai-quant-lab/install/claude-desktop-windows.ps1 | iex"
```

It installs [uv](https://docs.astral.sh/uv/) if needed (uv brings its own
Python), downloads ai-quant-lab, runs a self-test, and adds the server to
Claude's config. Every existing server and setting in that file is kept, and
a timestamped backup is written next to it first. Then quit Claude from the
tray icon, reopen it, and ask something like:

- "Test NBA betting systems with ai-quant-lab"
- "Is there any trading strategy that works on NQ=F? Use ai-quant-lab"
- "Someone is selling a Bitcoin bot. Check BTC-USD with ai-quant-lab"

**Mac or any other MCP client**: install uv, then add this to the client's
MCP server config (for Claude Desktop, Settings, Developer, Edit Config):

```json
{
  "mcpServers": {
    "ai-quant-lab": {
      "command": "uvx",
      "args": [
        "--from",
        "https://github.com/danieljkoo92/ECC/archive/refs/heads/main.zip#subdirectory=ai-quant-lab",
        "quantlab-mcp"
      ]
    }
  }
}
```

Downloaded market data and NBA seasons are cached in `~/.ai-quant-lab/cache`.
A first run downloads data and can take a minute or two. If a call comes back
with a job id instead of a report, Claude collects it with `get_report`.

## Running it without installing anything

Use the **Run quant report** action in the GitHub Actions tab: choose
`markets` and a symbol, or `nba`, press the button, and read the report in the
run summary when it finishes.

## Reading the output

- **Sharpe ratio** - return per unit of risk. Above 1 is good. Above 2 in a
  backtest usually means something is wrong.
- **Deflated Sharpe** - the probability the edge is real, after correcting for the
  size of the search. Below 0.95, treat the strategy as noise.
- **Luck benchmark** - the Sharpe a search this size is expected to produce from a
  strategy with no edge whatsoever. If the winner's Sharpe is near this number,
  the search found luck, not skill.
- **PBO** - above 50% means the in-sample winner does worse than a coin flip out
  of sample.

## What this is not

It is not a trading system, it does not connect to a broker, and it will not tell
you what to buy. It is a truth filter: it takes a strategy search and tells you
how much of the result survives honest statistics. Most of the time the answer is
"none of it", and that is the useful answer - it is the one that stops you funding
an account on the strength of a backtest.

Nothing here is financial advice. Past performance of a backtest, deflated or not,
is not a prediction.
