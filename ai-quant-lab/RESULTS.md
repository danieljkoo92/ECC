# Real-data results

Four markets, run with the default settings: 4,752 strategies each, 2 bps
commission + 3 bps slippage, the final 30% of history held out and never used
for selection, plus an identical control pass on a random walk.

Run on 2026-09-22.

| Market | History | Best backtest (in-sample Sharpe) | Sharpe luck alone produces at this search size | Same strategy out of sample | Deflated Sharpe | Survivors |
|---|---|---:|---:|---:|---:|---:|
| `NQ=F` Nasdaq 100 futures | 2005-2026, 5,472 bars | 0.60 | **0.77** | 0.67 | 0.178 | **0** |
| `QQQ` Nasdaq 100 ETF | 2005-2026, 5,463 bars | 0.60 | **0.75** | 1.02 | 0.455 | **0** |
| `EURUSD=X` | 2005-2026, 5,634 bars | 0.62 | **0.73** | -0.13 | 0.077 | **0** |
| `BTC-USD` | 2015-2026, 4,283 bars | 1.29 | 0.87 | 0.31 | 0.797 | **0** |

In three of the four markets the best backtest out of 4,752 was *worse* than
what a search that size produces from luck alone. Not one strategy in any market
reached a deflated Sharpe of 0.95.

## The selection process itself, walked forward

Instead of asking "is this one strategy real", this asks the question the pipeline
actually implies: pick the best strategy on everything known so far, trade it on
the next unseen window, repeat. That is what "the AI picks the winners" means in
practice.

| Market | Walk-forward of the process | Buy and hold, same history |
|---|---:|---:|
| `NQ=F` | **-15.4%**, Sharpe -0.02, max DD -42.3% | +1,804.8%, Sharpe 0.73 |
| `QQQ` | **-33.8%**, Sharpe -0.10, max DD -59.4% | +2,108.0%, Sharpe 0.77 |
| `EURUSD=X` | **+3.3%**, Sharpe 0.07, max DD -11.0% | n/a (spot FX) |
| `BTC-USD` | **+715.2%**, Sharpe 0.57, max DD -74.1% | +27,267.1%, Sharpe 0.88 |

Every market: the selection process lost to doing nothing. Two of four lost money
outright.

The per-window picks show why. On `NQ=F` the process chose RSI reversion with a
2-bar lookback, then a 3-bar, then a 14-bar, then Bollinger, then RSI again with
opposite thresholds. Five windows, five different "edges" - it is re-learning
noise each time, not tracking something stable.

## Control

The identical funnel on a random walk with no edge in it at all produced **0
survivors** and walk-forward Sharpes between -0.22 and +0.06 - statistically
indistinguishable from the real-market runs above.

## What this does and does not prove

It does not prove no profitable systematic strategy exists. It proves something
narrower and more useful: **a brute-force search over thousands of technical
strategies, validated properly, does not produce one** - on the exact markets and
the exact validation stack that "AI quant" pitches describe. Any real edge has to
come from somewhere the search cannot reach: better data, an execution advantage,
a structural inefficiency, or a genuinely novel signal.

Reproduce any row with, for example:

```bash
python run.py --symbol NQ=F --start 2005-01-01
```
