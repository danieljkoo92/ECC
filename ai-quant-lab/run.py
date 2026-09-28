#!/usr/bin/env python3
"""Run the strategy search and validation funnel on one market.

Examples
--------
    python run.py --symbol QQQ                      # Nasdaq 100 ETF (NQ futures proxy)
    python run.py --symbol BTC/USDT --source ccxt   # crypto, free exchange data
    python run.py --symbol SIM --source synthetic   # offline demo, no network
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from quantlab import data, report
from quantlab.backtest import Costs
from quantlab.pipeline import Thresholds, run_pipeline
from quantlab.strategies import generate_candidates


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--source", default="yfinance", choices=["yfinance", "ccxt", "synthetic"])
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--oos", type=float, default=0.3, help="fraction of history held out and never used for selection")
    ap.add_argument("--commission-bps", type=float, default=2.0)
    ap.add_argument("--slippage-bps", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=None, help="cap the number of candidates (for a quick run)")
    ap.add_argument("--no-control", action="store_true", help="skip the random-data control run")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    costs = Costs(args.commission_bps, args.slippage_bps)
    thresholds = Thresholds()
    candidates = generate_candidates(limit=args.limit)

    df = data.load(args.symbol, source=args.source, start=args.start, end=args.end)
    result = run_pipeline(df, args.symbol, costs=costs, thresholds=thresholds,
                          candidates=candidates, oos_fraction=args.oos)

    control = None
    if not args.no_control:
        print("\n[control] running the identical funnel on a random walk with no edge to find")
        control = run_pipeline(data.synthetic(n=len(df), seed=42), "RANDOM-CONTROL", costs=costs,
                               thresholds=thresholds, candidates=candidates, oos_fraction=args.oos,
                               verbose=False)

    os.makedirs(args.out, exist_ok=True)
    stem = args.symbol.replace("/", "-")
    md = report.to_markdown(result, control)
    with open(os.path.join(args.out, f"{stem}.md"), "w") as fh:
        fh.write(md)
    with open(os.path.join(args.out, f"{stem}.json"), "w") as fh:
        json.dump({"result": asdict(result), "control": asdict(control) if control else None}, fh, indent=2, default=str)

    print("\n" + md)
    print(f"\nwritten: {os.path.join(args.out, stem + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
