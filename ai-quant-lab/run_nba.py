#!/usr/bin/env python3
"""Search thousands of NBA betting systems and test whether any of them beat the bookmaker.

Examples
--------
    python run_nba.py                         # every season 2007-08 to 2022-23 (downloads once, then cached)
    python run_nba.py --seasons 2015-16 2016-17 2017-18 2018-19 2019-20
    python run_nba.py --synthetic             # offline demo on a fake league, no network
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from quantlab.nba import data
from quantlab.nba.pipeline import run_bet_pipeline
from quantlab.nba.report import to_markdown


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="*", default=None, help="seasons like 2018-19 (default: all available)")
    ap.add_argument("--synthetic", action="store_true", help="use a fake league instead of downloading")
    ap.add_argument("--no-control", action="store_true", help="skip the no-edge control run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    games = data.synthetic_games(n_seasons=8, seed=args.seed) if args.synthetic else data.load_seasons(args.seasons)
    label = "NBA (synthetic)" if args.synthetic else "NBA"
    result = run_bet_pipeline(games, label=label)

    control = None
    if not args.no_control:
        print("\n[control] same schedule and lines, outcomes redrawn so every line is exactly right")
        control = run_bet_pipeline(data.simulate_outcomes(games, seed=args.seed), label="NBA control (no edge)",
                                   verbose=False)

    os.makedirs(args.out, exist_ok=True)
    md = to_markdown(result, control)
    with open(os.path.join(args.out, "NBA.md"), "w") as fh:
        fh.write(md)
    with open(os.path.join(args.out, "NBA.json"), "w") as fh:
        json.dump({"result": asdict(result), "control": asdict(control) if control else None}, fh, indent=2, default=str)
    print("\n" + md)
    print(f"\nwritten: {os.path.join(args.out, 'NBA.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
