"""Free OHLCV data loading with on-disk caching.

Sources
-------
yfinance : stocks, ETFs, FX pairs and futures continuous contracts (free, daily).
ccxt     : crypto spot OHLCV from any major exchange (free, no key for public data).
synthetic: a geometric random walk, used by the tests and by --synthetic runs so
           the pipeline can be exercised with no network access at all.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]

CACHE_DIR = os.environ.get(
    "QUANTLAB_CACHE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache"),
)


def _cache_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = key.replace("/", "-").replace(":", "-").replace(" ", "_")
    return os.path.join(CACHE_DIR, f"{safe}.csv")


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"data is missing columns {missing}; got {list(df.columns)}")
    df = df[COLUMNS].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna()


def load_yfinance(symbol: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no rows for {symbol}")
    return _normalise(raw)


CCXT_EXCHANGES = ("kraken", "coinbase", "bitstamp", "binance")


def load_ccxt(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    exchange: Optional[str] = None,
) -> pd.DataFrame:
    """Daily crypto OHLCV from a public exchange endpoint.

    Exchanges geo-block by region and change symbol naming (BTC/USDT vs BTC/USD),
    so try several and take the first that answers rather than failing outright.
    """
    import ccxt

    candidates = [exchange] if exchange else list(CCXT_EXCHANGES)
    symbols = [symbol] if "/" in symbol else [f"{symbol}/USD", f"{symbol}/USDT"]
    if symbol.endswith("/USDT"):
        symbols.append(symbol.replace("/USDT", "/USD"))
    errors: list[str] = []
    rows: list[list] = []
    for name in candidates:
        for sym in symbols:
            try:
                ex = getattr(ccxt, name)({"enableRateLimit": True})
                since = ex.parse8601("2017-01-01T00:00:00Z")
                rows = []
                while True:
                    batch = ex.fetch_ohlcv(sym, timeframe=timeframe, since=since, limit=1000)
                    if not batch:
                        break
                    rows.extend(batch)
                    since = batch[-1][0] + 1
                    if len(batch) < 1000:
                        break
                if rows:
                    break
            except Exception as exc:  # exchange unavailable here, or symbol unknown
                errors.append(f"{name}:{sym}: {type(exc).__name__}")
        if rows:
            break
    if not rows:
        raise RuntimeError(f"no ccxt exchange returned rows for {symbol} ({'; '.join(errors[:6])})")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("ts"), unit="ms")
    return _normalise(df)


def synthetic(n: int = 3000, seed: int = 0, mu: float = 0.0002, sigma: float = 0.012) -> pd.DataFrame:
    """A pure random walk. Nothing in here is predictable - that is the point.

    Any 'strategy' that looks profitable on this data is measuring luck, which
    makes it the correct control experiment for a large strategy search.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, sigma, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    noise = np.abs(rng.normal(0, sigma / 2, n)) * close
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, sigma / 4, n)),
            "high": close + noise,
            "low": close - noise,
            "close": close,
            "volume": rng.integers(1_000, 100_000, n).astype(float),
        },
        index=pd.date_range("2005-01-03", periods=n, freq="B"),
    )
    return _normalise(df)


def load(
    symbol: str,
    source: str = "yfinance",
    start: str = "2005-01-01",
    end: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load OHLCV for one symbol, caching the result as CSV."""
    if source == "synthetic":
        return synthetic(seed=abs(hash(symbol)) % (2**31))

    path = _cache_path(f"{source}-{symbol}-{start}")
    if use_cache and os.path.exists(path):
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        if not cached.empty:
            return _normalise(cached)

    if source == "yfinance":
        df = load_yfinance(symbol, start=start, end=end)
    elif source == "ccxt":
        df = load_ccxt(symbol)
        df = df[df.index >= pd.Timestamp(start)]
    else:
        raise ValueError(f"unknown source {source!r}")

    if use_cache:
        df.to_csv(path)
    return df
