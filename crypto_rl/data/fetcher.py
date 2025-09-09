import os
from typing import List, Dict

import pandas as pd
import yfinance as yf


class DailyDataFetcher:
    """Downloads daily OHLCV for a list of tickers and caches to CSVs.

    The fetcher merges individual asset DataFrames into a panel-like format:
    A dict of {ticker: DataFrame}, each indexed by date with columns [Open, High, Low, Close, Adj Close, Volume].
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, ticker: str) -> str:
        safe_ticker = ticker.replace("/", "_")
        return os.path.join(self.cache_dir, f"{safe_ticker}.csv")

    def fetch(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str | None,
        interval: str = "1d",
        refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            path = self._cache_path(ticker)
            if os.path.exists(path) and not refresh:
                try:
                    df = pd.read_csv(path, parse_dates=[0], index_col=0)
                except Exception:
                    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=False, progress=False)
                    if df.empty:
                        raise RuntimeError(f"No data for {ticker}")
                    df = df.reset_index().rename(columns={"index": "Date"}).set_index("Date")
                    df = df[[c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]]
                    df.sort_index(inplace=True)
                    df.to_csv(path, index=True)
            else:
                df = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=False, progress=False)
                if df.empty:
                    raise RuntimeError(f"No data for {ticker}")
                df = df.reset_index().rename(columns={"index": "Date"}).set_index("Date")
                df = df[[c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]]
                df.sort_index(inplace=True)
                df.to_csv(path, index=True)
            # Ensure numeric dtypes
            for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            # Ensure datetime index
            try:
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df[~df.index.isna()]
                df.sort_index(inplace=True)
            except Exception:
                pass
            result[ticker] = df
        return result

