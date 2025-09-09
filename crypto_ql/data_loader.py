import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover
    ccxt = None


@dataclass
class DataConfig:
    use_ccxt: bool
    exchange: str
    timeframe: str
    limit: int
    cache_dir: str
    symbols: List[str]
    start: str
    end: str
    align: str


def _parse_timestamp(ts):
    if isinstance(ts, (int, float)):
        # assume ms
        return pd.to_datetime(int(ts), unit="ms", utc=True)
    return pd.to_datetime(ts, utc=True)


def _read_local_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    # normalize columns
    def pick(name):
        for k in [name, name.upper(), name.capitalize()]:
            if k in df.columns:
                return k
        for k, v in cols.items():
            if k == name:
                return v
        return None

    ts_col = pick("timestamp") or pick("date") or pick("time")
    if ts_col is None:
        raise ValueError("CSV must contain a timestamp column")
    df["timestamp"] = df[ts_col].apply(_parse_timestamp)
    for c in ["open", "high", "low", "close", "volume"]:
        cand = pick(c)
        if cand is None:
            raise ValueError(f"CSV missing column: {c}")
        df[c] = pd.to_numeric(df[cand], errors="coerce")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.tz_convert(None).dt.normalize()
    out = df.groupby("date").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).reset_index()
    return out


def _fetch_ccxt(symbol: str, cfg: DataConfig) -> pd.DataFrame:
    if ccxt is None:
        raise RuntimeError("ccxt not available")
    ex_cls = getattr(ccxt, cfg.exchange)
    ex = ex_cls({'enableRateLimit': True})
    since = int(pd.Timestamp(cfg.start, tz='UTC').timestamp() * 1000)
    end_ts = int(pd.Timestamp(cfg.end, tz='UTC').timestamp() * 1000)
    all_rows: List[List[float]] = []
    next_since = since
    while True:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=cfg.timeframe, since=next_since, limit=cfg.limit)
        if not ohlcv:
            break
        all_rows.extend(ohlcv)
        next_since = ohlcv[-1][0] + 1
        if next_since >= end_ts:
            break
        time.sleep(ex.rateLimit / 1000.0)
    if not all_rows:
        raise RuntimeError(f"No OHLCV for {symbol}")
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])  # type: ignore
    df["timestamp"] = df["timestamp"].apply(_parse_timestamp)
    df["date"] = df["timestamp"].dt.tz_convert(None).dt.normalize()
    out = df.groupby("date").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).reset_index()
    return out


def _generate_synthetic(symbol: str, start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start=start, end=end, freq="D")
    # geometric brownian motion with occasional jumps
    np.random.seed(abs(hash(symbol)) % (2**32))
    mu = 0.20 / 365.0
    sigma = 0.80 / 365.0 ** 0.5
    price = [100.0 + 10.0 * (abs(hash(symbol)) % 17)]
    for _ in range(1, len(dates)):
        drift = (mu - 0.5 * sigma * sigma)
        shock = sigma * np.random.randn()
        jump = 0.0
        if np.random.rand() < 0.02:
            jump = np.random.randn() * 0.05
        price.append(price[-1] * np.exp(drift + shock + jump))
    price = np.array(price)
    close = price
    high = close * (1.0 + 0.02 * np.random.rand(len(close)))
    low = close * (1.0 - 0.02 * np.random.rand(len(close)))
    open_ = close * (1.0 + 0.002 * np.random.randn(len(close)))
    volume = 1000.0 * (1.0 + 0.5 * np.random.randn(len(close)))
    df = pd.DataFrame({
        "date": dates.tz_localize(None),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return df


def load_ohlcv(cfg: DataConfig, data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    result: Dict[str, pd.DataFrame] = {}
    for sym in cfg.symbols:
        normalized = sym.replace("/", "")
        local_path = os.path.join(data_dir, f"{normalized}.csv")
        cache_path = os.path.join(cfg.cache_dir, f"{normalized}_{cfg.timeframe}.csv")
        df = None
        # priority: ccxt -> local csv -> synthetic
        if cfg.use_ccxt:
            try:
                df = _fetch_ccxt(sym, cfg)
            except Exception:
                df = None
        if df is None and os.path.exists(local_path):
            try:
                df = _read_local_csv(local_path)
            except Exception:
                df = None
        if df is None:
            df = _generate_synthetic(sym, cfg.start, cfg.end)
        # cache
        df.to_csv(cache_path, index=False)
        # clip by [start, end]
        df = df[(df["date"] >= pd.to_datetime(cfg.start)) & (df["date"] <= pd.to_datetime(cfg.end))]
        result[sym] = df.reset_index(drop=True)
    return result


def align_panel(symbol_to_df: Dict[str, pd.DataFrame], align: str = "intersection") -> Tuple[pd.DatetimeIndex, Dict[str, pd.DataFrame]]:
    all_dates = None
    for df in symbol_to_df.values():
        dates = pd.DatetimeIndex(df["date"])
        if all_dates is None:
            all_dates = dates
        else:
            if align == "intersection":
                all_dates = all_dates.intersection(dates)
            else:
                all_dates = all_dates.union(dates)
    assert all_dates is not None
    # forward fill/align
    aligned: Dict[str, pd.DataFrame] = {}
    for sym, df in symbol_to_df.items():
        tmp = df.set_index("date").reindex(all_dates)
        if align != "intersection":
            tmp = tmp.ffill()
        tmp = tmp.dropna()
        tmp.index.name = "date"
        aligned[sym] = tmp.reset_index()
    return all_dates, aligned

