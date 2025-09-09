import time
from typing import Dict, List, Optional

import ccxt
import numpy as np
import pandas as pd


def _init_exchange(exchange_id: str):
	exchange_class = getattr(ccxt, exchange_id)
	exchange = exchange_class({
		"enableRateLimit": True,
		"options": {"adjustForTimeDifference": True},
	})
	return exchange


def fetch_ohlcv_daily(exchange_id: str, symbols: List[str], timeframe: str = "1d", since: Optional[int] = None) -> Dict[str, pd.DataFrame]:
	exchange = _init_exchange(exchange_id)
	data: Dict[str, pd.DataFrame] = {}

	for symbol in symbols:
		all_rows: List[List[float]] = []
		since_ms = since
		while True:
			try:
				rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=1000)
			except Exception:
				time.sleep(2.0)
				continue
			if not rows:
				break
			all_rows.extend(rows)
			since_ms = rows[-1][0] + 1
			if len(rows) < 1000:
				break
			time.sleep(exchange.rateLimit / 1000.0)

		if not all_rows:
			continue

		df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"]).drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
		df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("UTC").dt.normalize()
		df = df.set_index("date").sort_index()
		data[symbol] = df[["open", "high", "low", "close", "volume"]]
	return data


def align_symbols_daily(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
	if not data:
		raise ValueError("No data provided")
	frames = []
	for symbol, df in data.items():
		df_symbol = df.copy()
		df_symbol.columns = pd.MultiIndex.from_product([[symbol], df_symbol.columns])
		frames.append(df_symbol)
	panel = pd.concat(frames, axis=1).sort_index()
	panel = panel.groupby(level=1, axis=1).apply(lambda x: x.ffill().bfill())
	return panel