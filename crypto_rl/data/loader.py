import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

def _read_one_csv(path: str) -> pd.DataFrame:
	df = pd.read_csv(path)
	if "timestamp" not in df.columns:
		raise ValueError(f"Missing timestamp in {path}")
	if np.issubdtype(df["timestamp"].dtype, np.number):
		df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
	else:
		df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
	for col in ["open", "high", "low", "close", "volume"]:
		if col not in df.columns:
			raise ValueError(f"Missing {col} in {path}")
	df = df.sort_values("timestamp").set_index("timestamp")
	agg = {
		"open": "first",
		"high": "max",
		"low": "min",
		"close": "last",
		"volume": "sum",
	}
	daily = df.resample("1D", origin="start_day").agg(agg).dropna(how="any")
	return daily

def load_panel(data_dir: str, symbols: List[str]) -> Dict[str, pd.DataFrame]:
	panel: Dict[str, pd.DataFrame] = {}
	for symbol in symbols:
		path = os.path.join(data_dir, f"{symbol}.csv")
		if not os.path.exists(path):
			raise FileNotFoundError(f"CSV for {symbol} not found at {path}")
		panel[symbol] = _read_one_csv(path)
	return panel

def align_panel(panel: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
	closes = []
	volumes = []
	for sym, df in panel.items():
		closes.append(df[["close"]].rename(columns={"close": sym}))
		volumes.append(df[["volume"]].rename(columns={"volume": sym}))
	close = pd.concat(closes, axis=1).sort_index().ffill().dropna(how="any")
	volume = pd.concat(volumes, axis=1).reindex(close.index).ffill().fillna(0.0)
	return close, volume

def compute_returns(close: pd.DataFrame) -> pd.DataFrame:
	ret = close.pct_change().fillna(0.0)
	return ret

def train_test_split_by_date(index: pd.DatetimeIndex, start: str, split: str, end: str) -> Tuple[np.ndarray, np.ndarray]:
	start_ts = pd.to_datetime(start, utc=True)
	split_ts = pd.to_datetime(split, utc=True)
	end_ts = pd.to_datetime(end, utc=True)
	mask_train = (index >= start_ts) & (index < split_ts)
	mask_test = (index >= split_ts) & (index <= end_ts)
	return mask_train.values, mask_test.values
