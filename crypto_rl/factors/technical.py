import pandas as pd
import numpy as np
from typing import Dict, List

def compute_momentum(close: pd.DataFrame, windows: List[int] = [5, 10, 20, 60]) -> Dict[str, pd.DataFrame]:
	mom = {}
	ret = close.pct_change()
	for w in windows:
		mom[f"mom_{w}"] = close.pct_change(w)
		mom[f"ret_mean_{w}"] = ret.rolling(w).mean()
	return mom

def compute_volatility(close: pd.DataFrame, windows: List[int] = [10, 20, 60]) -> Dict[str, pd.DataFrame]:
	ret = close.pct_change()
	vol = {}
	for w in windows:
		vol[f"vol_{w}"] = ret.rolling(w).std()
	return vol

def compute_volume_factors(volume: pd.DataFrame, windows: List[int] = [10, 20, 60]) -> Dict[str, pd.DataFrame]:
	factors = {}
	for w in windows:
		rolling_mean = volume.rolling(w).mean()
		rolling_std = volume.rolling(w).std()
		factors[f"vol_z_{w}"] = (volume - rolling_mean) / (rolling_std + 1e-8)
	return factors

def compute_rsi(close: pd.DataFrame, window: int = 14) -> Dict[str, pd.DataFrame]:
	delta = close.diff()
	gain = delta.clip(lower=0)
	loss = -delta.clip(upper=0)
	avg_gain = gain.rolling(window).mean()
	avg_loss = loss.rolling(window).mean()
	rs = avg_gain / (avg_loss + 1e-8)
	rsi = 100 - (100 / (1 + rs))
	return {f"rsi_{window}": rsi}

def compute_macd(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.DataFrame]:
	ewm_fast = close.ewm(span=fast, adjust=False).mean()
	ewm_slow = close.ewm(span=slow, adjust=False).mean()
	macd = ewm_fast - ewm_slow
	signal_line = macd.ewm(span=signal, adjust=False).mean()
	hist = macd - signal_line
	return {"macd": macd, "macd_signal": signal_line, "macd_hist": hist}

def compute_obv(close: pd.DataFrame, volume: pd.DataFrame) -> Dict[str, pd.DataFrame]:
	ret = close.pct_change()
	direction = np.sign(ret).fillna(0.0)
	obv = (direction * volume).cumsum()
	return {"obv": obv}

def build_feature_panel(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
	features: Dict[str, pd.DataFrame] = {}
	features.update(compute_momentum(close))
	features.update(compute_volatility(close))
	features.update(compute_volume_factors(volume))
	features.update(compute_rsi(close))
	features.update(compute_macd(close))
	features.update(compute_obv(close, volume))
	stacked = []
	for name, df in features.items():
		df_named = df.copy()
		df_named.columns = pd.MultiIndex.from_product([[name], df.columns])
		stacked.append(df_named)
	feature_panel = pd.concat(stacked, axis=1).sort_index()
	feature_panel = feature_panel.ffill().bfill()
	return feature_panel
