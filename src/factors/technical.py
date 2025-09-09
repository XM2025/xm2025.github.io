from typing import Dict, List

import numpy as np
import pandas as pd


def compute_log_returns(panel: pd.DataFrame) -> pd.DataFrame:
	close = panel.xs("close", axis=1, level=1)
	logret = np.log(close / close.shift(1))
	logret.columns.name = "symbol"
	return logret


def rolling_volatility(panel: pd.DataFrame, window: int) -> pd.DataFrame:
	logret = compute_log_returns(panel)
	vol = logret.rolling(window).std() * np.sqrt(365)
	vol.columns.name = "symbol"
	return vol.add_suffix(f"_vol_{window}")


def momentum(panel: pd.DataFrame, window: int) -> pd.DataFrame:
	close = panel.xs("close", axis=1, level=1)
	mom = close.pct_change(window)
	mom.columns.name = "symbol"
	return mom.add_suffix(f"_mom_{window}")


def moving_average_crossover(panel: pd.DataFrame, short: int, long: int) -> pd.DataFrame:
	close = panel.xs("close", axis=1, level=1)
	short_ma = close.rolling(short).mean()
	long_ma = close.rolling(long).mean()
	cross = (short_ma / long_ma) - 1.0
	cross.columns.name = "symbol"
	return cross.add_suffix(f"_ma{short}x{long}")


def volume_change(panel: pd.DataFrame, window: int) -> pd.DataFrame:
	vol = panel.xs("volume", axis=1, level=1)
	vchg = vol.pct_change(window)
	vchg.columns.name = "symbol"
	return vchg.add_suffix(f"_vchg_{window}")


def factor_matrix(panel: pd.DataFrame, lookbacks: List[int]) -> pd.DataFrame:
	factors: List[pd.DataFrame] = []
	for lb in lookbacks:
		factors.append(momentum(panel, lb))
		factors.append(rolling_volatility(panel, lb))
		factors.append(volume_change(panel, lb))
	for s, l in [(7, 21), (14, 60), (30, 120)]:
		if l in lookbacks and s in lookbacks:
			factors.append(moving_average_crossover(panel, s, l))
	X = pd.concat(factors, axis=1)
	X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
	return X