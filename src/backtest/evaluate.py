from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
	total_return: float
	annual_return: float
	annual_vol: float
	sharpe: float
	max_drawdown: float
	equity_curve: pd.Series


def compute_metrics(returns: pd.Series, freq_per_year: int = 365) -> BacktestResult:
	cum = (1.0 + returns).cumprod()
	total_return = cum.iloc[-1] - 1.0
	ann_ret = (1.0 + returns.mean()) ** freq_per_year - 1.0
	ann_vol = returns.std() * np.sqrt(freq_per_year)
	sharpe = ann_ret / (ann_vol + 1e-8)
	roll_max = cum.cummax()
	drawdown = cum / roll_max - 1.0
	max_dd = drawdown.min()
	return BacktestResult(total_return, ann_ret, ann_vol, sharpe, float(max_dd), cum)


def backtest_static_weights(prices: pd.DataFrame, weights: pd.Series) -> BacktestResult:
	weights = weights / weights.sum()
	rets = prices.pct_change().fillna(0.0)
	port_rets = (rets * weights).sum(axis=1)
	return compute_metrics(port_rets)