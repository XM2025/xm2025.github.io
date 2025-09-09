from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(equity_curve: pd.Series, turnover: pd.Series | None = None) -> dict:
    returns = equity_curve.pct_change().fillna(0.0)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    num_days = len(equity_curve)
    years = num_days / 252.0
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / max(years, 1e-9)) - 1.0
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / (vol + 1e-12)
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1.0
    max_dd = drawdown.min()
    metrics = {
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "MaxDrawdown": float(max_dd),
        "TotalReturn": float(total_return),
    }
    if turnover is not None and len(turnover) > 0:
        metrics["TurnoverAnnual"] = float(turnover.mean() * 252)
    return metrics


def backtest_weights(returns: pd.DataFrame, weight_series: pd.DataFrame) -> pd.Series:
    # Align
    common_index = returns.index.intersection(weight_series.index)
    returns = returns.loc[common_index]
    weight_series = weight_series.loc[common_index]

    port_ret = (weight_series * returns).sum(axis=1)
    equity = (1.0 + port_ret).cumprod()
    equity.name = "equity"
    return equity

