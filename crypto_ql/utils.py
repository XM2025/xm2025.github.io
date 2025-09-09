import os
import json
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(tensor: torch.Tensor) -> torch.Tensor:
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def compute_nav(returns: np.ndarray) -> np.ndarray:
    nav = np.cumprod(1.0 + returns)
    return nav


def max_drawdown(nav: np.ndarray) -> float:
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    return float(dd.min())


def sharpe_ratio(daily_returns: np.ndarray, risk_free: float = 0.0) -> float:
    if daily_returns.size == 0:
        return 0.0
    excess = daily_returns - risk_free / 365.0
    std = excess.std(ddof=1)
    if std == 0:
        return 0.0
    return float(np.sqrt(365.0) * excess.mean() / std)


def sortino_ratio(daily_returns: np.ndarray, risk_free: float = 0.0) -> float:
    excess = daily_returns - risk_free / 365.0
    downside = excess[excess < 0.0]
    denom = downside.std(ddof=1) if downside.size > 0 else 0.0
    if denom == 0:
        return 0.0
    return float(np.sqrt(365.0) * excess.mean() / denom)


def annual_return(daily_returns: np.ndarray) -> float:
    nav = compute_nav(daily_returns)
    if nav.size == 0:
        return 0.0
    total_return = nav[-1] - 1.0
    years = max(1e-9, len(daily_returns) / 365.0)
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def turnover_rate(weights_today: np.ndarray, weights_yesterday: np.ndarray) -> float:
    return float(np.abs(weights_today - weights_yesterday).sum())


def metrics_summary(daily_returns: np.ndarray, daily_turnovers: np.ndarray | None = None) -> Dict[str, float]:
    nav = compute_nav(daily_returns)
    summary = {
        "CAGR": annual_return(daily_returns),
        "Sharpe": sharpe_ratio(daily_returns),
        "Sortino": sortino_ratio(daily_returns),
        "MaxDrawdown": max_drawdown(nav),
        "Volatility": float(np.sqrt(365.0) * daily_returns.std(ddof=1)) if len(daily_returns) > 1 else 0.0,
    }
    if daily_turnovers is not None and len(daily_turnovers) > 0:
        summary["AvgTurnover"] = float(np.mean(daily_turnovers))
    return summary


def plot_nav(dates: List[pd.Timestamp], nav: np.ndarray, save_path: str) -> None:
    ensure_dir(os.path.dirname(save_path))
    plt.figure(figsize=(10, 4))
    plt.plot(dates, nav, label="Strategy NAV")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_json(obj: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

