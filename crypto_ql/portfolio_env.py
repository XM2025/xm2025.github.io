from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class EnvConfig:
    trading_cost_rate: float = 0.0005
    reward: str = "log_return"  # or "simple_return"


class PortfolioEnv:
    """
    Daily frequency portfolio environment.
    State: per-asset features + previous weights (optional; appended by caller)
    Action: raw logits -> softmax -> weights on N assets + cash (N+1)
    """

    def __init__(self, dates, features: np.ndarray, returns: np.ndarray, include_cash: bool = True, prev_weight_in_features: bool = False, cfg: EnvConfig | None = None):
        self.dates = dates
        self.features = features  # shape T x N x F or T x N x (F+1) if prev weight appended externally
        self.returns = returns    # shape T x N (asset simple returns)
        self.include_cash = include_cash
        self.prev_weight_in_features = prev_weight_in_features
        self.cfg = cfg or EnvConfig()

        self.num_steps = features.shape[0]
        self.num_assets = features.shape[1]
        self.num_obs_features = features.shape[2]
        self.action_dim = self.num_assets + (1 if include_cash else 0)

        self.reset()

    def reset(self, start_index: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        # start from t = 1 to use previous returns if needed
        self.t = start_index
        self.weights = np.zeros(self.action_dim, dtype=np.float32)
        if self.include_cash:
            self.weights[-1] = 1.0  # start fully in cash
        self.nav = 1.0
        self.turnovers = []
        self.episode_returns = []
        # set prev-weight channel on first observation if needed
        if self.prev_weight_in_features:
            if self.include_cash:
                asset_prev = self.weights[:-1]
            else:
                asset_prev = self.weights
            if self.t < self.num_steps:
                self.features[self.t, :, -1] = asset_prev
        return self._get_obs(), self.weights.copy()

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        ex = np.exp(x)
        s = ex / ex.sum()
        return s

    def _get_obs(self) -> np.ndarray:
        # obs shape: N x F (+ optionally we concatenate prev weights as additional feature outside)
        return self.features[self.t]

    def step(self, action_logits: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # action is treated as logits; map to weights via softmax on N (+ cash)
        new_weights = self._softmax(action_logits.astype(np.float32))
        # effective asset weights exclude cash
        if self.include_cash:
            asset_weights = new_weights[:-1]
            cash_weight = new_weights[-1]
        else:
            asset_weights = new_weights
            cash_weight = 0.0
        # realized portfolio return at t (using returns[t] for assets)
        r_assets = self.returns[self.t]
        port_ret = float((asset_weights * r_assets).sum())
        # deduct trading cost by turnover
        turnover = float(np.abs(new_weights - self.weights).sum())
        cost = self.cfg.trading_cost_rate * turnover
        port_ret_after = port_ret - cost
        if self.cfg.reward == "log_return":
            reward = float(np.log(max(1e-9, 1.0 + port_ret_after)))
        else:
            reward = port_ret_after
        self.nav *= (1.0 + port_ret_after)
        self.turnovers.append(turnover)
        self.episode_returns.append(port_ret_after)
        # write prev weight feature for next step if applicable
        if self.prev_weight_in_features:
            t_next = self.t + 1
            if t_next < self.num_steps:
                self.features[t_next, :, -1] = asset_weights
        self.weights = new_weights
        self.t += 1
        done = self.t >= self.num_steps
        info = {
            "date": self.dates[self.t - 1] if self.t - 1 < len(self.dates) else None,
            "turnover": turnover,
            "nav": self.nav,
            "raw_return": port_ret,
            "after_cost_return": port_ret_after,
        }
        next_obs = self._get_obs() if not done else np.zeros_like(self.features[0])
        return next_obs, reward, done, info

