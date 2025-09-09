from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd


class DailyPortfolioEnv(gym.Env):
    """Daily portfolio allocation environment.

    Observation: concatenated factor vector for all assets at day t.
    Action: unconstrained real vector -> mapped to weights via softmax. Optional cash asset.
    Reward: next-day portfolio return minus transaction costs minus risk penalty.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        factor_matrix: pd.DataFrame,
        close_prices: pd.DataFrame,
        include_cash: bool = True,
        transaction_cost_bps: float = 5.0,
        risk_aversion: float = 2.0,
        max_weight: float = 0.35,
        train_slice: slice | None = None,
    ) -> None:
        super().__init__()

        # Align indices and assets
        common_index = factor_matrix.index.intersection(close_prices.index)
        self.factors = factor_matrix.loc[common_index]
        self.close = close_prices.loc[common_index]

        self.assets: List[str] = list({c[0] for c in self.factors.columns})
        self.assets.sort()
        self.asset_to_idx = {a: i for i, a in enumerate(self.assets)}

        # Reindex factor columns by asset order
        ordered_cols = []
        for asset in self.assets:
            cols = [c for c in self.factors.columns if c[0] == asset]
            cols_sorted = sorted(cols, key=lambda x: x[1])
            ordered_cols += cols_sorted
        self.factors = self.factors[ordered_cols]

        # Factor dimension per asset
        self.factors_per_asset = len([c for c in self.factors.columns if c[0] == self.assets[0]])
        self.include_cash = include_cash
        self.transaction_cost = transaction_cost_bps / 10000.0
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight

        # Compute daily simple returns
        self.returns = self.close.pct_change().shift(-1).reindex(common_index)
        self.returns = self.returns.fillna(0.0)

        # Split train/val if provided
        self.train_slice = train_slice
        if train_slice is not None:
            self.idx = self.factors.index[train_slice]
        else:
            self.idx = self.factors.index

        num_assets = len(self.assets) + (1 if self.include_cash else 0)
        obs_dim = len(self.assets) * self.factors_per_asset

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # Stable-Baselines3 requires finite bounds for action space
        self.action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(num_assets,), dtype=np.float32)

        self._t = 0
        self._weights = np.zeros(num_assets, dtype=np.float32)
        self._weights[-1] = 1.0 if self.include_cash else 0.0

    def _current_obs(self) -> np.ndarray:
        row = self.factors.loc[self.idx[self._t]]
        return row.values.astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._t = 0
        num_assets = len(self.assets) + (1 if self.include_cash else 0)
        self._weights = np.zeros(num_assets, dtype=np.float32)
        if self.include_cash:
            self._weights[-1] = 1.0
        return self._current_obs(), {}

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        z = x - np.max(x)
        exp_z = np.exp(z)
        return exp_z / (np.sum(exp_z) + 1e-12)

    def step(self, action: np.ndarray):
        raw_weights = np.array(action, dtype=np.float32)
        weights = self._softmax(raw_weights)

        # Enforce max weight per asset (projective clipping then renorm keeping cash optional)
        max_w = self.max_weight
        if max_w is not None:
            weights = np.minimum(weights, max_w)
            total = np.sum(weights)
            if total > 0:
                weights = weights / total

        # Transaction cost on turnover
        turnover = np.sum(np.abs(weights - self._weights))
        cost = turnover * self.transaction_cost

        # Compute next-day return
        date = self.idx[self._t]
        next_ret_assets = self.returns.loc[date, self.assets].values.astype(np.float32)
        if self.include_cash:
            next_ret = np.concatenate([next_ret_assets, np.array([0.0], dtype=np.float32)])
        else:
            next_ret = next_ret_assets

        portfolio_ret = float(np.dot(weights, next_ret))

        # Risk penalty: variance of asset returns (ex-ante risk proxy)
        risk_penalty = self.risk_aversion * float(np.var(next_ret_assets))
        reward = portfolio_ret - cost - risk_penalty

        self._weights = weights
        self._t += 1

        terminated = self._t >= (len(self.idx) - 1)
        truncated = False
        info = {
            "portfolio_ret": portfolio_ret,
            "turnover": turnover,
            "cost": cost,
            "risk_penalty": risk_penalty,
            "weights": weights,
        }
        return self._current_obs(), reward, terminated, truncated, info

    def render(self):
        return None

