from typing import Dict, Tuple, Optional

import gymnasium as gym
import numpy as np
import pandas as pd


class DailyPortfolioEnv(gym.Env):
	metadata = {"render.modes": ["human"]}

	def __init__(self, prices: pd.DataFrame, features: pd.DataFrame, trading_cost_bps: float = 5.0, slippage_bps: float = 2.0, max_weight_leverage: float = 1.0, warmup_days: int = 60, seed: int = 42):
		super().__init__()
		rng = np.random.RandomState(seed)
		self.random_state = rng
		self.asset_symbols = prices.columns.tolist()
		self.num_assets = len(self.asset_symbols)
		self.prices = prices
		self.returns = prices.pct_change().fillna(0.0)
		self.features = features.reindex(self.returns.index).fillna(0.0)
		self.trading_cost = trading_cost_bps / 10000.0
		self.slippage = slippage_bps / 10000.0
		self.max_weight_leverage = max_weight_leverage
		self.warmup_days = warmup_days
		self.action_space = gym.spaces.Box(low=0.0, high=max_weight_leverage, shape=(self.num_assets,), dtype=np.float32)
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.features.shape[1] + self.num_assets,), dtype=np.float32)
		self._t = 0
		self._weights = np.zeros(self.num_assets, dtype=np.float32)

	def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
		if seed is not None:
			self.random_state.seed(seed)
		self._t = self.warmup_days
		self._weights = np.zeros(self.num_assets, dtype=np.float32)
		return self._get_observation(), {}

	def step(self, action: np.ndarray):
		action = np.clip(action, 0.0, self.max_weight_leverage)
		if action.sum() > 0:
			action = action / action.sum()
		else:
			action = np.zeros_like(action)
		prev_w = self._weights
		turnover = np.abs(action - prev_w).sum()
		trade_cost = turnover * self.trading_cost
		asset_ret = self.returns.iloc[self._t].values
		slippage_penalty = np.abs(action - prev_w).sum() * self.slippage
		portfolio_ret = (action * asset_ret).sum() - trade_cost - slippage_penalty
		self._weights = action.astype(np.float32)
		reward = float(portfolio_ret)
		self._t += 1
		terminated = self._t >= len(self.returns)
		obs = self._get_observation()
		info = {"portfolio_return": reward}
		return obs, reward, terminated, False, info

	def _get_observation(self) -> np.ndarray:
		feat_t = self.features.iloc[self._t].values.astype(np.float32)
		obs = np.concatenate([feat_t, self._weights.astype(np.float32)], axis=0)
		return obs

	def render(self):
		pass