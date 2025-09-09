import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple

class DailyPortfolioEnv(gym.Env):
	metadata = {"render.modes": ["human"]}
	def __init__(self, features: np.ndarray, returns: np.ndarray, transaction_cost: float = 0.001):
		# features: [T, N, F]; returns: [T, N]
		assert features.ndim == 3 and returns.ndim == 2
		assert features.shape[0] == returns.shape[0]
		self.features = features
		self.returns = returns
		self.T, self.N, self.F = features.shape
		self.transaction_cost = float(transaction_cost)
		self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.N,), dtype=np.float32)
		self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.N, self.F), dtype=np.float32)
		self._t = 0
		self._w = np.zeros(self.N, dtype=np.float32)
		self._nav = 1.0
	def reset(self, *, seed: int | None = None, options: Dict | None = None) -> Tuple[np.ndarray, Dict]:
		super().reset(seed=seed)
		self._t = 0
		self._w = np.zeros(self.N, dtype=np.float32)
		self._nav = 1.0
		return self.features[self._t].astype(np.float32), {"nav": self._nav, "weights": self._w.copy()}
	def step(self, action: np.ndarray):
		w = np.asarray(action, dtype=np.float32)
		w = np.clip(w, 0.0, 1.0)
		if w.sum() <= 0:
			w = np.ones_like(w) / len(w)
		else:
			w = w / (w.sum() + 1e-8)
		turnover = np.abs(w - self._w).sum()
		cost = self.transaction_cost * turnover
		r = float((w * self.returns[self._t]).sum()) - cost
		self._nav *= (1.0 + r)
		self._w = w
		self._t += 1
		done = self._t >= (self.T - 1)
		reward = r
		obs = self.features[self._t].astype(np.float32) if not done else self.features[self.T - 1].astype(np.float32)
		info = {"nav": self._nav, "weights": self._w.copy(), "turnover": turnover}
		return obs, reward, done, False, info
