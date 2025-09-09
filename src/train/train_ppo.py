from typing import Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env.portfolio_env import DailyPortfolioEnv


def make_env(prices: pd.DataFrame, features: pd.DataFrame, seed: int = 42, trading_cost_bps: float = 5.0, slippage_bps: float = 2.0, max_weight_leverage: float = 1.0, warmup_days: int = 60):
	def _init():
		return DailyPortfolioEnv(
			prices=prices,
			features=features,
			trading_cost_bps=trading_cost_bps,
			slippage_bps=slippage_bps,
			max_weight_leverage=max_weight_leverage,
			warmup_days=warmup_days,
			seed=seed,
		)
	return _init


def train_ppo(prices: pd.DataFrame, features: pd.DataFrame, total_timesteps: int = 200_000, seed: int = 42) -> Tuple[PPO, DummyVecEnv]:
	env_fn = make_env(prices, features, seed)
	env = DummyVecEnv([env_fn])
	model = PPO("MlpPolicy", env, verbose=0, seed=seed, tensorboard_log=None)
	model.learn(total_timesteps=total_timesteps)
	return model, env