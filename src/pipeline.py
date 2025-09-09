import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import Config
from src.data.ccxt_loader import fetch_ohlcv_daily, align_symbols_daily
from src.factors.technical import factor_matrix
from src.train.train_ppo import train_ppo
from src.backtest.evaluate import compute_metrics
from src.env.portfolio_env import DailyPortfolioEnv


def to_ms(date_str: str) -> int:
	dt = pd.to_datetime(date_str).tz_localize("UTC")
	return int(dt.timestamp() * 1000)


def run_pipeline():
	cfg = Config()
	data = fetch_ohlcv_daily(cfg.data.exchange, cfg.data.symbols, cfg.data.timeframe, since=to_ms(cfg.data.since))
	panel = align_symbols_daily(data)
	close = panel.xs("close", axis=1, level=1)
	features = factor_matrix(panel, cfg.factor.lookbacks)
	features = features.reindex(close.index).fillna(0.0)
	train_mask = (close.index >= cfg.train.train_start_date) & (close.index <= cfg.train.train_end_date)
	test_mask = (close.index >= cfg.train.test_start_date) & (close.index <= cfg.train.test_end_date)
	train_prices = close.loc[train_mask]
	test_prices = close.loc[test_mask]
	train_feats = features.loc[train_mask]
	test_feats = features.loc[test_mask]
	model, env = train_ppo(train_prices, train_feats, total_timesteps=cfg.train.total_timesteps, seed=cfg.train.seed)
	# Evaluate on test set
	test_env = DailyPortfolioEnv(test_prices, test_feats, trading_cost_bps=cfg.env.trading_cost_bps, slippage_bps=cfg.env.slippage_bps, max_weight_leverage=cfg.env.max_weight_leverage, warmup_days=cfg.env.warmup_days, seed=cfg.train.seed)
	obs, _ = test_env.reset()
	done = False
	returns = []
	while not done:
		action, _ = model.predict(obs, deterministic=True)
		obs, reward, done, _, info = test_env.step(action)
		returns.append(info["portfolio_return"])
	ret_s = pd.Series(returns, index=test_prices.index[test_env.warmup_days:])
	metrics = compute_metrics(ret_s)
	print("Test metrics:")
	print(metrics)


if __name__ == "__main__":
	run_pipeline()