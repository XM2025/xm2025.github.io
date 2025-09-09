import argparse
import os

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from crypto_rl.data.fetcher import DailyDataFetcher
from crypto_rl.factors.technical import compute_technical_factors
from crypto_rl.env.portfolio_env import DailyPortfolioEnv
from crypto_rl.backtest.evaluator import backtest_weights, compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    factor_cfg = cfg["factors"]
    env_cfg = cfg["env"]
    train_cfg = cfg["train"]

    fetcher = DailyDataFetcher(cache_dir=data_cfg["cache_dir"])
    panels = fetcher.fetch(
        tickers=data_cfg["tickers"],
        start_date=data_cfg["start_date"],
        end_date=data_cfg.get("end_date"),
        interval=data_cfg.get("interval", "1d"),
        refresh=data_cfg.get("refresh", False),
    )

    # Build close prices matrix
    close_frames = []
    for asset, df in panels.items():
        s = df["Close"].copy()
        s.name = asset
        close_frames.append(s)
    close_prices = pd.concat(close_frames, axis=1).sort_index()

    factors = compute_technical_factors(
        ohlcv_panels=panels,
        windows=factor_cfg.get("windows", {}),
        include_cross_sectional=factor_cfg.get("include_cross_sectional", True),
    )

    # Build environment over the validation range
    env = DailyPortfolioEnv(
        factor_matrix=factors,
        close_prices=close_prices,
        include_cash=env_cfg.get("include_cash", True),
        transaction_cost_bps=env_cfg.get("transaction_cost_bps", 5.0),
        risk_aversion=env_cfg.get("risk_aversion", 2.0),
        max_weight=env_cfg.get("max_weight", 0.35),
    )
    vec_env = DummyVecEnv([lambda: env])

    model = PPO.load(args.model_path, device=train_cfg.get("device", "auto"))

    obs, _ = env.reset()
    weights_records = []
    dates = []
    turnovers = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        weights = info["weights"]
        if env.include_cash:
            asset_weights = weights[:-1]
        else:
            asset_weights = weights
        weights_records.append(asset_weights)
        dates.append(env.idx[env._t])
        turnovers.append(info["turnover"])  # approximate per-step turnover
        if terminated or truncated:
            break

    weights_df = pd.DataFrame(weights_records, index=dates, columns=env.assets)

    equity = backtest_weights(
        returns=env.returns[env.assets],
        weight_series=weights_df,
    )
    turnover_s = pd.Series(turnovers, index=weights_df.index)
    metrics = compute_metrics(equity, turnover_s)
    print("Backtest Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()

