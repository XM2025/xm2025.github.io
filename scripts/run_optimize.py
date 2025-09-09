import argparse
import os
from tempfile import TemporaryDirectory

import numpy as np
import optuna
import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from crypto_rl.data.fetcher import DailyDataFetcher
from crypto_rl.factors.technical import compute_technical_factors
from crypto_rl.env.portfolio_env import DailyPortfolioEnv
from crypto_rl.backtest.evaluator import backtest_weights, compute_metrics


def objective(trial: optuna.Trial, cfg: dict, factors: pd.DataFrame, close_prices: pd.DataFrame):
    env_cfg = cfg["env"].copy()
    train_cfg = cfg["train"].copy()

    # Sample hyperparameters
    env_cfg["transaction_cost_bps"] = trial.suggest_float("tc_bps", 0.0, 20.0)
    env_cfg["risk_aversion"] = trial.suggest_float("risk_aversion", 0.0, 5.0)
    env_cfg["max_weight"] = trial.suggest_float("max_weight", 0.1, 0.6)

    learning_rate = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.05, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096])

    # Build env for train and val
    train_env = DailyPortfolioEnv(
        factor_matrix=factors.loc[(factors.index >= train_cfg["train_start"]) & (factors.index <= train_cfg["train_end"])],
        close_prices=close_prices,
        include_cash=env_cfg.get("include_cash", True),
        transaction_cost_bps=env_cfg["transaction_cost_bps"],
        risk_aversion=env_cfg["risk_aversion"],
        max_weight=env_cfg["max_weight"],
    )
    val_env = DailyPortfolioEnv(
        factor_matrix=factors.loc[(factors.index >= train_cfg["val_start"]) & (factors.index <= train_cfg["val_end"])],
        close_prices=close_prices,
        include_cash=env_cfg.get("include_cash", True),
        transaction_cost_bps=env_cfg["transaction_cost_bps"],
        risk_aversion=env_cfg["risk_aversion"],
        max_weight=env_cfg["max_weight"],
    )

    model = PPO(
        policy=train_cfg["policy"],
        env=DummyVecEnv([lambda: train_env]),
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=float(train_cfg["vf_coef"]),
        verbose=0,
        device=train_cfg.get("device", "auto"),
        seed=cfg.get("random_seed", 42),
    )
    model.learn(total_timesteps=int(train_cfg["total_timesteps"]))

    # Evaluate on validation set
    obs, _ = val_env.reset()
    weights_records = []
    dates = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = val_env.step(action)
        weights = info["weights"]
        if val_env.include_cash:
            asset_weights = weights[:-1]
        else:
            asset_weights = weights
        weights_records.append(asset_weights)
        dates.append(val_env.idx[val_env._t])
        if terminated or truncated:
            break

    weights_df = pd.DataFrame(weights_records, index=dates, columns=val_env.assets)
    equity = backtest_weights(
        returns=val_env.returns[val_env.assets],
        weight_series=weights_df,
    )
    metrics = compute_metrics(equity)

    # Optimize for Sharpe with drawdown penalty
    score = metrics["Sharpe"] + 0.5 * metrics["CAGR"] + 0.2 * metrics["TotalReturn"] + 0.3 * min(metrics["MaxDrawdown"], 0)
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--n_trials", type=int, default=20)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    factor_cfg = cfg["factors"]

    fetcher = DailyDataFetcher(cache_dir=data_cfg["cache_dir"])
    panels = fetcher.fetch(
        tickers=data_cfg["tickers"],
        start_date=data_cfg["start_date"],
        end_date=data_cfg.get("end_date"),
        interval=data_cfg.get("interval", "1d"),
        refresh=data_cfg.get("refresh", False),
    )
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

    study = optuna.create_study(direction="maximize", study_name=cfg["optuna"]["study_name"])
    study.optimize(lambda t: objective(t, cfg, factors, close_prices), n_trials=args.n_trials)
    print("Best trial:")
    print(study.best_trial.params)
    print(study.best_value)


if __name__ == "__main__":
    main()

