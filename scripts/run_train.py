import argparse
import os

import pandas as pd
import yaml

from crypto_rl.data.fetcher import DailyDataFetcher
from crypto_rl.factors.technical import compute_technical_factors
from crypto_rl.models.train import TrainConfig, train_ppo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    factor_cfg = cfg["factors"]
    env_cfg = cfg["env"]
    train_cfg = cfg["train"]
    seed = cfg.get("random_seed", 42)

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

    tc = TrainConfig(
        total_timesteps=int(train_cfg["total_timesteps"]),
        policy=train_cfg["policy"],
        learning_rate=float(train_cfg["learning_rate"]),
        batch_size=int(train_cfg["batch_size"]),
        n_steps=int(train_cfg["n_steps"]),
        gamma=float(train_cfg["gamma"]),
        gae_lambda=float(train_cfg["gae_lambda"]),
        clip_range=float(train_cfg["clip_range"]),
        ent_coef=float(train_cfg["ent_coef"]),
        vf_coef=float(train_cfg["vf_coef"]),
        device=train_cfg.get("device", "auto"),
        artifacts_dir=train_cfg["artifacts_dir"],
    )

    save_path = train_ppo(
        factors=factors,
        close_prices=close_prices,
        env_cfg=env_cfg,
        train_cfg=tc,
        train_range=(train_cfg["train_start"], train_cfg["train_end"]),
        val_range=(train_cfg["val_start"], train_cfg["val_end"]),
        seed=seed,
    )

    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()

