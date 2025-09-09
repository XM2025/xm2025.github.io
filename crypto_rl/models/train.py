from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from crypto_rl.env.portfolio_env import DailyPortfolioEnv


@dataclass
class TrainConfig:
    total_timesteps: int
    policy: str
    learning_rate: float
    batch_size: int
    n_steps: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    device: str
    artifacts_dir: str


def split_data_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))
    return df.loc[mask]


def build_env(
    factors: pd.DataFrame,
    close_prices: pd.DataFrame,
    include_cash: bool,
    transaction_cost_bps: float,
    risk_aversion: float,
    max_weight: float,
    start: str,
    end: str,
):
    idx = factors.index
    mask = (idx >= pd.to_datetime(start)) & (idx <= pd.to_datetime(end))
    env = DailyPortfolioEnv(
        factor_matrix=factors.loc[mask],
        close_prices=close_prices.loc[mask],
        include_cash=include_cash,
        transaction_cost_bps=transaction_cost_bps,
        risk_aversion=risk_aversion,
        max_weight=max_weight,
    )
    return env


def train_ppo(
    factors: pd.DataFrame,
    close_prices: pd.DataFrame,
    env_cfg: dict,
    train_cfg: TrainConfig,
    train_range: tuple[str, str],
    val_range: tuple[str, str] | None = None,
    seed: int = 42,
):
    os.makedirs(train_cfg.artifacts_dir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = build_env(
        factors=factors,
        close_prices=close_prices,
        include_cash=env_cfg.get("include_cash", True),
        transaction_cost_bps=env_cfg.get("transaction_cost_bps", 5.0),
        risk_aversion=env_cfg.get("risk_aversion", 2.0),
        max_weight=env_cfg.get("max_weight", 0.35),
        start=train_range[0],
        end=train_range[1],
    )

    vec_env = DummyVecEnv([lambda: env])

    model = PPO(
        policy=train_cfg.policy,
        env=vec_env,
        learning_rate=train_cfg.learning_rate,
        n_steps=train_cfg.n_steps,
        batch_size=train_cfg.batch_size,
        gamma=train_cfg.gamma,
        gae_lambda=train_cfg.gae_lambda,
        clip_range=train_cfg.clip_range,
        ent_coef=train_cfg.ent_coef,
        vf_coef=train_cfg.vf_coef,
        verbose=1,
        device=train_cfg.device,
        seed=seed,
    )

    model.learn(total_timesteps=train_cfg.total_timesteps)

    save_path = os.path.join(train_cfg.artifacts_dir, "ppo_model")
    model.save(save_path)

    return save_path

