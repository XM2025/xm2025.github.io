import os
import argparse
import yaml
from typing import Dict

import numpy as np
import pandas as pd
import torch

from crypto_ql.utils import ensure_dir, compute_nav, metrics_summary, plot_nav, save_json
from crypto_ql.data_loader import DataConfig, load_ohlcv, align_panel
from crypto_ql.factors import build_feature_panel
from crypto_ql.portfolio_env import PortfolioEnv, EnvConfig
from crypto_ql.ppo import PPOAgent, PPOConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='config.yaml')
    p.add_argument('--checkpoint', type=str, required=True)
    return p.parse_args()


def read_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = read_config(args.config)

    data_cfg = DataConfig(
        use_ccxt=cfg['data']['use_ccxt'],
        exchange=cfg['data']['exchange'],
        timeframe=cfg['data']['timeframe'],
        limit=int(cfg['data']['limit']),
        cache_dir=cfg['data']['cache_dir'],
        symbols=list(cfg['data']['symbols']),
        start=cfg['data']['start'],
        end=cfg['data']['end'],
        align=cfg['data']['align'],
    )
    symbol_to_df = load_ohlcv(data_cfg, data_dir='data')
    dates, aligned = align_panel(symbol_to_df, align=data_cfg.align)
    dates, X, R, feature_names = build_feature_panel(
        aligned,
        feature_list=list(cfg['features']['list']),
        normalize_cross_section=bool(cfg['features']['normalize_cross_section']),
        include_prev_weights=bool(cfg['features']['include_prev_weights']),
    )

    include_prev_weights = bool(cfg['features']['include_prev_weights'])
    if include_prev_weights:
        prev_w = np.zeros((X.shape[0], X.shape[1], 1), dtype=np.float32)
        X = np.concatenate([X, prev_w], axis=2)

    idx_test = np.where(dates > pd.to_datetime(cfg['split']['val_end']))[0]
    X_test, R_test, dates_test = X[idx_test], R[idx_test], dates[idx_test]

    env = PortfolioEnv(dates_test, X_test, R_test, include_cash=True, prev_weight_in_features=include_prev_weights, cfg=EnvConfig(
        trading_cost_rate=float(cfg['env']['trading_cost_rate']),
        reward=str(cfg['env']['reward'])
    ))

    obs_dim = X_test.shape[1] * X_test.shape[2]
    action_dim = env.action_dim
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, cfg=PPOConfig(
        learning_rate=float(cfg['rl']['learning_rate']),
        gamma=float(cfg['rl']['gamma']),
        gae_lambda=float(cfg['rl']['gae_lambda']),
        clip_range=float(cfg['rl']['clip_range']),
        entropy_coef=float(cfg['rl']['entropy_coef']),
        value_coef=float(cfg['rl']['value_coef']),
        max_grad_norm=float(cfg['rl']['max_grad_norm']),
        hidden_sizes=list(cfg['rl']['hidden_sizes']),
        minibatch_size=int(cfg['rl']['minibatch_size']),
        n_epochs=int(cfg['rl']['n_epochs']),
    ))
    agent.model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    agent.model.eval()

    ensure_dir(cfg['output']['outputs_dir'])
    out_dir = cfg['output']['outputs_dir']
    turns = []
    rets = []
    env.reset(start_index=1)
    done = False
    while not done:
        obs = env._get_obs().reshape(1, -1)
        with torch.no_grad():
            mu, log_std, _ = agent.model(torch.as_tensor(obs, dtype=torch.float32))
            action = mu.squeeze(0).cpu().numpy()
        obs, r, done, info = env.step(action)
        rets.append(info['after_cost_return'])
        turns.append(info['turnover'])

    rets = np.array(rets, dtype=np.float32)
    turns = np.array(turns, dtype=np.float32)
    nav = compute_nav(rets)
    metrics = metrics_summary(rets, turns)
    plot_nav(list(dates_test[1:1+len(nav)]), nav, os.path.join(out_dir, 'nav.png'))
    save_json(metrics, os.path.join(out_dir, 'metrics.json'))
    print(metrics)


if __name__ == '__main__':
    main()

