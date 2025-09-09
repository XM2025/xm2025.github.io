import os
import argparse
import yaml
import random
from typing import Dict, List

import numpy as np
import torch

from crypto_ql.utils import ensure_dir
from crypto_ql.data_loader import DataConfig, load_ohlcv, align_panel
from crypto_ql.factors import build_feature_panel
from crypto_ql.portfolio_env import PortfolioEnv, EnvConfig
from crypto_ql.ppo import PPOAgent, PPOConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='config.yaml')
    p.add_argument('--trials', type=int, default=10)
    return p.parse_args()


def read_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def sample_hparams(base_cfg: Dict) -> Dict:
    cfg = dict(base_cfg)
    rl = dict(cfg['rl'])
    rl['learning_rate'] = 10 ** random.uniform(-4.5, -3)
    rl['entropy_coef'] = 10 ** random.uniform(-3, -1)
    rl['clip_range'] = random.uniform(0.1, 0.3)
    rl['hidden_sizes'] = [random.choice([128, 256, 512]), random.choice([128, 256, 512])]
    cfg['rl'] = rl
    return cfg


def evaluate_once(cfg: Dict) -> float:
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
    if cfg['features']['include_prev_weights']:
        prev_w = np.zeros((X.shape[0], X.shape[1], 1), dtype=np.float32)
        X = np.concatenate([X, prev_w], axis=2)

    idx_train = np.where(dates <= np.datetime64(cfg['split']['train_end']))[0]
    idx_val = np.where((dates > np.datetime64(cfg['split']['train_end'])) & (dates <= np.datetime64(cfg['split']['val_end'])))[0]
    X_train, R_train, dates_train = X[idx_train], R[idx_train], dates[idx_train]
    X_val, R_val, dates_val = X[idx_val], R[idx_val], dates[idx_val]

    env = PortfolioEnv(dates_train, X_train, R_train, include_cash=True, prev_weight_in_features=cfg['features']['include_prev_weights'], cfg=EnvConfig(
        trading_cost_rate=float(cfg['env']['trading_cost_rate']),
        reward=str(cfg['env']['reward'])
    ))
    obs_dim = X_train.shape[1] * X_train.shape[2]
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
    # short training
    total_timesteps = int(cfg['rl']['total_timesteps']) // 4
    rollout_length = int(cfg['rl']['rollout_length'])
    env.reset(start_index=1)
    t = 0
    while t < total_timesteps:
        env.features[env.t, :, -1] = env.weights[:-1] if env.action_dim == env.num_assets + 1 else env.weights
        buffer = {
            **__import__('train').rollout_collect(env, agent, rollout_length)
        }
        agent.update(buffer)
        t += len(buffer['obs'])
    # validation NAV
    if len(dates_val) < 2:
        return 0.0
    val_env = PortfolioEnv(dates_val, X_val.copy(), R_val.copy(), include_cash=True, prev_weight_in_features=cfg['features']['include_prev_weights'], cfg=EnvConfig(
        trading_cost_rate=float(cfg['env']['trading_cost_rate']),
        reward=str(cfg['env']['reward'])
    ))
    val_env.reset(start_index=1)
    nav = 1.0
    done = False
    while not done:
        obs = val_env._get_obs().reshape(1, -1)
        with torch.no_grad():
            mu, log_std, _ = agent.model(torch.as_tensor(obs, dtype=torch.float32))
            action = mu.squeeze(0).cpu().numpy()
        _, r, done, info = val_env.step(action)
        nav = info['nav']
    return nav


def main():
    args = parse_args()
    base_cfg = read_config(args.config)
    best_nav = -1.0
    best_cfg = None
    for i in range(args.trials):
        trial_cfg = sample_hparams(base_cfg)
        nav = evaluate_once(trial_cfg)
        print(f"trial {i+1}/{args.trials}: NAV={nav:.4f} | lr={trial_cfg['rl']['learning_rate']:.2e} ent={trial_cfg['rl']['entropy_coef']:.2e} clip={trial_cfg['rl']['clip_range']:.2f} hidden={trial_cfg['rl']['hidden_sizes']}")
        if nav > best_nav:
            best_nav = nav
            best_cfg = trial_cfg
    ensure_dir(base_cfg['output']['outputs_dir'])
    if best_cfg is not None:
        import json
        # convert any non-serializable types to str
        def default(o):
            try:
                return str(o)
            except Exception:
                return repr(o)
        with open(os.path.join(base_cfg['output']['outputs_dir'], 'best_config.json'), 'w') as f:
            json.dump(best_cfg, f, indent=2, default=default)
        print("Saved best_config.json")


if __name__ == '__main__':
    main()

