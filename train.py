import os
import argparse
import yaml
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from crypto_ql.utils import ensure_dir, set_seed
from crypto_ql.data_loader import DataConfig, load_ohlcv, align_panel
from crypto_ql.factors import build_feature_panel
from crypto_ql.portfolio_env import PortfolioEnv, EnvConfig
from crypto_ql.ppo import PPOAgent, PPOConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='config.yaml')
    return p.parse_args()


def read_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def split_indices(dates: pd.DatetimeIndex, train_end: str, val_end: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_train = np.where(dates <= pd.to_datetime(train_end))[0]
    idx_val = np.where((dates > pd.to_datetime(train_end)) & (dates <= pd.to_datetime(val_end)))[0]
    idx_test = np.where(dates > pd.to_datetime(val_end))[0]
    return idx_train, idx_val, idx_test


def rollout_collect(env: PortfolioEnv, agent: PPOAgent, rollout_length: int) -> Dict[str, torch.Tensor]:
    obs_list = []
    act_list = []
    logp_list = []
    val_list = []
    rew_list = []
    done_list = []
    obs = env._get_obs()
    for _ in range(rollout_length):
        x = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            mu, log_std, value = agent.model(x)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            logp = dist.log_prob(action).sum(-1)
        action_np = action.squeeze(0).cpu().numpy()
        next_obs, reward, done, info = env.step(action_np)
        obs_list.append(obs.reshape(-1))
        act_list.append(action_np)
        logp_list.append(float(logp.squeeze(0).item()))
        val_list.append(float(value.squeeze(0).item()))
        rew_list.append(reward)
        done_list.append(done)
        obs = next_obs
        if done:
            break
    # compute GAE
    rewards = torch.as_tensor(rew_list, dtype=torch.float32)
    values = torch.as_tensor(val_list, dtype=torch.float32)
    dones = torch.as_tensor(done_list, dtype=torch.float32)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0.0
    for t in reversed(range(len(rewards))):
        nextnonterminal = 1.0 - (dones[t] if t < len(dones) else 1.0)
        nextvalue = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + agent.cfg.gamma * nextvalue * nextnonterminal - values[t]
        lastgaelam = delta + agent.cfg.gamma * agent.cfg.gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    # pack buffer
    buffer = {
        'obs': torch.as_tensor(np.stack(obs_list), dtype=torch.float32),
        'actions': torch.as_tensor(np.array(act_list), dtype=torch.float32),
        'logp': torch.as_tensor(np.array(logp_list), dtype=torch.float32),
        'returns': returns.detach(),
        'advantages': (advantages - advantages.mean()) / (advantages.std() + 1e-8),
    }
    return buffer


def main():
    args = parse_args()
    cfg = read_config(args.config)
    set_seed(cfg.get('seed', 42))

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

    # append prev weights as extra feature (initialized to zero, updated online during env stepping handled externally if needed)
    include_prev_weights = bool(cfg['features']['include_prev_weights'])
    if include_prev_weights:
        prev_w = np.zeros((X.shape[0], X.shape[1], 1), dtype=np.float32)
        X = np.concatenate([X, prev_w], axis=2)

    idx_train, idx_val, idx_test = split_indices(dates, cfg['split']['train_end'], cfg['split']['val_end'])
    # Train slice
    X_train, R_train, dates_train = X[idx_train], R[idx_train], dates[idx_train]

    env = PortfolioEnv(dates_train, X_train, R_train, include_cash=True, prev_weight_in_features=include_prev_weights, cfg=EnvConfig(
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

    total_timesteps = int(cfg['rl']['total_timesteps'])
    rollout_length = int(cfg['rl']['rollout_length'])
    eval_interval = int(cfg['rl']['eval_interval'])

    ensure_dir(cfg['output']['checkpoints_dir'])
    ensure_dir(cfg['output']['logs_dir'])

    t = 0
    best_val_nav = -1.0
    best_path = None
    obs, _ = env.reset(start_index=1)
    while t < total_timesteps:
        if env.t >= env.num_steps:
            env.reset(start_index=1)
        # env maintains prev-weight channel internally now
        buffer = rollout_collect(env, agent, rollout_length)
        losses = agent.update(buffer)
        t += len(buffer['obs'])

        if t % eval_interval < rollout_length:
            # validation
            X_val, R_val, dates_val = X[idx_val], R[idx_val], dates[idx_val]
            if len(dates_val) > 10:
                val_env = PortfolioEnv(dates_val, X_val.copy(), R_val.copy(), include_cash=True, cfg=EnvConfig(
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
                    obs, r, done, info = val_env.step(action)
                    nav = info['nav']
                if nav > best_val_nav:
                    best_val_nav = nav
                    best_path = os.path.join(cfg['output']['checkpoints_dir'], 'best.pt')
                    torch.save(agent.model.state_dict(), best_path)

    if best_path is None:
        best_path = os.path.join(cfg['output']['checkpoints_dir'], 'last.pt')
        torch.save(agent.model.state_dict(), best_path)
    print(f"Saved model to {best_path}")


if __name__ == '__main__':
    main()

