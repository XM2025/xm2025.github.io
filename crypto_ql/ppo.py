from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: List[int] = None  # type: ignore
    minibatch_size: int = 64
    n_epochs: int = 10

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 256]


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.mlp = nn.Sequential(*layers)
        # Policy outputs mean of Gaussian over action logits (pre-softmax)
        self.mu = nn.Linear(last, action_dim)
        # Log standard deviation as a parameter per action dim (state-independent)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # Value head
        self.v = nn.Linear(last, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.mlp(x)
        mu = self.mu(z)
        log_std = self.log_std.clamp(-5.0, 2.0).expand_as(mu)
        value = self.v(z).squeeze(-1)
        return mu, log_std, value


class PPOAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: PPOConfig):
        self.cfg = cfg
        self.model = ActorCritic(obs_dim, action_dim, cfg.hidden_sizes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        self.model.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32)
            mu, log_std, value = self.model(x)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            logp = dist.log_prob(action).sum(-1)
        return action.cpu().numpy(), float(logp.cpu().numpy()), float(value.cpu().numpy())

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std, value = self.model(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, value

    def update(self, buffer: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = buffer["obs"]
        actions = buffer["actions"]
        old_logp = buffer["logp"]
        returns = buffer["returns"]
        advantages = buffer["advantages"]
        n = obs.shape[0]
        idx = torch.randperm(n)
        losses = {}
        for _ in range(self.cfg.n_epochs):
            for start in range(0, n, self.cfg.minibatch_size):
                end = min(n, start + self.cfg.minibatch_size)
                b = idx[start:end]
                logp, entropy, value = self.evaluate_actions(obs[b], actions[b])
                ratio = torch.exp(logp - old_logp[b])
                adv = advantages[b]
                pg1 = ratio * adv
                pg2 = torch.clamp(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * adv
                policy_loss = -torch.min(pg1, pg2).mean()
                value_loss = F.mse_loss(value, returns[b])
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.cfg.value_coef * value_loss + self.cfg.entropy_coef * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                losses = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.mean().item()),
                }
        return losses

