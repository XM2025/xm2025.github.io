"""
深度强化学习智能体
实现PPO、A2C、SAC等算法用于加密货币交易
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import deque
import random

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class Actor(nn.Module):
    """策略网络 (Actor)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [512, 256, 128]):
        super(Actor, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # 输出层
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.feature_layers(state)
        
        mean = torch.tanh(self.mean_layer(features))  # 限制在 [-1, 1]
        log_std = torch.clamp(self.log_std_layer(features), -20, 2)  # 限制标准差范围
        
        return mean, log_std
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取动作"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            action = mean
            log_prob = None
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob

class Critic(nn.Module):
    """价值网络 (Critic)"""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [512, 256, 128]):
        super(Critic, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(state)

class PPOAgent:
    """PPO智能体"""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 gamma: float = 0.99,
                 lambda_gae: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coeff: float = 0.01,
                 value_coeff: float = 0.5,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 10,
                 batch_size: int = 64):
        """
        初始化PPO智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            gamma: 折扣因子
            lambda_gae: GAE参数
            clip_epsilon: PPO裁剪参数
            entropy_coeff: 熵系数
            value_coeff: 价值损失系数
            max_grad_norm: 梯度裁剪阈值
            ppo_epochs: PPO更新轮数
            batch_size: 批次大小
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # 创建网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        
        # 创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 经验缓冲区
        self.buffer = PPOBuffer()
        
        # 日志
        self.logger = logging.getLogger(__name__)
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """获取动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_tensor, deterministic)
            value = self.critic(state_tensor)
        
        action = action.cpu().numpy().squeeze()
        log_prob = log_prob.cpu().numpy().item() if log_prob is not None else 0.0
        value = value.cpu().numpy().item()
        
        return action, log_prob, value
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, 
                        next_state: np.ndarray, done: bool, log_prob: float, value: float):
        """存储转换"""
        self.buffer.store(state, action, reward, next_state, done, log_prob, value)
    
    def update(self) -> Dict[str, float]:
        """更新网络"""
        if len(self.buffer) < self.batch_size:
            return {}
        
        # 获取批次数据
        batch = self.buffer.get()
        
        states = torch.FloatTensor(batch['states']).to(device)
        actions = torch.FloatTensor(batch['actions']).to(device)
        rewards = torch.FloatTensor(batch['rewards']).to(device)
        next_states = torch.FloatTensor(batch['next_states']).to(device)
        dones = torch.BoolTensor(batch['dones']).to(device)
        old_log_probs = torch.FloatTensor(batch['log_probs']).to(device)
        old_values = torch.FloatTensor(batch['values']).to(device)
        
        # 计算GAE和目标值
        advantages, returns = self._compute_gae(rewards, old_values, dones)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for _ in range(self.ppo_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # 计算当前策略的动作概率
                mean, log_std = self.actor(batch_states)
                std = log_std.exp()
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1, keepdim=True)
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO裁剪损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 熵损失
                entropy_loss = -self.entropy_coeff * entropy.mean()
                
                # 总Actor损失
                total_actor_loss_batch = actor_loss + entropy_loss
                
                # 更新Actor
                self.actor_optimizer.zero_grad()
                total_actor_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # 计算Critic损失
                values = self.critic(batch_states)
                critic_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # 更新Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
        
        # 清空缓冲区
        self.buffer.clear()
        
        return {
            'actor_loss': total_actor_loss / self.ppo_epochs,
            'critic_loss': total_critic_loss / self.ppo_epochs,
            'entropy': total_entropy / self.ppo_epochs
        }
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算GAE优势和回报"""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t].float()
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t].float()
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.lambda_gae * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        return advantages, returns
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

class PPOBuffer:
    """PPO经验缓冲区"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = {
            'states': deque(maxlen=capacity),
            'actions': deque(maxlen=capacity),
            'rewards': deque(maxlen=capacity),
            'next_states': deque(maxlen=capacity),
            'dones': deque(maxlen=capacity),
            'log_probs': deque(maxlen=capacity),
            'values': deque(maxlen=capacity)
        }
    
    def store(self, state, action, reward, next_state, done, log_prob, value):
        """存储经验"""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['next_states'].append(next_state)
        self.buffer['dones'].append(done)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['values'].append(value)
    
    def get(self):
        """获取所有经验"""
        return {key: np.array(list(values)) for key, values in self.buffer.items()}
    
    def clear(self):
        """清空缓冲区"""
        for key in self.buffer:
            self.buffer[key].clear()
    
    def __len__(self):
        return len(self.buffer['states'])

class A2CAgent:
    """A2C智能体"""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 entropy_coeff: float = 0.01,
                 value_coeff: float = 0.5,
                 max_grad_norm: float = 0.5):
        """
        初始化A2C智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 学习率
            gamma: 折扣因子
            entropy_coeff: 熵系数
            value_coeff: 价值损失系数
            max_grad_norm: 梯度裁剪阈值
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        
        # 创建网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        
        # 创建优化器
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=lr
        )
        
        # 经验缓冲区
        self.buffer = []
        
        # 日志
        self.logger = logging.getLogger(__name__)
    
    def get_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """获取动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_tensor)
            value = self.critic(state_tensor)
        
        action = action.cpu().numpy().squeeze()
        log_prob = log_prob.cpu().numpy().item()
        value = value.cpu().numpy().item()
        
        return action, log_prob, value
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """存储转换"""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value
        })
    
    def update(self) -> Dict[str, float]:
        """更新网络"""
        if len(self.buffer) == 0:
            return {}
        
        # 计算折扣回报
        returns = []
        discounted_sum = 0
        
        for transition in reversed(self.buffer):
            if transition['done']:
                discounted_sum = 0
            discounted_sum = transition['reward'] + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        
        # 转换为张量
        states = torch.FloatTensor([t['state'] for t in self.buffer]).to(device)
        actions = torch.FloatTensor([t['action'] for t in self.buffer]).to(device)
        old_log_probs = torch.FloatTensor([t['log_prob'] for t in self.buffer]).to(device)
        old_values = torch.FloatTensor([t['value'] for t in self.buffer]).to(device)
        returns = torch.FloatTensor(returns).to(device)
        
        # 计算优势
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算当前策略的动作概率和价值
        mean, log_std = self.actor(states)
        std = log_std.exp()
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        values = self.critic(states)
        
        # 计算损失
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values.squeeze(), returns)
        entropy_loss = -self.entropy_coeff * entropy.mean()
        
        total_loss = actor_loss + self.value_coeff * critic_loss + entropy_loss
        
        # 更新网络
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            self.max_grad_norm
        )
        self.optimizer.step()
        
        # 清空缓冲区
        self.buffer.clear()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.mean().item(),
            'total_loss': total_loss.item()
        }
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class SACAgent:
    """SAC (Soft Actor-Critic) 智能体"""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 buffer_size: int = 1000000,
                 batch_size: int = 256):
        """
        初始化SAC智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 学习率
            gamma: 折扣因子
            tau: 软更新参数
            alpha: 熵正则化参数
            buffer_size: 经验回放缓冲区大小
            batch_size: 批次大小
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        
        # 创建网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic1 = Critic(state_dim + action_dim).to(device)
        self.critic2 = Critic(state_dim + action_dim).to(device)
        self.target_critic1 = Critic(state_dim + action_dim).to(device)
        self.target_critic2 = Critic(state_dim + action_dim).to(device)
        
        # 复制参数到目标网络
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 日志
        self.logger = logging.getLogger(__name__)
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """获取动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action, _ = self.actor.get_action(state_tensor, deterministic)
        
        return action.cpu().numpy().squeeze()
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转换"""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """更新网络"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # 采样批次
        batch = self.replay_buffer.sample(self.batch_size)
        
        state = torch.FloatTensor(batch['state']).to(device)
        action = torch.FloatTensor(batch['action']).to(device)
        reward = torch.FloatTensor(batch['reward']).to(device)
        next_state = torch.FloatTensor(batch['next_state']).to(device)
        done = torch.BoolTensor(batch['done']).to(device)
        
        # 更新Critic
        with torch.no_grad():
            next_action, next_log_prob = self.actor.get_action(next_state)
            next_state_action = torch.cat([next_state, next_action], dim=1)
            
            target_q1 = self.target_critic1(next_state_action)
            target_q2 = self.target_critic2(next_state_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            
            target_q = reward.unsqueeze(1) + (~done).unsqueeze(1) * self.gamma * target_q
        
        state_action = torch.cat([state, action], dim=1)
        current_q1 = self.critic1(state_action)
        current_q2 = self.critic2(state_action)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # 更新Actor
        new_action, log_prob = self.actor.get_action(state)
        new_state_action = torch.cat([state, new_action], dim=1)
        
        q1 = self.critic1(new_state_action)
        q2 = self.critic2(new_state_action)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_prob - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item()
        }
    
    def _soft_update(self, local_model, target_model):
        """软更新目标网络"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, state, action, reward, next_state, done):
        """添加经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """采样批次"""
        batch = random.sample(self.buffer, batch_size)
        
        return {
            'state': np.array([e['state'] for e in batch]),
            'action': np.array([e['action'] for e in batch]),
            'reward': np.array([e['reward'] for e in batch]),
            'next_state': np.array([e['next_state'] for e in batch]),
            'done': np.array([e['done'] for e in batch])
        }
    
    def __len__(self):
        return len(self.buffer)

# 使用示例
if __name__ == "__main__":
    # 测试参数
    state_dim = 100
    action_dim = 4
    
    print("测试PPO智能体...")
    ppo_agent = PPOAgent(state_dim, action_dim)
    
    # 测试动作获取
    state = np.random.randn(state_dim)
    action, log_prob, value = ppo_agent.get_action(state)
    print(f"PPO动作形状: {action.shape}, 对数概率: {log_prob:.4f}, 价值: {value:.4f}")
    
    print("\n测试A2C智能体...")
    a2c_agent = A2CAgent(state_dim, action_dim)
    
    # 测试动作获取
    action, log_prob, value = a2c_agent.get_action(state)
    print(f"A2C动作形状: {action.shape}, 对数概率: {log_prob:.4f}, 价值: {value:.4f}")
    
    print("\n测试SAC智能体...")
    sac_agent = SACAgent(state_dim, action_dim)
    
    # 测试动作获取
    action = sac_agent.get_action(state)
    print(f"SAC动作形状: {action.shape}")
    
    print("\n所有智能体测试完成!")