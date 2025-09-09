"""
强化学习交易环境
模拟真实的加密货币交易场景
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from enum import Enum

class TradingAction(Enum):
    """交易动作枚举"""
    HOLD = 0
    BUY = 1
    SELL = 2

class TradingEnvironment(gym.Env):
    """加密货币交易环境"""
    
    def __init__(self, 
                 data_dict: Dict[str, pd.DataFrame],
                 initial_balance: float = 100000.0,
                 transaction_fee: float = 0.001,
                 max_position_size: float = 0.95,
                 lookback_window: int = 60,
                 reward_scaling: float = 1.0,
                 risk_free_rate: float = 0.02):
        """
        初始化交易环境
        
        Args:
            data_dict: 包含特征的数据字典
            initial_balance: 初始资金
            transaction_fee: 交易手续费率
            max_position_size: 最大仓位比例
            lookback_window: 观察窗口大小
            reward_scaling: 奖励缩放因子
            risk_free_rate: 无风险利率
        """
        super(TradingEnvironment, self).__init__()
        
        self.data_dict = data_dict
        self.symbols = list(data_dict.keys())
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        self.reward_scaling = reward_scaling
        self.risk_free_rate = risk_free_rate
        
        # 环境状态
        self.current_step = 0
        self.balance = initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}  # 持仓数量
        self.portfolio_value = initial_balance
        self.transaction_costs = 0.0
        
        # 历史记录
        self.portfolio_history = []
        self.action_history = []
        self.reward_history = []
        self.trade_history = []
        
        # 准备数据
        self._prepare_data()
        
        # 定义动作空间和观察空间
        self._define_spaces()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _prepare_data(self):
        """准备训练数据"""
        # 获取所有数据的共同时间索引
        common_index = None
        for symbol, df in self.data_dict.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        # 对齐所有数据到共同索引
        self.aligned_data = {}
        for symbol, df in self.data_dict.items():
            self.aligned_data[symbol] = df.loc[common_index].copy()
        
        # 获取特征列名
        self.feature_columns = [col for col in self.aligned_data[self.symbols[0]].columns 
                               if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        self.data_length = len(common_index)
        self.timestamps = common_index
        
        print(f"数据准备完成: {len(self.symbols)} 个交易对, {self.data_length} 个时间点, {len(self.feature_columns)} 个特征")
    
    def _define_spaces(self):
        """定义动作空间和观察空间"""
        # 动作空间: 每个资产的动作 (持有/买入/卖出) + 交易量比例
        # 动作维度: len(symbols) * 2 (动作类型 + 交易量)
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(len(self.symbols) * 2,), 
            dtype=np.float32
        )
        
        # 观察空间维度计算
        # 特征维度: lookback_window * len(symbols) * len(feature_columns)
        # 投资组合状态: len(symbols) + 3 (各资产持仓比例 + 现金比例 + 总收益率 + 夏普比率)
        feature_dim = self.lookback_window * len(self.symbols) * len(self.feature_columns)
        portfolio_dim = len(self.symbols) + 3
        
        obs_dim = feature_dim + portfolio_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        print(f"动作空间维度: {self.action_space.shape}")
        print(f"观察空间维度: {self.observation_space.shape}")
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.portfolio_value = self.initial_balance
        self.transaction_costs = 0.0
        
        # 清空历史记录
        self.portfolio_history = []
        self.action_history = []
        self.reward_history = []
        self.trade_history = []
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步交易"""
        # 解析动作
        actions = self._parse_action(action)
        
        # 记录当前投资组合价值
        prev_portfolio_value = self.portfolio_value
        
        # 执行交易
        trade_info = self._execute_trades(actions)
        
        # 更新投资组合价值
        self._update_portfolio_value()
        
        # 计算奖励
        reward = self._calculate_reward(prev_portfolio_value)
        
        # 记录历史
        self.portfolio_history.append(self.portfolio_value)
        self.action_history.append(actions)
        self.reward_history.append(reward)
        self.trade_history.extend(trade_info)
        
        # 移动到下一步
        self.current_step += 1
        
        # 检查是否结束
        done = self.current_step >= self.data_length - 1
        
        # 准备info字典
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'transaction_costs': self.transaction_costs,
            'trades': trade_info,
            'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance
        }
        
        return self._get_observation(), reward * self.reward_scaling, done, info
    
    def _parse_action(self, action: np.ndarray) -> Dict[str, Dict]:
        """解析动作向量"""
        actions = {}
        
        for i, symbol in enumerate(self.symbols):
            # 动作类型 (-1: 卖出, 0: 持有, 1: 买入)
            action_type = np.tanh(action[i * 2])  # 映射到 [-1, 1]
            # 交易量比例 [0, 1]
            amount_ratio = (np.tanh(action[i * 2 + 1]) + 1) / 2  # 映射到 [0, 1]
            
            actions[symbol] = {
                'type': action_type,
                'amount_ratio': amount_ratio
            }
        
        return actions
    
    def _execute_trades(self, actions: Dict[str, Dict]) -> List[Dict]:
        """执行交易"""
        trade_info = []
        
        for symbol, action in actions.items():
            current_price = self.aligned_data[symbol].iloc[self.current_step]['close']
            action_type = action['type']
            amount_ratio = action['amount_ratio']
            
            # 确定交易方向和数量
            if action_type > 0.1:  # 买入
                trade_type = 'BUY'
                # 计算可买入的最大金额
                max_buy_value = self.balance * self.max_position_size * amount_ratio
                # 扣除手续费后的实际买入金额
                actual_buy_value = max_buy_value / (1 + self.transaction_fee)
                buy_quantity = actual_buy_value / current_price
                
                if actual_buy_value > 1.0:  # 最小交易金额
                    # 执行买入
                    cost = actual_buy_value + actual_buy_value * self.transaction_fee
                    if cost <= self.balance:
                        self.balance -= cost
                        self.positions[symbol] += buy_quantity
                        self.transaction_costs += actual_buy_value * self.transaction_fee
                        
                        trade_info.append({
                            'symbol': symbol,
                            'type': trade_type,
                            'quantity': buy_quantity,
                            'price': current_price,
                            'value': actual_buy_value,
                            'fee': actual_buy_value * self.transaction_fee,
                            'timestamp': self.timestamps[self.current_step]
                        })
            
            elif action_type < -0.1:  # 卖出
                trade_type = 'SELL'
                # 计算要卖出的数量
                sell_quantity = self.positions[symbol] * amount_ratio
                
                if sell_quantity > 0:
                    # 执行卖出
                    sell_value = sell_quantity * current_price
                    fee = sell_value * self.transaction_fee
                    net_proceeds = sell_value - fee
                    
                    self.balance += net_proceeds
                    self.positions[symbol] -= sell_quantity
                    self.transaction_costs += fee
                    
                    trade_info.append({
                        'symbol': symbol,
                        'type': trade_type,
                        'quantity': sell_quantity,
                        'price': current_price,
                        'value': sell_value,
                        'fee': fee,
                        'timestamp': self.timestamps[self.current_step]
                    })
        
        return trade_info
    
    def _update_portfolio_value(self):
        """更新投资组合价值"""
        total_position_value = 0.0
        
        for symbol, quantity in self.positions.items():
            current_price = self.aligned_data[symbol].iloc[self.current_step]['close']
            total_position_value += quantity * current_price
        
        self.portfolio_value = self.balance + total_position_value
    
    def _calculate_reward(self, prev_portfolio_value: float) -> float:
        """计算奖励"""
        # 基础收益奖励
        portfolio_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # 风险调整奖励
        if len(self.portfolio_history) > 1:
            returns = np.array(self.portfolio_history[-30:])  # 最近30天
            returns = np.diff(returns) / returns[:-1]
            
            if len(returns) > 1:
                volatility = np.std(returns)
                if volatility > 0:
                    sharpe_ratio = (np.mean(returns) - self.risk_free_rate/365) / volatility
                    risk_adjusted_reward = portfolio_return + 0.1 * sharpe_ratio
                else:
                    risk_adjusted_reward = portfolio_return
            else:
                risk_adjusted_reward = portfolio_return
        else:
            risk_adjusted_reward = portfolio_return
        
        # 交易成本惩罚
        if len(self.trade_history) > 0:
            recent_fees = sum([trade['fee'] for trade in self.trade_history[-10:]])  # 最近10笔交易
            fee_penalty = recent_fees / self.portfolio_value
        else:
            fee_penalty = 0
        
        # 持仓分散度奖励
        diversification_reward = self._calculate_diversification_reward()
        
        # 最终奖励
        total_reward = risk_adjusted_reward - fee_penalty + diversification_reward
        
        return total_reward
    
    def _calculate_diversification_reward(self) -> float:
        """计算分散度奖励"""
        if self.portfolio_value == 0:
            return 0
        
        # 计算各资产的权重
        weights = []
        for symbol in self.symbols:
            current_price = self.aligned_data[symbol].iloc[self.current_step]['close']
            position_value = self.positions[symbol] * current_price
            weight = position_value / self.portfolio_value
            weights.append(weight)
        
        weights = np.array(weights)
        
        # 计算Herfindahl指数 (集中度指标)
        hhi = np.sum(weights ** 2)
        
        # 分散度奖励 (HHI越小，分散度越高)
        diversification_score = (1 - hhi) * 0.01  # 小的正奖励
        
        return diversification_score
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观察状态"""
        # 特征数据 (lookback_window * symbols * features)
        feature_data = []
        
        for symbol in self.symbols:
            symbol_data = self.aligned_data[symbol]
            # 获取当前时间窗口的特征
            window_data = symbol_data.iloc[self.current_step - self.lookback_window:self.current_step]
            symbol_features = window_data[self.feature_columns].values.flatten()
            feature_data.extend(symbol_features)
        
        # 投资组合状态
        portfolio_state = []
        
        # 各资产持仓比例
        for symbol in self.symbols:
            current_price = self.aligned_data[symbol].iloc[self.current_step]['close']
            position_value = self.positions[symbol] * current_price
            position_ratio = position_value / self.portfolio_value if self.portfolio_value > 0 else 0
            portfolio_state.append(position_ratio)
        
        # 现金比例
        cash_ratio = self.balance / self.portfolio_value if self.portfolio_value > 0 else 1.0
        portfolio_state.append(cash_ratio)
        
        # 总收益率
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        portfolio_state.append(total_return)
        
        # 夏普比率
        if len(self.portfolio_history) > 1:
            returns = np.array(self.portfolio_history[-30:])
            returns = np.diff(returns) / returns[:-1]
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_ratio = (np.mean(returns) - self.risk_free_rate/365) / np.std(returns)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        portfolio_state.append(sharpe_ratio)
        
        # 合并所有观察
        observation = np.concatenate([feature_data, portfolio_state])
        
        # 标准化处理
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation.astype(np.float32)
    
    def get_portfolio_stats(self) -> Dict:
        """获取投资组合统计信息"""
        if len(self.portfolio_history) < 2:
            return {}
        
        portfolio_values = np.array(self.portfolio_history)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        stats = {
            'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance,
            'annualized_return': np.mean(returns) * 365,
            'volatility': np.std(returns) * np.sqrt(365),
            'sharpe_ratio': (np.mean(returns) - self.risk_free_rate/365) / np.std(returns) if np.std(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'total_trades': len(self.trade_history),
            'transaction_costs': self.transaction_costs,
            'final_balance': self.balance,
            'positions': self.positions.copy()
        }
        
        return stats
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """计算最大回撤"""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return np.min(drawdown)
    
    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            stats = self.get_portfolio_stats()
            print(f"\n=== 交易环境状态 ===")
            print(f"当前步数: {self.current_step}/{self.data_length}")
            print(f"投资组合价值: ${self.portfolio_value:,.2f}")
            print(f"现金余额: ${self.balance:,.2f}")
            print(f"总收益率: {stats.get('total_return', 0):.2%}")
            print(f"夏普比率: {stats.get('sharpe_ratio', 0):.3f}")
            print(f"最大回撤: {stats.get('max_drawdown', 0):.2%}")
            print(f"交易次数: {stats.get('total_trades', 0)}")
            print(f"交易成本: ${stats.get('transaction_costs', 0):,.2f}")
            
            print("\n持仓情况:")
            for symbol, quantity in self.positions.items():
                if quantity > 0:
                    current_price = self.aligned_data[symbol].iloc[self.current_step]['close']
                    value = quantity * current_price
                    print(f"  {symbol}: {quantity:.6f} (${value:,.2f})")

# 多资产投资组合环境
class MultiAssetPortfolioEnv(TradingEnvironment):
    """多资产投资组合环境 - 支持同时交易多个资产"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 重新定义动作空间为连续权重分配
        # 动作为各资产的目标权重 [0, 1]，总和为1
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(len(self.symbols),), 
            dtype=np.float32
        )
    
    def _parse_action(self, action: np.ndarray) -> Dict[str, float]:
        """解析动作为目标权重"""
        # 归一化权重
        weights = action / (np.sum(action) + 1e-8)
        
        target_weights = {}
        for i, symbol in enumerate(self.symbols):
            target_weights[symbol] = weights[i]
        
        return target_weights
    
    def _execute_trades(self, target_weights: Dict[str, float]) -> List[Dict]:
        """根据目标权重执行再平衡"""
        trade_info = []
        
        # 计算当前权重
        current_weights = {}
        for symbol in self.symbols:
            current_price = self.aligned_data[symbol].iloc[self.current_step]['close']
            position_value = self.positions[symbol] * current_price
            current_weights[symbol] = position_value / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # 执行再平衡
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights[symbol]
            weight_diff = target_weight - current_weight
            
            current_price = self.aligned_data[symbol].iloc[self.current_step]['close']
            
            if abs(weight_diff) > 0.01:  # 权重差异阈值
                target_value = target_weight * self.portfolio_value
                current_value = current_weight * self.portfolio_value
                trade_value = target_value - current_value
                
                if trade_value > 0:  # 买入
                    # 计算实际买入金额 (考虑手续费)
                    actual_buy_value = min(trade_value / (1 + self.transaction_fee), self.balance)
                    if actual_buy_value > 1.0:
                        buy_quantity = actual_buy_value / current_price
                        cost = actual_buy_value * (1 + self.transaction_fee)
                        
                        self.balance -= cost
                        self.positions[symbol] += buy_quantity
                        self.transaction_costs += actual_buy_value * self.transaction_fee
                        
                        trade_info.append({
                            'symbol': symbol,
                            'type': 'BUY',
                            'quantity': buy_quantity,
                            'price': current_price,
                            'value': actual_buy_value,
                            'fee': actual_buy_value * self.transaction_fee,
                            'timestamp': self.timestamps[self.current_step]
                        })
                
                elif trade_value < 0:  # 卖出
                    sell_value = abs(trade_value)
                    sell_quantity = min(sell_value / current_price, self.positions[symbol])
                    
                    if sell_quantity > 0:
                        actual_sell_value = sell_quantity * current_price
                        fee = actual_sell_value * self.transaction_fee
                        net_proceeds = actual_sell_value - fee
                        
                        self.balance += net_proceeds
                        self.positions[symbol] -= sell_quantity
                        self.transaction_costs += fee
                        
                        trade_info.append({
                            'symbol': symbol,
                            'type': 'SELL',
                            'quantity': sell_quantity,
                            'price': current_price,
                            'value': actual_sell_value,
                            'fee': fee,
                            'timestamp': self.timestamps[self.current_step]
                        })
        
        return trade_info

# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    n_days = len(dates)
    
    sample_data = {}
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        df = pd.DataFrame({
            'open': np.random.randn(n_days).cumsum() + 40000,
            'high': np.random.randn(n_days).cumsum() + 41000,
            'low': np.random.randn(n_days).cumsum() + 39000,
            'close': np.random.randn(n_days).cumsum() + 40000,
            'volume': np.random.rand(n_days) * 1000000,
            'returns': np.random.randn(n_days) * 0.02,
            'rsi_14': np.random.rand(n_days) * 100,
            'macd': np.random.randn(n_days) * 100
        }, index=dates)
        sample_data[symbol] = df
    
    # 创建交易环境
    env = TradingEnvironment(sample_data, initial_balance=100000)
    
    # 测试环境
    obs = env.reset()
    print(f"初始观察维度: {obs.shape}")
    
    # 执行几步随机动作
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        print(f"步骤 {step}: 奖励={reward:.4f}, 投资组合价值=${info['portfolio_value']:,.2f}")
        
        if done:
            break
    
    # 显示最终统计
    stats = env.get_portfolio_stats()
    print(f"\n最终统计:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")