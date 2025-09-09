"""
主程序管道
整合完整的训练和回测流程
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

# 导入自定义模块
from data_collector import CryptoDataCollector
from feature_engineering import FeatureEngineer
from trading_environment import TradingEnvironment, MultiAssetPortfolioEnv
from drl_agent import PPOAgent, A2CAgent, SACAgent
from backtesting_engine import BacktestEngine, PerformanceAnalyzer
from optimization_framework import HyperparameterOptimizer, WalkForwardOptimizer, OptimizationPipeline

warnings.filterwarnings('ignore')

class CryptoTradingPipeline:
    """加密货币交易策略管道"""
    
    def __init__(self, config_path: str = None):
        """
        初始化交易管道
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化组件
        self.data_collector = None
        self.feature_engineer = None
        self.trading_env = None
        self.agent = None
        self.backtest_engine = None
        
        # 数据存储
        self.raw_data = {}
        self.processed_data = {}
        self.training_data = {}
        self.testing_data = {}
        
        # 结果存储
        self.training_results = {}
        self.backtest_results = {}
        self.optimization_results = {}
        
        # 设置日志
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 创建必要的目录
        self._create_directories()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            # 数据配置
            'symbols': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT'],
            'data_days': 730,  # 2年数据
            'data_interval': '1d',
            
            # 特征工程配置
            'feature_engineering': {
                'technical_indicators': True,
                'cross_asset_features': True,
                'time_features': True,
                'lag_features': True,
                'rolling_features': True
            },
            
            # 环境配置
            'environment': {
                'initial_balance': 100000.0,
                'transaction_fee': 0.001,
                'max_position_size': 0.95,
                'lookback_window': 60,
                'reward_scaling': 1.0,
                'risk_free_rate': 0.02
            },
            
            # 智能体配置
            'agent': {
                'type': 'PPO',  # PPO, A2C, SAC
                'lr_actor': 3e-4,
                'lr_critic': 3e-4,
                'gamma': 0.99,
                'batch_size': 64,
                'hidden_dims': [512, 256, 128]
            },
            
            # 训练配置
            'training': {
                'total_timesteps': 100000,
                'eval_freq': 5000,
                'save_freq': 10000,
                'train_test_split': 0.8,
                'validation_split': 0.1
            },
            
            # 优化配置
            'optimization': {
                'run_hyperparameter_optimization': True,
                'run_walk_forward_optimization': False,
                'n_trials': 50,
                'optimization_metric': 'sharpe_ratio'
            },
            
            # 回测配置
            'backtesting': {
                'benchmark_symbol': 'BTCUSDT',
                'save_results': True,
                'generate_plots': True
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            # 递归更新配置
            def update_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = update_dict(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d
            
            default_config = update_dict(default_config, user_config)
        
        return default_config
    
    def _setup_logging(self):
        """设置日志"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO'))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/workspace/logs/pipeline.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            '/workspace/data',
            '/workspace/models',
            '/workspace/results',
            '/workspace/logs',
            '/workspace/optimization_results',
            '/workspace/plots'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def collect_data(self) -> Dict[str, pd.DataFrame]:
        """收集数据"""
        self.logger.info("开始数据收集...")
        
        # 初始化数据收集器
        self.data_collector = CryptoDataCollector()
        
        # 收集数据
        symbols = [f"{symbol}" for symbol in self.config['symbols']]
        self.raw_data = self.data_collector.get_multiple_cryptos_data(
            symbols=symbols,
            interval=self.config['data_interval'],
            days=self.config['data_days']
        )
        
        self.logger.info(f"成功收集 {len(self.raw_data)} 个交易对的数据")
        
        # 保存原始数据
        self.data_collector.save_data(self.raw_data)
        
        return self.raw_data
    
    def engineer_features(self) -> Dict[str, pd.DataFrame]:
        """特征工程"""
        self.logger.info("开始特征工程...")
        
        if not self.raw_data:
            raise ValueError("请先收集数据")
        
        # 初始化特征工程器
        self.feature_engineer = FeatureEngineer()
        
        # 执行特征工程
        self.processed_data = self.feature_engineer.engineer_features(self.raw_data)
        
        # 根据配置选择性添加特征
        feature_config = self.config['feature_engineering']
        
        for symbol in self.processed_data:
            df = self.processed_data[symbol]
            
            if not feature_config.get('technical_indicators', True):
                # 移除技术指标特征
                tech_cols = [col for col in df.columns if any(indicator in col.lower() 
                           for indicator in ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr'])]
                df = df.drop(columns=tech_cols, errors='ignore')
            
            if not feature_config.get('time_features', True):
                # 移除时间特征
                time_cols = [col for col in df.columns if any(time_feat in col.lower() 
                           for time_feat in ['hour', 'day', 'month', 'weekend'])]
                df = df.drop(columns=time_cols, errors='ignore')
            
            self.processed_data[symbol] = df
        
        self.logger.info("特征工程完成")
        return self.processed_data
    
    def split_data(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """分割训练、验证和测试数据"""
        self.logger.info("分割数据...")
        
        if not self.processed_data:
            raise ValueError("请先进行特征工程")
        
        train_ratio = self.config['training']['train_test_split']
        val_ratio = self.config['training']['validation_split']
        
        train_data = {}
        val_data = {}
        test_data = {}
        
        for symbol, df in self.processed_data.items():
            n_samples = len(df)
            train_end = int(n_samples * train_ratio)
            val_end = int(n_samples * (train_ratio + val_ratio))
            
            train_data[symbol] = df.iloc[:train_end].copy()
            val_data[symbol] = df.iloc[train_end:val_end].copy()
            test_data[symbol] = df.iloc[val_end:].copy()
        
        self.training_data = train_data
        self.validation_data = val_data
        self.testing_data = test_data
        
        self.logger.info(f"数据分割完成 - 训练: {len(train_data[list(train_data.keys())[0]])}, "
                        f"验证: {len(val_data[list(val_data.keys())[0]])}, "
                        f"测试: {len(test_data[list(test_data.keys())[0]])}")
        
        return train_data, val_data, test_data
    
    def create_environment(self, data_dict: Dict[str, pd.DataFrame]) -> TradingEnvironment:
        """创建交易环境"""
        env_config = self.config['environment']
        
        self.trading_env = TradingEnvironment(
            data_dict=data_dict,
            initial_balance=env_config['initial_balance'],
            transaction_fee=env_config['transaction_fee'],
            max_position_size=env_config['max_position_size'],
            lookback_window=env_config['lookback_window'],
            reward_scaling=env_config['reward_scaling'],
            risk_free_rate=env_config['risk_free_rate']
        )
        
        return self.trading_env
    
    def create_agent(self, env: TradingEnvironment) -> Any:
        """创建智能体"""
        agent_config = self.config['agent']
        agent_type = agent_config['type'].upper()
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        if agent_type == 'PPO':
            self.agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                lr_actor=agent_config['lr_actor'],
                lr_critic=agent_config['lr_critic'],
                gamma=agent_config['gamma'],
                batch_size=agent_config['batch_size']
            )
        elif agent_type == 'A2C':
            self.agent = A2CAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=agent_config['lr_actor'],
                gamma=agent_config['gamma']
            )
        elif agent_type == 'SAC':
            self.agent = SACAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=agent_config['lr_actor'],
                gamma=agent_config['gamma'],
                batch_size=agent_config['batch_size']
            )
        else:
            raise ValueError(f"不支持的智能体类型: {agent_type}")
        
        self.logger.info(f"创建 {agent_type} 智能体成功")
        return self.agent
    
    def train_agent(self) -> Dict[str, Any]:
        """训练智能体"""
        self.logger.info("开始训练智能体...")
        
        if not self.agent or not self.trading_env:
            raise ValueError("请先创建智能体和环境")
        
        training_config = self.config['training']
        
        # 训练循环
        total_timesteps = training_config['total_timesteps']
        eval_freq = training_config['eval_freq']
        save_freq = training_config['save_freq']
        
        # 训练历史记录
        training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'loss_history': [],
            'evaluation_scores': []
        }
        
        obs = self.trading_env.reset()
        episode_reward = 0
        episode_length = 0
        episode_count = 0
        
        for timestep in range(total_timesteps):
            # 获取动作
            if hasattr(self.agent, 'actor'):  # PPO/A2C
                action, log_prob, value = self.agent.get_action(obs)
            else:  # SAC
                action = self.agent.get_action(obs)
                log_prob, value = 0, 0
            
            # 执行动作
            next_obs, reward, done, info = self.trading_env.step(action)
            
            # 存储转换
            if hasattr(self.agent, 'store_transition'):
                self.agent.store_transition(obs, action, reward, next_obs, done, log_prob, value)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            # 更新智能体
            if timestep % self.agent.batch_size == 0:
                loss_info = self.agent.update()
                if loss_info:
                    training_history['loss_history'].append(loss_info)
            
            # 回合结束
            if done:
                training_history['episode_rewards'].append(episode_reward)
                training_history['episode_lengths'].append(episode_length)
                
                if episode_count % 10 == 0:
                    self.logger.info(f"Episode {episode_count}: Reward={episode_reward:.4f}, "
                                   f"Length={episode_length}, Portfolio=${info.get('portfolio_value', 0):,.2f}")
                
                obs = self.trading_env.reset()
                episode_reward = 0
                episode_length = 0
                episode_count += 1
            
            # 评估
            if timestep % eval_freq == 0 and timestep > 0:
                eval_score = self._evaluate_agent()
                training_history['evaluation_scores'].append(eval_score)
                self.logger.info(f"Timestep {timestep}: Evaluation Score={eval_score:.4f}")
            
            # 保存模型
            if timestep % save_freq == 0 and timestep > 0:
                model_path = f"/workspace/models/agent_checkpoint_{timestep}.pth"
                self.agent.save(model_path)
                self.logger.info(f"模型已保存到: {model_path}")
        
        # 保存最终模型
        final_model_path = "/workspace/models/agent_final.pth"
        self.agent.save(final_model_path)
        
        self.training_results = training_history
        self.logger.info("智能体训练完成")
        
        return training_history
    
    def _evaluate_agent(self) -> float:
        """评估智能体性能"""
        if not hasattr(self, 'validation_data') or not self.validation_data:
            return 0.0
        
        # 创建验证环境
        val_env = self.create_environment(self.validation_data)
        
        # 运行验证回合
        obs = val_env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if hasattr(self.agent, 'actor'):  # PPO/A2C
                action, _, _ = self.agent.get_action(obs, deterministic=True)
            else:  # SAC
                action = self.agent.get_action(obs, deterministic=True)
            
            obs, reward, done, info = val_env.step(action)
            total_reward += reward
        
        # 计算性能指标
        portfolio_stats = val_env.get_portfolio_stats()
        return portfolio_stats.get('sharpe_ratio', 0)
    
    def run_backtest(self) -> Dict[str, Any]:
        """运行回测"""
        self.logger.info("开始回测...")
        
        if not self.agent or not self.testing_data:
            raise ValueError("请先训练智能体和准备测试数据")
        
        # 创建回测引擎
        self.backtest_engine = BacktestEngine(
            initial_balance=self.config['environment']['initial_balance'],
            transaction_fee=self.config['environment']['transaction_fee'],
            benchmark_symbol=self.config['backtesting']['benchmark_symbol']
        )
        
        # 创建测试环境
        test_env = self.create_environment(self.testing_data)
        
        # 运行回测
        self.backtest_results = self.backtest_engine.run_backtest(
            agent=self.agent,
            env=test_env,
            save_results=self.config['backtesting']['save_results']
        )
        
        # 生成报告
        performance_report = self.backtest_engine.generate_performance_report(self.backtest_results)
        print(performance_report)
        
        # 生成图表
        if self.config['backtesting']['generate_plots']:
            plot_path = f"/workspace/plots/backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.backtest_engine.plot_results(self.backtest_results, save_path=plot_path)
        
        self.logger.info("回测完成")
        return self.backtest_results
    
    def optimize_hyperparameters(self) -> Dict[str, Any]:
        """优化超参数"""
        if not self.config['optimization']['run_hyperparameter_optimization']:
            return {}
        
        self.logger.info("开始超参数优化...")
        
        def objective_function(params, data_dict):
            """目标函数"""
            try:
                # 使用参数创建环境和智能体
                temp_env = TradingEnvironment(
                    data_dict=data_dict,
                    initial_balance=params.get('initial_balance', 100000),
                    transaction_fee=params.get('transaction_fee', 0.001),
                    lookback_window=params.get('lookback_window', 60)
                )
                
                # 创建智能体
                agent_type = params.get('agent_type', 'PPO')
                state_dim = temp_env.observation_space.shape[0]
                action_dim = temp_env.action_space.shape[0]
                
                if agent_type == 'PPO':
                    temp_agent = PPOAgent(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        lr_actor=params.get('learning_rate_actor', 3e-4),
                        lr_critic=params.get('learning_rate_critic', 3e-4),
                        gamma=params.get('gamma', 0.99),
                        batch_size=params.get('batch_size', 64)
                    )
                elif agent_type == 'A2C':
                    temp_agent = A2CAgent(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        lr=params.get('learning_rate_actor', 3e-4),
                        gamma=params.get('gamma', 0.99)
                    )
                else:  # SAC
                    temp_agent = SACAgent(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        lr=params.get('learning_rate_actor', 3e-4),
                        gamma=params.get('gamma', 0.99)
                    )
                
                # 简化训练（用于优化）
                obs = temp_env.reset()
                total_reward = 0
                
                for _ in range(1000):  # 短期训练
                    if hasattr(temp_agent, 'actor'):
                        action, log_prob, value = temp_agent.get_action(obs)
                    else:
                        action = temp_agent.get_action(obs)
                        log_prob, value = 0, 0
                    
                    next_obs, reward, done, info = temp_env.step(action)
                    
                    if hasattr(temp_agent, 'store_transition'):
                        temp_agent.store_transition(obs, action, reward, next_obs, done, log_prob, value)
                    
                    if len(getattr(temp_agent, 'buffer', [])) >= temp_agent.batch_size:
                        temp_agent.update()
                    
                    obs = next_obs
                    total_reward += reward
                    
                    if done:
                        obs = temp_env.reset()
                
                # 返回性能指标
                portfolio_stats = temp_env.get_portfolio_stats()
                return portfolio_stats.get('sharpe_ratio', 0)
                
            except Exception as e:
                self.logger.warning(f"优化试验失败: {e}")
                return -1.0  # 返回较差的分数
        
        # 创建优化器
        optimizer = HyperparameterOptimizer(
            objective_function=objective_function,
            n_trials=self.config['optimization']['n_trials']
        )
        
        # 执行优化
        self.optimization_results = optimizer.optimize(self.training_data)
        
        self.logger.info("超参数优化完成")
        return self.optimization_results
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """运行完整管道"""
        self.logger.info("开始运行完整交易策略管道...")
        
        results = {}
        
        try:
            # 1. 数据收集
            results['data_collection'] = {
                'status': 'success',
                'data_shapes': {symbol: df.shape for symbol, df in self.collect_data().items()}
            }
            
            # 2. 特征工程
            results['feature_engineering'] = {
                'status': 'success',
                'feature_counts': {symbol: df.shape[1] for symbol, df in self.engineer_features().items()}
            }
            
            # 3. 数据分割
            train_data, val_data, test_data = self.split_data()
            results['data_splitting'] = {
                'status': 'success',
                'train_size': len(train_data[list(train_data.keys())[0]]),
                'val_size': len(val_data[list(val_data.keys())[0]]),
                'test_size': len(test_data[list(test_data.keys())[0]])
            }
            
            # 4. 创建环境和智能体
            env = self.create_environment(train_data)
            agent = self.create_agent(env)
            results['model_creation'] = {
                'status': 'success',
                'agent_type': self.config['agent']['type'],
                'state_dim': env.observation_space.shape[0],
                'action_dim': env.action_space.shape[0]
            }
            
            # 5. 超参数优化（可选）
            if self.config['optimization']['run_hyperparameter_optimization']:
                optimization_results = self.optimize_hyperparameters()
                results['hyperparameter_optimization'] = {
                    'status': 'success',
                    'best_params': optimization_results.get('best_params', {}),
                    'best_value': optimization_results.get('best_value', 0)
                }
                
                # 使用最佳参数更新配置
                if optimization_results.get('best_params'):
                    self._update_config_with_best_params(optimization_results['best_params'])
                    
                    # 重新创建环境和智能体
                    env = self.create_environment(train_data)
                    agent = self.create_agent(env)
            
            # 6. 训练智能体
            training_results = self.train_agent()
            results['training'] = {
                'status': 'success',
                'total_episodes': len(training_results['episode_rewards']),
                'final_reward': training_results['episode_rewards'][-1] if training_results['episode_rewards'] else 0,
                'best_evaluation': max(training_results['evaluation_scores']) if training_results['evaluation_scores'] else 0
            }
            
            # 7. 回测
            backtest_results = self.run_backtest()
            results['backtesting'] = {
                'status': 'success',
                'total_return': backtest_results['performance_metrics'].get('total_return', 0),
                'sharpe_ratio': backtest_results['performance_metrics'].get('sharpe_ratio', 0),
                'max_drawdown': backtest_results['performance_metrics'].get('max_drawdown', 0)
            }
            
            # 8. 保存完整结果
            self._save_pipeline_results(results)
            
            self.logger.info("完整管道运行成功!")
            
        except Exception as e:
            self.logger.error(f"管道运行失败: {e}")
            results['error'] = str(e)
            raise
        
        return results
    
    def _update_config_with_best_params(self, best_params: Dict[str, Any]):
        """使用最佳参数更新配置"""
        # 更新智能体配置
        if 'learning_rate_actor' in best_params:
            self.config['agent']['lr_actor'] = best_params['learning_rate_actor']
        if 'learning_rate_critic' in best_params:
            self.config['agent']['lr_critic'] = best_params['learning_rate_critic']
        if 'gamma' in best_params:
            self.config['agent']['gamma'] = best_params['gamma']
        if 'batch_size' in best_params:
            self.config['agent']['batch_size'] = best_params['batch_size']
        
        # 更新环境配置
        if 'lookback_window' in best_params:
            self.config['environment']['lookback_window'] = best_params['lookback_window']
        if 'transaction_fee' in best_params:
            self.config['environment']['transaction_fee'] = best_params['transaction_fee']
        if 'max_position_size' in best_params:
            self.config['environment']['max_position_size'] = best_params['max_position_size']
        if 'reward_scaling' in best_params:
            self.config['environment']['reward_scaling'] = best_params['reward_scaling']
        
        self.logger.info("配置已使用最佳参数更新")
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """保存管道结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"/workspace/results/pipeline_results_{timestamp}.json"
        
        # 准备可序列化的结果
        serializable_results = {
            'timestamp': timestamp,
            'config': self.config,
            'results': results,
            'training_history': self.training_results,
            'backtest_results': self.backtest_results,
            'optimization_results': self.optimization_results
        }
        
        # 转换numpy数组为列表
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {key: make_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
                return obj.item()
            else:
                return obj
        
        serializable_results = make_serializable(serializable_results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"管道结果已保存到: {results_file}")

def main():
    """主函数"""
    print("=== 加密货币量化交易策略系统 ===")
    print("基于深度强化学习的多资产交易策略")
    print("支持主流加密货币，多样化技术指标，日频交易")
    print()
    
    # 创建管道
    pipeline = CryptoTradingPipeline()
    
    try:
        # 运行完整管道
        results = pipeline.run_full_pipeline()
        
        print("\n=== 管道运行完成 ===")
        print(f"数据收集: ✓")
        print(f"特征工程: ✓")
        print(f"模型训练: ✓")
        print(f"策略回测: ✓")
        
        if 'backtesting' in results:
            backtest_metrics = results['backtesting']
            print(f"\n=== 回测结果 ===")
            print(f"总收益率: {backtest_metrics.get('total_return', 0):.2%}")
            print(f"夏普比率: {backtest_metrics.get('sharpe_ratio', 0):.3f}")
            print(f"最大回撤: {backtest_metrics.get('max_drawdown', 0):.2%}")
        
        print(f"\n结果已保存到 /workspace/results/ 目录")
        print(f"模型已保存到 /workspace/models/ 目录")
        print(f"图表已保存到 /workspace/plots/ 目录")
        
    except Exception as e:
        print(f"\n❌ 管道运行失败: {e}")
        raise

if __name__ == "__main__":
    main()