"""
参数优化框架
实现自动调参和策略迭代优化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from datetime import datetime
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self, 
                 objective_function: Callable,
                 optimization_direction: str = 'maximize',
                 n_trials: int = 100,
                 n_jobs: int = 1):
        """
        初始化超参数优化器
        
        Args:
            objective_function: 目标函数
            optimization_direction: 优化方向 ('maximize' 或 'minimize')
            n_trials: 优化试验次数
            n_jobs: 并行任务数
        """
        self.objective_function = objective_function
        self.optimization_direction = optimization_direction
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        
        # 创建Optuna研究对象
        self.study = optuna.create_study(direction=optimization_direction)
        
        # 日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def define_search_space(self) -> Dict[str, Any]:
        """定义搜索空间"""
        search_space = {
            # 强化学习智能体参数
            'agent_type': ['PPO', 'A2C', 'SAC'],
            'learning_rate_actor': (1e-5, 1e-2),
            'learning_rate_critic': (1e-5, 1e-2),
            'gamma': (0.9, 0.999),
            'batch_size': [32, 64, 128, 256],
            'hidden_dims': [(256, 128), (512, 256), (512, 256, 128), (1024, 512, 256)],
            
            # PPO特定参数
            'clip_epsilon': (0.1, 0.3),
            'entropy_coeff': (0.001, 0.1),
            'ppo_epochs': [5, 10, 15, 20],
            
            # 环境参数
            'lookback_window': [30, 60, 90, 120],
            'transaction_fee': (0.0005, 0.002),
            'max_position_size': (0.8, 1.0),
            'reward_scaling': (0.1, 10.0),
            
            # 特征工程参数
            'technical_indicators': [True, False],
            'cross_asset_features': [True, False],
            'time_features': [True, False],
            'lag_features': [True, False],
            'rolling_features': [True, False],
            
            # 数据参数
            'train_test_split': (0.6, 0.8),
            'validation_split': (0.1, 0.3)
        }
        
        return search_space
    
    def suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """建议参数组合"""
        params = {}
        
        # 智能体参数
        params['agent_type'] = trial.suggest_categorical('agent_type', ['PPO', 'A2C', 'SAC'])
        params['learning_rate_actor'] = trial.suggest_float('learning_rate_actor', 1e-5, 1e-2, log=True)
        params['learning_rate_critic'] = trial.suggest_float('learning_rate_critic', 1e-5, 1e-2, log=True)
        params['gamma'] = trial.suggest_float('gamma', 0.9, 0.999)
        params['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        
        # 网络结构
        hidden_dim_choices = [(256, 128), (512, 256), (512, 256, 128), (1024, 512, 256)]
        params['hidden_dims'] = trial.suggest_categorical('hidden_dims', hidden_dim_choices)
        
        # PPO特定参数
        if params['agent_type'] == 'PPO':
            params['clip_epsilon'] = trial.suggest_float('clip_epsilon', 0.1, 0.3)
            params['entropy_coeff'] = trial.suggest_float('entropy_coeff', 0.001, 0.1, log=True)
            params['ppo_epochs'] = trial.suggest_categorical('ppo_epochs', [5, 10, 15, 20])
        
        # 环境参数
        params['lookback_window'] = trial.suggest_categorical('lookback_window', [30, 60, 90, 120])
        params['transaction_fee'] = trial.suggest_float('transaction_fee', 0.0005, 0.002)
        params['max_position_size'] = trial.suggest_float('max_position_size', 0.8, 1.0)
        params['reward_scaling'] = trial.suggest_float('reward_scaling', 0.1, 10.0, log=True)
        
        # 特征工程参数
        params['technical_indicators'] = trial.suggest_categorical('technical_indicators', [True, False])
        params['cross_asset_features'] = trial.suggest_categorical('cross_asset_features', [True, False])
        params['time_features'] = trial.suggest_categorical('time_features', [True, False])
        params['lag_features'] = trial.suggest_categorical('lag_features', [True, False])
        params['rolling_features'] = trial.suggest_categorical('rolling_features', [True, False])
        
        # 数据分割参数
        params['train_test_split'] = trial.suggest_float('train_test_split', 0.6, 0.8)
        params['validation_split'] = trial.suggest_float('validation_split', 0.1, 0.3)
        
        return params
    
    def optimize(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """执行超参数优化"""
        self.logger.info(f"开始超参数优化，共 {self.n_trials} 次试验...")
        
        def objective(trial):
            params = self.suggest_parameters(trial)
            return self.objective_function(params, data_dict)
        
        # 运行优化
        self.study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        
        # 获取最佳参数
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        self.logger.info(f"优化完成! 最佳目标值: {best_value:.4f}")
        self.logger.info(f"最佳参数: {best_params}")
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'study': self.study,
            'optimization_history': self._get_optimization_history()
        }
    
    def _get_optimization_history(self) -> List[Dict]:
        """获取优化历史"""
        history = []
        for trial in self.study.trials:
            history.append({
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            })
        return history
    
    def plot_optimization_history(self, save_path: str = None):
        """绘制优化历史"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 优化历史
        optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=ax1)
        ax1.set_title('优化历史')
        
        # 参数重要性
        try:
            optuna.visualization.matplotlib.plot_param_importances(self.study, ax=ax2)
            ax2.set_title('参数重要性')
        except:
            ax2.text(0.5, 0.5, '参数重要性分析需要更多试验', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"优化历史图表已保存到: {save_path}")
        
        plt.show()

class WalkForwardOptimizer:
    """走向前优化器"""
    
    def __init__(self, 
                 train_window: int = 252,  # 训练窗口（天）
                 test_window: int = 63,    # 测试窗口（天）
                 step_size: int = 21):     # 步长（天）
        """
        初始化走向前优化器
        
        Args:
            train_window: 训练窗口大小
            test_window: 测试窗口大小
            step_size: 向前移动步长
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        
        # 日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_walk_forward_splits(self, data_length: int) -> List[Tuple[slice, slice]]:
        """创建走向前分割"""
        splits = []
        
        start = 0
        while start + self.train_window + self.test_window <= data_length:
            train_end = start + self.train_window
            test_end = train_end + self.test_window
            
            train_slice = slice(start, train_end)
            test_slice = slice(train_end, test_end)
            
            splits.append((train_slice, test_slice))
            
            start += self.step_size
        
        return splits
    
    def optimize_walk_forward(self, 
                             objective_function: Callable,
                             data_dict: Dict[str, pd.DataFrame],
                             param_grid: Dict[str, List],
                             n_trials_per_split: int = 50) -> Dict[str, Any]:
        """执行走向前优化"""
        data_length = len(list(data_dict.values())[0])
        splits = self.create_walk_forward_splits(data_length)
        
        self.logger.info(f"创建了 {len(splits)} 个走向前分割")
        
        all_results = []
        best_params_per_split = []
        
        for i, (train_slice, test_slice) in enumerate(splits):
            self.logger.info(f"优化分割 {i+1}/{len(splits)}")
            
            # 创建训练和测试数据
            train_data = {symbol: df.iloc[train_slice] for symbol, df in data_dict.items()}
            test_data = {symbol: df.iloc[test_slice] for symbol, df in data_dict.items()}
            
            # 为当前分割创建优化器
            split_optimizer = HyperparameterOptimizer(
                objective_function=lambda params, data: objective_function(params, train_data, test_data),
                n_trials=n_trials_per_split
            )
            
            # 执行优化
            split_results = split_optimizer.optimize(train_data)
            
            all_results.append({
                'split_index': i,
                'train_slice': train_slice,
                'test_slice': test_slice,
                'best_params': split_results['best_params'],
                'best_value': split_results['best_value']
            })
            
            best_params_per_split.append(split_results['best_params'])
        
        # 分析参数稳定性
        param_stability = self._analyze_parameter_stability(best_params_per_split)
        
        return {
            'all_results': all_results,
            'param_stability': param_stability,
            'average_performance': np.mean([r['best_value'] for r in all_results]),
            'performance_std': np.std([r['best_value'] for r in all_results])
        }
    
    def _analyze_parameter_stability(self, params_list: List[Dict]) -> Dict[str, Any]:
        """分析参数稳定性"""
        if not params_list:
            return {}
        
        stability_analysis = {}
        
        # 获取所有参数名
        all_param_names = set()
        for params in params_list:
            all_param_names.update(params.keys())
        
        for param_name in all_param_names:
            values = []
            for params in params_list:
                if param_name in params:
                    value = params[param_name]
                    if isinstance(value, (int, float)):
                        values.append(value)
            
            if values:
                stability_analysis[param_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
                }
        
        return stability_analysis

class StrategyEnsemble:
    """策略集成器"""
    
    def __init__(self, strategies: List[Any], weights: Optional[List[float]] = None):
        """
        初始化策略集成器
        
        Args:
            strategies: 策略列表
            weights: 策略权重
        """
        self.strategies = strategies
        self.weights = weights if weights else [1.0 / len(strategies)] * len(strategies)
        
        # 确保权重和为1
        self.weights = np.array(self.weights)
        self.weights = self.weights / np.sum(self.weights)
        
        # 日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_ensemble_action(self, state: np.ndarray) -> np.ndarray:
        """获取集成动作"""
        actions = []
        
        for strategy in self.strategies:
            if hasattr(strategy, 'get_action'):
                if hasattr(strategy, 'actor'):  # PPO/A2C
                    action, _, _ = strategy.get_action(state, deterministic=True)
                else:  # SAC
                    action = strategy.get_action(state, deterministic=True)
            else:
                action = strategy.predict(state)
            
            actions.append(action)
        
        # 加权平均
        actions = np.array(actions)
        ensemble_action = np.average(actions, axis=0, weights=self.weights)
        
        return ensemble_action
    
    def optimize_weights(self, 
                        validation_data: Dict[str, pd.DataFrame],
                        objective_function: Callable) -> np.ndarray:
        """优化集成权重"""
        from scipy.optimize import minimize
        
        def weight_objective(weights):
            # 确保权重和为1
            weights = weights / np.sum(weights)
            self.weights = weights
            
            # 计算集成策略的性能
            performance = objective_function(self, validation_data)
            return -performance  # 最小化负性能等于最大化性能
        
        # 初始权重
        initial_weights = np.ones(len(self.strategies)) / len(self.strategies)
        
        # 约束条件
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0, 1) for _ in range(len(self.strategies))]
        
        # 优化
        result = minimize(weight_objective, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x / np.sum(result.x)
            self.weights = optimal_weights
            self.logger.info(f"权重优化成功: {optimal_weights}")
        else:
            self.logger.warning("权重优化失败，使用等权重")
        
        return self.weights

class AdaptiveOptimizer:
    """自适应优化器"""
    
    def __init__(self, 
                 base_optimizer: HyperparameterOptimizer,
                 adaptation_frequency: int = 50,  # 适应频率（交易日）
                 performance_threshold: float = 0.05):  # 性能阈值
        """
        初始化自适应优化器
        
        Args:
            base_optimizer: 基础优化器
            adaptation_frequency: 适应频率
            performance_threshold: 性能阈值
        """
        self.base_optimizer = base_optimizer
        self.adaptation_frequency = adaptation_frequency
        self.performance_threshold = performance_threshold
        
        # 性能历史
        self.performance_history = []
        self.last_optimization_step = 0
        
        # 日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def should_reoptimize(self, current_step: int, current_performance: float) -> bool:
        """判断是否需要重新优化"""
        # 检查时间条件
        time_condition = (current_step - self.last_optimization_step) >= self.adaptation_frequency
        
        # 检查性能条件
        performance_condition = False
        if len(self.performance_history) >= 10:  # 至少有10个性能记录
            recent_avg = np.mean(self.performance_history[-10:])
            performance_condition = abs(current_performance - recent_avg) > self.performance_threshold
        
        return time_condition or performance_condition
    
    def adaptive_optimize(self, 
                         current_step: int,
                         current_performance: float,
                         data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """自适应优化"""
        self.performance_history.append(current_performance)
        
        if self.should_reoptimize(current_step, current_performance):
            self.logger.info(f"在步骤 {current_step} 触发重新优化")
            
            # 使用最近的数据进行优化
            recent_data = {}
            lookback_period = min(500, len(list(data_dict.values())[0]))  # 最多使用500天数据
            
            for symbol, df in data_dict.items():
                recent_data[symbol] = df.iloc[-lookback_period:]
            
            # 执行优化
            results = self.base_optimizer.optimize(recent_data)
            
            self.last_optimization_step = current_step
            
            return results
        
        return None

class OptimizationPipeline:
    """优化管道"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化优化管道
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 创建组件
        self.hyperparameter_optimizer = None
        self.walk_forward_optimizer = None
        self.strategy_ensemble = None
        self.adaptive_optimizer = None
        
        # 结果存储
        self.optimization_results = {}
        
        # 日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def run_full_optimization(self, 
                             data_dict: Dict[str, pd.DataFrame],
                             objective_function: Callable) -> Dict[str, Any]:
        """运行完整优化流程"""
        self.logger.info("开始完整优化流程...")
        
        results = {}
        
        # 1. 超参数优化
        if self.config.get('run_hyperparameter_optimization', True):
            self.logger.info("步骤 1: 超参数优化")
            self.hyperparameter_optimizer = HyperparameterOptimizer(
                objective_function=objective_function,
                n_trials=self.config.get('n_trials', 100)
            )
            results['hyperparameter_optimization'] = self.hyperparameter_optimizer.optimize(data_dict)
        
        # 2. 走向前优化
        if self.config.get('run_walk_forward_optimization', True):
            self.logger.info("步骤 2: 走向前优化")
            self.walk_forward_optimizer = WalkForwardOptimizer(
                train_window=self.config.get('train_window', 252),
                test_window=self.config.get('test_window', 63),
                step_size=self.config.get('step_size', 21)
            )
            results['walk_forward_optimization'] = self.walk_forward_optimizer.optimize_walk_forward(
                objective_function, data_dict, {}, 
                n_trials_per_split=self.config.get('n_trials_per_split', 20)
            )
        
        # 3. 保存结果
        self.optimization_results = results
        self._save_optimization_results()
        
        self.logger.info("完整优化流程完成!")
        return results
    
    def _save_optimization_results(self):
        """保存优化结果"""
        os.makedirs('/workspace/optimization_results', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'/workspace/optimization_results/optimization_results_{timestamp}.json'
        
        # 转换结果为可序列化格式
        serializable_results = self._make_serializable(self.optimization_results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"优化结果已保存到: {results_file}")
    
    def _make_serializable(self, obj):
        """使对象可序列化"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj

# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    sample_data = {}
    
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        df = pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 40000,
            'high': np.random.randn(len(dates)).cumsum() + 41000,
            'low': np.random.randn(len(dates)).cumsum() + 39000,
            'close': np.random.randn(len(dates)).cumsum() + 40000,
            'volume': np.random.rand(len(dates)) * 1000000,
            'returns': np.random.randn(len(dates)) * 0.02
        }, index=dates)
        sample_data[symbol] = df
    
    # 定义示例目标函数
    def example_objective(params, data_dict):
        # 模拟策略性能评估
        # 实际中这里应该是完整的训练和评估流程
        performance = np.random.uniform(0.5, 1.5)  # 模拟夏普比率
        return performance
    
    # 测试超参数优化
    print("测试超参数优化...")
    optimizer = HyperparameterOptimizer(
        objective_function=example_objective,
        n_trials=10  # 减少试验次数以便快速测试
    )
    
    results = optimizer.optimize(sample_data)
    print(f"最佳参数: {results['best_params']}")
    print(f"最佳性能: {results['best_value']:.4f}")
    
    # 测试走向前优化
    print("\n测试走向前优化...")
    wf_optimizer = WalkForwardOptimizer(
        train_window=100,
        test_window=30,
        step_size=10
    )
    
    splits = wf_optimizer.create_walk_forward_splits(len(dates))
    print(f"创建了 {len(splits)} 个走向前分割")
    
    print("\n优化框架测试完成!")