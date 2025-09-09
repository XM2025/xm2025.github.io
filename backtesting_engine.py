"""
回测引擎
支持日频交易和全面的性能评估
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, 
                 initial_balance: float = 100000.0,
                 transaction_fee: float = 0.001,
                 slippage: float = 0.0001,
                 benchmark_symbol: str = 'BTCUSDT'):
        """
        初始化回测引擎
        
        Args:
            initial_balance: 初始资金
            transaction_fee: 交易手续费率
            slippage: 滑点
            benchmark_symbol: 基准资产
        """
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.benchmark_symbol = benchmark_symbol
        
        # 回测结果存储
        self.portfolio_history = []
        self.trade_history = []
        self.daily_returns = []
        self.positions_history = []
        self.balance_history = []
        
        # 性能指标
        self.performance_metrics = {}
        
        # 日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def run_backtest(self, 
                    agent, 
                    env, 
                    start_date: str = None,
                    end_date: str = None,
                    save_results: bool = True) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            agent: 训练好的智能体
            env: 交易环境
            start_date: 开始日期
            end_date: 结束日期
            save_results: 是否保存结果
            
        Returns:
            回测结果字典
        """
        self.logger.info("开始运行回测...")
        
        # 重置环境和历史记录
        obs = env.reset()
        self._reset_history()
        
        # 记录初始状态
        self.balance_history.append(env.balance)
        self.portfolio_history.append(env.portfolio_value)
        self.positions_history.append(env.positions.copy())
        
        step = 0
        done = False
        
        while not done:
            # 获取智能体动作
            if hasattr(agent, 'get_action'):
                if hasattr(agent, 'actor'):  # PPO/A2C
                    action, _, _ = agent.get_action(obs, deterministic=True)
                else:  # SAC
                    action = agent.get_action(obs, deterministic=True)
            else:
                # 如果是自定义策略
                action = agent.predict(obs)
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            
            # 记录状态
            self.balance_history.append(env.balance)
            self.portfolio_history.append(info['portfolio_value'])
            self.positions_history.append(info['positions'].copy())
            
            # 记录交易
            if 'trades' in info and info['trades']:
                self.trade_history.extend(info['trades'])
            
            step += 1
            
            if step % 100 == 0:
                self.logger.info(f"回测进度: {step} 步, 投资组合价值: ${info['portfolio_value']:,.2f}")
        
        # 计算日收益率
        self._calculate_daily_returns()
        
        # 计算性能指标
        benchmark_data = self._get_benchmark_data(env)
        self.performance_metrics = self._calculate_performance_metrics(benchmark_data)
        
        # 生成报告
        results = self._generate_backtest_report()
        
        if save_results:
            self._save_results(results)
        
        self.logger.info("回测完成!")
        return results
    
    def _reset_history(self):
        """重置历史记录"""
        self.portfolio_history = []
        self.trade_history = []
        self.daily_returns = []
        self.positions_history = []
        self.balance_history = []
    
    def _calculate_daily_returns(self):
        """计算日收益率"""
        portfolio_values = np.array(self.portfolio_history)
        self.daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    def _get_benchmark_data(self, env) -> np.ndarray:
        """获取基准数据"""
        if self.benchmark_symbol in env.aligned_data:
            benchmark_prices = env.aligned_data[self.benchmark_symbol]['close'].values
            benchmark_returns = np.diff(benchmark_prices) / benchmark_prices[:-1]
            
            # 对齐长度
            min_length = min(len(self.daily_returns), len(benchmark_returns))
            return benchmark_returns[:min_length]
        else:
            # 如果没有基准数据，返回零收益率
            return np.zeros(len(self.daily_returns))
    
    def _calculate_performance_metrics(self, benchmark_returns: np.ndarray) -> Dict[str, float]:
        """计算性能指标"""
        if len(self.daily_returns) == 0:
            return {}
        
        portfolio_returns = self.daily_returns
        
        # 基础指标
        total_return = (self.portfolio_history[-1] - self.initial_balance) / self.initial_balance
        annualized_return = np.mean(portfolio_returns) * 365
        volatility = np.std(portfolio_returns) * np.sqrt(365)
        
        # 风险调整指标
        risk_free_rate = 0.02  # 假设无风险利率为2%
        excess_returns = portfolio_returns - risk_free_rate / 365
        sharpe_ratio = np.mean(excess_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0
        
        # 最大回撤
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar比率
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino比率
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(365) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
        
        # 胜率
        win_rate = np.sum(portfolio_returns > 0) / len(portfolio_returns)
        
        # 盈亏比
        winning_returns = portfolio_returns[portfolio_returns > 0]
        losing_returns = portfolio_returns[portfolio_returns < 0]
        avg_win = np.mean(winning_returns) if len(winning_returns) > 0 else 0
        avg_loss = np.mean(losing_returns) if len(losing_returns) > 0 else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # 与基准的比较
        if len(benchmark_returns) > 0:
            benchmark_total_return = np.prod(1 + benchmark_returns) - 1
            alpha = total_return - benchmark_total_return
            
            # Beta计算
            if np.var(benchmark_returns) > 0:
                beta = np.cov(portfolio_returns[:len(benchmark_returns)], benchmark_returns)[0, 1] / np.var(benchmark_returns)
            else:
                beta = 0
            
            # 信息比率
            active_returns = portfolio_returns[:len(benchmark_returns)] - benchmark_returns
            information_ratio = np.mean(active_returns) / np.std(active_returns) if np.std(active_returns) > 0 else 0
        else:
            benchmark_total_return = 0
            alpha = total_return
            beta = 0
            information_ratio = 0
        
        # 交易相关指标
        total_trades = len(self.trade_history)
        total_fees = sum([trade['fee'] for trade in self.trade_history])
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'benchmark_return': benchmark_total_return,
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'total_trades': total_trades,
            'total_fees': total_fees,
            'fee_ratio': total_fees / self.initial_balance
        }
    
    def _generate_backtest_report(self) -> Dict[str, Any]:
        """生成回测报告"""
        report = {
            'summary': {
                'initial_balance': self.initial_balance,
                'final_balance': self.portfolio_history[-1] if self.portfolio_history else self.initial_balance,
                'total_trades': len(self.trade_history),
                'backtest_period': len(self.portfolio_history),
            },
            'performance_metrics': self.performance_metrics,
            'portfolio_history': self.portfolio_history,
            'daily_returns': self.daily_returns.tolist() if len(self.daily_returns) > 0 else [],
            'trade_history': self.trade_history,
            'positions_history': self.positions_history,
            'balance_history': self.balance_history
        }
        
        return report
    
    def _save_results(self, results: Dict[str, Any]):
        """保存回测结果"""
        import json
        import os
        
        os.makedirs('/workspace/results', exist_ok=True)
        
        # 保存JSON报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_file = f'/workspace/results/backtest_report_{timestamp}.json'
        
        # 转换numpy数组为列表以便JSON序列化
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_results[key][k] = v.tolist()
                    else:
                        json_results[key][k] = v
            else:
                json_results[key] = value
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"回测结果已保存到: {json_file}")
    
    def plot_results(self, results: Dict[str, Any], save_path: str = None):
        """绘制回测结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 投资组合价值曲线
        axes[0, 0].plot(results['portfolio_history'], label='投资组合价值', linewidth=2)
        axes[0, 0].axhline(y=self.initial_balance, color='r', linestyle='--', alpha=0.5, label='初始资金')
        axes[0, 0].set_title('投资组合价值曲线')
        axes[0, 0].set_xlabel('交易日')
        axes[0, 0].set_ylabel('价值 ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 日收益率分布
        if len(results['daily_returns']) > 0:
            axes[0, 1].hist(results['daily_returns'], bins=50, alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(x=np.mean(results['daily_returns']), color='r', linestyle='--', 
                             label=f'均值: {np.mean(results["daily_returns"]):.4f}')
            axes[0, 1].set_title('日收益率分布')
            axes[0, 1].set_xlabel('日收益率')
            axes[0, 1].set_ylabel('频次')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 回撤曲线
        if len(results['daily_returns']) > 0:
            cumulative_returns = np.cumprod(1 + np.array(results['daily_returns']))
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            
            axes[1, 0].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
            axes[1, 0].plot(drawdown, color='red', linewidth=1)
            axes[1, 0].set_title('回撤曲线')
            axes[1, 0].set_xlabel('交易日')
            axes[1, 0].set_ylabel('回撤比例')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 累计收益率对比
        if len(results['daily_returns']) > 0:
            cumulative_returns = np.cumprod(1 + np.array(results['daily_returns'])) - 1
            axes[1, 1].plot(cumulative_returns * 100, label='策略累计收益率', linewidth=2)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].set_title('累计收益率')
            axes[1, 1].set_xlabel('交易日')
            axes[1, 1].set_ylabel('累计收益率 (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """生成性能报告文本"""
        metrics = results['performance_metrics']
        
        report = f"""
=== 投资组合回测报告 ===

基本信息:
- 初始资金: ${results['summary']['initial_balance']:,.2f}
- 最终价值: ${results['summary']['final_balance']:,.2f}
- 回测期间: {results['summary']['backtest_period']} 个交易日
- 总交易次数: {results['summary']['total_trades']}

收益指标:
- 总收益率: {metrics.get('total_return', 0):.2%}
- 年化收益率: {metrics.get('annualized_return', 0):.2%}
- 基准收益率: {metrics.get('benchmark_return', 0):.2%}
- 超额收益 (Alpha): {metrics.get('alpha', 0):.2%}

风险指标:
- 年化波动率: {metrics.get('volatility', 0):.2%}
- 最大回撤: {metrics.get('max_drawdown', 0):.2%}
- Beta系数: {metrics.get('beta', 0):.3f}

风险调整指标:
- 夏普比率: {metrics.get('sharpe_ratio', 0):.3f}
- Calmar比率: {metrics.get('calmar_ratio', 0):.3f}
- Sortino比率: {metrics.get('sortino_ratio', 0):.3f}
- 信息比率: {metrics.get('information_ratio', 0):.3f}

交易指标:
- 胜率: {metrics.get('win_rate', 0):.2%}
- 盈亏比: {metrics.get('profit_loss_ratio', 0):.2f}
- 总手续费: ${metrics.get('total_fees', 0):,.2f}
- 手续费占比: {metrics.get('fee_ratio', 0):.2%}

"""
        
        return report

class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_trades(self, trade_history: List[Dict]) -> Dict[str, Any]:
        """分析交易记录"""
        if not trade_history:
            return {}
        
        df_trades = pd.DataFrame(trade_history)
        
        # 按交易对分组分析
        trade_analysis = {}
        
        for symbol in df_trades['symbol'].unique():
            symbol_trades = df_trades[df_trades['symbol'] == symbol]
            
            buy_trades = symbol_trades[symbol_trades['type'] == 'BUY']
            sell_trades = symbol_trades[symbol_trades['type'] == 'SELL']
            
            trade_analysis[symbol] = {
                'total_trades': len(symbol_trades),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'total_volume': symbol_trades['value'].sum(),
                'total_fees': symbol_trades['fee'].sum(),
                'avg_trade_size': symbol_trades['value'].mean(),
                'price_range': {
                    'min': symbol_trades['price'].min(),
                    'max': symbol_trades['price'].max(),
                    'mean': symbol_trades['price'].mean()
                }
            }
        
        return trade_analysis
    
    def calculate_rolling_metrics(self, returns: np.ndarray, window: int = 30) -> Dict[str, np.ndarray]:
        """计算滚动性能指标"""
        if len(returns) < window:
            return {}
        
        rolling_metrics = {}
        
        # 滚动收益率
        rolling_returns = pd.Series(returns).rolling(window).mean().values
        rolling_metrics['rolling_returns'] = rolling_returns
        
        # 滚动波动率
        rolling_volatility = pd.Series(returns).rolling(window).std().values * np.sqrt(365)
        rolling_metrics['rolling_volatility'] = rolling_volatility
        
        # 滚动夏普比率
        rolling_sharpe = rolling_returns / (rolling_volatility / np.sqrt(365))
        rolling_metrics['rolling_sharpe'] = rolling_sharpe
        
        # 滚动最大回撤
        cumulative_returns = np.cumprod(1 + returns)
        rolling_max_dd = []
        
        for i in range(window, len(returns)):
            window_returns = cumulative_returns[i-window:i+1]
            running_max = np.maximum.accumulate(window_returns)
            drawdown = (window_returns - running_max) / running_max
            rolling_max_dd.append(np.min(drawdown))
        
        rolling_metrics['rolling_max_drawdown'] = np.array([np.nan] * window + rolling_max_dd)
        
        return rolling_metrics
    
    def compare_strategies(self, results_list: List[Dict[str, Any]], strategy_names: List[str]) -> pd.DataFrame:
        """比较多个策略的性能"""
        comparison_data = []
        
        for i, results in enumerate(results_list):
            metrics = results['performance_metrics']
            strategy_name = strategy_names[i] if i < len(strategy_names) else f'策略{i+1}'
            
            comparison_data.append({
                '策略名称': strategy_name,
                '总收益率': metrics.get('total_return', 0),
                '年化收益率': metrics.get('annualized_return', 0),
                '波动率': metrics.get('volatility', 0),
                '夏普比率': metrics.get('sharpe_ratio', 0),
                '最大回撤': metrics.get('max_drawdown', 0),
                'Calmar比率': metrics.get('calmar_ratio', 0),
                '胜率': metrics.get('win_rate', 0),
                '总交易次数': metrics.get('total_trades', 0),
                '手续费占比': metrics.get('fee_ratio', 0)
            })
        
        return pd.DataFrame(comparison_data)

# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    
    # 模拟投资组合价值序列
    initial_value = 100000
    returns = np.random.normal(0.001, 0.02, 252)  # 一年的日收益率
    portfolio_values = [initial_value]
    
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
    
    # 创建模拟交易记录
    trade_history = []
    for i in range(50):
        trade_history.append({
            'symbol': np.random.choice(['BTCUSDT', 'ETHUSDT']),
            'type': np.random.choice(['BUY', 'SELL']),
            'quantity': np.random.uniform(0.1, 2.0),
            'price': np.random.uniform(30000, 50000),
            'value': np.random.uniform(1000, 10000),
            'fee': np.random.uniform(1, 50),
            'timestamp': pd.Timestamp.now()
        })
    
    # 创建模拟结果
    mock_results = {
        'summary': {
            'initial_balance': initial_value,
            'final_balance': portfolio_values[-1],
            'total_trades': len(trade_history),
            'backtest_period': len(portfolio_values)
        },
        'performance_metrics': {
            'total_return': (portfolio_values[-1] - initial_value) / initial_value,
            'annualized_return': np.mean(returns) * 365,
            'volatility': np.std(returns) * np.sqrt(365),
            'sharpe_ratio': (np.mean(returns) * 365) / (np.std(returns) * np.sqrt(365)),
            'max_drawdown': -0.15,
            'win_rate': 0.55,
            'total_trades': len(trade_history),
            'total_fees': sum([trade['fee'] for trade in trade_history])
        },
        'portfolio_history': portfolio_values,
        'daily_returns': returns.tolist(),
        'trade_history': trade_history
    }
    
    # 测试回测引擎
    engine = BacktestEngine()
    
    # 生成性能报告
    report = engine.generate_performance_report(mock_results)
    print(report)
    
    # 测试性能分析器
    analyzer = PerformanceAnalyzer()
    trade_analysis = analyzer.analyze_trades(trade_history)
    print("\n交易分析:")
    for symbol, analysis in trade_analysis.items():
        print(f"{symbol}: {analysis}")
    
    print("\n回测引擎测试完成!")