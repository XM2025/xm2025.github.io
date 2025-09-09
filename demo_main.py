"""
演示主程序
使用模拟数据展示加密货币量化交易策略系统的核心功能
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入演示模块
from demo_data_collector import DemoDataCollector
from feature_engineering import FeatureEngineer

class DemoTradingStrategy:
    """演示交易策略"""
    
    def __init__(self, initial_balance: float = 100000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.portfolio_history = []
        self.trade_history = []
    
    def simple_strategy(self, data: pd.DataFrame) -> pd.Series:
        """
        简单的移动平均交叉策略
        
        Args:
            data: 包含技术指标的数据
            
        Returns:
            交易信号序列 (1=买入, -1=卖出, 0=持有)
        """
        signals = pd.Series(0, index=data.index)
        
        if 'sma_5' in data.columns and 'sma_20' in data.columns:
            # 短期均线上穿长期均线 - 买入信号
            buy_signals = (data['sma_5'] > data['sma_20']) & (data['sma_5'].shift(1) <= data['sma_20'].shift(1))
            # 短期均线下穿长期均线 - 卖出信号
            sell_signals = (data['sma_5'] < data['sma_20']) & (data['sma_5'].shift(1) >= data['sma_20'].shift(1))
            
            signals[buy_signals] = 1
            signals[sell_signals] = -1
        
        return signals
    
    def backtest_strategy(self, data: pd.DataFrame, signals: pd.Series) -> dict:
        """
        回测策略
        
        Args:
            data: 价格数据
            signals: 交易信号
            
        Returns:
            回测结果
        """
        portfolio_value = self.initial_balance
        position = 0  # 持仓数量
        cash = self.initial_balance
        
        portfolio_values = []
        
        for i, (date, row) in enumerate(data.iterrows()):
            current_price = row['close']
            signal = signals.iloc[i] if i < len(signals) else 0
            
            # 执行交易
            if signal == 1 and position == 0:  # 买入
                position = cash * 0.95 / current_price  # 95%仓位，留5%现金
                cash = cash * 0.05
                self.trade_history.append({
                    'date': date,
                    'type': 'BUY',
                    'price': current_price,
                    'quantity': position,
                    'value': position * current_price
                })
            elif signal == -1 and position > 0:  # 卖出
                cash += position * current_price * 0.999  # 扣除0.1%手续费
                position = 0
                self.trade_history.append({
                    'date': date,
                    'type': 'SELL',
                    'price': current_price,
                    'quantity': position,
                    'value': cash
                })
            
            # 计算投资组合价值
            portfolio_value = cash + position * current_price
            portfolio_values.append(portfolio_value)
        
        # 计算性能指标
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (portfolio_values[-1] - self.initial_balance) / self.initial_balance
        annualized_return = np.mean(returns) * 365
        volatility = np.std(returns) * np.sqrt(365)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # 最大回撤
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        return {
            'portfolio_values': portfolio_values,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trade_history)
        }

def run_demo():
    """运行演示"""
    print("=== 加密货币量化交易策略演示 ===")
    print("使用模拟数据展示系统核心功能\n")
    
    # 1. 数据收集
    print("1. 数据收集...")
    collector = DemoDataCollector()
    data_dict = collector.get_multiple_cryptos_data(
        symbols=['BTCUSDT', 'ETHUSDT'], 
        days=365
    )
    print(f"✅ 生成了 {len(data_dict)} 个交易对的模拟数据")
    
    # 2. 特征工程
    print("\n2. 特征工程...")
    feature_engineer = FeatureEngineer()
    processed_data = feature_engineer.engineer_features(data_dict)
    print(f"✅ 完成特征工程，每个资产包含 {processed_data['BTCUSDT'].shape[1]} 个特征")
    
    # 3. 策略回测
    print("\n3. 策略回测...")
    strategy = DemoTradingStrategy()
    
    results = {}
    for symbol, df in processed_data.items():
        print(f"   回测 {symbol}...")
        signals = strategy.simple_strategy(df)
        result = strategy.backtest_strategy(df, signals)
        results[symbol] = result
    
    # 4. 结果展示
    print("\n" + "="*60)
    print("                    回测结果")
    print("="*60)
    
    for symbol, result in results.items():
        print(f"\n{symbol}:")
        print(f"  总收益率:     {result['total_return']:.2%}")
        print(f"  年化收益率:   {result['annualized_return']:.2%}")
        print(f"  夏普比率:     {result['sharpe_ratio']:.3f}")
        print(f"  最大回撤:     {result['max_drawdown']:.2%}")
        print(f"  交易次数:     {result['total_trades']}")
        
        # 性能评级
        sharpe = result['sharpe_ratio']
        if sharpe > 2.0:
            rating = "🌟🌟🌟🌟🌟 优秀"
        elif sharpe > 1.5:
            rating = "🌟🌟🌟🌟 良好"
        elif sharpe > 1.0:
            rating = "🌟🌟🌟 一般"
        elif sharpe > 0.5:
            rating = "🌟🌟 较差"
        else:
            rating = "🌟 需改进"
        
        print(f"  策略评级:     {rating}")
    
    # 5. 可视化结果
    print(f"\n5. 生成可视化图表...")
    plot_results(results, processed_data)
    
    print("\n" + "="*60)
    print("演示完成！")
    print("这个演示展示了系统的核心功能：")
    print("✅ 数据收集与处理")
    print("✅ 多样化特征工程")
    print("✅ 策略回测与评估")
    print("✅ 性能指标计算")
    print("✅ 结果可视化")
    print("\n完整系统还包括深度强化学习训练、超参数优化等高级功能。")
    print("="*60)

def plot_results(results: dict, data_dict: dict):
    """绘制结果图表"""
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = ['blue', 'red', 'green', 'orange']
        
        # 投资组合价值曲线
        for i, (symbol, result) in enumerate(results.items()):
            axes[0, 0].plot(result['portfolio_values'], 
                          label=symbol, color=colors[i % len(colors)], linewidth=2)
        
        axes[0, 0].axhline(y=100000, color='black', linestyle='--', alpha=0.5, label='Initial Balance')
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 价格走势
        for i, (symbol, df) in enumerate(data_dict.items()):
            if symbol in results:
                axes[0, 1].plot(df.index, df['close'], 
                              label=f'{symbol} Price', color=colors[i % len(colors)])
        
        axes[0, 1].set_title('Price Trends')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Price ($)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 性能指标对比
        symbols = list(results.keys())
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        metric_names = ['Total Return', 'Sharpe Ratio', 'Max Drawdown']
        
        x = np.arange(len(symbols))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [results[symbol][metric] for symbol in symbols]
            if metric == 'max_drawdown':
                values = [abs(v) for v in values]  # 取绝对值便于显示
            
            axes[1, 0].bar(x + i * width, values, width, 
                          label=metric_names[i], color=colors[i])
        
        axes[1, 0].set_title('Performance Metrics Comparison')
        axes[1, 0].set_xlabel('Assets')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_xticks(x + width)
        axes[1, 0].set_xticklabels(symbols)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 收益分布
        for i, (symbol, result) in enumerate(results.items()):
            portfolio_values = result['portfolio_values']
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            axes[1, 1].hist(returns, bins=30, alpha=0.7, 
                          label=f'{symbol} Returns', color=colors[i % len(colors)])
        
        axes[1, 1].set_title('Daily Returns Distribution')
        axes[1, 1].set_xlabel('Daily Return')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        os.makedirs('/workspace/plots', exist_ok=True)
        plot_path = f"/workspace/plots/demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ 图表已保存到: {plot_path}")
        
        # 显示图表
        plt.show()
        
    except Exception as e:
        print(f"绘图时出现错误: {e}")
        print("跳过可视化步骤...")

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n演示被用户中断")
    except Exception as e:
        print(f"\n演示运行出错: {e}")
        import traceback
        traceback.print_exc()