"""
系统运行脚本
提供多种运行模式的统一入口
"""

import os
import sys
import argparse
from datetime import datetime

def print_banner():
    """打印系统横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     加密货币量化交易策略系统                                  ║
║                  Cryptocurrency Quantitative Trading System                  ║
║                                                                              ║
║  🚀 基于深度强化学习的多资产交易策略                                          ║
║  📊 支持30+主流加密货币，100+技术指标                                         ║
║  🤖 集成PPO、A2C、SAC等先进算法                                               ║
║  📈 完整的回测和性能评估体系                                                  ║
║  ⚙️  自动超参数优化和策略迭代                                                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """检查依赖"""
    required_packages = ['numpy', 'pandas', 'matplotlib', 'torch', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n请运行以下命令安装依赖:")
        print("pip install --break-system-packages numpy pandas matplotlib torch scikit-learn")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def run_demo():
    """运行演示模式"""
    print("启动演示模式...")
    try:
        from demo_main import run_demo
        run_demo()
    except ImportError as e:
        print(f"❌ 导入演示模块失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 演示运行失败: {e}")
        return False
    return True

def run_data_collection():
    """运行数据收集"""
    print("启动数据收集...")
    try:
        from demo_data_collector import DemoDataCollector
        
        collector = DemoDataCollector()
        data = collector.get_multiple_cryptos_data(days=365)
        collector.save_data(data)
        
        print(f"✅ 成功收集 {len(data)} 个交易对的数据")
        return True
    except Exception as e:
        print(f"❌ 数据收集失败: {e}")
        return False

def run_feature_engineering():
    """运行特征工程测试"""
    print("启动特征工程测试...")
    try:
        from demo_data_collector import DemoDataCollector
        from feature_engineering import FeatureEngineer
        
        # 生成测试数据
        collector = DemoDataCollector()
        raw_data = collector.get_multiple_cryptos_data(
            symbols=['BTCUSDT'], days=100
        )
        
        # 特征工程
        engineer = FeatureEngineer()
        processed_data = engineer.engineer_features(raw_data)
        
        symbol = list(processed_data.keys())[0]
        df = processed_data[symbol]
        
        print(f"✅ 特征工程完成")
        print(f"   - 数据形状: {df.shape}")
        print(f"   - 特征数量: {df.shape[1]}")
        print(f"   - 样本数量: {df.shape[0]}")
        
        return True
    except Exception as e:
        print(f"❌ 特征工程测试失败: {e}")
        return False

def show_project_structure():
    """显示项目结构"""
    print("📁 项目结构:")
    structure = """
workspace/
├── 🗂️  核心模块
│   ├── data_collector.py          # 数据收集模块 (完整版)
│   ├── demo_data_collector.py     # 数据收集模块 (演示版)
│   ├── feature_engineering.py     # 特征工程模块
│   ├── trading_environment.py     # 强化学习交易环境
│   ├── drl_agent.py              # 深度强化学习智能体
│   ├── backtesting_engine.py     # 回测引擎
│   ├── optimization_framework.py  # 参数优化框架
│   └── main_pipeline.py          # 主程序管道
│
├── 🚀 运行脚本
│   ├── run_system.py             # 系统运行脚本 (当前)
│   ├── demo_main.py              # 演示程序
│   └── run_example.py            # 示例运行脚本
│
├── ⚙️  配置文件
│   ├── config.json               # 主配置文件
│   ├── requirements.txt          # 依赖列表
│   └── install.sh                # 安装脚本
│
├── 📁 数据目录
│   ├── data/                     # 数据存储
│   ├── models/                   # 模型存储
│   ├── results/                  # 结果存储
│   ├── logs/                     # 日志文件
│   ├── plots/                    # 图表文件
│   └── optimization_results/     # 优化结果
│
└── 📖 文档
    └── README.md                 # 项目说明
    """
    print(structure)

def show_system_features():
    """显示系统特性"""
    print("🌟 系统特性:")
    features = """
📊 数据处理
  • 支持30+主流加密货币
  • 多种数据源 (Binance, CCXT, Yahoo Finance)
  • 实时和历史数据获取
  • 数据清洗和预处理

🔧 特征工程  
  • 100+技术指标 (趋势、动量、波动率、成交量)
  • 跨资产特征 (相关性、相对强度、Beta)
  • 时间特征 (周期性、季节性)
  • 滞后和滚动特征

🤖 深度强化学习
  • PPO (Proximal Policy Optimization)
  • A2C (Advantage Actor-Critic)  
  • SAC (Soft Actor-Critic)
  • 自定义神经网络架构

📈 回测系统
  • 完整的性能评估指标
  • 风险调整收益分析
  • 交易成本和滑点模拟
  • 可视化结果展示

⚙️ 优化框架
  • 超参数自动优化 (Optuna)
  • 走向前验证
  • 策略集成
  • 自适应参数调整

🛡️ 风险管理
  • 仓位管理
  • 止损止盈
  • 风险预算控制
  • 分散化约束
    """
    print(features)

def show_performance_metrics():
    """显示性能指标"""
    print("📊 性能指标:")
    metrics = """
收益指标:
  • 总收益率 (Total Return)
  • 年化收益率 (Annualized Return)  
  • 超额收益 (Alpha)

风险指标:
  • 年化波动率 (Volatility)
  • 最大回撤 (Maximum Drawdown)
  • VaR / CVaR (风险价值)

风险调整指标:
  • 夏普比率 (Sharpe Ratio)
  • Sortino比率 (Sortino Ratio)
  • Calmar比率 (Calmar Ratio)
  • 信息比率 (Information Ratio)

交易指标:
  • 胜率 (Win Rate)
  • 盈亏比 (Profit/Loss Ratio)
  • 交易频率 (Trading Frequency)
  • 手续费占比 (Fee Ratio)
    """
    print(metrics)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='加密货币量化交易策略系统')
    parser.add_argument('--mode', '-m', 
                       choices=['demo', 'data', 'features', 'info', 'structure', 'metrics'],
                       default='demo',
                       help='运行模式')
    parser.add_argument('--check-deps', action='store_true',
                       help='检查依赖')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.check_deps:
        if not check_dependencies():
            sys.exit(1)
        return
    
    if args.mode == 'demo':
        print("🚀 运行演示模式")
        print("=" * 60)
        if not run_demo():
            sys.exit(1)
            
    elif args.mode == 'data':
        print("📊 运行数据收集")
        print("=" * 60)
        if not run_data_collection():
            sys.exit(1)
            
    elif args.mode == 'features':
        print("🔧 运行特征工程测试")
        print("=" * 60)
        if not run_feature_engineering():
            sys.exit(1)
            
    elif args.mode == 'info':
        show_system_features()
        
    elif args.mode == 'structure':
        show_project_structure()
        
    elif args.mode == 'metrics':
        show_performance_metrics()
    
    print(f"\n⏰ 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎉 运行完成!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()