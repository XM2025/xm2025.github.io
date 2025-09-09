"""
运行示例脚本
演示如何使用加密货币量化交易策略系统
"""

import os
import sys
import json
import warnings
import logging
from datetime import datetime

# 添加项目路径
sys.path.append('/workspace')

from main_pipeline import CryptoTradingPipeline

warnings.filterwarnings('ignore')

def run_quick_demo():
    """运行快速演示"""
    print("=== 快速演示模式 ===")
    print("使用少量数据和简化配置进行快速测试")
    
    # 创建演示配置
    demo_config = {
        "symbols": ["BTCUSDT", "ETHUSDT"],  # 只使用2个交易对
        "data_days": 180,  # 6个月数据
        "data_interval": "1d",
        
        "feature_engineering": {
            "technical_indicators": True,
            "cross_asset_features": True,
            "time_features": False,  # 简化特征
            "lag_features": False,
            "rolling_features": False
        },
        
        "environment": {
            "initial_balance": 100000.0,
            "transaction_fee": 0.001,
            "max_position_size": 0.9,
            "lookback_window": 30,  # 减小窗口
            "reward_scaling": 1.0,
            "risk_free_rate": 0.02
        },
        
        "agent": {
            "type": "PPO",
            "lr_actor": 3e-4,
            "lr_critic": 3e-4,
            "gamma": 0.99,
            "batch_size": 32,  # 减小批次
            "hidden_dims": [256, 128]  # 简化网络
        },
        
        "training": {
            "total_timesteps": 10000,  # 减少训练步数
            "eval_freq": 2000,
            "save_freq": 5000,
            "train_test_split": 0.8,
            "validation_split": 0.1
        },
        
        "optimization": {
            "run_hyperparameter_optimization": False,  # 关闭优化
            "run_walk_forward_optimization": False,
            "n_trials": 10
        },
        
        "backtesting": {
            "benchmark_symbol": "BTCUSDT",
            "save_results": True,
            "generate_plots": True
        }
    }
    
    # 保存演示配置
    with open('/workspace/demo_config.json', 'w') as f:
        json.dump(demo_config, f, indent=2)
    
    # 创建管道
    pipeline = CryptoTradingPipeline('/workspace/demo_config.json')
    
    try:
        # 运行演示
        results = pipeline.run_full_pipeline()
        
        print("\n=== 演示完成 ===")
        print_results_summary(results)
        
    except Exception as e:
        print(f"演示运行失败: {e}")
        raise

def run_full_system():
    """运行完整系统"""
    print("=== 完整系统模式 ===")
    print("使用完整配置和所有功能")
    
    # 创建管道
    pipeline = CryptoTradingPipeline('/workspace/config.json')
    
    try:
        # 运行完整系统
        results = pipeline.run_full_pipeline()
        
        print("\n=== 系统运行完成 ===")
        print_results_summary(results)
        
    except Exception as e:
        print(f"系统运行失败: {e}")
        raise

def run_custom_strategy():
    """运行自定义策略示例"""
    print("=== 自定义策略示例 ===")
    
    # 自定义配置
    custom_config = {
        "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"],
        "data_days": 365,
        "agent": {
            "type": "SAC",  # 使用SAC算法
            "lr_actor": 1e-4,
            "lr_critic": 1e-4,
            "gamma": 0.995,
            "batch_size": 128
        },
        "training": {
            "total_timesteps": 50000,
            "eval_freq": 5000
        },
        "optimization": {
            "run_hyperparameter_optimization": True,
            "n_trials": 20
        }
    }
    
    # 保存自定义配置
    with open('/workspace/custom_config.json', 'w') as f:
        json.dump(custom_config, f, indent=2)
    
    # 创建管道
    pipeline = CryptoTradingPipeline('/workspace/custom_config.json')
    
    try:
        # 分步执行
        print("1. 数据收集...")
        raw_data = pipeline.collect_data()
        
        print("2. 特征工程...")
        processed_data = pipeline.engineer_features()
        
        print("3. 数据分割...")
        train_data, val_data, test_data = pipeline.split_data()
        
        print("4. 超参数优化...")
        optimization_results = pipeline.optimize_hyperparameters()
        
        print("5. 创建环境和智能体...")
        env = pipeline.create_environment(train_data)
        agent = pipeline.create_agent(env)
        
        print("6. 训练智能体...")
        training_results = pipeline.train_agent()
        
        print("7. 回测...")
        backtest_results = pipeline.run_backtest()
        
        print("\n=== 自定义策略完成 ===")
        print_results_summary({
            'training': {'status': 'success'},
            'backtesting': backtest_results['performance_metrics']
        })
        
    except Exception as e:
        print(f"自定义策略运行失败: {e}")
        raise

def print_results_summary(results):
    """打印结果摘要"""
    print("\n" + "="*50)
    print("           结果摘要")
    print("="*50)
    
    # 数据收集结果
    if 'data_collection' in results:
        data_info = results['data_collection']
        print(f"📊 数据收集: {data_info.get('status', 'Unknown')}")
        if 'data_shapes' in data_info:
            total_samples = sum([shape[0] for shape in data_info['data_shapes'].values()])
            total_features = list(data_info['data_shapes'].values())[0][1] if data_info['data_shapes'] else 0
            print(f"   - 总样本数: {total_samples:,}")
            print(f"   - 特征维度: {total_features}")
    
    # 训练结果
    if 'training' in results:
        training_info = results['training']
        print(f"🤖 模型训练: {training_info.get('status', 'Unknown')}")
        if 'total_episodes' in training_info:
            print(f"   - 训练回合: {training_info['total_episodes']}")
            print(f"   - 最终奖励: {training_info.get('final_reward', 0):.4f}")
            print(f"   - 最佳评估: {training_info.get('best_evaluation', 0):.4f}")
    
    # 超参数优化结果
    if 'hyperparameter_optimization' in results:
        opt_info = results['hyperparameter_optimization']
        print(f"⚙️ 参数优化: {opt_info.get('status', 'Unknown')}")
        if 'best_value' in opt_info:
            print(f"   - 最佳性能: {opt_info['best_value']:.4f}")
    
    # 回测结果
    if 'backtesting' in results:
        backtest_info = results['backtesting']
        print(f"📈 策略回测: {backtest_info.get('status', 'Unknown')}")
        
        if isinstance(backtest_info, dict):
            # 如果是完整的回测结果
            total_return = backtest_info.get('total_return', 0)
            sharpe_ratio = backtest_info.get('sharpe_ratio', 0)
            max_drawdown = backtest_info.get('max_drawdown', 0)
            
            print(f"   - 总收益率: {total_return:.2%}")
            print(f"   - 夏普比率: {sharpe_ratio:.3f}")
            print(f"   - 最大回撤: {max_drawdown:.2%}")
            
            # 性能评级
            if sharpe_ratio > 2.0:
                rating = "🌟🌟🌟🌟🌟 优秀"
            elif sharpe_ratio > 1.5:
                rating = "🌟🌟🌟🌟 良好"
            elif sharpe_ratio > 1.0:
                rating = "🌟🌟🌟 一般"
            elif sharpe_ratio > 0.5:
                rating = "🌟🌟 较差"
            else:
                rating = "🌟 需改进"
            
            print(f"   - 策略评级: {rating}")
    
    print("="*50)
    print("📁 结果文件保存位置:")
    print("   - 模型文件: /workspace/models/")
    print("   - 回测结果: /workspace/results/")
    print("   - 图表文件: /workspace/plots/")
    print("   - 日志文件: /workspace/logs/")
    print("="*50)

def main():
    """主函数"""
    print("🚀 加密货币量化交易策略系统")
    print("基于深度强化学习的多资产交易策略")
    print()
    
    # 选择运行模式
    print("请选择运行模式:")
    print("1. 快速演示 (推荐新手)")
    print("2. 完整系统")
    print("3. 自定义策略")
    print("4. 退出")
    
    try:
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == '1':
            run_quick_demo()
        elif choice == '2':
            run_full_system()
        elif choice == '3':
            run_custom_strategy()
        elif choice == '4':
            print("退出程序")
            return
        else:
            print("无效选择，运行快速演示...")
            run_quick_demo()
            
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n运行出错: {e}")
        logging.exception("程序运行异常")

if __name__ == "__main__":
    main()