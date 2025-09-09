#!/bin/bash

echo "=== 加密货币量化交易策略系统 - 安装脚本 ==="
echo "正在安装必要的Python依赖包..."

# 更新pip
echo "更新pip..."
python3 -m pip install --upgrade pip

# 安装基础科学计算包
echo "安装基础科学计算包..."
python3 -m pip install numpy pandas scipy matplotlib seaborn

# 安装机器学习和深度学习包
echo "安装机器学习包..."
python3 -m pip install scikit-learn

# 安装PyTorch (CPU版本，适合大多数环境)
echo "安装PyTorch..."
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安装强化学习环境
echo "安装强化学习环境..."
python3 -m pip install gym gymnasium

# 安装金融数据包
echo "安装金融数据包..."
python3 -m pip install yfinance ccxt python-binance

# 安装技术分析包 (简化版本，避免TA-Lib编译问题)
echo "安装技术分析包..."
python3 -m pip install pandas-ta

# 安装超参数优化包
echo "安装优化包..."
python3 -m pip install optuna

# 安装其他必要工具
echo "安装其他工具..."
python3 -m pip install requests tqdm python-dotenv joblib

echo ""
echo "=== 安装完成 ==="
echo "正在测试基本功能..."

# 测试基本导入
python3 -c "
import pandas as pd
import numpy as np
import torch
print('✅ 基础包导入成功!')
print(f'   - pandas: {pd.__version__}')
print(f'   - numpy: {np.__version__}')
print(f'   - torch: {torch.__version__}')
"

echo ""
echo "系统已准备就绪！"
echo "运行方式："
echo "  python3 run_example.py    # 交互式运行"
echo "  python3 main_pipeline.py  # 直接运行完整管道"