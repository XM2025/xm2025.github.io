# 加密货币量化交易策略系统

基于深度强化学习的多资产加密货币交易策略系统，支持主流币种丰富的技术指标和日频交易。

## 🚀 项目特点

- **多资产支持**: 支持30+主流加密货币交易对
- **丰富特征**: 100+技术指标和市场因子
- **深度强化学习**: 集成PPO、A2C、SAC等先进算法
- **日频交易**: 专为日频数据优化的交易策略
- **自动优化**: 超参数自动调优和策略迭代
- **完整回测**: 全面的性能评估和风险分析

## 📁 项目结构

```
workspace/
├── data_collector.py          # 数据收集模块
├── feature_engineering.py     # 特征工程模块
├── trading_environment.py     # 强化学习交易环境
├── drl_agent.py              # 深度强化学习智能体
├── backtesting_engine.py     # 回测引擎
├── optimization_framework.py  # 参数优化框架
├── main_pipeline.py          # 主程序管道
├── requirements.txt          # 依赖包列表
├── config.json              # 配置文件
├── README.md               # 项目说明
├── data/                   # 数据存储目录
├── models/                 # 模型存储目录
├── results/                # 结果存储目录
├── logs/                   # 日志目录
├── plots/                  # 图表目录
└── optimization_results/   # 优化结果目录
```

## 🛠️ 安装依赖

```bash
# 安装Python依赖
pip install -r requirements.txt

# 安装TA-Lib (技术分析库)
# Ubuntu/Debian:
sudo apt-get install libta-lib-dev
pip install TA-Lib

# macOS:
brew install ta-lib
pip install TA-Lib

# Windows:
# 下载预编译的whl文件安装
```

## 🚀 快速开始

### 1. 基本使用

```python
from main_pipeline import CryptoTradingPipeline

# 创建交易管道
pipeline = CryptoTradingPipeline()

# 运行完整流程
results = pipeline.run_full_pipeline()

print(f"总收益率: {results['backtesting']['total_return']:.2%}")
print(f"夏普比率: {results['backtesting']['sharpe_ratio']:.3f}")
```

### 2. 自定义配置

```python
# 使用自定义配置
pipeline = CryptoTradingPipeline(config_path='my_config.json')
results = pipeline.run_full_pipeline()
```

### 3. 分步执行

```python
pipeline = CryptoTradingPipeline()

# 1. 数据收集
raw_data = pipeline.collect_data()

# 2. 特征工程
processed_data = pipeline.engineer_features()

# 3. 数据分割
train_data, val_data, test_data = pipeline.split_data()

# 4. 创建环境和智能体
env = pipeline.create_environment(train_data)
agent = pipeline.create_agent(env)

# 5. 训练
training_results = pipeline.train_agent()

# 6. 回测
backtest_results = pipeline.run_backtest()
```

## 📊 支持的加密货币

默认支持以下主流加密货币：

- **主流币**: BTC, ETH, BNB, ADA, SOL, XRP, DOT, AVAX
- **DeFi币**: UNI, LINK, AAVE, CAKE, SUSHI
- **平台币**: BNB, FTT, HT, OKB
- **稳定币对**: 所有交易对均以USDT计价

可通过配置文件自定义交易对列表。

## 🔧 技术指标

系统集成了100+技术指标，包括：

### 趋势指标
- 移动平均线 (SMA, EMA, WMA)
- MACD (指数平滑异同移动平均线)
- ADX (平均趋向指数)
- Parabolic SAR
- Aroon指标

### 动量指标
- RSI (相对强弱指数)
- 随机指标 (Stochastic)
- Williams %R
- CCI (商品通道指数)
- ROC (变化率)
- MFI (资金流量指数)

### 波动率指标
- 布林带 (Bollinger Bands)
- ATR (平均真实波幅)
- 历史波动率
- Keltner通道

### 成交量指标
- OBV (能量潮)
- A/D Line (累积/派发线)
- Chaikin A/D振荡器
- 成交量价格趋势 (VPT)

### 跨资产特征
- 市场整体动量
- 相关性分析
- 相对强度
- Beta系数

## 🤖 深度强化学习算法

### 支持的算法

1. **PPO (Proximal Policy Optimization)**
   - 稳定的策略梯度算法
   - 适合连续动作空间
   - 良好的样本效率

2. **A2C (Advantage Actor-Critic)**
   - 经典的Actor-Critic算法
   - 快速收敛
   - 内存效率高

3. **SAC (Soft Actor-Critic)**
   - 基于最大熵的算法
   - 优秀的探索能力
   - 适合复杂环境

### 网络架构

- **Actor网络**: 多层全连接网络 + LayerNorm + Dropout
- **Critic网络**: 价值函数估计网络
- **激活函数**: ReLU / Tanh
- **优化器**: Adam优化器
- **正则化**: 梯度裁剪 + 权重衰减

## 📈 回测和评估

### 性能指标

- **收益指标**: 总收益率、年化收益率、超额收益
- **风险指标**: 波动率、最大回撤、VaR、CVaR
- **风险调整指标**: 夏普比率、Sortino比率、Calmar比率
- **交易指标**: 胜率、盈亏比、交易频率、手续费占比

### 可视化分析

- 投资组合价值曲线
- 日收益率分布
- 回撤曲线
- 滚动性能指标
- 持仓分析图表

## ⚙️ 超参数优化

### 优化算法

- **Optuna**: 基于贝叶斯优化的超参数调优
- **网格搜索**: 传统的网格搜索方法
- **随机搜索**: 随机采样优化
- **走向前优化**: 时间序列特化的优化方法

### 可优化参数

- 学习率 (Actor/Critic)
- 网络结构 (隐藏层维度)
- 折扣因子 (Gamma)
- 批次大小
- 环境参数 (手续费、窗口大小等)

## 🔄 策略迭代

### 自适应优化

- **性能监控**: 实时监控策略性能
- **自动重训练**: 性能下降时自动触发重训练
- **参数调整**: 根据市场变化调整参数
- **模型集成**: 多个模型的集成策略

### 风险管理

- **仓位管理**: Kelly公式、固定比例等
- **止损止盈**: 动态止损、固定止盈
- **风险预算**: VaR控制、最大回撤限制
- **分散化**: 相关性控制、行业分散

## 📊 配置说明

主要配置参数：

```json
{
  "symbols": ["BTCUSDT", "ETHUSDT", ...],  // 交易对列表
  "agent": {
    "type": "PPO",                         // 算法类型
    "lr_actor": 3e-4,                      // Actor学习率
    "lr_critic": 3e-4,                     // Critic学习率
    "gamma": 0.99,                         // 折扣因子
    "batch_size": 64                       // 批次大小
  },
  "environment": {
    "initial_balance": 100000,             // 初始资金
    "transaction_fee": 0.001,              // 交易手续费
    "lookback_window": 60                  // 观察窗口
  },
  "training": {
    "total_timesteps": 100000,             // 训练步数
    "eval_freq": 5000                      // 评估频率
  }
}
```

## 🚨 注意事项

### 风险提示

1. **历史表现不代表未来收益**
2. **加密货币市场波动巨大**
3. **算法可能存在过拟合风险**
4. **实盘交易需谨慎验证**

### 使用建议

1. **充分回测**: 使用足够长的历史数据
2. **参数优化**: 避免过度优化
3. **风险控制**: 设置合理的风险限制
4. **持续监控**: 定期评估策略性能
5. **逐步部署**: 从小资金开始实盘验证

## 📝 开发计划

### 近期计划

- [ ] 添加更多技术指标
- [ ] 支持分钟级数据
- [ ] 实现实时交易接口
- [ ] 添加更多风险管理工具

### 长期计划

- [ ] 支持期货和期权交易
- [ ] 集成新闻情感分析
- [ ] 添加链上数据分析
- [ ] 实现多策略组合

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见LICENSE文件

## 📞 联系方式

- 项目地址: https://github.com/your-username/crypto-trading-rl
- 问题反馈: 通过GitHub Issues
- 邮件联系: your-email@example.com

---

**免责声明**: 本项目仅供学习研究使用，不构成投资建议。使用本系统进行实际交易的风险由用户自行承担。