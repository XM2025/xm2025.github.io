# Crypto Quant RL (Daily Frequency)

面向加密数字货币市场的量化投资策略项目，覆盖：
- 主流币种多资产日频数据
- 多样化因子工程（动量、波动、成交、趋势、相关等）
- 深度强化学习（PPO，连续权重，Softmax+熵正则）
- 回测指标（收益曲线、夏普、回撤、换手率）
- 超参搜索与自动迭代优化

## 快速开始

1) 安装依赖
```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

2) 可选：准备本地数据（若无法联网或CCXT不可用）。将 CSV 放入 `data/` 目录，文件名与交易对一致，如 `BTCUSDT.csv`，列包含：`timestamp,open,high,low,close,volume`，时间戳（毫秒或ISO 8601）。

3) 训练（含自动数据加载与合成数据回退）
```bash
python train.py --config config.yaml
```

4) 回测
```bash
python backtest.py --config config.yaml --checkpoint checkpoints/best.pt
```

5) 超参搜索
```bash
python optimize.py --config config.yaml
```

### 快速演示配置

若要快速验证端到端流程（更少数据、更短训练）：
```bash
python train.py --config config.quick.yaml
python backtest.py --config config.quick.yaml --checkpoint checkpoints/best.pt
python optimize.py --config config.quick.yaml --trials 3
```

## 设计要点

- 数据日频，训练/验证/测试分割
- 权重向量位于概率单纯形（通过 Softmax 映射实现），含现金权重
- 交易成本按换手率计提
- 可在无网环境下自动生成合成数据，以便完整离线运行

## 目录结构

```
.
├── backtest.py
├── config.yaml
├── crypto_ql
│   ├── __init__.py
│   ├── backtest.py
│   ├── data_loader.py
│   ├── factors.py
│   ├── portfolio_env.py
│   ├── ppo.py
│   └── utils.py
├── data
├── optimize.py
├── README.md
├── requirements.txt
└── train.py
```

## 免责声明

本项目仅用于研究与教育，不构成投资建议。加密资产波动剧烈，请谨慎使用。

# xm2025.github.io