## Crypto Quant RL (Daily, Multi-Asset)

A Python project for quantitative crypto portfolio management using diverse factors and a deep reinforcement learning agent (A2C). Supports daily frequency, multiple mainstream coins, transaction costs, backtesting, and hyperparameter optimization.

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Data
- Place per-asset CSVs in a directory, one file per symbol (e.g., `BTCUSDT.csv`, `ETHUSDT.csv`).
- Required columns: `timestamp` (UTC ISO or epoch), `open`, `high`, `low`, `close`, `volume`.
- The loader will resample/align to daily.

### Train
```bash
python train.py --data_dir ./data --symbols BTCUSDT,ETHUSDT,BNBUSDT --start 2020-01-01 --split 2022-01-01 --end 2024-12-31
```

### Optimize
```bash
python optimize.py --data_dir ./data --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT --start 2020-01-01 --split 2022-01-01 --end 2024-12-31 --trials 20
```

### Notes
- Action space is portfolio weights; allocations are long-only and sum to 1 via softmax.
- Reward is daily net portfolio return after transaction costs on turnover.
- Factors include momentum, volatility, volume, RSI, MACD, OBV, etc.
