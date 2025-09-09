# Crypto Quant RL (Daily Frequency)

A Python project for daily-frequency multi-asset crypto portfolio allocation using diverse technical factors and deep reinforcement learning (PPO). Includes data loading via CCXT, feature engineering, a Gymnasium trading environment with transaction costs and slippage, backtesting, and Optuna-based hyperparameter optimization.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Project Structure

- `src/config.py`: Global configuration (symbols, dates, costs, lookbacks, etc.)
- `src/data/ccxt_loader.py`: Daily OHLCV data loader (CCXT) and alignment
- `src/factors/technical.py`: Factor library (momentum, volatility, volume, MA cross)
- `src/env/portfolio_env.py`: Gymnasium daily portfolio environment with costs
- `src/train/train_ppo.py`: PPO training utilities
- `src/backtest/evaluate.py`: Backtesting metrics
- `src/optimize/optuna_search.py`: Optuna hyperparameter search
- `src/pipeline.py`: End-to-end pipeline

## Usage

1) Configure symbols and dates in `src/config.py` as needed.

2) Run the pipeline:
```bash
python -m src.pipeline
```
This will:
- Fetch daily OHLCV for major symbols from `binance`
- Build factor matrix with multiple lookbacks
- Split train/test by dates
- Train PPO agent
- Evaluate on test and print key metrics

## Notes
- Transaction cost and slippage are set via `EnvConfig` in `src/config.py` (in bps).
- The environment normalizes actions to a long-only weight vector (sum to 1) with optional leverage cap.
- You can extend `src/factors/technical.py` with additional alpha factors.

## Hyperparameter Optimization (Optional)
Example of running an Optuna study:
```python
from src.optimize.optuna_search import run_study
from src.env.portfolio_env import DailyPortfolioEnv
from src.train.train_ppo import make_env
# Build your train prices/features first (see src/pipeline.py), then:
study = run_study(lambda: DailyPortfolioEnv(train_prices, train_feats), n_trials=25, total_timesteps=150_000)
print(study.best_params)
```

## Disclaimer
For research and educational purposes only. Crypto trading involves significant risk.