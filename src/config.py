from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    exchange: str = "binance"
    symbols: List[str] = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
        "TRX/USDT", "MATIC/USDT", "ATOM/USDT", "LTC/USDT", "UNI/USDT",
    ])
    timeframe: str = "1d"
    since: str = "2019-01-01"
    max_lookback_days: int = 3650


@dataclass
class EnvConfig:
    trading_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    risk_free_rate: float = 0.0
    warmup_days: int = 60
    max_weight_leverage: float = 1.0


@dataclass
class TrainConfig:
    seed: int = 42
    total_timesteps: int = 400_000
    eval_episodes: int = 5
    train_start_date: str = "2020-01-01"
    train_end_date: str = "2023-12-31"
    test_start_date: str = "2024-01-01"
    test_end_date: str = "2024-12-31"


@dataclass
class FactorConfig:
    # lookbacks in days for factors
    lookbacks: List[int] = field(default_factory=lambda: [3, 7, 14, 21, 30, 60, 120])


@dataclass
class Config:
    data: DataConfig = DataConfig()
    env: EnvConfig = EnvConfig()
    train: TrainConfig = TrainConfig()
    factor: FactorConfig = FactorConfig()