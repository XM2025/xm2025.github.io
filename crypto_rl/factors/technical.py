from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange


def compute_technical_factors(
    ohlcv_panels: Dict[str, pd.DataFrame],
    windows: dict,
    include_cross_sectional: bool = True,
) -> pd.DataFrame:
    """Compute a multi-asset factor matrix indexed by date with multiindex columns (asset, factor).

    Factors:
      - Momentum: close/close.shift(w) - 1
      - Volatility: rolling std of returns for windows
      - RSI: RSI at given windows
      - EMA gaps: (close - ema_w) / ema_w
      - ATR: average true range normalized by close
    Cross-sectional variants (z-scores per day) included if enabled.
    """
    assets: List[str] = list(ohlcv_panels.keys())
    all_frames: List[pd.DataFrame] = []

    for asset, df in ohlcv_panels.items():
        df = df.copy()
        df["ret_1d"] = df["Close"].pct_change()

        # Momentum
        for w in windows.get("mom", [21]):
            df[f"mom_{w}"] = df["Close"].pct_change(w)

        # Volatility
        for w in windows.get("vol", [21]):
            df[f"vol_{w}"] = df["ret_1d"].rolling(w).std()

        # RSI
        for w in windows.get("rsi", [14]):
            rsi = RSIIndicator(close=df["Close"], window=w).rsi()
            df[f"rsi_{w}"] = rsi / 100.0

        # EMA gaps
        for w in windows.get("ema", [21, 50]):
            ema_val = EMAIndicator(close=df["Close"], window=w).ema_indicator()
            df[f"ema_gap_{w}"] = (df["Close"] - ema_val) / ema_val

        # ATR normalized
        for w in windows.get("atr", [14]):
            atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=w).average_true_range()
            df[f"atr_norm_{w}"] = atr / df["Close"]

        # Select factor columns
        factor_cols = [c for c in df.columns if any(c.startswith(prefix) for prefix in ["mom_", "vol_", "rsi_", "ema_gap_", "atr_norm_"])]
        asset_frame = df[factor_cols].copy()
        asset_frame.columns = pd.MultiIndex.from_product([[asset], asset_frame.columns])
        all_frames.append(asset_frame)

    combined = pd.concat(all_frames, axis=1).sort_index()

    if include_cross_sectional:
        # cross-sectional z-scores per day for each base factor name across assets
        cs_frames: List[pd.DataFrame] = []
        for factor_name in set(col[1] for col in combined.columns):
            slice_df = combined.xs(factor_name, axis=1, level=1)
            cs_z = (slice_df - slice_df.mean(axis=1, skipna=True).values.reshape(-1, 1)) / (
                slice_df.std(axis=1, ddof=0, skipna=True).replace(0, np.nan).values.reshape(-1, 1)
            )
            cs_z.columns = pd.MultiIndex.from_product([slice_df.columns, [f"{factor_name}_csz"]])
            cs_frames.append(cs_z)
        combined = pd.concat([combined] + cs_frames, axis=1)

    # Clean: forward-fill per asset across time; keep MultiIndex columns intact
    combined = combined.sort_index()
    combined = combined.groupby(level=0, axis=1).ffill()
    combined = combined.fillna(0.0)
    return combined

