from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _zscore(series: pd.Series, window: int) -> pd.Series:
    r = series.rolling(window)
    return (series - r.mean()) / (r.std(ddof=0) + 1e-9)


def compute_factors(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    returns = close.pct_change().fillna(0.0)
    out["ret_1"] = returns
    if "momentum_7" in feature_list:
        out["mom_7"] = close.pct_change(7)
    if "momentum_30" in feature_list:
        out["mom_30"] = close.pct_change(30)
    if "volatility_30" in feature_list:
        out["vol_30"] = returns.rolling(30).std(ddof=0)
    if "volume_z_30" in feature_list:
        out["volu_z_30"] = _zscore(out["volume"], 30)
    if "rsi_14" in feature_list:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    if "ema_slope_20" in feature_list:
        ema = close.ewm(span=20, adjust=False).mean()
        out["ema_slope_20"] = ema.pct_change()
    if "price_to_ma_30" in feature_list:
        ma30 = close.rolling(30).mean()
        out["price_to_ma_30"] = close / (ma30 + 1e-9) - 1.0
    if "reversal_1" in feature_list:
        out["reversal_1"] = -returns.shift(1)
    # corr_with_btc_60 will be computed in panel step
    return out


def build_feature_panel(aligned: Dict[str, pd.DataFrame], feature_list: List[str], normalize_cross_section: bool = True, include_prev_weights: bool = True) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, List[str]]:
    symbols = list(aligned.keys())
    dates = pd.DatetimeIndex(sorted(set.union(*[set(df['date']) for df in aligned.values()])))
    enriched: Dict[str, pd.DataFrame] = {}
    for sym, df in aligned.items():
        enriched[sym] = compute_factors(df, feature_list)
    # special factor: corr_with_btc_60
    if "corr_with_btc_60" in feature_list and "BTC/USDT" in symbols:
        btc_ret = enriched["BTC/USDT"]["close"].pct_change()
        for sym in symbols:
            r = enriched[sym]["close"].pct_change()
            corr = r.rolling(60).corr(btc_ret)
            enriched[sym]["corr_btc_60"] = corr
    # assemble feature names
    cols = []
    if "momentum_7" in feature_list:
        cols.append("mom_7")
    if "momentum_30" in feature_list:
        cols.append("mom_30")
    if "volatility_30" in feature_list:
        cols.append("vol_30")
    if "volume_z_30" in feature_list:
        cols.append("volu_z_30")
    if "rsi_14" in feature_list:
        cols.append("rsi_14")
    if "ema_slope_20" in feature_list:
        cols.append("ema_slope_20")
    if "price_to_ma_30" in feature_list:
        cols.append("price_to_ma_30")
    if "reversal_1" in feature_list:
        cols.append("reversal_1")
    if "corr_with_btc_60" in feature_list:
        cols.append("corr_btc_60")

    num_assets = len(symbols)
    num_features = len(cols)
    # shape: T x N x F (+ optional prev weights)
    T = len(dates)
    X = np.full((T, num_assets, num_features), np.nan, dtype=np.float32)
    R = np.full((T, num_assets), np.nan, dtype=np.float32)
    for i, sym in enumerate(symbols):
        df = enriched[sym].set_index("date").reindex(dates)
        R[:, i] = df["close"].pct_change().fillna(0.0).values.astype(np.float32)
        X[:, i, :] = df[cols].values.astype(np.float32)
    # cross-sectional normalize per day per feature
    if normalize_cross_section and num_features > 0:
        for t in range(T):
            for f in range(num_features):
                v = X[t, :, f]
                mu = np.nanmean(v)
                sd = np.nanstd(v)
                if np.isfinite(sd) and sd > 1e-9:
                    X[t, :, f] = (v - mu) / sd
                else:
                    X[t, :, f] = 0.0
    # fill nans
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    # final feature names
    feature_names = cols.copy()
    if include_prev_weights:
        feature_names.append("prev_weight")
    return dates, X, R, feature_names

