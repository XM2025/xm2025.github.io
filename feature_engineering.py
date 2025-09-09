"""
特征工程模块
实现技术指标、市场情绪、链上数据等多样化因子
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 简化的技术指标实现，避免依赖TA-Lib
class TechnicalIndicators:
    """简化的技术指标实现"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """简单移动平均"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """指数移动平均"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """相对强弱指数"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD指标"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """布林带"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """平均真实波幅"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """随机指标"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent

# 创建全局实例
talib = TechnicalIndicators()

class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self):
        """初始化特征工程器"""
        self.feature_names = []
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加技术指标特征
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            添加技术指标后的DataFrame
        """
        df = df.copy()
        
        # 价格相关指标
        df = self._add_price_features(df)
        
        # 趋势指标
        df = self._add_trend_indicators(df)
        
        # 动量指标
        df = self._add_momentum_indicators(df)
        
        # 波动率指标
        df = self._add_volatility_indicators(df)
        
        # 成交量指标
        df = self._add_volume_indicators(df)
        
        # 支撑阻力指标
        df = self._add_support_resistance_indicators(df)
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加价格相关特征"""
        # 收益率
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 价格位置
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # 真实波幅
        df['true_range'] = talib.atr(df['high'], df['low'], df['close'], window=1)
        
        # 典型价格
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # 加权收盘价
        df['weighted_close'] = (df['high'] + df['low'] + 2 * df['close']) / 4
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加趋势指标"""
        # 移动平均线
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = talib.sma(df['close'], window=period)
            df[f'ema_{period}'] = talib.ema(df['close'], window=period)
            
            # 价格相对于移动平均线的位置
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
            df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.macd(df['close'])
        
        # 简化的趋势指标
        # 价格动量
        df['price_momentum_14'] = df['close'] / df['close'].shift(14) - 1
        df['price_momentum_30'] = df['close'] / df['close'].shift(30) - 1
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加动量指标"""
        # RSI
        for period in [6, 14, 21]:
            df[f'rsi_{period}'] = talib.rsi(df['close'], window=period)
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.stochastic(df['high'], df['low'], df['close'])
        
        # ROC (变化率)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(periods=period) * 100
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加波动率指标"""
        # 布林带
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.bollinger_bands(df['close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (平均真实波幅)
        df['atr'] = talib.atr(df['high'], df['low'], df['close'], window=14)
        df['atr_ratio'] = df['atr'] / df['close']
        
        # 历史波动率
        for period in [5, 10, 20, 30]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std() * np.sqrt(365)
        
        # Keltner Channels
        ema_20 = df['close'].ewm(span=20).mean()
        atr_10 = df['atr'].rolling(10).mean()
        df['kc_upper'] = ema_20 + (2 * atr_10)
        df['kc_lower'] = ema_20 - (2 * atr_10)
        df['kc_position'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'])
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加成交量指标"""
        # 成交量移动平均
        for period in [5, 10, 20]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
        
        # 简化的成交量指标
        # 价格成交量趋势
        df['price_volume_trend'] = (df['close'].pct_change() * df['volume']).cumsum()
        
        # 成交量加权平均价格
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # 成交量价格趋势 (VPT)
        df['vpt'] = (df['volume'] * df['returns']).cumsum()
        
        # 负成交量指数 (NVI)
        df['nvi'] = self._calculate_nvi(df)
        
        # 正成交量指数 (PVI)
        df['pvi'] = self._calculate_pvi(df)
        
        return df
    
    def _add_support_resistance_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加支撑阻力指标"""
        # Pivot Points
        df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['r1'] = 2 * df['pivot'] - df['low'].shift(1)
        df['s1'] = 2 * df['pivot'] - df['high'].shift(1)
        df['r2'] = df['pivot'] + (df['high'].shift(1) - df['low'].shift(1))
        df['s2'] = df['pivot'] - (df['high'].shift(1) - df['low'].shift(1))
        
        # 价格距离支撑阻力的位置
        df['distance_to_pivot'] = (df['close'] - df['pivot']) / df['pivot']
        
        return df
    
    def _calculate_nvi(self, df: pd.DataFrame) -> pd.Series:
        """计算负成交量指数"""
        nvi = pd.Series(index=df.index, dtype=float)
        nvi.iloc[0] = 1000
        
        for i in range(1, len(df)):
            if df['volume'].iloc[i] < df['volume'].iloc[i-1]:
                nvi.iloc[i] = nvi.iloc[i-1] + (df['returns'].iloc[i] * nvi.iloc[i-1])
            else:
                nvi.iloc[i] = nvi.iloc[i-1]
        
        return nvi
    
    def _calculate_pvi(self, df: pd.DataFrame) -> pd.Series:
        """计算正成交量指数"""
        pvi = pd.Series(index=df.index, dtype=float)
        pvi.iloc[0] = 1000
        
        for i in range(1, len(df)):
            if df['volume'].iloc[i] > df['volume'].iloc[i-1]:
                pvi.iloc[i] = pvi.iloc[i-1] + (df['returns'].iloc[i] * pvi.iloc[i-1])
            else:
                pvi.iloc[i] = pvi.iloc[i-1]
        
        return pvi
    
    def add_cross_asset_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        添加跨资产特征
        
        Args:
            data_dict: 多个资产的数据字典
            
        Returns:
            添加跨资产特征后的数据字典
        """
        # 计算市场整体指标
        market_data = self._calculate_market_indicators(data_dict)
        
        # 为每个资产添加市场特征
        enhanced_data = {}
        for symbol, df in data_dict.items():
            enhanced_df = df.copy()
            
            # 添加市场特征
            enhanced_df = enhanced_df.join(market_data, how='left')
            
            # 添加相对强度特征
            enhanced_df = self._add_relative_strength_features(enhanced_df, market_data)
            
            # 添加相关性特征
            enhanced_df = self._add_correlation_features(enhanced_df, data_dict, symbol)
            
            enhanced_data[symbol] = enhanced_df
        
        return enhanced_data
    
    def _calculate_market_indicators(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """计算市场整体指标"""
        # 获取所有资产的收益率
        returns_df = pd.DataFrame()
        for symbol, df in data_dict.items():
            if 'returns' not in df.columns:
                df['returns'] = df['close'].pct_change()
            returns_df[symbol] = df['returns']
        
        market_data = pd.DataFrame(index=returns_df.index)
        
        # 市场平均收益率
        market_data['market_return'] = returns_df.mean(axis=1)
        
        # 市场波动率
        market_data['market_volatility'] = returns_df.std(axis=1)
        
        # 上涨资产比例
        market_data['up_ratio'] = (returns_df > 0).mean(axis=1)
        
        # 市场动量
        market_data['market_momentum'] = returns_df.mean(axis=1).rolling(20).sum()
        
        # VIX类似指标 (基于收益率分布)
        market_data['crypto_vix'] = returns_df.std(axis=1).rolling(20).mean()
        
        return market_data
    
    def _add_relative_strength_features(self, df: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """添加相对强度特征"""
        # 相对于市场的收益率
        df['relative_return'] = df['returns'] - market_data['market_return']
        
        # 相对强度指数 (相对于市场)
        df['relative_strength'] = df['returns'].rolling(20).mean() / market_data['market_return'].rolling(20).mean()
        
        # Beta系数
        df['beta'] = self._calculate_beta(df['returns'], market_data['market_return'])
        
        return df
    
    def _add_correlation_features(self, df: pd.DataFrame, data_dict: Dict[str, pd.DataFrame], current_symbol: str) -> pd.DataFrame:
        """添加相关性特征"""
        # 与BTC的相关性 (如果当前资产不是BTC)
        if current_symbol != 'BTCUSDT' and 'BTCUSDT' in data_dict:
            btc_returns = data_dict['BTCUSDT']['returns'] if 'returns' in data_dict['BTCUSDT'].columns else data_dict['BTCUSDT']['close'].pct_change()
            df['btc_correlation'] = df['returns'].rolling(30).corr(btc_returns)
        
        # 与ETH的相关性 (如果当前资产不是ETH)
        if current_symbol != 'ETHUSDT' and 'ETHUSDT' in data_dict:
            eth_returns = data_dict['ETHUSDT']['returns'] if 'returns' in data_dict['ETHUSDT'].columns else data_dict['ETHUSDT']['close'].pct_change()
            df['eth_correlation'] = df['returns'].rolling(30).corr(eth_returns)
        
        return df
    
    def _calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series, window: int = 60) -> pd.Series:
        """计算Beta系数"""
        def rolling_beta(x, y):
            if len(x) < 2 or len(y) < 2:
                return np.nan
            covariance = np.cov(x, y)[0, 1]
            variance = np.var(y)
            return covariance / variance if variance != 0 else np.nan
        
        beta_series = pd.Series(index=asset_returns.index, dtype=float)
        for i in range(window, len(asset_returns)):
            x = asset_returns.iloc[i-window:i].dropna()
            y = market_returns.iloc[i-window:i].dropna()
            if len(x) > 1 and len(y) > 1:
                beta_series.iloc[i] = rolling_beta(x, y)
        
        return beta_series
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加时间特征
        
        Args:
            df: 数据DataFrame
            
        Returns:
            添加时间特征后的DataFrame
        """
        df = df.copy()
        
        # 确保索引是datetime类型
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 基本时间特征
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        # 周期性特征 (使用sin/cos编码)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 是否为周末
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # 是否为月初/月末
        df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10, 20]) -> pd.DataFrame:
        """
        添加滞后特征
        
        Args:
            df: 数据DataFrame
            lags: 滞后期数列表
            
        Returns:
            添加滞后特征后的DataFrame
        """
        df = df.copy()
        
        # 主要特征的滞后值
        key_features = ['returns', 'volume', 'rsi_14', 'macd', 'bb_position', 'atr_ratio']
        
        for feature in key_features:
            if feature in df.columns:
                for lag in lags:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20, 30]) -> pd.DataFrame:
        """
        添加滚动窗口特征
        
        Args:
            df: 数据DataFrame
            windows: 窗口大小列表
            
        Returns:
            添加滚动特征后的DataFrame
        """
        df = df.copy()
        
        # 主要特征的滚动统计
        key_features = ['returns', 'volume', 'close']
        
        for feature in key_features:
            if feature in df.columns:
                for window in windows:
                    # 滚动均值
                    df[f'{feature}_mean_{window}'] = df[feature].rolling(window).mean()
                    # 滚动标准差
                    df[f'{feature}_std_{window}'] = df[feature].rolling(window).std()
                    # 滚动最大值
                    df[f'{feature}_max_{window}'] = df[feature].rolling(window).max()
                    # 滚动最小值
                    df[f'{feature}_min_{window}'] = df[feature].rolling(window).min()
                    # 当前值相对于滚动均值的位置
                    df[f'{feature}_zscore_{window}'] = (df[feature] - df[f'{feature}_mean_{window}']) / df[f'{feature}_std_{window}']
        
        return df
    
    def engineer_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        完整的特征工程流程
        
        Args:
            data_dict: 原始数据字典
            
        Returns:
            完整特征工程后的数据字典
        """
        print("开始特征工程...")
        
        # 1. 添加技术指标
        print("添加技术指标...")
        enhanced_data = {}
        for symbol, df in data_dict.items():
            enhanced_data[symbol] = self.add_technical_indicators(df)
        
        # 2. 添加跨资产特征
        print("添加跨资产特征...")
        enhanced_data = self.add_cross_asset_features(enhanced_data)
        
        # 3. 添加时间特征
        print("添加时间特征...")
        for symbol in enhanced_data:
            enhanced_data[symbol] = self.add_time_features(enhanced_data[symbol])
        
        # 4. 添加滞后特征
        print("添加滞后特征...")
        for symbol in enhanced_data:
            enhanced_data[symbol] = self.add_lag_features(enhanced_data[symbol])
        
        # 5. 添加滚动特征
        print("添加滚动特征...")
        for symbol in enhanced_data:
            enhanced_data[symbol] = self.add_rolling_features(enhanced_data[symbol])
        
        # 6. 清理数据
        print("清理数据...")
        for symbol in enhanced_data:
            # 移除无限值和NaN
            enhanced_data[symbol] = enhanced_data[symbol].replace([np.inf, -np.inf], np.nan)
            enhanced_data[symbol] = enhanced_data[symbol].fillna(method='ffill').fillna(0)
        
        print(f"特征工程完成! 每个资产包含 {enhanced_data[list(enhanced_data.keys())[0]].shape[1]} 个特征")
        
        return enhanced_data

# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    sample_data = {
        'BTCUSDT': pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 40000,
            'high': np.random.randn(len(dates)).cumsum() + 41000,
            'low': np.random.randn(len(dates)).cumsum() + 39000,
            'close': np.random.randn(len(dates)).cumsum() + 40000,
            'volume': np.random.rand(len(dates)) * 1000000
        }, index=dates)
    }
    
    # 初始化特征工程器
    feature_engineer = FeatureEngineer()
    
    # 执行特征工程
    enhanced_data = feature_engineer.engineer_features(sample_data)
    
    print("特征工程完成!")
    print(f"BTCUSDT数据形状: {enhanced_data['BTCUSDT'].shape}")
    print(f"特征列数: {len(enhanced_data['BTCUSDT'].columns)}")