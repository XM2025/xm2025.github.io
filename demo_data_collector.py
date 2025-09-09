"""
演示数据收集模块
生成模拟的加密货币数据用于演示
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import os

class DemoDataCollector:
    """演示数据收集器 - 生成模拟数据"""
    
    def __init__(self):
        """初始化演示数据收集器"""
        # 主流加密货币列表
        self.major_cryptos = [
            'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'DOT', 
            'AVAX', 'MATIC', 'LTC', 'LINK', 'UNI', 'ATOM', 'XLM'
        ]
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_crypto_data(self, 
                           symbol: str, 
                           days: int = 365,
                           initial_price: float = None) -> pd.DataFrame:
        """
        生成模拟的加密货币数据
        
        Args:
            symbol: 交易对符号
            days: 生成天数
            initial_price: 初始价格
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        # 设置初始价格
        if initial_price is None:
            price_ranges = {
                'BTC': 40000, 'ETH': 2500, 'BNB': 300, 'XRP': 0.5,
                'ADA': 0.4, 'SOL': 80, 'DOGE': 0.08, 'DOT': 6,
                'AVAX': 15, 'MATIC': 0.8, 'LTC': 70, 'LINK': 8,
                'UNI': 6, 'ATOM': 10, 'XLM': 0.1
            }
            base_symbol = symbol.replace('USDT', '')
            initial_price = price_ranges.get(base_symbol, 100)
        
        # 生成日期序列
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 生成价格数据 (几何布朗运动)
        np.random.seed(42 + hash(symbol) % 1000)  # 为每个符号设置不同的随机种子
        
        # 参数设置
        dt = 1/365  # 日频数据
        mu = 0.0002  # 年化漂移率
        sigma = 0.03  # 年化波动率
        
        # 生成随机游走
        returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), len(dates))
        
        # 添加一些趋势和周期性
        trend = np.linspace(-0.001, 0.001, len(dates))
        cycle = 0.0005 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)  # 30天周期
        
        returns = returns + trend + cycle
        
        # 生成价格序列
        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices)
        
        # 生成OHLC数据
        data = []
        for i, price in enumerate(prices):
            # 生成日内波动
            daily_volatility = 0.02  # 日内2%波动
            high = price * (1 + np.random.uniform(0, daily_volatility))
            low = price * (1 - np.random.uniform(0, daily_volatility))
            
            # 确保开盘价在合理范围内
            if i == 0:
                open_price = price
            else:
                open_price = prices[i-1] * (1 + np.random.uniform(-0.01, 0.01))
            
            # 收盘价就是计算出的价格
            close_price = price
            
            # 确保OHLC逻辑正确
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            # 生成成交量 (与价格变动相关)
            volume_base = 1000000
            price_change = abs(returns[i]) if i < len(returns) else 0.01
            volume = volume_base * (1 + price_change * 10) * np.random.uniform(0.5, 1.5)
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_multiple_cryptos_data(self, 
                                 symbols: List[str] = None,
                                 interval: str = '1d',
                                 days: int = 365) -> Dict[str, pd.DataFrame]:
        """
        批量生成多个加密货币的模拟数据
        
        Args:
            symbols: 交易对列表，默认使用主流币种
            interval: 时间间隔 (暂时只支持1d)
            days: 历史天数
            
        Returns:
            字典，键为交易对，值为对应的DataFrame
        """
        if symbols is None:
            symbols = [f"{crypto}USDT" for crypto in self.major_cryptos[:5]]  # 只生成前5个
        
        data_dict = {}
        
        for symbol in symbols:
            self.logger.info(f"生成 {symbol} 模拟数据...")
            df = self.generate_crypto_data(symbol, days)
            data_dict[symbol] = df
        
        return data_dict
    
    def get_market_cap_data(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        生成模拟市值数据
        
        Args:
            symbols: 币种列表
            
        Returns:
            包含市值信息的DataFrame
        """
        if symbols is None:
            symbols = self.major_cryptos[:5]
        
        market_data = []
        
        # 模拟市值数据
        market_caps = {
            'BTC': 800000000000, 'ETH': 300000000000, 'BNB': 40000000000,
            'XRP': 25000000000, 'ADA': 15000000000, 'SOL': 35000000000
        }
        
        for i, symbol in enumerate(symbols):
            base_symbol = symbol.replace('USDT', '') if 'USDT' in symbol else symbol
            market_data.append({
                'symbol': base_symbol,
                'name': f"{base_symbol} Token",
                'market_cap': market_caps.get(base_symbol, 1000000000),
                'market_cap_rank': i + 1,
                'current_price': np.random.uniform(1, 50000),
                'price_change_24h': np.random.uniform(-10, 10),
                'volume_24h': np.random.uniform(1000000, 10000000000)
            })
        
        return pd.DataFrame(market_data)
    
    def get_fear_greed_index(self) -> Dict:
        """
        生成模拟恐慌贪婪指数
        
        Returns:
            包含恐慌贪婪指数的字典
        """
        value = np.random.randint(10, 90)
        
        if value < 25:
            classification = "Extreme Fear"
        elif value < 45:
            classification = "Fear"
        elif value < 55:
            classification = "Neutral"
        elif value < 75:
            classification = "Greed"
        else:
            classification = "Extreme Greed"
        
        return {
            'value': value,
            'classification': classification,
            'timestamp': datetime.now().timestamp()
        }
    
    def save_data(self, data_dict: Dict[str, pd.DataFrame], path: str = "/workspace/data/"):
        """
        保存数据到本地
        
        Args:
            data_dict: 数据字典
            path: 保存路径
        """
        os.makedirs(path, exist_ok=True)
        
        for symbol, df in data_dict.items():
            filename = f"{path}/{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename)
            self.logger.info(f"已保存 {symbol} 数据到 {filename}")

# 使用示例
if __name__ == "__main__":
    # 初始化演示数据收集器
    collector = DemoDataCollector()
    
    # 生成主流币种数据
    print("开始生成模拟加密货币数据...")
    crypto_data = collector.get_multiple_cryptos_data(days=500)
    
    # 生成市值数据
    market_cap_data = collector.get_market_cap_data()
    print(f"生成 {len(market_cap_data)} 个币种的市值数据")
    
    # 生成恐慌贪婪指数
    fear_greed = collector.get_fear_greed_index()
    print(f"当前恐慌贪婪指数: {fear_greed}")
    
    # 保存数据
    collector.save_data(crypto_data)
    
    print(f"成功生成 {len(crypto_data)} 个交易对的模拟数据")
    
    # 显示数据统计
    for symbol, df in crypto_data.items():
        print(f"{symbol}: {len(df)} 天数据, 价格范围 ${df['close'].min():.2f} - ${df['close'].max():.2f}")