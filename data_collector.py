"""
加密货币数据收集模块
支持多种主流币种的历史和实时数据获取
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import ccxt
import yfinance as yf
from binance.client import Client
import warnings
warnings.filterwarnings('ignore')

class CryptoDataCollector:
    """加密货币数据收集器"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        """
        初始化数据收集器
        
        Args:
            api_key: Binance API密钥
            api_secret: Binance API密钥
        """
        self.api_key = api_key
        self.api_secret = api_secret
        
        # 主流加密货币列表
        self.major_cryptos = [
            'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'DOT', 
            'AVAX', 'MATIC', 'LTC', 'LINK', 'UNI', 'ATOM', 'XLM',
            'BCH', 'ALGO', 'VET', 'ICP', 'FIL', 'TRX', 'ETC', 'THETA',
            'XMR', 'AAVE', 'CAKE', 'NEAR', 'FTT', 'HBAR', 'FLOW'
        ]
        
        # 初始化交易所连接
        self.exchanges = self._initialize_exchanges()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_exchanges(self) -> Dict:
        """初始化交易所连接"""
        exchanges = {}
        
        try:
            # Binance
            if self.api_key and self.api_secret:
                exchanges['binance'] = Client(self.api_key, self.api_secret)
            
            # CCXT交易所
            exchanges['binance_ccxt'] = ccxt.binance()
            exchanges['okx'] = ccxt.okx()
            exchanges['huobi'] = ccxt.huobi()
            
        except Exception as e:
            self.logger.warning(f"交易所初始化警告: {e}")
        
        return exchanges
    
    def get_historical_data(self, 
                          symbol: str, 
                          interval: str = '1d',
                          days: int = 365,
                          exchange: str = 'binance') -> pd.DataFrame:
        """
        获取历史价格数据
        
        Args:
            symbol: 交易对符号 (如 'BTCUSDT')
            interval: 时间间隔 ('1m', '5m', '15m', '1h', '4h', '1d')
            days: 历史天数
            exchange: 交易所名称
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        try:
            if exchange == 'binance' and 'binance' in self.exchanges:
                return self._get_binance_data(symbol, interval, days)
            elif exchange in ['binance_ccxt', 'okx', 'huobi']:
                return self._get_ccxt_data(symbol, interval, days, exchange)
            else:
                # 使用yfinance作为备选
                return self._get_yfinance_data(symbol, interval, days)
                
        except Exception as e:
            self.logger.error(f"获取{symbol}历史数据失败: {e}")
            return pd.DataFrame()
    
    def _get_binance_data(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """从Binance获取数据"""
        client = self.exchanges['binance']
        
        # 转换时间间隔格式
        interval_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY
        }
        
        klines = client.get_historical_klines(
            symbol, 
            interval_map.get(interval, Client.KLINE_INTERVAL_1DAY),
            f"{days} days ago UTC"
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'count', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        # 数据类型转换
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].set_index('timestamp')
    
    def _get_ccxt_data(self, symbol: str, interval: str, days: int, exchange: str) -> pd.DataFrame:
        """从CCXT交易所获取数据"""
        ex = self.exchanges[exchange]
        
        # 计算开始时间
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        # 获取OHLCV数据
        ohlcv = ex.fetch_ohlcv(symbol, interval, since)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df.set_index('timestamp')
    
    def _get_yfinance_data(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """从Yahoo Finance获取数据（备选方案）"""
        # 转换符号格式
        yf_symbol = f"{symbol[:3]}-USD" if symbol.endswith('USDT') else f"{symbol}-USD"
        
        # 转换时间间隔
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', 
            '1h': '1h', '4h': '4h', '1d': '1d'
        }
        
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(
            period=f"{days}d",
            interval=interval_map.get(interval, '1d')
        )
        
        # 重命名列
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume'
        })
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def get_multiple_cryptos_data(self, 
                                 symbols: List[str] = None,
                                 interval: str = '1d',
                                 days: int = 365) -> Dict[str, pd.DataFrame]:
        """
        批量获取多个加密货币的历史数据
        
        Args:
            symbols: 交易对列表，默认使用主流币种
            interval: 时间间隔
            days: 历史天数
            
        Returns:
            字典，键为交易对，值为对应的DataFrame
        """
        if symbols is None:
            symbols = [f"{crypto}USDT" for crypto in self.major_cryptos]
        
        data_dict = {}
        
        for symbol in symbols:
            self.logger.info(f"获取 {symbol} 数据...")
            df = self.get_historical_data(symbol, interval, days)
            
            if not df.empty:
                data_dict[symbol] = df
                time.sleep(0.1)  # 避免API限制
            else:
                self.logger.warning(f"无法获取 {symbol} 数据")
        
        return data_dict
    
    def get_market_cap_data(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        获取市值数据
        
        Args:
            symbols: 币种列表
            
        Returns:
            包含市值信息的DataFrame
        """
        if symbols is None:
            symbols = self.major_cryptos
        
        market_data = []
        
        try:
            # 使用CoinGecko API获取市值数据
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 100,
                'page': 1,
                'sparkline': False
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            for item in data:
                if item['symbol'].upper() in [s.upper() for s in symbols]:
                    market_data.append({
                        'symbol': item['symbol'].upper(),
                        'name': item['name'],
                        'market_cap': item['market_cap'],
                        'market_cap_rank': item['market_cap_rank'],
                        'current_price': item['current_price'],
                        'price_change_24h': item['price_change_percentage_24h'],
                        'volume_24h': item['total_volume']
                    })
        
        except Exception as e:
            self.logger.error(f"获取市值数据失败: {e}")
        
        return pd.DataFrame(market_data)
    
    def get_fear_greed_index(self) -> Dict:
        """
        获取恐慌贪婪指数
        
        Returns:
            包含恐慌贪婪指数的字典
        """
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url)
            data = response.json()
            
            return {
                'value': int(data['data'][0]['value']),
                'classification': data['data'][0]['value_classification'],
                'timestamp': data['data'][0]['timestamp']
            }
        
        except Exception as e:
            self.logger.error(f"获取恐慌贪婪指数失败: {e}")
            return {}
    
    def save_data(self, data_dict: Dict[str, pd.DataFrame], path: str = "/workspace/data/"):
        """
        保存数据到本地
        
        Args:
            data_dict: 数据字典
            path: 保存路径
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        for symbol, df in data_dict.items():
            filename = f"{path}/{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename)
            self.logger.info(f"已保存 {symbol} 数据到 {filename}")

# 使用示例
if __name__ == "__main__":
    # 初始化数据收集器
    collector = CryptoDataCollector()
    
    # 获取主流币种数据
    print("开始收集主流加密货币数据...")
    crypto_data = collector.get_multiple_cryptos_data(days=500)
    
    # 获取市值数据
    market_cap_data = collector.get_market_cap_data()
    print(f"获取到 {len(market_cap_data)} 个币种的市值数据")
    
    # 获取恐慌贪婪指数
    fear_greed = collector.get_fear_greed_index()
    print(f"当前恐慌贪婪指数: {fear_greed}")
    
    # 保存数据
    collector.save_data(crypto_data)
    
    print(f"成功收集 {len(crypto_data)} 个交易对的数据")