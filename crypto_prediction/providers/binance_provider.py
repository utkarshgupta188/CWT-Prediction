import pandas as pd
import httpx
from crypto_prediction.providers.base import MarketDataProvider
from crypto_prediction.utils.helpers import request_with_retry
from crypto_prediction.utils.logger import setup_logger
from crypto_prediction.schemas.config import settings

logger = setup_logger(settings.LOG_LEVEL)

class BinanceProvider(MarketDataProvider):
    def __init__(self, base_url: str = "https://data-api.binance.vision"):
        self.base_url = base_url

    async def get_klines(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """
        Fetches public OHLCV data from Binance data API.
        Binance Kline/Candlestick endpoint returns:
        [
          [
            1499040000000,      // Open time
            "0.01634790",       // Open
            "0.80000000",       // High
            "0.01575800",       // Low
            "0.01577100",       // Close
            "148976.11400000",  // Volume
            1499644799999,      // Close time
            "2434.19055334",    // Quote asset volume
            308,                // Number of trades
            "1756.87400000",    // Taker buy base asset volume
            "28.46694368",      // Taker buy quote asset volume
            "17928899.62484339" // Ignore.
          ]
        ]
        """
        endpoint = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }
        
        logger.info(f"Fetching Binance candles for {symbol} ({interval}, limit={limit})...")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await request_with_retry(client, "GET", endpoint, params=params)
            data = response.json()
            
        # Parse into DataFrame
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count", "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        
        # Keep only required columns and convert to correct types
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
            
        logger.info(f"Successfully fetched {len(df)} candles for {symbol}.")
        return df

    async def get_latest_price(self, symbol: str) -> float:
        """
        Fetch the current ticker price.
        """
        endpoint = f"{self.base_url}/api/v3/ticker/price"
        params = {"symbol": symbol.upper()}
        
        logger.info(f"Fetching latest price for {symbol}...")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await request_with_retry(client, "GET", endpoint, params=params)
            data = response.json()
            
        price = float(data["price"])
        logger.info(f"Latest price for {symbol} is {price}.")
        return price
