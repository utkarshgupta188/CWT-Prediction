import pandas as pd
from crypto_prediction.providers.base import MarketDataProvider
from crypto_prediction.providers.binance_provider import BinanceProvider
from crypto_prediction.utils.logger import setup_logger
from crypto_prediction.schemas.config import settings

logger = setup_logger(settings.LOG_LEVEL)

class MarketDataAgent:
    def __init__(self, provider: MarketDataProvider = None):
        self.provider = provider or BinanceProvider()

    async def execute(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical candle data for symbol.
        """
        logger.info(f"MarketDataAgent: Fetching data for {symbol} ({interval}, limit={limit})...")
        df = await self.provider.get_klines(symbol, interval, limit)
        return df
