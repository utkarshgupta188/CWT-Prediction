from abc import ABC, abstractmethod
import pandas as pd

class MarketDataProvider(ABC):
    @abstractmethod
    async def get_klines(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        Returns:
            pd.DataFrame with columns: timestamp, open, high, low, close, volume.
            Index should be timestamps or a default integer index, but timestamps must be present.
        """
        pass

    @abstractmethod
    async def get_latest_price(self, symbol: str) -> float:
        """
        Fetch the latest price of the symbol.
        """
        pass
