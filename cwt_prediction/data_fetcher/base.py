from abc import ABC, abstractmethod
import pandas as pd
from typing import List
from .models import OHLCVBar

class DataFetcher(ABC):
    @abstractmethod
    async def fetch_history(self, asset: str, limit: int = 400) -> pd.DataFrame:
        """
        Fetches N history bars of OHLCV data.
        Returns a pandas DataFrame with columns:
        ['open', 'high', 'low', 'close', 'volume'] and a DatetimeIndex or 'timestamps' column.
        """
        pass
