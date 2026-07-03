from abc import ABC, abstractmethod
from typing import List
from .models import PredictionMarket

class MarketFinder(ABC):
    @abstractmethod
    async def find_markets(self) -> List[PredictionMarket]:
        """
        Finds open crypto short-term prediction markets.
        """
        pass
