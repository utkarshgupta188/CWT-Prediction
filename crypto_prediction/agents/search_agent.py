from typing import List, Dict
from crypto_prediction.services.polymarket import PolymarketService
from crypto_prediction.services.kalshi import KalshiService
from crypto_prediction.utils.logger import setup_logger
from crypto_prediction.schemas.config import settings

logger = setup_logger(settings.LOG_LEVEL)

class SearchAgent:
    def __init__(self):
        self.polymarket = PolymarketService()
        self.kalshi = KalshiService()

    async def execute(self, limit_per_platform: int = 10) -> List[dict]:
        """
        Search prediction markets on Polymarket and Kalshi, normalize them and return.
        """
        logger.info("SearchAgent: Starting search for crypto prediction markets...")
        
        # Run queries in parallel
        import asyncio
        poly_task = self.polymarket.get_active_markets(limit=limit_per_platform)
        kalshi_task = self.kalshi.get_active_markets(limit=limit_per_platform)
        
        poly_markets, kalshi_markets = await asyncio.gather(poly_task, kalshi_task)
        
        combined = poly_markets + kalshi_markets
        logger.info(f"SearchAgent: Found {len(combined)} normalized markets total.")
        return combined
