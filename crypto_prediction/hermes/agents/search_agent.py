import json
import time
from typing import List
from loguru import logger
from tools.registry import registry

ALLOWED_TOOLS = ["search_polymarket", "search_kalshi"]

class HermesSearchAgent:
    """Hermes Search Agent - restricted to Polymarket and Kalshi search tools."""

    @property
    def allowed_tools(self) -> List[str]:
        return list(ALLOWED_TOOLS)

    async def execute(self, limit_per_platform: int = 10) -> List[dict]:
        logger.info("HermesSearchAgent: Started")
        start = time.time()
        try:
            poly_result_str = registry.dispatch("search_polymarket", {"limit": limit_per_platform})
            kalshi_result_str = registry.dispatch("search_kalshi", {"limit": limit_per_platform})
            poly_result = json.loads(poly_result_str)
            kalshi_result = json.loads(kalshi_result_str)
            markets = []
            if "error" not in poly_result:
                markets.extend(poly_result.get("markets", []))
            if "error" not in kalshi_result:
                markets.extend(kalshi_result.get("markets", []))
            elapsed = time.time() - start
            logger.info(f"HermesSearchAgent: Finished in {elapsed:.2f}s, found {len(markets)} markets total")
            return markets
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"HermesSearchAgent: Error after {elapsed:.2f}s: {e}")
            raise
