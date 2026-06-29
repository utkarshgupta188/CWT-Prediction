import json
import time
from loguru import logger
from tools.registry import registry

SEARCH_KALSHI_SCHEMA = {
    "name": "search_kalshi",
    "description": "Search active prediction markets on Kalshi for crypto-related events",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of markets to fetch",
                "default": 20
            }
        }
    }
}

async def _search_kalshi_handler(args):
    limit = int(args.get("limit", 20))
    logger.info("HermesTool[search_kalshi]: Started")
    start = time.time()
    try:
        from crypto_prediction.services.kalshi import KalshiService
        service = KalshiService()
        markets = await service.get_active_markets(limit=limit)
        elapsed = time.time() - start
        logger.info(f"HermesTool[search_kalshi]: Finished in {elapsed:.2f}s, found {len(markets)} markets")
        return json.dumps({"success": True, "markets": markets, "platform": "Kalshi"}, default=str)
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"HermesTool[search_kalshi]: Error after {elapsed:.2f}s: {e}")
        return json.dumps({"error": str(e), "markets": []})

registry.register(
    name="search_kalshi",
    toolset="crypto_search",
    schema=SEARCH_KALSHI_SCHEMA,
    handler=_search_kalshi_handler,
    is_async=True,
    description="Search active prediction markets on Kalshi",
    emoji="",
)
