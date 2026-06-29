import json
import time
from loguru import logger
from tools.registry import registry

SEARCH_POLYMARKET_SCHEMA = {
    "name": "search_polymarket",
    "description": "Search active prediction markets on Polymarket for crypto-related events",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional search term to filter markets (e.g., BTC, ETH)",
                "default": ""
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of markets to fetch",
                "default": 3
            }
        }
    }
}

async def _search_polymarket_handler(args, **kwargs):
    limit = min(int(args.get("limit", 3)), 5)
    query = args.get("query", "").strip() or None
    logger.info(f"HermesTool[search_polymarket]: Started (query={query})")
    start = time.time()
    try:
        from crypto_prediction.services.polymarket import PolymarketService
        service = PolymarketService()
        markets = await service.get_active_markets(query=query, limit=limit, max_retries=1)
        elapsed = time.time() - start
        logger.info(f"HermesTool[search_polymarket]: Finished in {elapsed:.2f}s, found {len(markets)} markets")
        return json.dumps({"success": True, "markets": markets, "platform": "Polymarket"}, default=str)
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"HermesTool[search_polymarket]: Error after {elapsed:.2f}s: {e}")
        return json.dumps({"error": str(e), "markets": []})

registry.register(
    name="search_polymarket",
    toolset="crypto-prediction",
    schema=SEARCH_POLYMARKET_SCHEMA,
    handler=_search_polymarket_handler,
    is_async=True,
    description="Search active prediction markets on Polymarket",
    emoji="",
)
