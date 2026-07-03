import httpx
from typing import List, Dict
import datetime
from crypto_prediction.utils.helpers import request_with_retry
from crypto_prediction.utils.logger import setup_logger
from crypto_prediction.schemas.config import settings

logger = setup_logger(settings.LOG_LEVEL)

# Known crypto tickers/keywords for client-side filtering
CRYPTO_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "DOGE": ["doge", "dogecoin"],
    "ADA": ["cardano", "ada"],
    "XRP": ["ripple", "xrp"],
}


class KalshiService:
    def __init__(self, base_url: str = "https://api.elections.kalshi.com/trade-api/v2"):
        self.base_url = base_url

    async def get_active_markets(self, query: str = None, limit: int = 20, max_retries: int = 2) -> List[dict]:
        """
        Fetch open markets from Kalshi and normalize them.
        Uses client-side filtering since Kalshi's API doesn't support text search.
        """
        endpoint = f"{self.base_url}/markets"
        params = {
            "status": "open",
            "limit": min(limit * 10, 200),  # Fetch more to filter client-side
        }

        logger.info(f"Fetching markets from Kalshi (query={query})...")
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await request_with_retry(
                    client, "GET", endpoint, params=params, max_retries=max_retries
                )
                data = response.json()
                markets_data = data.get("markets", [])
        except Exception as e:
            logger.warning(f"Failed to fetch from Kalshi: {e}")
            return []

        # Build query keywords for filtering
        query_keywords = []
        if query:
            query_lower = query.lower().strip()
            # Check if query maps to a known crypto asset
            for asset_code, keywords in CRYPTO_KEYWORDS.items():
                if query_lower in keywords or query_lower == asset_code.lower():
                    query_keywords = keywords
                    break
            if not query_keywords:
                query_keywords = [query_lower]

        normalized_markets = []
        for m in markets_data:
            try:
                ticker = m.get("ticker")
                title = m.get("title") or m.get("subtitle") or ""
                if not ticker or not title:
                    continue

                # Determine asset (heuristic based on title/ticker)
                title_lower = title.lower()
                ticker_lower = ticker.lower()

                asset = "OTHER"
                for asset_code, keywords in CRYPTO_KEYWORDS.items():
                    if any(kw in title_lower or kw in ticker_lower for kw in keywords):
                        asset = asset_code
                        break

                # Client-side query filtering: skip non-matching markets
                if query_keywords:
                    text_to_search = f"{title_lower} {ticker_lower}"
                    if not any(kw in text_to_search for kw in query_keywords):
                        continue

                # Get market probability
                yes_bid = m.get("yes_bid") or m.get("last_price")
                floor = m.get("floor_price", 0)
                cap = m.get("cap_price", 100)

                market_prob = 0.5
                if yes_bid is not None and (cap - floor) > 0:
                    try:
                        market_prob = float(yes_bid - floor) / float(cap - floor)
                    except (ValueError, TypeError):
                        pass

                expiration = None
                exp_ts = m.get("expiration_ts")
                if exp_ts:
                    try:
                        expiration = datetime.datetime.utcfromtimestamp(exp_ts).isoformat() + "Z"
                    except Exception:
                        pass

                normalized_markets.append({
                    "asset": asset,
                    "platform": "Kalshi",
                    "question": title,
                    "market_probability": market_prob,
                    "expiration": expiration,
                    "market_id": ticker
                })

                if len(normalized_markets) >= limit:
                    break
            except Exception as ex:
                logger.warning(f"Error normalizing Kalshi market: {ex}")
                continue

        logger.info(f"Retrieved {len(normalized_markets)} normalized markets from Kalshi.")
        return normalized_markets
