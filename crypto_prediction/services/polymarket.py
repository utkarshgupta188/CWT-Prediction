import httpx
from typing import List, Dict
from crypto_prediction.utils.helpers import request_with_retry
from crypto_prediction.utils.logger import setup_logger
from crypto_prediction.schemas.config import settings

logger = setup_logger(settings.LOG_LEVEL)

class PolymarketService:
    def __init__(self, base_url: str = "https://gamma-api.polymarket.com"):
        self.base_url = base_url

    async def get_active_markets(self, limit: int = 20, max_retries: int = 5) -> List[dict]:
        """
        Fetch active markets from Polymarket and normalize them.
        """
        endpoint = f"{self.base_url}/markets"
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
        }
        
        logger.info("Fetching markets from Polymarket...")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await request_with_retry(client, "GET", endpoint, params=params, max_retries=max_retries)
                markets_data = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch from Polymarket: {e}")
            return []

        normalized_markets = []
        for m in markets_data:
            try:
                # Basic validation
                market_id = str(m.get("id") or m.get("conditionId"))
                if not market_id or not m.get("question"):
                    continue
                
                # Determine asset (heuristic based on title/slug/category)
                question_lower = m["question"].lower()
                category_lower = (m.get("category") or "").lower()
                
                asset = "OTHER"
                if "bitcoin" in question_lower or "btc" in question_lower or "bitcoin" in category_lower:
                    asset = "BTC"
                elif "ethereum" in question_lower or "eth" in question_lower or "ethereum" in category_lower:
                    asset = "ETH"
                elif "solana" in question_lower or "sol" in question_lower or "solana" in category_lower:
                    asset = "SOL"
                elif "doge" in question_lower:
                    asset = "DOGE"
                elif "cardano" in question_lower or "ada" in question_lower:
                    asset = "ADA"
                elif "ripple" in question_lower or "xrp" in question_lower:
                    asset = "XRP"

                # Get market probability (outcomePrices contains prices of Yes and No, e.g. ["0.65", "0.35"])
                # We default to the first outcome price (Yes)
                outcome_prices = m.get("outcomePrices")
                market_prob = 0.5
                if outcome_prices and len(outcome_prices) > 0:
                    try:
                        market_prob = float(outcome_prices[0])
                    except (ValueError, TypeError):
                        pass

                normalized_markets.append({
                    "asset": asset,
                    "platform": "Polymarket",
                    "question": m["question"],
                    "market_probability": market_prob,
                    "expiration": m.get("endDate"),
                    "market_id": market_id
                })
            except Exception as ex:
                logger.warning(f"Error normalizing Polymarket market: {ex}")
                continue

        logger.info(f"Retrieved {len(normalized_markets)} normalized markets from Polymarket.")
        return normalized_markets
