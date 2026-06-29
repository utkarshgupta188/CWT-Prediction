import httpx
from typing import List, Dict
import datetime
from crypto_prediction.utils.helpers import request_with_retry
from crypto_prediction.utils.logger import setup_logger
from crypto_prediction.schemas.config import settings

logger = setup_logger(settings.LOG_LEVEL)

class KalshiService:
    def __init__(self, base_url: str = "https://external-api.kalshi.com/trade-api/v2"):
        self.base_url = base_url

    async def get_active_markets(self, limit: int = 20) -> List[dict]:
        """
        Fetch open markets from Kalshi and normalize them.
        """
        endpoint = f"{self.base_url}/markets"
        params = {
            "status": "open",
            "limit": limit
        }
        
        logger.info("Fetching markets from Kalshi...")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await request_with_retry(client, "GET", endpoint, params=params)
                data = response.json()
                markets_data = data.get("markets", [])
        except Exception as e:
            logger.error(f"Failed to fetch from Kalshi: {e}")
            return []

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
                if "bitcoin" in title_lower or "btc" in title_lower or "btc" in ticker_lower:
                    asset = "BTC"
                elif "ethereum" in title_lower or "eth" in title_lower or "eth" in ticker_lower:
                    asset = "ETH"
                elif "solana" in title_lower or "sol" in title_lower or "sol" in ticker_lower:
                    asset = "SOL"
                elif "doge" in title_lower or "doge" in ticker_lower:
                    asset = "DOGE"

                # Get market probability (yes_ask/no_ask or floor/cap price ranges if available, but Kalshi uses cents for price, e.g. 0 to 100)
                # We can estimate probability as yes_bid or yes_ask / 100 or simply a placeholder if not present.
                # In Kalshi API, yes_price/no_price is standard if we query market details, but /markets response gives tick sizes/cap price.
                # Let's check: floor_price, cap_price, yes_price is cents, so (yes_price - floor_price) / (cap_price - floor_price)
                # If yes_price is missing, let's use recent price or default to 0.5.
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
            except Exception as ex:
                logger.warning(f"Error normalizing Kalshi market: {ex}")
                continue

        logger.info(f"Retrieved {len(normalized_markets)} normalized markets from Kalshi.")
        return normalized_markets
