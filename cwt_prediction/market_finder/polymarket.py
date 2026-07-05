import logging
import httpx
from datetime import datetime, timezone
from typing import List, Dict, Any
from .base import MarketFinder
from .models import PredictionMarket

logger = logging.getLogger("cwt_prediction.market_finder.polymarket")

class PolymarketFinder(MarketFinder):
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.api_url = "https://gamma-api.polymarket.com/markets"

    async def find_markets(self) -> List[PredictionMarket]:
        markets = []
        try:
            # We query open, active markets with a search filter
            # Let's search for both Bitcoin and Ethereum markets
            for query in ["Bitcoin", "Ethereum"]:
                params = {
                    "closed": "false",
                    "active": "true",
                    "limit": "10",
                    "search": query
                }
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "application/json"
                }
                with httpx.Client(timeout=self.timeout, headers=headers, trust_env=True) as client:
                    response = client.get(self.api_url, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        markets.extend(self._parse_markets(data, query))
                    else:
                        logger.warning(
                            f"Polymarket API returned status code {response.status_code} for search: {query}"
                        )
        except Exception as e:
            logger.error(f"Error querying Polymarket API: {e}", exc_info=True)
            # Graceful degradation: Return empty list if API fails, no crash
            return []

        return markets

    def _parse_markets(self, data: List[Dict[str, Any]], asset_search: str) -> List[PredictionMarket]:
        parsed = []
        asset = "BTC" if "bitcoin" in asset_search.lower() else "ETH"

        for item in data:
            try:
                question = item.get("question", "")
                # Filter for price direction questions (e.g., "Will Bitcoin go above...", "Will Ethereum price...")
                # E.g., we look for short-term prediction markets.
                # In 5-min or daily markets, the question usually mentions a price level or a specific time.
                question_lower = question.lower()
                is_crypto_price_market = (
                    "above" in question_lower or 
                    "price" in question_lower or 
                    "reaches" in question_lower or
                    "goes" in question_lower or
                    "high" in question_lower
                )
                if not is_crypto_price_market:
                    continue

                market_id = item.get("id")
                if not market_id:
                    continue

                # Parse outcome prices (YES token price is index 0)
                outcome_prices = item.get("outcomePrices")
                if not outcome_prices or len(outcome_prices) < 1:
                    continue
                
                try:
                    implied_prob_yes = float(outcome_prices[0])
                except (ValueError, TypeError):
                    continue

                # Parse expiry (defaults to a timestamp far in the future if missing)
                end_date_str = item.get("endDate")
                if end_date_str:
                    try:
                        # Polymarket standard ISO format, strip Z
                        end_date_str = end_date_str.replace("Z", "+00:00")
                        expiry = datetime.fromisoformat(end_date_str)
                    except ValueError:
                        expiry = datetime.now(timezone.utc)
                else:
                    expiry = datetime.now(timezone.utc)

                url = f"https://polymarket.com/market/{item.get('slug', '')}"

                parsed.append(PredictionMarket(
                    id=str(market_id),
                    platform="polymarket",
                    asset=asset,
                    question=question,
                    expiry=expiry,
                    implied_prob_yes=implied_prob_yes,
                    url=url,
                    ticker=item.get("ticker")
                ))
            except Exception as e:
                logger.warning(f"Error parsing Polymarket market item: {e}")
                continue

        return parsed
