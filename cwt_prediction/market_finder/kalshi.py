import logging
import httpx
from datetime import datetime, timezone
from typing import List, Dict, Any
from .base import MarketFinder
from .models import PredictionMarket

logger = logging.getLogger("cwt_prediction.market_finder.kalshi")

class KalshiFinder(MarketFinder):
    def __init__(self, timeout: float = 10.0, use_demo: bool = False):
        self.timeout = timeout
        # Using public Kalshi API
        domain = "demo-api.kalshi.co" if use_demo else "api.kalshi.com"
        self.api_url = f"https://{domain}/trade-api/v2/markets"

    async def find_markets(self) -> List[PredictionMarket]:
        markets = []
        try:
            # Kalshi API allows querying markets by ticker series
            # E.g. BTC or ETH
            for ticker in ["BTC", "ETH"]:
                params = {
                    "limit": "20",
                    "status": "open",
                    "series_ticker": ticker
                }
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(self.api_url, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        markets_data = data.get("markets", [])
                        markets.extend(self._parse_markets(markets_data, ticker))
                    else:
                        logger.warning(
                            f"Kalshi API returned status code {response.status_code} for ticker: {ticker}"
                        )
        except Exception as e:
            logger.error(f"Error querying Kalshi API: {e}", exc_info=True)
            # Return empty list gracefully
            return []

        return markets

    def _parse_markets(self, data: List[Dict[str, Any]], asset: str) -> List[PredictionMarket]:
        parsed = []
        for item in data:
            try:
                question = item.get("title", "")
                market_ticker = item.get("ticker")
                if not market_ticker:
                    continue

                # In Kalshi, prices are represented in cents, e.g. yes_ask=55 means $0.55 or 55% prob.
                # We can calculate probability as the average of yes_bid and yes_ask if available,
                # or default to yes_ask, or fallback to yes_price.
                yes_ask = item.get("yes_ask")
                yes_bid = item.get("yes_bid")
                
                if yes_ask is not None and yes_bid is not None:
                    implied_prob_yes = ((yes_ask + yes_bid) / 2.0) / 100.0
                elif yes_ask is not None:
                    implied_prob_yes = yes_ask / 100.0
                elif yes_bid is not None:
                    implied_prob_yes = yes_bid / 100.0
                else:
                    # Sometimes last_price is available
                    last_price = item.get("last_price")
                    if last_price is not None:
                        implied_prob_yes = last_price / 100.0
                    else:
                        # Skip if price data is completely missing
                        continue

                # Ensure prob is bounded 0-1
                implied_prob_yes = max(0.0, min(1.0, implied_prob_yes))

                # Parse expiry date
                close_time_str = item.get("close_time")
                if close_time_str:
                    try:
                        # Kalshi ISO format
                        close_time_str = close_time_str.replace("Z", "+00:00")
                        expiry = datetime.fromisoformat(close_time_str)
                    except ValueError:
                        expiry = datetime.now(timezone.utc)
                else:
                    expiry = datetime.now(timezone.utc)

                url = f"https://kalshi.com/markets/{market_ticker.lower()}"

                parsed.append(PredictionMarket(
                    id=market_ticker,
                    platform="kalshi",
                    asset=asset,
                    question=question,
                    expiry=expiry,
                    implied_prob_yes=implied_prob_yes,
                    url=url,
                    ticker=market_ticker
                ))
            except Exception as e:
                logger.warning(f"Error parsing Kalshi market item: {e}")
                continue

        return parsed
