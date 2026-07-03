import logging
import httpx
import pandas as pd
from datetime import datetime, timezone
from .base import DataFetcher

logger = logging.getLogger("cwt_prediction.data_fetcher.coingecko")

class CoinGeckoFetcher(DataFetcher):
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.base_url = "https://api.coingecko.com/api/v3/coins"

    def _normalize_id(self, asset: str) -> str:
        asset_upper = asset.upper().strip()
        if "BTC" in asset_upper or "BITCOIN" in asset_upper:
            return "bitcoin"
        if "ETH" in asset_upper or "ETHEREUM" in asset_upper:
            return "ethereum"
        return asset.lower()

    async def fetch_history(self, asset: str, limit: int = 400) -> pd.DataFrame:
        coin_id = self._normalize_id(asset)
        # CoinGecko OHLC public endpoint: /coins/{id}/ohlc?vs_currency=usd&days=1
        url = f"{self.base_url}/{coin_id}/ohlc"
        params = {
            "vs_currency": "usd",
            "days": "1"  # Returns 30-min interval data for the last 1 day
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    return self._parse_ohlc(data, limit)
                else:
                    logger.error(
                        f"CoinGecko API returned status code {response.status_code} for coin {coin_id}"
                    )
                    raise httpx.HTTPStatusError(
                        f"Status {response.status_code}", request=response.request, response=response
                    )
        except Exception as e:
            logger.error(f"Error fetching history from CoinGecko: {e}", exc_info=True)
            raise

    def _parse_ohlc(self, ohlc_data: list, limit: int) -> pd.DataFrame:
        # CoinGecko item format: [timestamp_ms, open, high, low, close]
        # CoinGecko OHLC does not provide volume or amount, so we fill with 0
        rows = []
        for bar in ohlc_data:
            timestamp_ms = bar[0]
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
            rows.append({
                "timestamps": timestamp,
                "open": float(bar[1]),
                "high": float(bar[2]),
                "low": float(bar[3]),
                "close": float(bar[4]),
                "volume": 0.0,
                "amount": 0.0
            })

        df = pd.DataFrame(rows)
        # Sort by timestamp
        df = df.sort_values("timestamps").reset_index(drop=True)
        # Apply limit to match requested length (from the end)
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)
        
        df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
        return df
