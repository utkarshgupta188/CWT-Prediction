import logging
import httpx
import pandas as pd
from datetime import datetime, timezone
from .base import DataFetcher

logger = logging.getLogger("cwt_prediction.data_fetcher.binance")

class BinanceFetcher(DataFetcher):
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.base_url = "https://api.binance.com/api/v3/klines"

    def _normalize_symbol(self, asset: str) -> str:
        asset_upper = asset.upper().strip()
        if asset_upper in ["BTC", "ETH"]:
            return f"{asset_upper}USDT"
        if not asset_upper.endswith("USDT") and not asset_upper.endswith("USD"):
            return f"{asset_upper}USDT"
        return asset_upper

    async def fetch_history(self, asset: str, limit: int = 400, start_time: datetime = None, end_time: datetime = None) -> pd.DataFrame:
        symbol = self._normalize_symbol(asset)
        params = {
            "symbol": symbol,
            "interval": "5m",
            "limit": str(limit)
        }
        if start_time:
            # Convert to milliseconds timestamp
            params["startTime"] = str(int(start_time.timestamp() * 1000))
        if end_time:
            params["endTime"] = str(int(end_time.timestamp() * 1000))
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.base_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    return self._parse_klines(data)
                else:
                    logger.error(
                        f"Binance API returned status code {response.status_code} for symbol {symbol}"
                    )
                    raise httpx.HTTPStatusError(
                        f"Status {response.status_code}", request=response.request, response=response
                    )
        except Exception as e:
            logger.error(f"Error fetching history from Binance: {e}", exc_info=True)
            raise

    def _parse_klines(self, klines_data: list) -> pd.DataFrame:
        # Binance kline columns:
        # 0: Open time (ms)
        # 1: Open
        # 2: High
        # 3: Low
        # 4: Close
        # 5: Volume
        # 6: Close time
        # 7: Quote asset volume (this maps to 'amount' in Chinese/some datasets, but we can also use it)
        # We parse these into floats
        rows = []
        for bar in klines_data:
            open_time_ms = bar[0]
            # Convert ms timestamp to datetime
            timestamp = datetime.fromtimestamp(open_time_ms / 1000.0, tz=timezone.utc)
            rows.append({
                "timestamps": timestamp,
                "open": float(bar[1]),
                "high": float(bar[2]),
                "low": float(bar[3]),
                "close": float(bar[4]),
                "volume": float(bar[5]),
                "amount": float(bar[7]) # Quote asset volume
            })

        df = pd.DataFrame(rows)
        # Ensure correct column order and types
        df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
        return df
