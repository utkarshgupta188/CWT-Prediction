import json
import time
from typing import List
import pandas as pd
from loguru import logger
from tools.registry import registry

ALLOWED_TOOLS = ["get_market_data"]

class HermesMarketDataAgent:
    """Hermes Market Data Agent - restricted to Binance market data tool."""

    @property
    def allowed_tools(self) -> List[str]:
        return list(ALLOWED_TOOLS)

    async def execute(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        logger.info(f"HermesMarketDataAgent: Started for {symbol} ({interval}, limit={limit})")
        start = time.time()
        try:
            result_str = registry.dispatch("get_market_data", {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            })
            result = json.loads(result_str)
            if "error" in result:
                raise ValueError(result["error"])
            df = pd.DataFrame(result["data"])
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            elapsed = time.time() - start
            logger.info(f"HermesMarketDataAgent: Finished in {elapsed:.2f}s, got {len(df)} candles")
            return df
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"HermesMarketDataAgent: Error after {elapsed:.2f}s: {e}")
            raise
