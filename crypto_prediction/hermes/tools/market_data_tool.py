import json
import time
from loguru import logger
from tools.registry import registry

MARKET_DATA_SCHEMA = {
    "name": "get_market_data",
    "description": "Fetch OHLCV candle data from Binance for a given symbol and interval",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Trading pair symbol e.g. BTCUSDT"
            },
            "interval": {
                "type": "string",
                "description": "Candle interval e.g. 5m, 1h, 1d"
            },
            "limit": {
                "type": "integer",
                "description": "Number of candles to fetch (max 1000)",
                "default": 1000
            }
        },
        "required": ["symbol", "interval"]
    }
}

async def _market_data_handler(args, **kwargs):
    symbol = args.get("symbol", "").upper()
    interval = args.get("interval", "5m")
    limit = min(int(args.get("limit", 1000)), 1000)
    logger.info(f"HermesTool[get_market_data]: Started for {symbol} ({interval}, limit={limit})")
    start = time.time()
    try:
        from crypto_prediction.providers.binance_provider import BinanceProvider
        provider = BinanceProvider()
        df = await provider.get_klines(symbol, interval, limit)
        elapsed = time.time() - start
        logger.info(f"HermesTool[get_market_data]: Finished in {elapsed:.2f}s, got {len(df)} candles")
        
        # Save tokens by only returning the last 5 candles in detail, and a summary
        latest_candles = df.tail(5).to_dict(orient="records")
        return json.dumps({
            "success": True,
            "message": f"Successfully fetched {len(df)} candles for {symbol} ({interval}). Showing latest 5 candles.",
            "latest_candles": latest_candles,
            "columns": list(df.columns),
            "shape": list(df.shape),
            "symbol": symbol,
            "interval": interval
        }, default=str)
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"HermesTool[get_market_data]: Error after {elapsed:.2f}s: {e}")
        return json.dumps({"error": str(e)})

registry.register(
    name="get_market_data",
    toolset="crypto-prediction",
    schema=MARKET_DATA_SCHEMA,
    handler=_market_data_handler,
    is_async=True,
    description="Fetch OHLCV candle data from Binance for a given symbol and interval",
    emoji="",
)
