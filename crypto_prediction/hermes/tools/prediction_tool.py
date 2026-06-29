import json
import time
import pandas as pd
from loguru import logger
from tools.registry import registry

PREDICTION_SCHEMA = {
    "name": "get_prediction",
    "description": "Run Kronos model prediction for a given symbol and interval to forecast next movement",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Trading pair symbol e.g. BTCUSDT"
            },
            "candles": {
                "type": "string",
                "description": "JSON string of OHLCV candle data (optional, fallback if symbol not provided)"
            },
            "interval": {
                "type": "string",
                "description": "Candle interval e.g. 5m, 1h, 1d",
                "default": "5m"
            },
            "limit": {
                "type": "integer",
                "description": "Number of candles to fetch (max 1000)",
                "default": 1000
            },
            "pred_len": {
                "type": "integer",
                "description": "Number of future candles to predict",
                "default": 5
            }
        },
        "required": ["candles", "symbol"]
    }
}

async def _prediction_handler(args, **kwargs):
    symbol = args.get("symbol", "").upper()
    interval = args.get("interval", "5m")
    limit = min(int(args.get("limit", 1000)), 1000)
    pred_len = int(args.get("pred_len", 5))
    logger.info(f"HermesTool[get_prediction]: Started for {symbol} ({interval})")
    start = time.time()
    try:
        candles_json = args.get("candles")
        if candles_json and candles_json.strip() not in ["", "[]", "fetch"]:
            candles = json.loads(candles_json)
            df = pd.DataFrame(candles)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            from crypto_prediction.providers.binance_provider import BinanceProvider
            provider = BinanceProvider()
            df = await provider.get_klines(symbol, interval, limit)
            if df.empty:
                raise ValueError(f"No OHLCV data returned for symbol {symbol}")
        from crypto_prediction.prediction.kronos_service import predict_next_movement
        result = await predict_next_movement(df, pred_len=pred_len)
        elapsed = time.time() - start
        logger.info(f"HermesTool[get_prediction]: Finished in {elapsed:.2f}s -> {result['direction']}")
        return json.dumps(result, default=str)
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"HermesTool[get_prediction]: Error after {elapsed:.2f}s: {e}")
        return json.dumps({"error": str(e)})

registry.register(
    name="get_prediction",
    toolset="crypto-prediction",
    schema=PREDICTION_SCHEMA,
    handler=_prediction_handler,
    is_async=True,
    description="Run Kronos model prediction on OHLCV price data",
    emoji="",
)
