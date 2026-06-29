import json
import time
import pandas as pd
from loguru import logger
from tools.registry import registry

PREDICTION_SCHEMA = {
    "name": "get_prediction",
    "description": "Run Kronos model prediction on OHLCV price data to forecast next movement",
    "parameters": {
        "type": "object",
        "properties": {
            "candles": {
                "type": "string",
                "description": "JSON string of OHLCV candle data with columns: timestamp, open, high, low, close, volume"
            },
            "pred_len": {
                "type": "integer",
                "description": "Number of future candles to predict",
                "default": 5
            }
        },
        "required": ["candles"]
    }
}

async def _prediction_handler(args):
    candles_json = args.get("candles", "[]")
    pred_len = int(args.get("pred_len", 5))
    logger.info("HermesTool[get_prediction]: Started")
    start = time.time()
    try:
        candles = json.loads(candles_json)
        df = pd.DataFrame(candles)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
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
    toolset="crypto_prediction",
    schema=PREDICTION_SCHEMA,
    handler=_prediction_handler,
    is_async=True,
    description="Run Kronos model prediction on OHLCV price data",
    emoji="",
)
