import json
import time
from loguru import logger
from tools.registry import registry

RISK_SCHEMA = {
    "name": "calculate_risk",
    "description": "Calculate Kelly Criterion optimal position size given market and model probabilities",
    "parameters": {
        "type": "object",
        "properties": {
            "market_probability": {
                "type": "number",
                "description": "Market-implied probability (from prediction market)"
            },
            "model_probability": {
                "type": "number",
                "description": "Model-calculated probability (from Kronos)"
            }
        },
        "required": ["market_probability", "model_probability"]
    }
}

def _risk_handler(args):
    market_prob = float(args.get("market_probability", 0.5))
    model_prob = float(args.get("model_probability", 0.5))
    logger.info(f"HermesTool[calculate_risk]: Started (market={market_prob:.4f}, model={model_prob:.4f})")
    start = time.time()
    try:
        from crypto_prediction.risk.kelly import KellyCalculator
        result = KellyCalculator.calculate(market_prob, model_prob)
        elapsed = time.time() - start
        logger.info(f"HermesTool[calculate_risk]: Finished in {elapsed:.3f}s -> {result['recommended_direction']} ({result['recommended_position_size']:.4f})")
        return json.dumps(result, default=str)
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"HermesTool[calculate_risk]: Error after {elapsed:.3f}s: {e}")
        return json.dumps({"error": str(e)})

registry.register(
    name="calculate_risk",
    toolset="crypto_risk",
    schema=RISK_SCHEMA,
    handler=_risk_handler,
    is_async=False,
    description="Calculate Kelly Criterion optimal position size",
    emoji="",
)
