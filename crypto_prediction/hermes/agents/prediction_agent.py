import json
import time
from typing import List
import pandas as pd
from loguru import logger
from tools.registry import registry

ALLOWED_TOOLS = ["get_prediction"]

class HermesPredictionAgent:
    """Hermes Prediction Agent - restricted to Kronos prediction tool."""

    @property
    def allowed_tools(self) -> List[str]:
        return list(ALLOWED_TOOLS)

    async def execute(self, df: pd.DataFrame, pred_len: int = 5) -> dict:
        logger.info(f"HermesPredictionAgent: Started (candles={len(df)}, pred_len={pred_len})")
        start = time.time()
        try:
            candles_json = df.to_dict(orient="records")
            result_str = registry.dispatch("get_prediction", {
                "candles": json.dumps(candles_json, default=str),
                "pred_len": pred_len
            })
            result = json.loads(result_str)
            if "error" in result:
                raise ValueError(result["error"])
            elapsed = time.time() - start
            logger.info(f"HermesPredictionAgent: Finished in {elapsed:.2f}s -> {result['direction']} ({result['confidence']:.4f})")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"HermesPredictionAgent: Error after {elapsed:.2f}s: {e}")
            raise
