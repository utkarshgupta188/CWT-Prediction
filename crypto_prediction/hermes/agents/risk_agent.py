import json
import time
from typing import List
from loguru import logger
from tools.registry import registry

ALLOWED_TOOLS = ["calculate_risk"]

class HermesRiskAgent:
    """Hermes Risk Agent - restricted to Kelly Criterion risk calculation tool."""

    @property
    def allowed_tools(self) -> List[str]:
        return list(ALLOWED_TOOLS)

    async def execute(self, market_probability: float, model_probability: float) -> dict:
        logger.info(f"HermesRiskAgent: Started (market={market_probability:.4f}, model={model_probability:.4f})")
        start = time.time()
        try:
            result_str = registry.dispatch("calculate_risk", {
                "market_probability": market_probability,
                "model_probability": model_probability
            })
            result = json.loads(result_str)
            if "error" in result:
                raise ValueError(result["error"])
            elapsed = time.time() - start
            logger.info(f"HermesRiskAgent: Finished in {elapsed:.3f}s -> {result['recommended_direction']} ({result['recommended_position_size']:.4f})")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"HermesRiskAgent: Error after {elapsed:.3f}s: {e}")
            raise
