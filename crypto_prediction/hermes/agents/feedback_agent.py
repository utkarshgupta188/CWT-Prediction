import json
import time
from typing import List
from loguru import logger
from tools.registry import registry

ALLOWED_TOOLS = ["save_feedback"]

class HermesFeedbackAgent:
    """Hermes Feedback Agent - restricted to feedback/save tool."""

    @property
    def allowed_tools(self) -> List[str]:
        return list(ALLOWED_TOOLS)

    async def execute(self, repo, prediction_id: int, actual_movement: str) -> dict:
        logger.info(f"HermesFeedbackAgent: Started for prediction {prediction_id} -> {actual_movement}")
        start = time.time()
        try:
            feedback_str = registry.dispatch("save_feedback", {
                "prediction_id": prediction_id,
                "actual_movement": actual_movement
            })
            feedback_result = json.loads(feedback_str)
            if "error" in feedback_result:
                raise ValueError(feedback_result["error"])
            elapsed = time.time() - start
            logger.info(f"HermesFeedbackAgent: Finished in {elapsed:.2f}s -> "
                        f"{'CORRECT' if feedback_result.get('correct') else 'INCORRECT'}")
            return {
                "prediction_id": prediction_id,
                "predicted_direction": feedback_result.get("predicted_direction",
                    "UP" if feedback_result.get("correct") else "DOWN"),
                "actual_movement": actual_movement,
                "correct": feedback_result["correct"],
                "total_predictions": feedback_result["total_predictions"],
                "correct_predictions": feedback_result.get("correct_predictions", 0),
                "accuracy": feedback_result["accuracy"]
            }
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"HermesFeedbackAgent: Error after {elapsed:.2f}s: {e}")
            raise
