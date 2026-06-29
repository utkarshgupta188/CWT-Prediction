import json
import time
from loguru import logger
from tools.registry import registry

FEEDBACK_SCHEMA = {
    "name": "save_feedback",
    "description": "Save prediction feedback with actual movement outcome and update accuracy statistics",
    "parameters": {
        "type": "object",
        "properties": {
            "prediction_id": {
                "type": "integer",
                "description": "ID of the prediction to provide feedback for"
            },
            "actual_movement": {
                "type": "string",
                "enum": ["UP", "DOWN"],
                "description": "Actual price movement that occurred"
            }
        },
        "required": ["prediction_id", "actual_movement"]
    }
}

async def _feedback_handler(args):
    prediction_id = int(args.get("prediction_id", 0))
    actual_movement = args.get("actual_movement", "").upper()
    logger.info(f"HermesTool[save_feedback]: Started for prediction {prediction_id} -> {actual_movement}")
    start = time.time()
    try:
        from crypto_prediction.database.repository import AsyncSessionLocal, PredictionRepository
        async with AsyncSessionLocal() as session:
            repo = PredictionRepository(session)
            prediction = await repo.get_prediction_by_id(prediction_id)
            if not prediction:
                return json.dumps({"error": f"Prediction ID {prediction_id} not found"})
            correct = (prediction.prediction_direction == actual_movement)
            feedback_data = {
                "prediction_id": prediction_id,
                "actual_movement": actual_movement,
                "correct": correct
            }
            feedback = await repo.save_feedback(feedback_data)
            all_feedbacks = await repo.get_feedbacks()
            total = len(all_feedbacks)
            correct_count = sum(1 for fb in all_feedbacks if fb.correct)
            accuracy = float(correct_count) / total if total > 0 else 0.0
            await repo.update_statistic("total_predictions", float(total))
            await repo.update_statistic("correct_predictions", float(correct_count))
            await repo.update_statistic("accuracy", accuracy)
            elapsed = time.time() - start
            logger.info(f"HermesTool[save_feedback]: Finished in {elapsed:.2f}s -> {'CORRECT' if correct else 'INCORRECT'}")
            return json.dumps({
                "success": True,
                "prediction_id": prediction_id,
                "predicted_direction": prediction.prediction_direction,
                "actual_movement": actual_movement,
                "correct": correct,
                "correct_predictions": correct_count,
                "total_predictions": total,
                "accuracy": accuracy
            }, default=str)
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"HermesTool[save_feedback]: Error after {elapsed:.2f}s: {e}")
        return json.dumps({"error": str(e)})

registry.register(
    name="save_feedback",
    toolset="hermes-cli",
    schema=FEEDBACK_SCHEMA,
    handler=_feedback_handler,
    is_async=True,
    description="Save prediction feedback and update accuracy statistics",
    emoji="",
)
