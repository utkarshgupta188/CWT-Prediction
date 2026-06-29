from typing import Dict
from crypto_prediction.database.repository import PredictionRepository
from crypto_prediction.utils.logger import setup_logger
from crypto_prediction.schemas.config import settings

logger = setup_logger(settings.LOG_LEVEL)

class FeedbackAgent:
    async def execute(self, repo: PredictionRepository, prediction_id: int, actual_movement: str) -> dict:
        """
        Evaluate prediction correctness, store feedback and update aggregate statistics.
        """
        logger.info(f"FeedbackAgent: Evaluating prediction ID {prediction_id} against actual movement '{actual_movement}'...")
        
        # Retrieve prediction
        prediction = await repo.get_prediction_by_id(prediction_id)
        if not prediction:
            raise ValueError(f"Prediction with ID {prediction_id} not found in database.")
            
        predicted_direction = prediction.prediction_direction
        correct = (predicted_direction == actual_movement)
        
        # Save feedback
        feedback_data = {
            "prediction_id": prediction_id,
            "actual_movement": actual_movement,
            "correct": correct
        }
        feedback = await repo.save_feedback(feedback_data)
        
        # Recalculate statistics
        all_feedbacks = await repo.get_feedbacks()
        total_predictions = len(all_feedbacks)
        correct_predictions = sum(1 for fb in all_feedbacks if fb.correct)
        accuracy = float(correct_predictions) / total_predictions if total_predictions > 0 else 0.0
        
        # Update running statistics in DB
        await repo.update_statistic("total_predictions", float(total_predictions))
        await repo.update_statistic("correct_predictions", float(correct_predictions))
        await repo.update_statistic("accuracy", accuracy)
        
        logger.info(f"FeedbackAgent: Prediction {prediction_id} was {'CORRECT' if correct else 'INCORRECT'}. Total count: {total_predictions}, Running accuracy: {accuracy:.2%}.")
        
        return {
            "prediction_id": prediction_id,
            "predicted_direction": predicted_direction,
            "actual_movement": actual_movement,
            "correct": correct,
            "total_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy
        }
