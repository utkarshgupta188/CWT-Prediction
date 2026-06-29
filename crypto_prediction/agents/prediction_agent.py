import pandas as pd
from crypto_prediction.prediction.kronos_service import predict_next_movement
from crypto_prediction.utils.logger import setup_logger
from crypto_prediction.schemas.config import settings

logger = setup_logger(settings.LOG_LEVEL)

class PredictionAgent:
    async def execute(self, df: pd.DataFrame, pred_len: int = 5) -> dict:
        """
        Run Kronos prediction on the input candles.
        Returns:
            dict containing prediction direction, confidence, and model probability.
        """
        logger.info("PredictionAgent: Running Kronos forecasting pipeline...")
        result = await predict_next_movement(df, pred_len=pred_len)
        logger.info(f"PredictionAgent: Kronos predicted {result['direction']} with confidence {result['confidence']}.")
        return result
