from crypto_prediction.risk.kelly import KellyCalculator
from crypto_prediction.utils.logger import setup_logger
from crypto_prediction.schemas.config import settings

logger = setup_logger(settings.LOG_LEVEL)

class RiskAgent:
    def __init__(self):
        self.calculator = KellyCalculator()

    async def execute(self, market_probability: float, model_probability: float) -> dict:
        """
        Calculate risk and optimal betting sizes using the Kelly Criterion.
        """
        logger.info(f"RiskAgent: Calculating risk metrics (market_prob={market_probability}, model_prob={model_probability})...")
        metrics = self.calculator.calculate(market_probability, model_probability)
        logger.info(f"RiskAgent: Recommendation = {metrics['recommended_direction']}, Fraction = {metrics['recommended_position_size']}, Risk Level = {metrics['risk_level']}.")
        return metrics
