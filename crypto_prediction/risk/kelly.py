from typing import Dict

class KellyCalculator:
    @staticmethod
    def calculate(
        market_prob: float,
        model_prob: float,
        multiplier: float = 0.5  # Half-Kelly for risk mitigation
    ) -> dict:
        """
        Calculate Kelly Criterion for binary prediction markets.
        If model_prob > market_prob, we go YES/UP.
        If model_prob < market_prob, we go NO/DOWN.
        """
        # Ensure bounds
        market_prob = max(0.01, min(0.99, market_prob))
        model_prob = max(0.01, min(0.99, model_prob))

        # Check direction
        if model_prob > market_prob:
            # YES/UP bet
            edge = model_prob - market_prob
            odds = (1.0 - market_prob) / market_prob
            # kelly_fraction = (p * b - q) / b = (model_prob * odds - (1 - model_prob)) / odds
            kelly_fraction = edge / (1.0 - market_prob)
            recommended_direction = "YES"
        elif model_prob < market_prob:
            # NO/DOWN bet
            edge = market_prob - model_prob
            odds = market_prob / (1.0 - market_prob)
            # kelly_fraction for NO outcome (market_prob of NO is 1 - market_prob, model_prob is 1 - model_prob)
            kelly_fraction = edge / market_prob
            recommended_direction = "NO"
        else:
            edge = 0.0
            kelly_fraction = 0.0
            recommended_direction = "NONE"

        # Apply multiplier (Half-Kelly or similar) and bound to [0, 1]
        recommended_size = max(0.0, kelly_fraction * multiplier)
        
        # Determine risk level
        if recommended_size == 0:
            risk_level = "NONE"
        elif recommended_size < 0.05:
            risk_level = "LOW"
        elif recommended_size < 0.15:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return {
            "edge": round(edge, 4),
            "kelly_fraction": round(kelly_fraction, 4),
            "recommended_position_size": round(recommended_size, 4),
            "risk_level": risk_level,
            "recommended_direction": recommended_direction
        }
