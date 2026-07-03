from dataclasses import dataclass

@dataclass
class ForecastResult:
    direction: str          # "up" | "down"
    probability: float      # 0.0 - 1.0 (confidence)
    predicted_close: float  # Predicted close price
    current_close: float    # Current close price
    horizon_bars: int       # Number of bars predicted ahead
    sample_count: int       # Monte Carlo samples used
