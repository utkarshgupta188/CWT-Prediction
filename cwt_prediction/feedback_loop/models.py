from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class PredictionRecord:
    id: Optional[int]
    timestamp: datetime
    asset: str              # "BTC" | "ETH"
    market_id: str
    platform: str           # "polymarket" | "kalshi"
    direction_predicted: str # "up" | "down"
    model_prob: float
    market_implied_prob: float
    kelly_fraction: float
    expiry: datetime
    resolved: bool = False
    actual_direction: Optional[str] = None
    pnl_if_bet: Optional[float] = None
    resolve_timestamp: Optional[datetime] = None
