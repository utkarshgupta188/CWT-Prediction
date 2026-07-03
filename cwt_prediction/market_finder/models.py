from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class PredictionMarket:
    id: str
    platform: str           # "polymarket" | "kalshi"
    asset: str              # "BTC" | "ETH"
    question: str           # Human-readable market question
    expiry: datetime
    implied_prob_yes: float # 0.0 - 1.0
    url: str
    ticker: Optional[str] = None # Platform-specific symbol
