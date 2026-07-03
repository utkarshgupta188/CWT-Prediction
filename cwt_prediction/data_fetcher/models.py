from dataclasses import dataclass
from datetime import datetime

@dataclass
class OHLCVBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
