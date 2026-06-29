import json
from datetime import datetime
from typing import List, Optional
from loguru import logger


class HermesMemory:
    """Stores previous prediction context for Hermes reasoning.
    
    Stores: predictions, confidence, market probability, Kelly sizing,
    market disagreement, historical accuracy.
    Does NOT store raw OHLCV data.
    """

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self._predictions: List[dict] = []

    def add_prediction(
        self,
        symbol: str,
        interval: str,
        prediction_direction: str,
        confidence: float,
        model_probability: float,
        market_probability: float,
        kelly_fraction: float,
        reasoning: str,
        accuracy: Optional[float] = None,
    ):
        market_disagreement = abs(model_probability - market_probability)
        entry = {
            "symbol": symbol,
            "interval": interval,
            "prediction_direction": prediction_direction,
            "confidence": confidence,
            "model_probability": model_probability,
            "market_probability": market_probability,
            "kelly_fraction": kelly_fraction,
            "market_disagreement": round(market_disagreement, 4),
            "reasoning": reasoning,
            "accuracy": accuracy,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._predictions.append(entry)
        if len(self._predictions) > self.max_history:
            self._predictions.pop(0)
        logger.debug(f"HermesMemory: Stored prediction for {symbol} ({interval})")

    def get_recent_context(self, limit: int = 5) -> str:
        """Return a formatted string of recent predictions for LLM context."""
        recent = self._predictions[-limit:]
        if not recent:
            return "No previous predictions available."
        lines = ["Recent prediction history:"]
        for p in recent:
            acc = f"{p['accuracy']:.2%}" if p['accuracy'] is not None else "pending"
            lines.append(
                f"- {p['symbol']} ({p['interval']}): {p['prediction_direction']} "
                f"(confidence={p['confidence']:.2%}, market_prob={p['market_probability']:.2%}, "
                f"kelly={p['kelly_fraction']:.4f}, disagreement={p['market_disagreement']:.4f}, "
                f"accuracy={acc})"
            )
        return "\n".join(lines)

    def get_all(self) -> List[dict]:
        return list(self._predictions)

    def clear(self):
        self._predictions.clear()

    def to_json(self) -> str:
        return json.dumps(self._predictions, default=str)

    @classmethod
    def from_json(cls, data: str):
        mem = cls()
        entries = json.loads(data)
        mem._predictions = entries[-mem.max_history:]
        return mem
