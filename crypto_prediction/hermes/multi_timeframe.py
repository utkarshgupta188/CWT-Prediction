import asyncio
import time
from loguru import logger
from crypto_prediction.hermes.supervisor import HermesSupervisorAgent

class MultiTimeframeOrchestrator:
    """Orchestrates predictions across multiple timeframes (1m, 5m, 15m).
    
    Provides:
      - Parallel execution of predictions for 1m, 5m, 15m timeframes
      - 1min n+5 -> 5min n+1 cross-timeframe validation: checking if the micro trend
        corroborates the macro trend.
      - Internal arbitrage detection: identifies discrepancies between 15m predictions
        and multiple 5m predictions (e.g. 15m UP but three 5m predictions are DOWN).
    """

    def __init__(self, supervisor: HermesSupervisorAgent = None):
        self.supervisor = supervisor or HermesSupervisorAgent()

    async def execute_multi_timeframe(
        self,
        repo,
        symbol: str,
        limit: int = 1000
    ) -> dict:
        """Run predictions for 1m, 5m, and 15m timeframes in parallel, then perform
        arbitrage detection and cross-validation analysis.
        """
        logger.info(f"MultiTimeframeOrchestrator: Starting multi-timeframe flow for {symbol}")
        start = time.time()

        # Run 1m, 5m, and 15m predictions in parallel
        timeframes = ["1m", "5m", "15m"]
        tasks = [
            self.supervisor.execute_prediction_flow(repo, symbol, tf, limit)
            for tf in timeframes
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        tf_results = {}
        for tf, res in zip(timeframes, results):
            if isinstance(res, Exception):
                logger.error(f"MultiTimeframeOrchestrator: Prediction failed for {tf}: {res}")
                tf_results[tf] = {"error": str(res), "prediction": "NONE", "confidence": 0.0, "market_probability": 0.5, "kelly": 0.0}
            else:
                tf_results[tf] = res

        # ── Cross-Timeframe Analysis & Internal Arbitrage ─────────
        analysis = self._analyze_results(symbol, tf_results)
        
        elapsed = time.time() - start
        logger.info(f"MultiTimeframeOrchestrator: Completed in {elapsed:.2f}s")
        
        return {
            "symbol": symbol,
            "elapsed_seconds": round(elapsed, 2),
            "timeframes": tf_results,
            "analysis": analysis
        }

    def _analyze_results(self, symbol: str, results: dict) -> dict:
        """Analyze prediction alignment and find internal arbitrage opportunities."""
        pred_1m = results["1m"].get("prediction", "NONE")
        pred_5m = results["5m"].get("prediction", "NONE")
        pred_15m = results["15m"].get("prediction", "NONE")
        
        conf_1m = results["1m"].get("confidence", 0.0)
        conf_5m = results["5m"].get("confidence", 0.0)
        conf_15m = results["15m"].get("confidence", 0.0)

        # 1. 1min n+5 -> 5min n+1 Validation
        # 1m prediction for n+5 represents the exact same end time as 5m n+1 prediction.
        # If they match, validation is high.
        micro_validation = "DISAGREEMENT"
        if pred_1m == pred_5m and pred_1m != "NONE":
            micro_validation = "CONFIRMED"
        elif "NONE" in (pred_1m, pred_5m):
            micro_validation = "INCOMPLETE"

        # 2. Internal Arbitrage
        # If 15m is UP, but 5m and 1m are DOWN, we have a short-term trend discrepancy.
        # Or if 15m is DOWN, but 5m and 1m are UP.
        arbitrage_opportunity = False
        arbitrage_type = "NONE"
        arbitrage_reason = ""

        if pred_15m == "UP" and pred_5m == "DOWN":
            arbitrage_opportunity = True
            arbitrage_type = "SHORT_TERM_MEAN_REVERSION_BUY"
            arbitrage_reason = f"Macro (15m) is UP, but micro (5m) shows DOWN. Potential buy-the-dip arbitrage opportunity."
        elif pred_15m == "DOWN" and pred_5m == "UP":
            arbitrage_opportunity = True
            arbitrage_type = "SHORT_TERM_MEAN_REVERSION_SELL"
            arbitrage_reason = f"Macro (15m) is DOWN, but micro (5m) shows UP. Potential sell-the-rally arbitrage opportunity."

        # 3. Consensus Direction
        directions = [d for d in [pred_1m, pred_5m, pred_15m] if d != "NONE"]
        if not directions:
            consensus_direction = "NONE"
            consensus_strength = 0.0
        else:
            up_count = sum(1 for d in directions if d == "UP")
            down_count = sum(1 for d in directions if d == "DOWN")
            
            if up_count > down_count:
                consensus_direction = "UP"
                consensus_strength = up_count / len(directions)
            elif down_count > up_count:
                consensus_direction = "DOWN"
                consensus_strength = down_count / len(directions)
            else:
                consensus_direction = "NEUTRAL"
                consensus_strength = 0.5

        # Weighted composite Kelly recommendation
        kelly_1m = results["1m"].get("kelly", 0.0)
        kelly_5m = results["5m"].get("kelly", 0.0)
        kelly_15m = results["15m"].get("kelly", 0.0)
        # Give higher weight to longer timeframe
        composite_kelly = (kelly_1m * 0.2) + (kelly_5m * 0.3) + (kelly_15m * 0.5)

        return {
            "micro_validation": micro_validation,
            "arbitrage_detected": arbitrage_opportunity,
            "arbitrage_type": arbitrage_type,
            "arbitrage_reason": arbitrage_reason,
            "consensus_direction": consensus_direction,
            "consensus_strength": round(consensus_strength, 2),
            "composite_kelly_recommendation": round(composite_kelly, 4)
        }
