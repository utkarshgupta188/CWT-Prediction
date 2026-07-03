import asyncio
import time
import json
import os
from loguru import logger
from dotenv import load_dotenv

# Load environment variables into os.environ
load_dotenv()

from crypto_prediction.hermes.hermes_bootstrap import ensure_hermes_on_path
ensure_hermes_on_path()


from crypto_prediction.hermes.memory import HermesMemory
from crypto_prediction.hermes.agents import (
    HermesSearchAgent,
    HermesMarketDataAgent,
    HermesPredictionAgent,
    HermesRiskAgent,
    HermesFeedbackAgent,
)

# Asset name mapping for LLM prompts and market search
ASSET_MAP = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana",
    "DOGE": "Dogecoin",
    "ADA": "Cardano",
    "XRP": "Ripple",
}


def _symbol_to_base(symbol: str) -> str:
    return symbol.upper().replace("USDT", "").replace("BUSD", "")


def _symbol_to_name(symbol: str) -> str:
    base = _symbol_to_base(symbol)
    return ASSET_MAP.get(base, base)


class HermesSupervisorAgent:
    """Orchestrates the prediction pipeline with parallel execution.

    Design:
      1. Independent data-fetching tasks (market search + data fetch + prediction)
         run in parallel via asyncio.gather.
      2. Consolidated results are fed to the Hermes AIAgent for reasoning.
      3. Previous prediction history (HermesMemory) is injected into the prompt
         for feedback-loop learning.
    """

    def __init__(self):
        self.search_agent = HermesSearchAgent()
        self.market_data_agent = HermesMarketDataAgent()
        self.prediction_agent = HermesPredictionAgent()
        self.risk_agent = HermesRiskAgent()
        self.feedback_agent = HermesFeedbackAgent()
        self.memory = HermesMemory()

    # ------------------------------------------------------------------
    # Parallel multi-asset entry point
    # ------------------------------------------------------------------
    async def execute_parallel_prediction(
        self,
        repo,
        symbols: list[str] = None,
        interval: str = "5m",
        limit: int = 1000,
    ) -> list[dict]:
        """Run predictions for multiple symbols in parallel."""
        symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        logger.info(f"HermesSupervisor: Parallel prediction for {symbols} ({interval})")
        start = time.time()

        tasks = [
            self.execute_prediction_flow(repo, sym, interval, limit)
            for sym in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for sym, res in zip(symbols, results):
            if isinstance(res, Exception):
                logger.error(f"HermesSupervisor: Prediction failed for {sym}: {res}")
                output.append({"symbol": sym, "error": str(res)})
            else:
                output.append(res)

        elapsed = time.time() - start
        logger.info(f"HermesSupervisor: All {len(symbols)} predictions completed in {elapsed:.2f}s")
        return output

    # ------------------------------------------------------------------
    # Core single-symbol prediction flow (parallelized internally)
    # ------------------------------------------------------------------
    async def execute_prediction_flow(
        self,
        repo,
        symbol: str,
        interval: str,
        limit: int = 1000,
    ) -> dict:
        """Execute the full prediction pipeline for a single symbol.

        Phase 1 — Parallel data gathering:
          • search_polymarket + search_kalshi  (via SearchAgent, already parallel)
          • binance_fetch + kronos_predict     (via MarketDataAgent → PredictionAgent)

        Phase 2 — LLM reasoning:
          • Feed consolidated results + memory context to Hermes AIAgent
          • Agent calls calculate_risk and produces final reasoning

        Phase 3 — Persist:
          • Save prediction to DB + update memory
        """
        logger.info(f"HermesSupervisor: Starting flow for {symbol} ({interval})")
        overall_start = time.time()

        try:
            base_asset = _symbol_to_base(symbol)
            asset_name = _symbol_to_name(symbol)

            # ── Phase 1: Parallel data gathering ─────────────────────
            phase1_start = time.time()

            async def _search_markets():
                try:
                    return await self.search_agent.execute(
                        symbol=symbol, limit_per_platform=5, per_call_timeout=15.0
                    )
                except Exception as e:
                    logger.warning(f"HermesSupervisor: Market search failed for {symbol}: {e}")
                    return []

            async def _fetch_and_predict():
                df = await self.market_data_agent.execute(symbol, interval, limit)
                prediction = await self.prediction_agent.execute(df, pred_len=5)
                return prediction

            # Run search and prediction concurrently
            search_task = asyncio.create_task(_search_markets())
            predict_task = asyncio.create_task(_fetch_and_predict())

            markets, prediction = await asyncio.gather(search_task, predict_task)

            phase1_elapsed = time.time() - phase1_start
            logger.info(
                f"HermesSupervisor: Phase 1 done in {phase1_elapsed:.2f}s "
                f"(markets={len(markets)}, prediction={prediction.get('direction')})"
            )

            # ── Resolve market probability ───────────────────────────
            market_probability = 0.5
            market_question = ""
            matching_markets = [
                m for m in markets
                if m.get("asset", "").upper() == base_asset
            ]
            if matching_markets:
                best = matching_markets[0]
                market_probability = best.get("market_probability", 0.5)
                market_question = best.get("question", "")
                logger.info(
                    f"HermesSupervisor: Matched market '{market_question}' "
                    f"prob={market_probability:.4f}"
                )

            # ── Phase 2: Risk calculation ────────────────────────────
            model_probability = prediction.get("probability", 0.5)
            prediction_direction = prediction.get("direction", "NONE")
            confidence = prediction.get("confidence", 0.5)

            risk_result = await self.risk_agent.execute(
                market_probability=market_probability,
                model_probability=model_probability,
            )
            kelly_fraction = risk_result.get("recommended_position_size", 0.0)

            # ── Phase 3: LLM reasoning with feedback loop ───────────
            reasoning = await self._llm_reasoning(
                symbol=symbol,
                interval=interval,
                asset_name=asset_name,
                prediction=prediction,
                markets=markets,
                matching_markets=matching_markets,
                market_probability=market_probability,
                risk_result=risk_result,
            )

            # ── Phase 4: Persist ─────────────────────────────────────
            prediction_data = {
                "symbol": symbol,
                "interval": interval,
                "prediction_direction": prediction_direction,
                "confidence": confidence,
                "model_probability": model_probability,
                "market_probability": market_probability,
                "kelly_fraction": kelly_fraction,
                "reasoning": reasoning,
            }
            db_pred = await repo.save_prediction(prediction_data)

            self.memory.add_prediction(
                symbol=symbol,
                interval=interval,
                prediction_direction=prediction_direction,
                confidence=confidence,
                model_probability=model_probability,
                market_probability=market_probability,
                kelly_fraction=kelly_fraction,
                reasoning=reasoning,
            )

            overall_elapsed = time.time() - overall_start
            logger.info(f"HermesSupervisor: Flow completed in {overall_elapsed:.2f}s")

            return {
                "prediction_id": db_pred.id,
                "symbol": symbol,
                "prediction": prediction_direction,
                "confidence": confidence,
                "market_probability": market_probability,
                "kelly": kelly_fraction,
                "reasoning": reasoning,
            }

        except Exception as e:
            elapsed = time.time() - overall_start
            logger.error(f"HermesSupervisor: Flow failed after {elapsed:.2f}s: {e}")
            raise

    # ------------------------------------------------------------------
    # LLM reasoning via Hermes AIAgent
    # ------------------------------------------------------------------
    async def _llm_reasoning(
        self,
        symbol: str,
        interval: str,
        asset_name: str,
        prediction: dict,
        markets: list,
        matching_markets: list,
        market_probability: float,
        risk_result: dict,
    ) -> str:
        """Use the Hermes AIAgent for reasoning about the prediction.

        Falls back to a structured summary if the LLM is unavailable.
        """
        try:
            from run_agent import AIAgent

            api_key = os.environ.get("OPENROUTER_API_KEY")
            model = os.environ.get("MODEL_NAME", "google/gemini-flash-1.5-free")
            model = os.environ.get("MODEL_NAME", "google/gemini-2.5-flash")

            if not api_key:
                logger.warning("HermesSupervisor: No OPENROUTER_API_KEY, skipping LLM reasoning")
                return self._fallback_reasoning(
                    symbol, prediction, markets, matching_markets,
                    market_probability, risk_result
                )

            agent = AIAgent(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                model=model,
                enabled_toolsets=["crypto-prediction"],
                verbose_logging=False,
                max_tokens=200,
                quiet_mode=True,
            )

            # Build prompt with memory context for feedback loop
            memory_context = self.memory.get_recent_context(limit=5)

            market_summary = "No matching prediction markets found."
            if matching_markets:
                lines = []
                for m in matching_markets[:3]:
                    lines.append(
                        f"  - [{m['platform']}] \"{m['question']}\" → prob={m['market_probability']:.2%}"
                    )
                market_summary = "\n".join(lines)

            prompt = (
                f"Analyze this crypto prediction for {symbol} ({asset_name}, {interval} interval).\n\n"
                f"MODEL PREDICTION:\n"
                f"  Direction: {prediction['direction']}\n"
                f"  Confidence: {prediction['confidence']:.2%}\n"
                f"  Model Probability: {prediction.get('probability', 0.5):.4f}\n"
                f"  Predicted Price: {prediction.get('predicted_price', 'N/A')}\n\n"
                f"MATCHING PREDICTION MARKETS:\n{market_summary}\n\n"
                f"RISK ANALYSIS (Kelly Criterion):\n"
                f"  Edge: {risk_result.get('edge', 0):.4f}\n"
                f"  Kelly Fraction: {risk_result.get('kelly_fraction', 0):.4f}\n"
                f"  Recommended Size: {risk_result.get('recommended_position_size', 0):.4f}\n"
                f"  Risk Level: {risk_result.get('risk_level', 'NONE')}\n"
                f"  Direction: {risk_result.get('recommended_direction', 'NONE')}\n\n"
                f"PREDICTION HISTORY (feedback loop):\n{memory_context}\n\n"
                f"Provide a concise analysis: Is the model's prediction supported by the market? "
                f"What is the edge? Should we act on this signal? "
                f"Consider the historical accuracy from the feedback loop."
            )

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, agent.run_conversation, prompt)

            reasoning = response.get("final_response", "")
            if not reasoning:
                # Extract from messages
                for msg in reversed(response.get("messages", [])):
                    if msg.get("role") == "assistant":
                        content = msg.get("content") or ""
                        if content:
                            reasoning = content
                            break

            return reasoning or self._fallback_reasoning(
                symbol, prediction, markets, matching_markets,
                market_probability, risk_result
            )

        except Exception as e:
            logger.warning(f"HermesSupervisor: LLM reasoning failed: {e}. Using fallback.")
            return self._fallback_reasoning(
                symbol, prediction, markets, matching_markets,
                market_probability, risk_result
            )

    def _fallback_reasoning(
        self, symbol, prediction, markets, matching_markets,
        market_probability, risk_result
    ) -> str:
        """Structured reasoning when LLM is unavailable."""
        direction = prediction.get("direction", "NONE")
        confidence = prediction.get("confidence", 0.5)
        model_prob = prediction.get("probability", 0.5)
        edge = risk_result.get("edge", 0)
        kelly = risk_result.get("recommended_position_size", 0)
        risk_level = risk_result.get("risk_level", "NONE")

        market_info = "No matching prediction markets found."
        if matching_markets:
            m = matching_markets[0]
            market_info = f"Market: \"{m['question']}\" ({m['platform']}) → prob={m['market_probability']:.2%}"

        return (
            f"Kronos predicts {symbol} next move: {direction} "
            f"(confidence={confidence:.2%}, model_prob={model_prob:.4f}).\n"
            f"{market_info}\n"
            f"Market probability: {market_probability:.4f}.\n"
            f"Kelly edge: {edge:.4f}, position size: {kelly:.4f}, risk: {risk_level}.\n"
            f"Total markets scanned: {len(markets)}."
        )

    # ------------------------------------------------------------------
    # Legacy helper (kept for test compatibility)
    # ------------------------------------------------------------------
    def _resolve_market_probability(self, markets: list, symbol: str):
        base_asset = _symbol_to_base(symbol)
        matching_markets = [m for m in markets if m.get("asset", "").upper() == base_asset]
        if not matching_markets:
            raise ValueError(f"No prediction market found for asset {base_asset}")
        best = matching_markets[0]
        logger.info(f"HermesSupervisor: Found matching market: '{best.get('question')}' prob={best['market_probability']}")
        return best["market_probability"], best.get("question", "")
