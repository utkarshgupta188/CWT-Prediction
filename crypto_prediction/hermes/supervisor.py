import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional
from loguru import logger

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
from crypto_prediction.schemas.config import settings


class HermesSupervisorAgent:
    """Hermes Supervisor Agent.
    
    Orchestrates the prediction pipeline using Hermes-registered tools.
    Uses AIAgent for reasoning generation only - all calculations remain
    in the original services.
    """

    def __init__(self):
        self.search_agent = HermesSearchAgent()
        self.market_data_agent = HermesMarketDataAgent()
        self.prediction_agent = HermesPredictionAgent()
        self.risk_agent = HermesRiskAgent()
        self.feedback_agent = HermesFeedbackAgent()
        self.memory = HermesMemory()
        self._ai_agent = None

    def _get_ai_agent(self):
        if self._ai_agent is None:
            try:
                from run_agent import AIAgent
                logger.info("HermesSupervisor: Initializing AIAgent for reasoning")
                self._ai_agent = AIAgent(
                    model=settings.MODEL_NAME,
                    quiet_mode=True,
                    enabled_toolsets=[],
                )
            except Exception as e:
                logger.warning(f"HermesSupervisor: Failed to create AIAgent: {e}")
                self._ai_agent = None
        return self._ai_agent

    async def execute_prediction_flow(
        self,
        repo,
        symbol: str,
        interval: str,
        limit: int = 1000,
    ) -> dict:
        logger.info(f"HermesSupervisor: Beginning execution flow for {symbol} ({interval})")
        overall_start = time.time()

        try:
            async def _market_and_prediction():
                df = await self._step_market_data(symbol, interval, limit)
                return await self._step_prediction(df)

            search_task = asyncio.create_task(self._step_search(symbol))
            prediction_task = asyncio.create_task(_market_and_prediction())

            markets, prediction_result = await asyncio.gather(
                search_task, prediction_task,
            )

            market_prob, question = self._resolve_market_probability(markets, symbol)

            risk_result = await self._step_risk(market_prob, prediction_result["probability"])

            reasoning = await self._step_reasoning(
                symbol=symbol,
                interval=interval,
                prediction_result=prediction_result,
                market_prob=market_prob,
                risk_result=risk_result,
                question=question,
            )

            prediction_data = {
                "symbol": symbol,
                "interval": interval,
                "prediction_direction": prediction_result["direction"],
                "confidence": prediction_result["confidence"],
                "model_probability": prediction_result["probability"],
                "market_probability": market_prob,
                "kelly_fraction": risk_result["recommended_position_size"],
                "reasoning": reasoning,
            }
            db_pred = await repo.save_prediction(prediction_data)

            self.memory.add_prediction(
                symbol=symbol,
                interval=interval,
                prediction_direction=prediction_result["direction"],
                confidence=prediction_result["confidence"],
                model_probability=prediction_result["probability"],
                market_probability=market_prob,
                kelly_fraction=risk_result["recommended_position_size"],
                reasoning=reasoning,
            )

            overall_elapsed = time.time() - overall_start
            logger.info(f"HermesSupervisor: Flow completed in {overall_elapsed:.2f}s")

            return {
                "prediction_id": db_pred.id,
                "symbol": symbol,
                "prediction": prediction_result["direction"],
                "confidence": prediction_result["confidence"],
                "market_probability": market_prob,
                "kelly": risk_result["recommended_position_size"],
                "reasoning": reasoning,
            }

        except Exception as e:
            elapsed = time.time() - overall_start
            logger.error(f"HermesSupervisor: Flow failed after {elapsed:.2f}s: {e}")
            raise

    async def _step_search(self, symbol: str, timeout: float = 30.0):
        logger.info("HermesSupervisor: Step 1 - Searching prediction markets (timeout=%ss)", timeout)
        start = time.time()
        try:
            markets = await asyncio.wait_for(self.search_agent.execute(), timeout=timeout)
            elapsed = time.time() - start
            logger.info(f"HermesSupervisor: Step 1 completed in {elapsed:.2f}s, found {len(markets)} markets")
            return markets
        except asyncio.TimeoutError:
            elapsed = time.time() - start
            logger.warning(f"HermesSupervisor: Step 1 timed out after {elapsed:.2f}s, returning empty")
            return []
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"HermesSupervisor: Step 1 failed after {elapsed:.2f}s: {e}")
            return []

    def _resolve_market_probability(self, markets: list, symbol: str):
        base_asset = symbol.upper().replace("USDT", "").replace("BUSD", "")
        matching_markets = [m for m in markets if m.get("asset", "").upper() == base_asset]
        if matching_markets:
            best = matching_markets[0]
            logger.info(f"HermesSupervisor: Found matching market: '{best.get('question')}' prob={best['market_probability']}")
            return best["market_probability"], best.get("question", "No question available")
        logger.info(f"HermesSupervisor: No matching market for {base_asset}, defaulting to 0.5")
        return 0.5, "No active prediction market found for this asset."

    async def _step_market_data(self, symbol: str, interval: str, limit: int):
        logger.info("HermesSupervisor: Step 2 - Fetching market data")
        start = time.time()
        try:
            df = await self.market_data_agent.execute(symbol, interval, limit)
            if df.empty:
                raise ValueError(f"No OHLCV data returned for symbol {symbol}")
            elapsed = time.time() - start
            logger.info(f"HermesSupervisor: Step 2 completed in {elapsed:.2f}s, got {len(df)} candles")
            return df
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"HermesSupervisor: Step 2 failed after {elapsed:.2f}s: {e}")
            raise

    async def _step_prediction(self, df):
        logger.info("HermesSupervisor: Step 3 - Running Kronos prediction")
        start = time.time()
        try:
            result = await self.prediction_agent.execute(df)
            elapsed = time.time() - start
            logger.info(f"HermesSupervisor: Step 3 completed in {elapsed:.2f}s -> {result['direction']} ({result['confidence']:.4f})")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"HermesSupervisor: Step 3 failed after {elapsed:.2f}s: {e}")
            raise

    async def _step_risk(self, market_prob: float, model_prob: float):
        logger.info("HermesSupervisor: Step 4 - Calculating risk")
        start = time.time()
        try:
            result = await self.risk_agent.execute(market_prob, model_prob)
            elapsed = time.time() - start
            logger.info(f"HermesSupervisor: Step 4 completed in {elapsed:.3f}s -> {result['recommended_direction']} ({result['recommended_position_size']:.4f})")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"HermesSupervisor: Step 4 failed after {elapsed:.3f}s: {e}")
            raise

    async def _step_reasoning(
        self,
        symbol: str,
        interval: str,
        prediction_result: dict,
        market_prob: float,
        risk_result: dict,
        question: str,
    ) -> str:
        logger.info("HermesSupervisor: Step 5 - Generating reasoning (deterministic fallback)")
        start = time.time()
        reasoning = self._fallback_reasoning(
            symbol, interval, prediction_result, market_prob, risk_result
        )
        elapsed = time.time() - start
        logger.info(f"HermesSupervisor: Step 5 completed in {elapsed:.3f}s")
        return reasoning

    def _build_reasoning_prompt(
        self,
        symbol: str,
        interval: str,
        prediction_result: dict,
        market_prob: float,
        risk_result: dict,
        question: str,
    ) -> str:
        memory_context = self.memory.get_recent_context(limit=3)
        return (
            f"Generate a clear, professional prediction summary based on these research metrics:\n\n"
            f"Symbol: {symbol}\n"
            f"Interval: {interval}\n"
            f"Kronos Prediction: {prediction_result['direction']} (Confidence: {prediction_result['confidence']:.2%})\n"
            f"Prediction Market Question: {question}\n"
            f"Market Probability: {market_prob:.2%}\n"
            f"Kelly Recommendation: Direction={risk_result['recommended_direction']}, "
            f"Fraction={risk_result['recommended_position_size']:.2%}, "
            f"Risk={risk_result['risk_level']}\n\n"
            f"{memory_context}\n\n"
            f"Explain the rationale behind the predicted price movement and how the "
            f"discrepancy between market probability and our model's probability led "
            f"to the recommended Kelly fraction size. Keep under 150 words, focused "
            f"on mathematical edge and risk management."
        )

    def _fallback_reasoning(
        self,
        symbol: str,
        interval: str,
        prediction_result: dict,
        market_prob: float,
        risk_result: dict,
    ) -> str:
        return (
            f"Kronos predicts next price movement is {prediction_result['direction']} with "
            f"confidence {prediction_result['confidence']:.2%}. Market probability is "
            f"{market_prob:.2%}, leading to Kelly position size of "
            f"{risk_result['recommended_position_size']:.2%} with "
            f"{risk_result['risk_level']} risk level."
        )
