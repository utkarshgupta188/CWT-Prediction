import asyncio
import time
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


class HermesSupervisorAgent:

    def __init__(self):
        self.search_agent = HermesSearchAgent()
        self.market_data_agent = HermesMarketDataAgent()
        self.prediction_agent = HermesPredictionAgent()
        self.risk_agent = HermesRiskAgent()
        self.feedback_agent = HermesFeedbackAgent()
        self.memory = HermesMemory()

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

            prediction_data = {
                "symbol": symbol,
                "interval": interval,
                "prediction_direction": prediction_result["direction"],
                "confidence": prediction_result["confidence"],
                "model_probability": prediction_result["probability"],
                "market_probability": market_prob,
                "kelly_fraction": risk_result["recommended_position_size"],
                "reasoning": "",
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
                reasoning="",
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
                "reasoning": "",
            }

        except Exception as e:
            elapsed = time.time() - overall_start
            logger.error(f"HermesSupervisor: Flow failed after {elapsed:.2f}s: {e}")
            raise

    async def _step_search(self, symbol: str, timeout: float = 30.0):
        logger.info("HermesSupervisor: Step 1 - Searching prediction markets (timeout=%ss)", timeout)
        start = time.time()
        markets = await asyncio.wait_for(self.search_agent.execute(), timeout=timeout)
        elapsed = time.time() - start
        logger.info(f"HermesSupervisor: Step 1 completed in {elapsed:.2f}s, found {len(markets)} markets")
        return markets

    def _resolve_market_probability(self, markets: list, symbol: str):
        base_asset = symbol.upper().replace("USDT", "").replace("BUSD", "")
        matching_markets = [m for m in markets if m.get("asset", "").upper() == base_asset]
        if not matching_markets:
            raise ValueError(f"No prediction market found for asset {base_asset}")
        best = matching_markets[0]
        logger.info(f"HermesSupervisor: Found matching market: '{best.get('question')}' prob={best['market_probability']}")
        return best["market_probability"], best.get("question", "")

    async def _step_market_data(self, symbol: str, interval: str, limit: int):
        logger.info("HermesSupervisor: Step 2 - Fetching market data")
        start = time.time()
        df = await self.market_data_agent.execute(symbol, interval, limit)
        if not df.empty:
            elapsed = time.time() - start
            logger.info(f"HermesSupervisor: Step 2 completed in {elapsed:.2f}s, got {len(df)} candles")
            return df
        elapsed = time.time() - start
        logger.error(f"HermesSupervisor: Step 2 failed after {elapsed:.2f}s - empty DataFrame")
        raise ValueError(f"No OHLCV data returned for symbol {symbol}")

    async def _step_prediction(self, df):
        logger.info("HermesSupervisor: Step 3 - Running Kronos prediction")
        start = time.time()
        result = await self.prediction_agent.execute(df)
        elapsed = time.time() - start
        logger.info(f"HermesSupervisor: Step 3 completed in {elapsed:.2f}s -> {result['direction']} ({result['confidence']:.4f})")
        return result

    async def _step_risk(self, market_prob: float, model_prob: float):
        logger.info("HermesSupervisor: Step 4 - Calculating risk")
        start = time.time()
        result = await self.risk_agent.execute(market_prob, model_prob)
        elapsed = time.time() - start
        logger.info(f"HermesSupervisor: Step 4 completed in {elapsed:.3f}s -> {result['recommended_direction']} ({result['recommended_position_size']:.4f})")
        return result
