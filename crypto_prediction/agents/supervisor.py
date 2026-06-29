import httpx
from typing import Optional
from loguru import logger
from crypto_prediction.agents.search_agent import SearchAgent
from crypto_prediction.agents.market_data_agent import MarketDataAgent
from crypto_prediction.agents.prediction_agent import PredictionAgent
from crypto_prediction.agents.risk_agent import RiskAgent
from crypto_prediction.database.repository import PredictionRepository
from crypto_prediction.schemas.config import settings

class SupervisorAgent:
    def __init__(self):
        self.search_agent = SearchAgent()
        self.market_data_agent = MarketDataAgent()
        self.prediction_agent = PredictionAgent()
        self.risk_agent = RiskAgent()

    async def execute_prediction_flow(
        self,
        repo: PredictionRepository,
        symbol: str,
        interval: str,
        limit: int = 1000
    ) -> dict:
        """
        Coordinates the multi-agent prediction research pipeline.
        Flow:
        1. Fetch prediction markets (Search Agent) to locate asset probabilities.
        2. Fetch Binance OHLCV data (Market Data Agent).
        3. Predict price direction using Kronos (Prediction Agent).
        4. Calculate Kelly position sizes (Risk Agent).
        5. Generate agent reasoning narrative (using LLM via OpenRouter).
        6. Store the results in the database and return final payload.
        """
        logger.info(f"SupervisorAgent: Beginning execution flow for {symbol} ({interval})")

        # 1. Search Prediction Markets
        markets = await self.search_agent.execute()
        
        # Determine the base asset for matching
        # e.g., if symbol is BTCUSDT, base asset is BTC
        base_asset = symbol.upper().replace("USDT", "").replace("BUSD", "")
        matching_markets = [m for m in markets if m["asset"].upper() == base_asset]
        
        market_prob = 0.5
        question = "No active prediction market found for this asset."
        if matching_markets:
            # Pick the market with probability closest to extremes or the first one
            best_market = matching_markets[0]
            market_prob = best_market["market_probability"]
            question = best_market["question"]
            logger.info(f"SupervisorAgent: Found matching prediction market: '{question}' with prob={market_prob}")
        else:
            logger.info(f"SupervisorAgent: No matching prediction market for {base_asset}. Defaulting to 0.5.")

        # 2. Fetch Market Data
        df = await self.market_data_agent.execute(symbol, interval, limit)
        if df.empty:
            raise ValueError(f"No OHLCV data returned for symbol {symbol}")

        # 3. Predict next movement
        prediction_result = await self.prediction_agent.execute(df)
        
        # 4. Calculate Risk
        risk_result = await self.risk_agent.execute(market_prob, prediction_result["probability"])

        # 5. Generate reasoning via OpenRouter/Hermes Agent
        reasoning = await self._generate_reasoning(
            symbol=symbol,
            interval=interval,
            prediction_result=prediction_result,
            market_prob=market_prob,
            risk_result=risk_result,
            question=question
        )

        # 6. Store Prediction
        prediction_data = {
            "symbol": symbol,
            "interval": interval,
            "prediction_direction": prediction_result["direction"],
            "confidence": prediction_result["confidence"],
            "model_probability": prediction_result["probability"],
            "market_probability": market_prob,
            "kelly_fraction": risk_result["recommended_position_size"],
            "reasoning": reasoning
        }
        
        db_pred = await repo.save_prediction(prediction_data)

        # Build final response payload
        return {
            "prediction_id": db_pred.id,
            "symbol": symbol,
            "prediction": prediction_result["direction"],
            "confidence": prediction_result["confidence"],
            "market_probability": market_prob,
            "kelly": risk_result["recommended_position_size"],
            "reasoning": reasoning
        }

    async def _generate_reasoning(
        self,
        symbol: str,
        interval: str,
        prediction_result: dict,
        market_prob: float,
        risk_result: dict,
        question: str
    ) -> str:
        """
        Generates reasoning text using OpenRouter LLM.
        """
        # Try to use Hermes Agent run_agent if available
        try:
            import sys
            HERMES_PATH = "C:/Users/ag065/AppData/Local/hermes/hermes-agent"
            if HERMES_PATH not in sys.path:
                sys.path.append(HERMES_PATH)
            from run_agent import AIAgent
            logger.info("SupervisorAgent: Attempting to use Hermes AIAgent for reasoning...")
            agent = AIAgent(model=settings.MODEL_NAME, quiet_mode=True)
            prompt = self._build_reasoning_prompt(symbol, interval, prediction_result, market_prob, risk_result, question)
            # Use chat() method
            response = agent.chat(prompt)
            if response:
                return response.strip()
        except Exception as e:
            logger.warning(f"SupervisorAgent: Failed to use Hermes AIAgent directly ({e}). Falling back to direct API call.")

        # Fallback to direct HTTP request to OpenRouter
        if not settings.OPENROUTER_API_KEY or settings.OPENROUTER_API_KEY == "dummy-key-for-testing-if-empty":
            return (
                f"Kronos predicts next price movement is {prediction_result['direction']} with confidence "
                f"{prediction_result['confidence']:.2%}. Market probability is {market_prob:.2%}, leading to Kelly position size "
                f"of {risk_result['recommended_position_size']:.2%} with {risk_result['risk_level']} risk level."
            )

        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt = self._build_reasoning_prompt(symbol, interval, prediction_result, market_prob, risk_result, question)
        
        payload = {
            "model": settings.MODEL_NAME,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    logger.error(f"OpenRouter API returned status code {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Error calling OpenRouter: {e}")

        # Final default fallback
        return (
            f"Kronos model predicts {prediction_result['direction']} ({prediction_result['confidence']:.2%} confidence) "
            f"against market probability of {market_prob:.2%}. Kelly recommendation is to bet {risk_result['recommended_position_size']:.2%} "
            f"on {risk_result['recommended_direction']}."
        )

    def _build_reasoning_prompt(
        self,
        symbol: str,
        interval: str,
        prediction_result: dict,
        market_prob: float,
        risk_result: dict,
        question: str
    ) -> str:
        return (
            f"You are the Supervisor Agent of a multi-agent crypto prediction research system.\n"
            f"Generate a clear, professional, and concise prediction summary based on the following research metrics:\n\n"
            f"- Symbol: {symbol}\n"
            f"- Interval: {interval}\n"
            f"- Kronos Prediction: {prediction_result['direction']} (Confidence: {prediction_result['confidence']:.2%})\n"
            f"- Prediction Market Question: {question}\n"
            f"- Market Probability: {market_prob:.2%}\n"
            f"- Kelly Recommendation: Direction={risk_result['recommended_direction']}, Fraction={risk_result['recommended_position_size']:.2%}, Risk={risk_result['risk_level']}\n\n"
            f"Explain the rationale behind the predicted price movement and how the discrepancy between market probability "
            f"and our model's probability led to the recommended Kelly fraction size. Keep the explanation under 150 words, "
            f"focused on mathematical edge and risk management."
        )
