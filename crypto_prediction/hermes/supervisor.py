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


import json

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
        logger.info(f"HermesSupervisor: Beginning agentic execution flow for {symbol} ({interval})")
        overall_start = time.time()

        try:
            import os
            from run_agent import AIAgent

            agent = AIAgent(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                model=os.environ.get("MODEL_NAME", "google/gemini-2.5-flash"),
                enabled_toolsets=["crypto-prediction"],
                verbose_logging=False,
                max_tokens=200
            )

            base_asset = symbol.upper().replace("USDT", "").replace("BUSD", "")
            asset_name = "Bitcoin" if base_asset == "BTC" else ("Ethereum" if base_asset == "ETH" else base_asset)

            prompt = (
                f"Perform a crypto prediction flow for {symbol} ({interval} interval). "
                f"The asset name is '{asset_name}'.\n"
                "You must perform the following actions:\n"
                f"1. Search active prediction markets for '{asset_name}' on both Polymarket and Kalshi using search_polymarket and search_kalshi.\n"
                f"2. Predict the next movement of {symbol} using get_prediction (pass '[]' for candles parameter, as the tool will fetch the data internally).\n"
                "3. Calculate risk using calculate_risk. Choose the most relevant market probability from the search results to compare with the model's prediction probability. If no matching market probability is found, default to 0.5.\n"
                "Output your reasoning and the final prediction details (direction, confidence, market probability, kelly fraction) in a clear format."
            )

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, agent.run_conversation, prompt)

            if response.get("failed") and not response.get("messages"):
                raise ValueError(f"Agent conversation failed: {response.get('error')}")

            # Extract structured parameters from the conversation trajectory
            prediction_direction = "NONE"
            confidence = 0.5
            model_probability = 0.5
            market_probability = 0.5
            kelly_fraction = 0.0
            reasoning = response.get("final_response") or ""

            # Extract reasoning from assistant messages
            reasoning_parts = []
            for msg in response.get("messages", []):
                if msg.get("role") == "assistant":
                    content = msg.get("content") or ""
                    rc = msg.get("reasoning_content") or msg.get("reasoning")
                    if rc:
                        reasoning_parts.append(rc)
                    elif content:
                        reasoning_parts.append(content)
            if reasoning_parts:
                reasoning = "\n\n".join(reasoning_parts)

            # Scan messages for tool call responses
            for msg in response.get("messages", []):
                if msg.get("role") == "tool":
                    tool_name = msg.get("tool_name") or msg.get("name")
                    content_str = msg.get("content") or ""
                    try:
                        content_data = json.loads(content_str)
                    except Exception:
                        continue

                    if tool_name == "get_prediction":
                        prediction_direction = content_data.get("direction", prediction_direction)
                        confidence = float(content_data.get("confidence", confidence))
                        model_probability = float(content_data.get("probability", model_probability))

                    elif tool_name == "calculate_risk":
                        kelly_fraction = float(content_data.get("recommended_position_size", kelly_fraction))

            # Inspect tool_calls in assistant messages to get the arguments passed to calculate_risk
            for msg in response.get("messages", []):
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        if func.get("name") == "calculate_risk":
                            args_str = func.get("arguments", "{}")
                            try:
                                args = json.loads(args_str)
                                market_probability = float(args.get("market_probability", market_probability))
                            except Exception:
                                pass

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
            logger.info(f"HermesSupervisor: Flow completed agentically in {overall_elapsed:.2f}s")

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

    def _resolve_market_probability(self, markets: list, symbol: str):
        base_asset = symbol.upper().replace("USDT", "").replace("BUSD", "")
        matching_markets = [m for m in markets if m.get("asset", "").upper() == base_asset]
        if not matching_markets:
            raise ValueError(f"No prediction market found for asset {base_asset}")
        best = matching_markets[0]
        logger.info(f"HermesSupervisor: Found matching market: '{best.get('question')}' prob={best['market_probability']}")
        return best["market_probability"], best.get("question", "")
