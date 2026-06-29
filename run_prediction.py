import asyncio
import argparse
import json
from crypto_prediction.hermes import HermesSupervisorAgent
from crypto_prediction.database.repository import AsyncSessionLocal, PredictionRepository
from crypto_prediction.utils.logger import setup_logger
from crypto_prediction.schemas.config import settings

logger = setup_logger(settings.LOG_LEVEL)

async def main():
    parser = argparse.ArgumentParser(description="Run Crypto Prediction Pipeline")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Crypto pair (e.g., BTCUSDT)")
    parser.add_argument("--interval", type=str, default="5m", help="Candle interval (e.g., 5m, 15m, 1h)")
    parser.add_argument("--limit", type=int, default=1000, help="Number of historical candles (100-1000)")
    args = parser.parse_args()

    logger.info(f"CLI: Running prediction flow for {args.symbol} ({args.interval}, limit={args.limit})...")
    
    async with AsyncSessionLocal() as session:
        repo = PredictionRepository(session)
        supervisor = HermesSupervisorAgent()
        result = await supervisor.execute_prediction_flow(
            repo=repo,
            symbol=args.symbol,
            interval=args.interval,
            limit=args.limit
        )
        
    print("\n--- PREDICTION RESULT ---")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
