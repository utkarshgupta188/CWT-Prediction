import asyncio
import argparse
import json
from crypto_prediction.hermes import HermesSupervisorAgent
from crypto_prediction.hermes.multi_timeframe import MultiTimeframeOrchestrator
from crypto_prediction.database.repository import AsyncSessionLocal, PredictionRepository, init_db
from crypto_prediction.utils.logger import setup_logger
from crypto_prediction.schemas.config import settings

logger = setup_logger(settings.LOG_LEVEL)


async def main():
    parser = argparse.ArgumentParser(description="Run Crypto Prediction Pipeline")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Crypto pair (e.g., BTCUSDT)")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT", help="Comma-separated symbols for parallel execution")
    parser.add_argument("--interval", type=str, default="5m", help="Candle interval (e.g., 5m, 15m, 1h)")
    parser.add_argument("--limit", type=int, default=1000, help="Number of historical candles (100-1000)")
    parser.add_argument("--parallel", action="store_true", help="Run parallel predictions for --symbols")
    parser.add_argument("--multi-timeframe", action="store_true", help="Run multi-timeframe analysis for --symbol")
    args = parser.parse_args()

    # Ensure DB is initialized
    await init_db()

    async with AsyncSessionLocal() as session:
        repo = PredictionRepository(session)
        supervisor = HermesSupervisorAgent()

        if args.parallel:
            symbol_list = [s.strip().upper() for s in args.symbols.split(",")]
            logger.info(f"CLI: Running parallel predictions for {symbol_list} ({args.interval}, limit={args.limit})...")
            result = await supervisor.execute_parallel_prediction(
                repo=repo,
                symbols=symbol_list,
                interval=args.interval,
                limit=args.limit
            )
            print("\n--- PARALLEL PREDICTION RESULTS ---")
            print(json.dumps(result, indent=2))

        elif args.multi_timeframe:
            logger.info(f"CLI: Running multi-timeframe analysis for {args.symbol} (limit={args.limit})...")
            orchestrator = MultiTimeframeOrchestrator(supervisor)
            result = await orchestrator.execute_multi_timeframe(
                repo=repo,
                symbol=args.symbol,
                limit=args.limit
            )
            print("\n--- MULTI-TIMEFRAME ANALYSIS RESULT ---")
            print(json.dumps(result, indent=2))

        else:
            logger.info(f"CLI: Running prediction flow for {args.symbol} ({args.interval}, limit={args.limit})...")
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
