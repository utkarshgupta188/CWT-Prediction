"""Script for Hermes to execute via execute_code tool.
Usage: Hermes runs this file to get a full prediction result.
"""
import asyncio
import json
import sys
sys.path.insert(0, r"D:\CWT prediction")
from crypto_prediction.hermes import HermesSupervisorAgent
from crypto_prediction.hermes.multi_timeframe import MultiTimeframeOrchestrator
from crypto_prediction.database.repository import AsyncSessionLocal, PredictionRepository, init_db


async def run(symbol="BTCUSDT", symbols=None, interval="5m", limit=1000, parallel=False, multi_timeframe=False):
    await init_db()
    async with AsyncSessionLocal() as session:
        repo = PredictionRepository(session)
        sup = HermesSupervisorAgent()
        
        if parallel:
            symbol_list = [s.strip().upper() for s in symbols.split(",")] if symbols else ["BTCUSDT", "ETHUSDT"]
            result = await sup.execute_parallel_prediction(repo, symbol_list, interval, limit)
        elif multi_timeframe:
            orchestrator = MultiTimeframeOrchestrator(sup)
            result = await orchestrator.execute_multi_timeframe(repo, symbol, limit)
        else:
            result = await sup.execute_prediction_flow(repo, symbol, interval, limit)
            
        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--multi-timeframe", action="store_true")
    args = parser.parse_args()
    
    result = asyncio.run(run(
        symbol=args.symbol,
        symbols=args.symbols,
        interval=args.interval,
        limit=args.limit,
        parallel=args.parallel,
        multi_timeframe=args.multi_timeframe
    ))
    print(json.dumps(result, indent=2))
