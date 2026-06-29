"""Script for Hermes to execute via execute_code tool.
Usage: Hermes runs this file to get a full prediction result.
"""
import asyncio, json, sys
sys.path.insert(0, r"D:\CWT prediction")
from crypto_prediction.hermes import HermesSupervisorAgent
from crypto_prediction.database.repository import AsyncSessionLocal, PredictionRepository, init_db

async def run(symbol="BTCUSDT", interval="5m", limit=1000):
    await init_db()
    async with AsyncSessionLocal() as session:
        repo = PredictionRepository(session)
        sup = HermesSupervisorAgent()
        result = await sup.execute_prediction_flow(repo, symbol, interval, limit)
        return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    result = asyncio.run(run(args.symbol, args.interval, args.limit))
    print(json.dumps(result, indent=2))
