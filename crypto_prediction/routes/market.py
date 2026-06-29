from fastapi import APIRouter, Depends
from typing import List
from crypto_prediction.hermes.agents import HermesSearchAgent
from crypto_prediction.database.repository import AsyncSessionLocal, PredictionRepository
from crypto_prediction.utils.logger import setup_logger
from crypto_prediction.schemas.config import settings

logger = setup_logger(settings.LOG_LEVEL)
router = APIRouter()
search_agent = HermesSearchAgent()

async def get_repo():
    async with AsyncSessionLocal() as session:
        yield PredictionRepository(session)

@router.get("/markets")
async def get_markets(repo: PredictionRepository = Depends(get_repo)):
    """
    Fetch active prediction markets from Polymarket and Kalshi,
    store them in the database, and return.
    """
    markets = await search_agent.execute(limit_per_platform=15)
    
    # Store in DB
    saved_markets = []
    for m in markets:
        try:
            db_m = await repo.save_market(m)
            saved_markets.append(db_m)
        except Exception as e:
            logger.error(f"Error saving market {m.get('market_id')}: {e}")
            
    # Return formatted response
    return [
        {
            "asset": m.asset,
            "platform": m.platform,
            "question": m.question,
            "market_probability": m.market_probability,
            "expiration": m.expiration,
            "market_id": m.market_id
        }
        for m in saved_markets
    ]
