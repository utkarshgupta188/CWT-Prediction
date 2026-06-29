import datetime
from typing import List, Optional
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.future import select
from crypto_prediction.database.models import Base, DBMarket, DBPrediction, DBFeedback, DBStatistics
from crypto_prediction.schemas.config import settings
from crypto_prediction.utils.logger import setup_logger

logger = setup_logger(settings.LOG_LEVEL)

# Setup async engine
engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        # Create all tables if they do not exist
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized successfully.")

class PredictionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_market(self, market_data: dict) -> DBMarket:
        # Check if market already exists
        query = select(DBMarket).where(DBMarket.market_id == market_data["market_id"])
        result = await self.session.execute(query)
        db_market = result.scalar_one_or_none()

        if db_market:
            # Update fields
            db_market.market_probability = market_data["market_probability"]
            db_market.expiration = market_data.get("expiration")
            db_market.question = market_data["question"]
            db_market.asset = market_data["asset"]
        else:
            db_market = DBMarket(
                asset=market_data["asset"],
                platform=market_data["platform"],
                question=market_data["question"],
                market_probability=market_data["market_probability"],
                expiration=market_data.get("expiration"),
                market_id=market_data["market_id"],
            )
            self.session.add(db_market)
        
        await self.session.commit()
        await self.session.refresh(db_market)
        return db_market

    async def get_markets(self) -> List[DBMarket]:
        query = select(DBMarket)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def save_prediction(self, prediction_data: dict) -> DBPrediction:
        db_pred = DBPrediction(
            symbol=prediction_data["symbol"],
            interval=prediction_data["interval"],
            prediction_direction=prediction_data["prediction_direction"],
            confidence=prediction_data["confidence"],
            model_probability=prediction_data["model_probability"],
            market_probability=prediction_data.get("market_probability"),
            kelly_fraction=prediction_data.get("kelly_fraction"),
            reasoning=prediction_data.get("reasoning"),
        )
        self.session.add(db_pred)
        await self.session.commit()
        await self.session.refresh(db_pred)
        return db_pred

    async def get_predictions(self, limit: int = 100) -> List[DBPrediction]:
        from sqlalchemy.orm import selectinload
        query = select(DBPrediction).options(selectinload(DBPrediction.feedbacks)).order_by(DBPrediction.created_at.desc()).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_prediction_by_id(self, pred_id: int) -> Optional[DBPrediction]:
        query = select(DBPrediction).where(DBPrediction.id == pred_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def save_feedback(self, feedback_data: dict) -> DBFeedback:
        db_fb = DBFeedback(
            prediction_id=feedback_data["prediction_id"],
            actual_movement=feedback_data["actual_movement"],
            correct=feedback_data["correct"],
        )
        self.session.add(db_fb)
        await self.session.commit()
        await self.session.refresh(db_fb)
        return db_fb

    async def get_feedbacks(self, limit: int = 100) -> List[DBFeedback]:
        query = select(DBFeedback).order_by(DBFeedback.created_at.desc()).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update_statistic(self, name: str, value: float) -> DBStatistics:
        query = select(DBStatistics).where(DBStatistics.metric_name == name)
        result = await self.session.execute(query)
        stat = result.scalar_one_or_none()
        if stat:
            stat.metric_value = value
            stat.updated_at = datetime.datetime.utcnow()
        else:
            stat = DBStatistics(metric_name=name, metric_value=value)
            self.session.add(stat)
        await self.session.commit()
        await self.session.refresh(stat)
        return stat

    async def get_statistics(self) -> List[DBStatistics]:
        query = select(DBStatistics)
        result = await self.session.execute(query)
        return list(result.scalars().all())
