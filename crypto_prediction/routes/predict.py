from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from crypto_prediction.hermes import HermesSupervisorAgent
from crypto_prediction.hermes.agents import HermesFeedbackAgent
from crypto_prediction.database.repository import AsyncSessionLocal, PredictionRepository
from crypto_prediction.utils.logger import setup_logger
from crypto_prediction.schemas.config import settings

logger = setup_logger(settings.LOG_LEVEL)
router = APIRouter()
supervisor = HermesSupervisorAgent()
feedback_agent = HermesFeedbackAgent()

async def get_repo():
    async with AsyncSessionLocal() as session:
        yield PredictionRepository(session)

class PredictRequest(BaseModel):
    symbol: str = Field(..., example="BTCUSDT")
    interval: str = Field(default="5m", example="5m")
    limit: int = Field(default=1000, ge=100, le=1000)

class PredictResponse(BaseModel):
    prediction_id: int
    symbol: str
    prediction: str
    confidence: float
    market_probability: float
    kelly: float
    reasoning: str

class FeedbackRequest(BaseModel):
    prediction_id: int
    actual_movement: str = Field(..., pattern="^(UP|DOWN)$", example="UP")

@router.post("/predict", response_model=PredictResponse)
async def predict_market(req: PredictRequest, repo: PredictionRepository = Depends(get_repo)):
    """
    Triggers the multi-agent prediction research pipeline.
    """
    try:
        result = await supervisor.execute_prediction_flow(
            repo=repo,
            symbol=req.symbol,
            interval=req.interval,
            limit=req.limit
        )
        return result
    except Exception as e:
        logger.error(f"Prediction flow failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_history(limit: int = 100, repo: PredictionRepository = Depends(get_repo)):
    """
    Retrieve prediction history with matching feedback status.
    """
    predictions = await repo.get_predictions(limit=limit)
    history = []
    for p in predictions:
        # Check if there is any feedback associated
        # Since we have relationship or can run a query
        feedback_status = None
        if p.feedbacks:
            fb = p.feedbacks[0]
            feedback_status = {
                "actual_movement": fb.actual_movement,
                "correct": fb.correct,
                "created_at": fb.created_at.isoformat()
            }
            
        history.append({
            "id": p.id,
            "symbol": p.symbol,
            "interval": p.interval,
            "prediction": p.prediction_direction,
            "confidence": p.confidence,
            "market_probability": p.market_probability,
            "kelly": p.kelly_fraction,
            "reasoning": p.reasoning,
            "created_at": p.created_at.isoformat(),
            "feedback": feedback_status
        })
    return history

@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest, repo: PredictionRepository = Depends(get_repo)):
    """
    Submit actual price movement for a prediction to evaluate accuracy.
    """
    try:
        feedback_result = await feedback_agent.execute(
            repo=repo,
            prediction_id=req.prediction_id,
            actual_movement=req.actual_movement
        )
        return feedback_result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Feedback processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics")
async def get_statistics(repo: PredictionRepository = Depends(get_repo)):
    """
    Retrieve run-time prediction accuracy and volume stats.
    """
    stats = await repo.get_statistics()
    return {s.metric_name: s.metric_value for s in stats}
