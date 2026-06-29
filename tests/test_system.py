import json
import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock
from crypto_prediction.risk.kelly import KellyCalculator
from crypto_prediction.providers.binance_provider import BinanceProvider
from crypto_prediction.hermes.agents import HermesSearchAgent, HermesPredictionAgent, HermesFeedbackAgent

def test_kelly_calculator_yes():
    # If model prob > market prob
    # model = 0.7, market = 0.5
    res = KellyCalculator.calculate(market_prob=0.5, model_prob=0.7, multiplier=1.0)
    assert res["recommended_direction"] == "YES"
    # edge = 0.2
    assert res["edge"] == 0.2
    # kelly_fraction = edge / (1 - market_prob) = 0.2 / 0.5 = 0.4
    assert res["kelly_fraction"] == 0.4
    assert res["recommended_position_size"] == 0.4

def test_kelly_calculator_no():
    # If model prob < market prob
    # model = 0.3, market = 0.6
    res = KellyCalculator.calculate(market_prob=0.6, model_prob=0.3, multiplier=1.0)
    assert res["recommended_direction"] == "NO"
    # edge = market_prob - model_prob = 0.3
    assert res["edge"] == 0.3
    # kelly_fraction = edge / market_prob = 0.3 / 0.6 = 0.5
    assert res["kelly_fraction"] == 0.5
    assert res["recommended_position_size"] == 0.5

@pytest.mark.asyncio
async def test_binance_provider_get_klines(mocker):
    # Mock httpx AsyncClient request
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        [1719628800000, "61000.0", "61500.0", "60800.0", "61200.0", "100.0", 1719629099999, "6100000.0", 50, "50.0", "3050000.0", "0"]
    ]
    
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.request = AsyncMock(return_value=mock_response)
    mocker.patch("httpx.AsyncClient", return_value=mock_client)
    
    provider = BinanceProvider()
    df = await provider.get_klines("BTCUSDT", "5m", 1)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["open"] == 61000.0
    assert df.iloc[0]["close"] == 61200.0

@pytest.mark.asyncio
async def test_prediction_agent_execute(mocker):
    agent = HermesPredictionAgent()
    mock_predict = AsyncMock(return_value={
        "direction": "UP",
        "confidence": 0.85,
        "probability": 0.85,
        "predicted_price": 62000.0
    })
    mocker.patch("crypto_prediction.prediction.kronos_service.predict_next_movement", mock_predict)
    
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-06-29", periods=100, freq="5min"),
        "open": [60000.0] * 100,
        "high": [60100.0] * 100,
        "low": [59900.0] * 100,
        "close": [60050.0] * 100,
        "volume": [10.0] * 100
    })
    
    res = await agent.execute(df)
    assert res["direction"] == "UP"
    assert res["confidence"] == 0.85

@pytest.mark.asyncio
async def test_feedback_agent_execute(mocker):
    mock_feedback_result = {
        "success": True,
        "prediction_id": 1,
        "predicted_direction": "UP",
        "actual_movement": "UP",
        "correct": True,
        "correct_predictions": 1,
        "total_predictions": 1,
        "accuracy": 1.0
    }
    mocker.patch(
        "crypto_prediction.hermes.agents.feedback_agent.registry.dispatch",
        return_value=json.dumps(mock_feedback_result)
    )
    
    agent = HermesFeedbackAgent()
    res = await agent.execute(repo=None, prediction_id=1, actual_movement="UP")
    
    assert res["correct"] is True
    assert res["accuracy"] == 1.0
