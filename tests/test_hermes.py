import pytest
import json
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch

from crypto_prediction.hermes.tools.market_data_tool import MARKET_DATA_SCHEMA
from crypto_prediction.hermes.tools.prediction_tool import PREDICTION_SCHEMA
from crypto_prediction.hermes.tools.risk_tool import RISK_SCHEMA
from crypto_prediction.hermes.tools.search_polymarket_tool import SEARCH_POLYMARKET_SCHEMA
from crypto_prediction.hermes.tools.search_kalshi_tool import SEARCH_KALSHI_SCHEMA
from crypto_prediction.hermes.tools.feedback_tool import FEEDBACK_SCHEMA


class TestHermesToolSchemas:
    """Tool schemas must have required parameters."""

    def test_market_data_schema(self):
        assert MARKET_DATA_SCHEMA["name"] == "get_market_data"
        props = MARKET_DATA_SCHEMA["parameters"]["properties"]
        assert "symbol" in props
        assert "interval" in props
        assert "limit" in props
        assert "symbol" in MARKET_DATA_SCHEMA["parameters"]["required"]
        assert "interval" in MARKET_DATA_SCHEMA["parameters"]["required"]

    def test_prediction_schema(self):
        assert PREDICTION_SCHEMA["name"] == "get_prediction"
        props = PREDICTION_SCHEMA["parameters"]["properties"]
        assert "candles" in props
        assert "pred_len" in props
        assert "candles" in PREDICTION_SCHEMA["parameters"]["required"]

    def test_risk_schema(self):
        assert RISK_SCHEMA["name"] == "calculate_risk"
        props = RISK_SCHEMA["parameters"]["properties"]
        assert "market_probability" in props
        assert "model_probability" in props
        assert "market_probability" in RISK_SCHEMA["parameters"]["required"]
        assert "model_probability" in RISK_SCHEMA["parameters"]["required"]

    def test_search_polymarket_schema(self):
        assert SEARCH_POLYMARKET_SCHEMA["name"] == "search_polymarket"
        assert "limit" in SEARCH_POLYMARKET_SCHEMA["parameters"]["properties"]

    def test_search_kalshi_schema(self):
        assert SEARCH_KALSHI_SCHEMA["name"] == "search_kalshi"
        assert "limit" in SEARCH_KALSHI_SCHEMA["parameters"]["properties"]

    def test_feedback_schema(self):
        assert FEEDBACK_SCHEMA["name"] == "save_feedback"
        props = FEEDBACK_SCHEMA["parameters"]["properties"]
        assert "prediction_id" in props
        assert "actual_movement" in props
        assert props["actual_movement"]["enum"] == ["UP", "DOWN"]


@pytest.mark.asyncio
async def test_market_data_agent_invokes_tool():
    from crypto_prediction.hermes.agents.market_data_agent import HermesMarketDataAgent

    agent = HermesMarketDataAgent()

    assert "get_market_data" in agent.allowed_tools
    assert len(agent.allowed_tools) == 1


@pytest.mark.asyncio
async def test_prediction_agent_invokes_tool():
    from crypto_prediction.hermes.agents.prediction_agent import HermesPredictionAgent

    agent = HermesPredictionAgent()

    assert "get_prediction" in agent.allowed_tools
    assert len(agent.allowed_tools) == 1


@pytest.mark.asyncio
async def test_risk_agent_invokes_tool():
    from crypto_prediction.hermes.agents.risk_agent import HermesRiskAgent

    agent = HermesRiskAgent()

    assert "calculate_risk" in agent.allowed_tools
    assert len(agent.allowed_tools) == 1


@pytest.mark.asyncio
async def test_search_agent_has_correct_tools():
    from crypto_prediction.hermes.agents.search_agent import HermesSearchAgent

    agent = HermesSearchAgent()

    assert "search_polymarket" in agent.allowed_tools
    assert "search_kalshi" in agent.allowed_tools
    assert len(agent.allowed_tools) == 2


@pytest.mark.asyncio
async def test_search_agent_no_polymarket_tool_to_prediction():
    from crypto_prediction.hermes.agents.search_agent import HermesSearchAgent
    from crypto_prediction.hermes.agents.prediction_agent import HermesPredictionAgent

    search_agent = HermesSearchAgent()
    prediction_agent = HermesPredictionAgent()

    assert "search_polymarket" not in prediction_agent.allowed_tools
    assert "search_kalshi" not in prediction_agent.allowed_tools
    assert "get_prediction" not in search_agent.allowed_tools


@pytest.mark.asyncio
async def test_feedback_agent_invokes_tool():
    from crypto_prediction.hermes.agents.feedback_agent import HermesFeedbackAgent

    agent = HermesFeedbackAgent()

    assert "save_feedback" in agent.allowed_tools
    assert len(agent.allowed_tools) == 1


class TestHermesMemory:
    def test_add_and_retrieve(self):
        from crypto_prediction.hermes.memory import HermesMemory

        mem = HermesMemory(max_history=5)
        mem.add_prediction(
            symbol="BTCUSDT", interval="5m",
            prediction_direction="UP", confidence=0.85,
            model_probability=0.85, market_probability=0.6,
            kelly_fraction=0.25, reasoning="Bullish trend",
        )
        context = mem.get_recent_context(limit=5)
        assert "BTCUSDT" in context
        assert "UP" in context
        assert "85.00%" in context
        assert "0.2500" in context

    def test_max_history(self):
        from crypto_prediction.hermes.memory import HermesMemory

        mem = HermesMemory(max_history=3)
        for i in range(5):
            mem.add_prediction(
                symbol=f"PAIR{i}", interval="5m",
                prediction_direction="UP", confidence=0.5,
                model_probability=0.5, market_probability=0.5,
                kelly_fraction=0.0, reasoning="",
            )
        assert len(mem.get_all()) == 3
        assert mem.get_all()[0]["symbol"] == "PAIR2"
        assert mem.get_all()[-1]["symbol"] == "PAIR4"

    def test_empty_memory(self):
        from crypto_prediction.hermes.memory import HermesMemory

        mem = HermesMemory()
        context = mem.get_recent_context()
        assert "No previous predictions" in context

    def test_accuracy_updates(self):
        from crypto_prediction.hermes.memory import HermesMemory

        mem = HermesMemory()
        mem.add_prediction(
            symbol="BTCUSDT", interval="5m",
            prediction_direction="UP", confidence=0.85,
            model_probability=0.85, market_probability=0.6,
            kelly_fraction=0.25, reasoning="Test", accuracy=0.75,
        )
        context = mem.get_recent_context()
        assert "75.00%" in context

    def test_serialization(self):
        from crypto_prediction.hermes.memory import HermesMemory

        mem = HermesMemory(max_history=10)
        mem.add_prediction(
            symbol="ETHUSDT", interval="1h",
            prediction_direction="DOWN", confidence=0.7,
            model_probability=0.3, market_probability=0.5,
            kelly_fraction=0.1, reasoning="Bearish",
        )
        data = mem.to_json()
        restored = HermesMemory.from_json(data)
        assert len(restored.get_all()) == 1
        assert restored.get_all()[0]["symbol"] == "ETHUSDT"


class TestHermesSupervisorStepMethods:
    @pytest.mark.asyncio
    async def test_resolve_market_probability_finds_match(self):
        from crypto_prediction.hermes.supervisor import HermesSupervisorAgent

        supervisor = HermesSupervisorAgent()
        markets = [
            {"asset": "BTC", "market_probability": 0.65, "question": "Will BTC reach 100k?"},
            {"asset": "ETH", "market_probability": 0.45, "question": "Will ETH reach 5k?"},
        ]
        prob, question = supervisor._resolve_market_probability(markets, "BTCUSDT")
        assert prob == 0.65
        assert "100k" in question

    @pytest.mark.asyncio
    async def test_resolve_market_probability_defaults(self):
        from crypto_prediction.hermes.supervisor import HermesSupervisorAgent

        supervisor = HermesSupervisorAgent()
        prob, question = supervisor._resolve_market_probability([], "SOLUSDT")
        assert prob == 0.5
        assert "No active" in question

    def test_fallback_reasoning(self):
        from crypto_prediction.hermes.supervisor import HermesSupervisorAgent

        supervisor = HermesSupervisorAgent()
        reasoning = supervisor._fallback_reasoning(
            "BTCUSDT", "5m",
            {"direction": "UP", "confidence": 0.85, "probability": 0.85, "predicted_price": 62000.0},
            0.6,
            {"recommended_direction": "YES", "recommended_position_size": 0.25, "risk_level": "MEDIUM"},
        )
        assert "UP" in reasoning
        assert "85.00%" in reasoning
        assert "25.00%" in reasoning
        assert "MEDIUM" in reasoning

    def test_build_reasoning_prompt(self):
        from crypto_prediction.hermes.supervisor import HermesSupervisorAgent

        supervisor = HermesSupervisorAgent()
        prompt = supervisor._build_reasoning_prompt(
            "BTCUSDT", "5m",
            {"direction": "UP", "confidence": 0.85, "probability": 0.85, "predicted_price": 62000.0},
            0.6,
            {"recommended_direction": "YES", "recommended_position_size": 0.25, "risk_level": "MEDIUM"},
            "Will BTC reach 100k?",
        )
        assert "BTCUSDT" in prompt
        assert "5m" in prompt
        assert "UP" in prompt
        assert "YES" in prompt
        assert "Will BTC reach 100k?" in prompt
        assert "150 words" in prompt


@pytest.mark.asyncio
async def test_hermes_supervisor_full_flow(mocker):
    from crypto_prediction.hermes import HermesSupervisorAgent

    supervisor = HermesSupervisorAgent()

    mock_repo = MagicMock()
    mock_db_pred = MagicMock()
    mock_db_pred.id = 42
    mock_repo.save_prediction = AsyncMock(return_value=mock_db_pred)

    mock_search_markets = [
        {"asset": "BTC", "market_probability": 0.65, "question": "Will BTC reach 100k?"},
    ]

    test_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-06-29", periods=100, freq="5min"),
        "open": [60000.0] * 100, "high": [60100.0] * 100,
        "low": [59900.0] * 100, "close": [60050.0] * 100,
        "volume": [10.0] * 100,
    })

    mocker.patch.object(supervisor.search_agent, "execute",
                        AsyncMock(return_value=mock_search_markets))
    mocker.patch.object(supervisor.market_data_agent, "execute",
                        AsyncMock(return_value=test_df))
    mocker.patch.object(supervisor.prediction_agent, "execute",
                        AsyncMock(return_value={
                            "direction": "UP", "confidence": 0.85,
                            "probability": 0.85, "predicted_price": 62000.0
                        }))
    mocker.patch.object(supervisor.risk_agent, "execute",
                        AsyncMock(return_value={
                            "recommended_direction": "YES",
                            "recommended_position_size": 0.25,
                            "risk_level": "MEDIUM", "edge": 0.2,
                            "kelly_fraction": 0.5
                        }))
    mocker.patch.object(supervisor, "_step_reasoning",
                        AsyncMock(return_value="Bullish momentum suggests upward movement."))

    result = await supervisor.execute_prediction_flow(
        repo=mock_repo,
        symbol="BTCUSDT",
        interval="5m",
        limit=100
    )

    assert result["prediction_id"] == 42
    assert result["symbol"] == "BTCUSDT"
    assert result["prediction"] == "UP"
    assert result["confidence"] == 0.85
    assert result["market_probability"] == 0.65
    assert result["kelly"] == 0.25
    assert result["reasoning"] == "Bullish momentum suggests upward movement."

    mock_repo.save_prediction.assert_called_once()
    saved_data = mock_repo.save_prediction.call_args[0][0]
    assert saved_data["symbol"] == "BTCUSDT"
    assert saved_data["prediction_direction"] == "UP"
    assert saved_data["kelly_fraction"] == 0.25
    assert saved_data["market_probability"] == 0.65
    supervisor.search_agent.execute.assert_called_once()
    supervisor.market_data_agent.execute.assert_called_once_with("BTCUSDT", "5m", 100)
    supervisor.prediction_agent.execute.assert_called_once()


@pytest.mark.asyncio
async def test_hermes_supervisor_empty_market_data_raises(mocker):
    from crypto_prediction.hermes import HermesSupervisorAgent

    supervisor = HermesSupervisorAgent()

    mock_repo = MagicMock()
    mock_search_markets = [
        {"asset": "BTC", "market_probability": 0.65, "question": "Will BTC reach 100k?"},
    ]
    empty_df = pd.DataFrame()

    mocker.patch.object(supervisor.search_agent, "execute",
                        AsyncMock(return_value=mock_search_markets))
    mocker.patch.object(supervisor.market_data_agent, "execute",
                        AsyncMock(return_value=empty_df))

    with pytest.raises(ValueError, match="No OHLCV data returned"):
        await supervisor.execute_prediction_flow(
            repo=mock_repo, symbol="BTCUSDT", interval="5m", limit=100
        )


@pytest.mark.asyncio
async def test_hermes_supervisor_search_failure_continues(mocker):
    from crypto_prediction.hermes import HermesSupervisorAgent

    supervisor = HermesSupervisorAgent()

    mock_repo = MagicMock()
    mock_db_pred = MagicMock()
    mock_db_pred.id = 1
    mock_repo.save_prediction = AsyncMock(return_value=mock_db_pred)

    test_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-06-29", periods=100, freq="5min"),
        "open": [60000.0] * 100, "high": [60100.0] * 100,
        "low": [59900.0] * 100, "close": [60050.0] * 100,
        "volume": [10.0] * 100,
    })

    mocker.patch.object(supervisor.search_agent, "execute",
                        AsyncMock(side_effect=Exception("Search failed")))
    mocker.patch.object(supervisor.market_data_agent, "execute",
                        AsyncMock(return_value=test_df))
    mocker.patch.object(supervisor.prediction_agent, "execute",
                        AsyncMock(return_value={
                            "direction": "DOWN", "confidence": 0.75,
                            "probability": 0.25, "predicted_price": 59000.0
                        }))
    mocker.patch.object(supervisor.risk_agent, "execute",
                        AsyncMock(return_value={
                            "recommended_direction": "NO",
                            "recommended_position_size": 0.15,
                            "risk_level": "MEDIUM", "edge": 0.1,
                            "kelly_fraction": 0.3
                        }))
    mocker.patch.object(supervisor, "_step_reasoning",
                        AsyncMock(return_value="Default reasoning."))

    result = await supervisor.execute_prediction_flow(
        repo=mock_repo, symbol="BTCUSDT", interval="5m", limit=100
    )

    assert result["prediction"] == "DOWN"
    assert result["market_probability"] == 0.5


def test_hermes_memory_in_supervisor():
    from crypto_prediction.hermes.supervisor import HermesSupervisorAgent

    supervisor = HermesSupervisorAgent()
    supervisor.memory.add_prediction(
        symbol="BTCUSDT", interval="5m",
        prediction_direction="UP", confidence=0.85,
        model_probability=0.85, market_probability=0.6,
        kelly_fraction=0.25, reasoning="Bullish",
    )
    context = supervisor.memory.get_recent_context()
    assert "BTCUSDT" in context
    assert "UP" in context
    assert "0.2500" in context
