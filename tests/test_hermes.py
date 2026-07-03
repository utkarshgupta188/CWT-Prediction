import pytest
import json

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
    async def test_resolve_market_probability_no_match_raises(self):
        from crypto_prediction.hermes.supervisor import HermesSupervisorAgent

        supervisor = HermesSupervisorAgent()
        with pytest.raises(ValueError, match="No prediction market found"):
            supervisor._resolve_market_probability([], "SOLUSDT")


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


@pytest.mark.asyncio
async def test_multi_timeframe_orchestrator():
    from crypto_prediction.hermes.multi_timeframe import MultiTimeframeOrchestrator
    
    # Simple unit test of analysis logic
    orchestrator = MultiTimeframeOrchestrator()
    mock_results = {
        "1m": {"prediction": "UP", "confidence": 0.8, "kelly": 0.1},
        "5m": {"prediction": "UP", "confidence": 0.7, "kelly": 0.15},
        "15m": {"prediction": "UP", "confidence": 0.9, "kelly": 0.2}
    }
    
    analysis = orchestrator._analyze_results("BTCUSDT", mock_results)
    assert analysis["micro_validation"] == "CONFIRMED"
    assert analysis["arbitrage_detected"] is False
    assert analysis["consensus_direction"] == "UP"
    assert analysis["consensus_strength"] == 1.0
    assert analysis["composite_kelly_recommendation"] > 0.0

    # Test arbitrage detection
    mock_results_arb = {
        "1m": {"prediction": "DOWN", "confidence": 0.8, "kelly": 0.0},
        "5m": {"prediction": "DOWN", "confidence": 0.7, "kelly": 0.0},
        "15m": {"prediction": "UP", "confidence": 0.9, "kelly": 0.2}
    }
    analysis_arb = orchestrator._analyze_results("BTCUSDT", mock_results_arb)
    assert analysis_arb["arbitrage_detected"] is True
    assert "MEAN_REVERSION" in analysis_arb["arbitrage_type"]

