import pytest
from crypto_prediction.risk.kelly import KellyCalculator


def test_kelly_calculator_yes():
    res = KellyCalculator.calculate(market_prob=0.5, model_prob=0.7, multiplier=1.0)
    assert res["recommended_direction"] == "YES"
    assert res["edge"] == 0.2
    assert res["kelly_fraction"] == 0.4
    assert res["recommended_position_size"] == 0.4


def test_kelly_calculator_no():
    res = KellyCalculator.calculate(market_prob=0.6, model_prob=0.3, multiplier=1.0)
    assert res["recommended_direction"] == "NO"
    assert res["edge"] == 0.3
    assert res["kelly_fraction"] == 0.5
    assert res["recommended_position_size"] == 0.5
