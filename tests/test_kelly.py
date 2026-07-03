import pytest
from cwt_prediction.risk_manager.kelly import kelly_fraction

def test_kelly_basic_positive_edge():
    # p = 0.60, m = 0.50 -> raw fraction = (0.6 - 0.5) / (1.0 - 0.5) = 0.1 / 0.5 = 0.20
    assert pytest.approx(kelly_fraction(0.60, 0.50)) == 0.20

def test_kelly_no_edge():
    # p = 0.50, m = 0.50 -> 0.0 (no positive edge)
    assert kelly_fraction(0.50, 0.50) == 0.0

def test_kelly_negative_edge():
    # p = 0.40, m = 0.50 -> 0.0
    assert kelly_fraction(0.40, 0.50) == 0.0

def test_kelly_max_fraction_cap():
    # p = 0.90, m = 0.50 -> raw fraction = (0.9 - 0.5) / 0.5 = 0.8
    # max_fraction = 0.25 (default) -> should be capped at 0.25
    assert kelly_fraction(0.90, 0.50) == 0.25
    
    # max_fraction custom = 0.15 -> should cap at 0.15
    assert kelly_fraction(0.90, 0.50, max_fraction=0.15) == 0.15

def test_kelly_edge_cases():
    # Edge case: model prob very high but capped
    assert kelly_fraction(0.999, 0.50) == 0.25
    # Edge case: market prob very close to 1
    assert pytest.approx(kelly_fraction(0.999, 0.99)) == 0.09 # (0.999 - 0.99) / (1 - 0.99) = 0.009 / 0.01 = 0.9 -> capped at 0.25
    assert kelly_fraction(0.999, 0.99) == 0.25

def test_kelly_validation_errors():
    # Out of bounds probabilities
    with pytest.raises(ValueError):
        kelly_fraction(-0.1, 0.5)
    with pytest.raises(ValueError):
        kelly_fraction(1.5, 0.5)
    with pytest.raises(ValueError):
        kelly_fraction(0.6, -0.2)
    with pytest.raises(ValueError):
        kelly_fraction(0.6, 1.2)
        
    # Invalid max_fraction
    with pytest.raises(ValueError):
        kelly_fraction(0.6, 0.5, max_fraction=-0.1)
    with pytest.raises(ValueError):
        kelly_fraction(0.6, 0.5, max_fraction=1.5)
