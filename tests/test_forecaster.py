import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
from cwt_prediction.forecaster.kronos_wrapper import KronosForecaster
from cwt_prediction.forecaster.models import ForecastResult

@pytest.fixture
def mock_ohlcv_data():
    now = datetime.now(timezone.utc)
    rows = []
    # Create 10 mock 5-minute bars
    for i in range(10):
        rows.append({
            "timestamps": now - timedelta(minutes=5 * (10 - i)),
            "open": 100.0 + i,
            "high": 105.0 + i,
            "low": 95.0 + i,
            "close": 101.0 + i,
            "volume": 10.0 * i,
            "amount": 1000.0 * i
        })
    return pd.DataFrame(rows)

@patch("cwt_prediction.forecaster.kronos_wrapper.KronosPredictor")
@patch("cwt_prediction.forecaster.kronos_wrapper.Kronos")
@patch("cwt_prediction.forecaster.kronos_wrapper.KronosTokenizer")
def test_forecaster_success(mock_tokenizer, mock_model, mock_predictor_cls, mock_ohlcv_data):
    # Setup mocks
    mock_predictor = MagicMock()
    mock_predictor_cls.return_value = mock_predictor
    
    # Mock predict output DataFrame (it contains the predicted future path)
    # The last close is higher than input last close (110.0 vs 101.0 + 9 = 110.0, wait last index is 9 so close is 110.0)
    # Let's mock a predict DataFrame that goes up
    mock_pred_df = pd.DataFrame({
        "open": [111.0], "high": [115.0], "low": [109.0], "close": [112.0],
        "volume": [50.0], "amount": [5600.0]
    })
    mock_predictor.predict.return_value = mock_pred_df
    
    forecaster = KronosForecaster()
    # Force mock predictor assignment to avoid loading from hub
    forecaster.predictor = mock_predictor
    forecaster.device = "cpu"
    
    # Run forecast with horizon = 1
    result = forecaster.forecast(mock_ohlcv_data, horizon=1)
    
    assert isinstance(result, ForecastResult)
    # Since predicted close (112.0) > last close (110.0), direction should be up
    assert result.direction == "up"
    assert result.probability == 1.0 # Since all mock runs succeeded and were up
    assert result.predicted_close == 112.0
    assert result.current_close == 110.0
    assert result.horizon_bars == 1
    assert result.sample_count == 10
