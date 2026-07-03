import os
import sys
import logging
import pandas as pd
import numpy as np
import torch
from datetime import timedelta

# Ensure the parent directory is in path so model folder can be imported as "model"
forecaster_dir = os.path.dirname(os.path.abspath(__file__))
if forecaster_dir not in sys.path:
    sys.path.insert(0, forecaster_dir)

# Now we can import the model classes
try:
    from model.kronos import Kronos, KronosTokenizer, KronosPredictor
except ImportError:
    # Fallback to local import if path resolution is tricky
    from .model.kronos import Kronos, KronosTokenizer, KronosPredictor

from .models import ForecastResult

logger = logging.getLogger("cwt_prediction.forecaster")

class KronosForecaster:
    def __init__(self, model_name: str = "NeoQuasar/Kronos-mini", tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-2k"):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.tokenizer = None
        self.model = None
        self.predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Lazily load tokenizer and model from HF Hub."""
        if self.predictor is not None:
            return

        logger.info(f"Loading tokenizer '{self.tokenizer_name}' and model '{self.model_name}' on {self.device}...")
        try:
            self.tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name)
            self.model = Kronos.from_pretrained(self.model_name)
            self.predictor = KronosPredictor(self.model, self.tokenizer, device=self.device, max_context=512)
            logger.info("Kronos model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Kronos model from Hugging Face Hub: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load Kronos: {e}") from e

    def forecast(self, ohlcv_df: pd.DataFrame, horizon: int = 6) -> ForecastResult:
        """
        Runs forecasting on the input DataFrame.
        Args:
            ohlcv_df: DataFrame with 'timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount'
            horizon: forecast length (in bars)

        Returns:
            ForecastResult
        """
        self.load_model()

        if len(ohlcv_df) < 5:
            raise ValueError(f"Insufficient history data: got {len(ohlcv_df)} bars, need at least 5.")

        # Ensure correct types and sort by timestamp
        df = ohlcv_df.copy()
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df = df.sort_values('timestamps').reset_index(drop=True)

        # Prepare future timestamps
        last_timestamp = df['timestamps'].iloc[-1]
        time_diff = df['timestamps'].iloc[-1] - df['timestamps'].iloc[-2]
        
        # If timestamps are irregular, default to 5-min intervals
        if pd.isnull(time_diff) or time_diff.total_seconds() == 0:
            time_diff = timedelta(minutes=5)
            
        future_timestamps = [last_timestamp + (time_diff * i) for i in range(1, horizon + 1)]
        y_timestamp = pd.Series(future_timestamps)

        # Prepare x inputs
        x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]
        x_timestamp = df['timestamps']

        current_close = float(df['close'].iloc[-1])

        # Run Monte Carlo sampling to estimate probability
        # We run 10 separate single-sample path generations
        num_runs = 10
        up_votes = 0
        predicted_closes = []

        logger.info(f"Generating {num_runs} forecast paths using Kronos...")
        for run_idx in range(num_runs):
            try:
                # We use T=1.0 for temperature sampling and top_p=0.9
                pred_df = self.predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=horizon,
                    T=1.0,
                    top_p=0.9,
                    sample_count=1,
                    verbose=False
                )
                pred_close = float(pred_df['close'].iloc[-1])
                predicted_closes.append(pred_close)
                
                if pred_close > current_close:
                    up_votes += 1
            except Exception as e:
                logger.warning(f"Error during forecast run {run_idx}: {e}")
                continue

        if not predicted_closes:
            raise RuntimeError("All forecasting runs failed.")

        avg_predicted_close = float(np.mean(predicted_closes))
        p_up = up_votes / len(predicted_closes)

        # Determine majority direction
        if p_up >= 0.5:
            direction = "up"
            probability = p_up
        else:
            direction = "down"
            probability = 1.0 - p_up

        logger.info(f"Forecast complete: direction={direction}, probability={probability:.2f}, current_close={current_close:.2f}, pred_close={avg_predicted_close:.2f}")

        return ForecastResult(
            direction=direction,
            probability=probability,
            predicted_close=avg_predicted_close,
            current_close=current_close,
            horizon_bars=horizon,
            sample_count=len(predicted_closes)
        )
