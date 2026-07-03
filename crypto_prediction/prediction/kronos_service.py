import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from loguru import logger
from crypto_prediction.schemas.config import settings

# Add Kronos folder to path to import model classes
KRONOS_DIR = Path(__file__).parent / "Kronos"
if str(KRONOS_DIR) not in sys.path:
    sys.path.append(str(KRONOS_DIR))

# Singleton predictor instance
_predictor = None
_USE_MOCK = os.environ.get("USE_MOCK_PREDICTOR", "false").lower() in ("true", "1", "yes")


class MockKronosPredictor:
    """Trend-biased mock predictor using SMA crossover for when Kronos
    model weights are unavailable or GPU is not present.
    Produces realistic-looking predictions without requiring the actual model."""

    def predict(self, df: pd.DataFrame, x_timestamp=None, y_timestamp=None,
                pred_len: int = 5, verbose: bool = False) -> pd.DataFrame:
        close_prices = df["close"].astype(float).values

        # Simple Moving Average crossover signal
        sma_short = pd.Series(close_prices).rolling(window=10, min_periods=1).mean().values
        sma_long = pd.Series(close_prices).rolling(window=50, min_periods=1).mean().values

        last_close = close_prices[-1]
        trend_bias = 1.0 if sma_short[-1] > sma_long[-1] else -1.0

        # Generate synthetic future prices with trend + noise
        rng = np.random.default_rng(seed=int(last_close * 100) % 2**31)
        volatility = float(np.std(close_prices[-50:]) / np.mean(close_prices[-50:])) if len(close_prices) >= 50 else 0.002
        pct_changes = rng.normal(loc=trend_bias * volatility * 0.3, scale=volatility * 0.5, size=pred_len)

        predicted_closes = [last_close]
        for pct in pct_changes:
            predicted_closes.append(predicted_closes[-1] * (1.0 + pct))
        predicted_closes = predicted_closes[1:]  # remove the seed value

        result = pd.DataFrame({"close": predicted_closes})
        if y_timestamp is not None:
            result["timestamp"] = y_timestamp.values[:pred_len] if hasattr(y_timestamp, 'values') else y_timestamp[:pred_len]

        logger.info(f"MockKronosPredictor: Generated {pred_len} predictions (trend_bias={trend_bias:+.0f}, vol={volatility:.5f})")
        return result


def get_kronos_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor

    if _USE_MOCK:
        logger.warning("USE_MOCK_PREDICTOR=True → using MockKronosPredictor (SMA crossover fallback)")
        _predictor = MockKronosPredictor()
        return _predictor

    logger.info("Initializing Kronos model and tokenizer...")
    try:
        from model.kronos import Kronos, KronosTokenizer, KronosPredictor
    except ImportError as e:
        logger.warning(f"Failed to import Kronos: {e}. Falling back to MockKronosPredictor.")
        _predictor = MockKronosPredictor()
        return _predictor

    # Determine device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    logger.info(f"Using device for Kronos inference: {device}")

    try:
        # Load from Hugging Face
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", token=False)
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small", token=False)

        _predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
        logger.info("Kronos predictor initialized successfully.")
    except Exception as e:
        logger.warning(f"Error loading Kronos models from HuggingFace: {e}. Falling back to MockKronosPredictor.")
        _predictor = MockKronosPredictor()

    return _predictor


async def predict_next_movement(df: pd.DataFrame, pred_len: int = 5) -> dict:
    """
    Predict price movement using the last candles of the dataframe.
    Works with both real Kronos and MockKronosPredictor.
    """
    if len(df) < 100:
        raise ValueError(f"Insufficient candles to run prediction. Required at least 100, got {len(df)}")

    # Use only last 512 rows as that is max_context size
    input_df = df.iloc[-512:].copy()

    # Historical timestamps
    x_ts = pd.Series(pd.to_datetime(input_df['timestamp'])).reset_index(drop=True)

    # Generate future timestamps
    last_ts = x_ts.iloc[-1]

    # Determine the frequency of the candles
    if len(x_ts) >= 2:
        freq = x_ts.iloc[-1] - x_ts.iloc[-2]
    else:
        freq = pd.Timedelta(minutes=5)

    y_ts_raw = pd.date_range(start=last_ts + freq, periods=pred_len, freq=freq)
    y_ts = pd.Series(pd.to_datetime(y_ts_raw)).reset_index(drop=True)

    # Fetch predictor (real or mock)
    predictor = get_kronos_predictor()

    # Run the predict method in executor since it may be CPU/GPU heavy
    import asyncio
    from functools import partial

    loop = asyncio.get_event_loop()
    pred_df = await loop.run_in_executor(
        None,
        partial(
            predictor.predict,
            df=input_df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=pred_len,
            verbose=False
        )
    )

    # Determine direction
    last_close = float(input_df['close'].iloc[-1])
    pred_final_close = float(pred_df['close'].iloc[-1])

    direction = "UP" if pred_final_close > last_close else "DOWN"

    # Confidence based on prediction trajectory
    pct_change = (pred_final_close - last_close) / last_close
    confidence = float(min(0.99, max(0.51, 0.5 + abs(pct_change) * 20.0)))
    probability = confidence if direction == "UP" else (1.0 - confidence)

    return {
        "direction": direction,
        "confidence": round(confidence, 4),
        "probability": round(probability, 4),
        "predicted_price": round(float(pred_final_close), 4)
    }
