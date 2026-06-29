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

def get_kronos_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor

    if settings.USE_MOCK_PREDICTOR:
        logger.info("USE_MOCK_PREDICTOR is enabled. Instantiating MockKronosPredictor.")
        _predictor = MockKronosPredictor()
        return _predictor

    logger.info("Initializing Kronos model and tokenizer...")
    try:
        from model.kronos import Kronos, KronosTokenizer, KronosPredictor
    except ImportError as e:
        logger.error(f"Failed to import Kronos: {e}. Make sure Kronos directory is correctly cloned and dependencies are installed.")
        raise e

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
        logger.error(f"Error loading Kronos models from Hugging Face: {e}")
        # Return a mock predictor for test environments or if loading fails
        logger.warning("Fallback/Mock predictor will be used if remote load failed.")
        _predictor = MockKronosPredictor()
        
    return _predictor

class MockKronosPredictor:
    """Fallback predictor if HuggingFace weights cannot be loaded or during tests."""
    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
        logger.warning("Mock predictor is being used.")
        last_close = df['close'].iloc[-1]
        
        # Determine trend-based bias from recent candles
        recent_trend = 0.0
        if len(df) >= 10:
            recent_trend = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
        mean_bias = np.clip(recent_trend * 0.1, -0.001, 0.001)

        # Generate some synthetic future prices dynamically
        seed = int(abs(last_close * 100000 + len(df))) % (2**32 - 1)
        np.random.seed(seed)
        prices = [last_close]
        for _ in range(pred_len):
            change = (np.random.normal(mean_bias, 0.002)) * last_close
            prices.append(prices[-1] + change)
            
        prices = prices[1:]
        pred_df = pd.DataFrame(index=y_timestamp)
        pred_df['open'] = prices
        pred_df['high'] = [p * 1.002 for p in prices]
        pred_df['low'] = [p * 0.998 for p in prices]
        pred_df['close'] = prices
        pred_df['volume'] = 0.0
        pred_df['amount'] = 0.0
        return pred_df

async def predict_next_movement(df: pd.DataFrame, pred_len: int = 5) -> dict:
    """
    Predict price movement using the last candles of the dataframe.
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
    # To compute timedelta correctly:
    if len(x_ts) >= 2:
        freq = x_ts.iloc[-1] - x_ts.iloc[-2]
    else:
        freq = pd.Timedelta(minutes=5)
        
    y_ts_raw = pd.date_range(start=last_ts + freq, periods=pred_len, freq=freq)
    y_ts = pd.Series(pd.to_datetime(y_ts_raw)).reset_index(drop=True)
    
    # Fetch predictor
    predictor = get_kronos_predictor()
    
    # Call prediction (run in executor since it's a CPU/GPU heavy synchronous call)
    import asyncio
    from functools import partial
    
    # Run the predict method
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
    last_close = input_df['close'].iloc[-1]
    pred_final_close = pred_df['close'].iloc[-1]
    
    direction = "UP" if pred_final_close > last_close else "DOWN"
    
    # Simulate a confidence/probability based on prediction trajectory volatility
    # In a real model, we can sample multiple times to get exact probabilistic counts.
    # For efficiency and clean functionality:
    pct_change = (pred_final_close - last_close) / last_close
    confidence = float(min(0.99, max(0.51, 0.5 + abs(pct_change) * 20.0)))
    probability = confidence if direction == "UP" else (1.0 - confidence)
    
    return {
        "direction": direction,
        "confidence": round(confidence, 4),
        "probability": round(probability, 4),
        "predicted_price": round(float(pred_final_close), 4)
    }
