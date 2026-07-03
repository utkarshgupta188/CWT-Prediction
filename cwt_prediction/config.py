import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class Config:
    openrouter_api_key: str
    kronos_model: str = "NeoQuasar/Kronos-mini"
    kronos_tokenizer: str = "NeoQuasar/Kronos-Tokenizer-2k"
    lookback_bars: int = 400
    forecast_horizon: int = 6  # 6 bars * 5 min = 30 min horizon
    kelly_max_fraction: float = 0.25
    db_path: str = "data/predictions.db"
    log_file: str = "logs/pipeline.log"

def load_config() -> Config:
    load_dotenv()
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    # Note: OpenRouter API key is optional for core math/model forecasting, 
    # but required for advanced LLM parsing if used.
    
    return Config(
        openrouter_api_key=openrouter_key,
        kronos_model=os.getenv("KRONOS_MODEL", "NeoQuasar/Kronos-mini"),
        kronos_tokenizer=os.getenv("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-2k"),
        lookback_bars=int(os.getenv("LOOKBACK_BARS", "400")),
        forecast_horizon=int(os.getenv("FORECAST_HORIZON", "6")),
        kelly_max_fraction=float(os.getenv("KELLY_MAX_FRACTION", "0.25")),
        db_path=os.getenv("DB_PATH", "data/predictions.db"),
        log_file=os.getenv("LOG_FILE", "logs/pipeline.log")
    )
