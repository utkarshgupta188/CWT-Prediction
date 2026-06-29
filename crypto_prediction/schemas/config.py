from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    OPENROUTER_API_KEY: str = Field(default="dummy-key-for-testing-if-empty")
    MODEL_NAME: str = Field(default="meta-llama/llama-3-8b-instruct:free")
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///crypto_prediction.db")
    LOG_LEVEL: str = Field(default="INFO")
    PORT: int = Field(default=8000)
    HOST: str = Field(default="0.0.0.0")
    USE_MOCK_PREDICTOR: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
