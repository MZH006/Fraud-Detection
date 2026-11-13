"""
Configuration management for Fraud Detection API
Uses pydantic-settings for type-safe configuration with environment variables
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional, List


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    APP_NAME: str = "Fraud Detection API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4

    # Database
    SUPABASE_URL: str
    SUPABASE_KEY: str

    # Model
    MODEL_PATH: str = "./artifacts/fraud_detection_model_final.pkl"
    # None = use the threshold saved with the model artifact (chosen on validation).
    MODEL_THRESHOLD: Optional[float] = None

    CORS_ORIGINS: List[str] = ["*"]

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # 'json' or 'text'

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure settings are only loaded once.

    Returns:
        Settings: Application settings instance
    """
    return Settings()
