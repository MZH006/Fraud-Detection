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
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # Redis Cache
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    CACHE_TTL: int = 300  # 5 minutes

    # Model
    MODEL_PATH: str = "./artifacts/fraud_detection_model_final.pkl"
    # None = use the threshold saved with the model artifact (chosen on validation).
    MODEL_THRESHOLD: Optional[float] = None

    # Feature Engineering Thresholds
    HIGH_AMOUNT_THRESHOLD: float = 200.0
    HIGH_DEVIATION_THRESHOLD: float = 2.0
    HIGH_RISK_FRAUD_RATE: float = 0.05

    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    # Security
    API_KEY_ENABLED: bool = False
    API_KEY: Optional[str] = None
    CORS_ORIGINS: List[str] = ["*"]
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds

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
