"""
EnerVision AI - Core Configuration
All settings are loaded from environment variables or .env file.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────
    APP_NAME: str = "EnerVision AI"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # ── Database ──────────────────────────────────────────────────────────
    DATABASE_URL: str = (
        "postgresql+asyncpg://neondb_owner:npg_u7kX3tKhCIqB"
        "@ep-icy-dawn-aqzg7m6z-pooler.c-8.us-east-1.aws.neon.tech"
        "/neondb"
    )
    DATABASE_URL_SYNC: str = (
        "postgresql://neondb_owner:npg_u7kX3tKhCIqB"
        "@ep-icy-dawn-aqzg7m6z-pooler.c-8.us-east-1.aws.neon.tech"
        "/neondb?sslmode=require&channel_binding=require"
    )
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_ECHO: bool = False

    # ── JWT ───────────────────────────────────────────────────────────────
    SECRET_KEY: str = "enervision-super-secret-key-change-in-production-2024"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── CORS ──────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "https://enervision.ai",
    ]

    # ── Rate Limiting ─────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # ── File Upload ───────────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 50
    UPLOAD_DIR: str = "uploads"

    # ── ML Pipeline ───────────────────────────────────────────────────────
    ML_MODELS_DIR: str = "ml/outputs/models"
    ML_OUTPUTS_DIR: str = "ml/outputs"

    # ── Admin ─────────────────────────────────────────────────────────────
    FIRST_SUPERUSER_EMAIL: str = "admin@enervision.ai"
    FIRST_SUPERUSER_PASSWORD: str = "admin123"

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
