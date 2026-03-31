from enum import StrEnum

from pydantic_settings import BaseSettings


class EngineType(StrEnum):
    RULE = "rule"
    ML = "ml"


class Settings(BaseSettings):
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    engine_type: EngineType = EngineType.RULE
    model_path: str = "models/v1/model.pkl"
    ml_high_threshold: float = 0.7
    ml_medium_threshold: float = 0.4

    model_config = {"env_prefix": "APP_", "case_sensitive": False}


settings = Settings()
