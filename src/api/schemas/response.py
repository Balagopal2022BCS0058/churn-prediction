from enum import Enum
from pydantic import BaseModel


class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class PredictRiskResponse(BaseModel):
    customer_id: str
    risk_level: RiskLevel
    triggered_rules: list[str] = []
    details: dict = {}
    churn_probability: float | None = None
    model_version: str | None = None
    feature_contributions: dict | None = None
