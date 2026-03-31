import structlog
from fastapi import APIRouter
from src.api.schemas.request import PredictRiskRequest
from src.api.schemas.response import PredictRiskResponse
from src.config import settings, EngineType
from src.monitoring.metrics import RISK_PREDICTIONS, CHURN_PROBABILITY

logger = structlog.get_logger()
router = APIRouter()


def _get_engine():
    if settings.engine_type == EngineType.ML:
        from src.engine.ml_engine import MLEngine
        return MLEngine()
    from src.engine.rule_engine import RuleBasedEngine
    return RuleBasedEngine()


@router.post("/predict-risk", response_model=PredictRiskResponse, tags=["prediction"])
async def predict_risk(request: PredictRiskRequest) -> PredictRiskResponse:
    engine = _get_engine()
    response = await engine.evaluate(request)

    RISK_PREDICTIONS.labels(
        risk_level=response.risk_level.value,
        engine=settings.engine_type.value,
    ).inc()

    if response.churn_probability is not None:
        CHURN_PROBABILITY.observe(response.churn_probability)

    logger.info(
        "prediction",
        customer_id=request.customer_id,
        risk_level=response.risk_level.value,
        triggered_rules=response.triggered_rules,
        engine=settings.engine_type.value,
    )
    return response
