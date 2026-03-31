import json
from pathlib import Path

import joblib
import structlog

from src.api.schemas.request import PredictRiskRequest
from src.api.schemas.response import PredictRiskResponse, RiskLevel
from src.config import settings
from src.engine.base import RiskEngine
from src.features.engineering import FEATURE_NAMES, extract_features

logger = structlog.get_logger()


class MLEngine(RiskEngine):
    _model = None
    _model_version: str = "unknown"

    def __init__(self):
        self._load_model()

    def _load_model(self):
        model_path = Path(settings.model_path)
        if not model_path.exists():
            logger.warning("model_not_found", path=str(model_path), fallback="rule_engine")
            self._model = None
            return
        self._model = joblib.load(model_path)
        info_path = model_path.parent / "model_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text())
            self._model_version = info.get("model_name", "v1")
        logger.info("model_loaded", path=str(model_path), version=self._model_version)

    def _probability_to_risk(self, prob: float) -> RiskLevel:
        if prob >= settings.ml_high_threshold:
            return RiskLevel.HIGH
        if prob >= settings.ml_medium_threshold:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    async def evaluate(self, request: PredictRiskRequest) -> PredictRiskResponse:
        if self._model is None:
            # Fallback to rule engine
            from src.engine.rule_engine import RuleBasedEngine
            logger.warning("ml_fallback", reason="model_not_loaded")
            return await RuleBasedEngine().evaluate(request)

        tickets = [t.model_dump() for t in request.tickets]
        features = extract_features(
            tickets=tickets,
            monthly_charges=request.monthly_charges,
            previous_monthly_charges=request.previous_monthly_charges,
            contract=request.contract,
        )
        X = [[features[k] for k in FEATURE_NAMES]]
        prob = float(self._model.predict_proba(X)[0][1])
        risk_level = self._probability_to_risk(prob)

        # Feature contributions (importance-weighted)
        contributions = None
        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
            contributions = {k: round(float(v), 4) for k, v in zip(FEATURE_NAMES, importances)}
        elif hasattr(self._model, "named_steps"):
            clf = self._model.named_steps.get("clf")
            if hasattr(clf, "coef_"):
                coefs = clf.coef_[0]
                contributions = {k: round(float(v), 4) for k, v in zip(FEATURE_NAMES, coefs)}

        return PredictRiskResponse(
            customer_id=request.customer_id,
            risk_level=risk_level,
            triggered_rules=[],
            details={"features": features},
            churn_probability=round(prob, 4),
            model_version=self._model_version,
            feature_contributions=contributions,
        )
