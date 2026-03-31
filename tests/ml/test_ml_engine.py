"""Tests for ML inference engine including fallback behavior."""
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src.api.schemas.request import PredictRiskRequest, TicketSchema
from src.api.schemas.response import RiskLevel
from src.config import settings
from src.engine.ml_engine import MLEngine


class FixedProbabilityClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible classifier that always predicts a fixed probability."""

    def __init__(self, prob: float = 0.8):
        self.prob = prob
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.ones(len(X), dtype=int) if self.prob >= 0.5 else np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.array([[1 - self.prob, self.prob]] * len(X))


def make_request(**kwargs) -> PredictRiskRequest:
    defaults = {
        "customer_id": "C001",
        "contract": "Month-to-Month",
        "monthly_charges": 85.0,
        "previous_monthly_charges": 70.0,
        "tickets": [
            TicketSchema(
                date=date.today() - timedelta(days=1), category="complaint", text="unhappy"
            )
        ],
    }
    defaults.update(kwargs)
    return PredictRiskRequest(**defaults)


def save_model(model, tmpdir: str) -> str:
    model_path = Path(tmpdir) / "model.pkl"
    joblib.dump(model, model_path)
    return str(model_path)


@pytest.mark.asyncio
async def test_fallback_when_model_missing():
    """When model file doesn't exist, falls back to rule engine."""
    with patch.object(settings, "model_path", "/nonexistent/model.pkl"):
        engine = MLEngine()
        assert engine._model is None
        response = await engine.evaluate(make_request())
        assert response.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]


@pytest.mark.asyncio
async def test_high_risk_high_probability():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = save_model(FixedProbabilityClassifier(prob=0.9), tmpdir)
        with patch.object(settings, "model_path", model_path):
            engine = MLEngine()
            response = await engine.evaluate(make_request())
    assert response.risk_level == RiskLevel.HIGH
    assert response.churn_probability == pytest.approx(0.9, abs=0.01)


@pytest.mark.asyncio
async def test_medium_risk_medium_probability():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = save_model(FixedProbabilityClassifier(prob=0.55), tmpdir)
        with patch.object(settings, "model_path", model_path):
            engine = MLEngine()
            response = await engine.evaluate(make_request())
    assert response.risk_level == RiskLevel.MEDIUM


@pytest.mark.asyncio
async def test_low_risk_low_probability():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = save_model(FixedProbabilityClassifier(prob=0.2), tmpdir)
        with patch.object(settings, "model_path", model_path):
            engine = MLEngine()
            response = await engine.evaluate(make_request())
    assert response.risk_level == RiskLevel.LOW


@pytest.mark.asyncio
async def test_response_includes_model_version():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = save_model(FixedProbabilityClassifier(prob=0.8), tmpdir)
        with patch.object(settings, "model_path", model_path):
            engine = MLEngine()
            response = await engine.evaluate(make_request())
    assert response.model_version is not None


@pytest.mark.asyncio
async def test_response_includes_probability_field():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = save_model(FixedProbabilityClassifier(prob=0.75), tmpdir)
        with patch.object(settings, "model_path", model_path):
            engine = MLEngine()
            response = await engine.evaluate(make_request())
    assert response.churn_probability is not None
    assert 0.0 <= response.churn_probability <= 1.0


@pytest.mark.asyncio
async def test_empty_tickets_handled():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = save_model(FixedProbabilityClassifier(prob=0.3), tmpdir)
        with patch.object(settings, "model_path", model_path):
            engine = MLEngine()
            request = make_request(tickets=[])
            response = await engine.evaluate(request)
    assert response.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
