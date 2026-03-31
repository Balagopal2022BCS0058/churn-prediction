import pytest
from datetime import date, timedelta


def make_payload(**kwargs):
    today = date.today()
    defaults = {
        "customer_id": "C001",
        "contract": "One year",
        "monthly_charges": 70.0,
        "previous_monthly_charges": 70.0,
        "tickets": [],
    }
    defaults.update(kwargs)
    return defaults


@pytest.mark.asyncio
async def test_predict_low_risk(client):
    response = await client.post("/predict-risk", json=make_payload())
    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] == "Low"
    assert data["customer_id"] == "C001"


@pytest.mark.asyncio
async def test_predict_high_risk_ticket_frequency(client):
    today = date.today()
    tickets = [
        {"date": str(today - timedelta(days=i)), "category": "billing", "text": ""}
        for i in range(1, 7)
    ]
    response = await client.post("/predict-risk", json=make_payload(tickets=tickets))
    assert response.status_code == 200
    assert response.json()["risk_level"] == "High"


@pytest.mark.asyncio
async def test_predict_high_risk_monthly_complaint(client):
    today = date.today()
    tickets = [{"date": str(today - timedelta(days=1)), "category": "complaint", "text": ""}]
    response = await client.post("/predict-risk", json=make_payload(
        contract="Month-to-Month", tickets=tickets
    ))
    assert response.status_code == 200
    assert response.json()["risk_level"] == "High"


@pytest.mark.asyncio
async def test_predict_medium_risk_charge_increase(client):
    today = date.today()
    tickets = [
        {"date": str(today - timedelta(days=i)), "category": "billing", "text": ""}
        for i in range(1, 4)
    ]
    response = await client.post("/predict-risk", json=make_payload(
        monthly_charges=90.0, previous_monthly_charges=70.0, tickets=tickets
    ))
    assert response.status_code == 200
    assert response.json()["risk_level"] == "Medium"


@pytest.mark.asyncio
async def test_predict_validation_error_missing_field(client):
    response = await client.post("/predict-risk", json={"customer_id": "C001"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_returns_triggered_rules(client):
    today = date.today()
    tickets = [{"date": str(today - timedelta(days=1)), "category": "complaint", "text": ""}]
    response = await client.post("/predict-risk", json=make_payload(
        contract="Month-to-Month", tickets=tickets
    ))
    data = response.json()
    assert "RULE_MONTH_TO_MONTH_COMPLAINT" in data["triggered_rules"]
