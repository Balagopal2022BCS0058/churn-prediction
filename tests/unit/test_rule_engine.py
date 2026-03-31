from datetime import date, timedelta

import pytest

from src.api.schemas.request import PredictRiskRequest, TicketSchema
from src.api.schemas.response import RiskLevel
from src.engine.rule_engine import RuleBasedEngine


def make_ticket(days_ago: int = 5, category: str = "general") -> dict:
    return {"date": (date.today() - timedelta(days=days_ago)).isoformat(),
            "category": category, "text": ""}


def make_request(**kwargs) -> PredictRiskRequest:
    defaults = {
        "customer_id": "C001",
        "contract": "One year",
        "monthly_charges": 70.0,
        "previous_monthly_charges": 70.0,
        "tickets": [],
    }
    defaults.update(kwargs)
    return PredictRiskRequest(**defaults)


@pytest.mark.asyncio
async def test_low_risk_no_triggers():
    engine = RuleBasedEngine()
    resp = await engine.evaluate(make_request())
    assert resp.risk_level == RiskLevel.LOW
    assert resp.triggered_rules == []


@pytest.mark.asyncio
async def test_high_risk_ticket_frequency():
    tickets = [TicketSchema(date=date.today() - timedelta(days=i), category="billing", text="")
               for i in range(1, 7)]
    engine = RuleBasedEngine()
    resp = await engine.evaluate(make_request(tickets=tickets))
    assert resp.risk_level == RiskLevel.HIGH
    assert "RULE_TICKET_FREQUENCY_HIGH" in resp.triggered_rules


@pytest.mark.asyncio
async def test_high_risk_monthly_contract_complaint():
    tickets = [TicketSchema(date=date.today() - timedelta(days=1), category="complaint", text="")]
    engine = RuleBasedEngine()
    resp = await engine.evaluate(make_request(contract="Month-to-Month", tickets=tickets))
    assert resp.risk_level == RiskLevel.HIGH
    assert "RULE_MONTH_TO_MONTH_COMPLAINT" in resp.triggered_rules


@pytest.mark.asyncio
async def test_medium_risk_charge_increase():
    tickets = [TicketSchema(date=date.today() - timedelta(days=i), category="billing", text="")
               for i in range(1, 4)]
    engine = RuleBasedEngine()
    resp = await engine.evaluate(
        make_request(monthly_charges=90.0, previous_monthly_charges=70.0, tickets=tickets)
    )
    assert resp.risk_level == RiskLevel.MEDIUM
    assert "RULE_CHARGE_INCREASE_MEDIUM" in resp.triggered_rules


@pytest.mark.asyncio
async def test_high_overrides_medium():
    """High risk rules take priority over medium."""
    tickets = [TicketSchema(date=date.today() - timedelta(days=i), category="complaint", text="")
               for i in range(1, 7)]
    engine = RuleBasedEngine()
    resp = await engine.evaluate(make_request(
        contract="Month-to-Month",
        monthly_charges=90.0,
        previous_monthly_charges=70.0,
        tickets=tickets,
    ))
    assert resp.risk_level == RiskLevel.HIGH
