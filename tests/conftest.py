import pytest
from datetime import date, timedelta
from httpx import AsyncClient, ASGITransport
from src.main import app


@pytest.fixture
def today():
    return date.today()


@pytest.fixture
def recent_tickets(today):
    """6 tickets in last 30 days — triggers RULE_TICKET_FREQUENCY_HIGH."""
    return [
        {"date": today - timedelta(days=i), "category": "complaint", "text": "unhappy"}
        for i in range(1, 7)
    ]


@pytest.fixture
def few_tickets(today):
    """2 tickets in last 30 days."""
    return [
        {"date": today - timedelta(days=5), "category": "billing", "text": "question"},
        {"date": today - timedelta(days=10), "category": "technical", "text": "slow"},
    ]


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
