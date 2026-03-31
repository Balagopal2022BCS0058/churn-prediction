from datetime import date, timedelta

import pytest

from src.features.engineering import (
    FEATURE_NAMES,
    compute_avg_days_between_tickets,
    compute_category_counts,
    compute_charge_delta,
    compute_ticket_frequency,
    extract_features,
)


def make_ticket(days_ago: int, category: str = "general", text: str = "") -> dict:
    return {"date": date.today() - timedelta(days=days_ago), "category": category, "text": text}


def test_ticket_frequency_windows():
    tickets = [make_ticket(3), make_ticket(10), make_ticket(40)]
    result = compute_ticket_frequency(tickets)
    assert result["ticket_freq_7d"] == 1
    assert result["ticket_freq_30d"] == 2
    assert result["ticket_freq_90d"] == 3


def test_ticket_frequency_empty():
    result = compute_ticket_frequency([])
    assert all(v == 0 for v in result.values())


def test_category_counts():
    tickets = [
        make_ticket(1, "complaint"), make_ticket(2, "complaint"),
        make_ticket(3, "billing"), make_ticket(4, "technical"),
    ]
    result = compute_category_counts(tickets)
    assert result["complaint_count"] == 2
    assert result["billing_count"] == 1
    assert result["technical_count"] == 1


def test_category_counts_empty():
    result = compute_category_counts([])
    assert result == {"complaint_count": 0, "billing_count": 0, "technical_count": 0}


def test_avg_days_between_single_ticket():
    assert compute_avg_days_between_tickets([make_ticket(5)]) == 0.0


def test_avg_days_between_empty():
    assert compute_avg_days_between_tickets([]) == 0.0


def test_avg_days_between_multiple():
    tickets = [make_ticket(10), make_ticket(5), make_ticket(1)]
    result = compute_avg_days_between_tickets(tickets)
    assert result == pytest.approx(4.5, abs=0.1)


def test_charge_delta():
    assert compute_charge_delta(90.0, 70.0) == pytest.approx(20.0)
    assert compute_charge_delta(70.0, 90.0) == pytest.approx(-20.0)


def test_extract_features_returns_all_features():
    tickets = [make_ticket(5, "complaint", "bad service")]
    result = extract_features(tickets, 85.0, 70.0, "Month-to-Month")
    assert set(result.keys()) == set(FEATURE_NAMES)


def test_extract_features_monthly_flag():
    result = extract_features([], 70.0, 70.0, "Month-to-Month")
    assert result["is_monthly_contract"] == 1
    result2 = extract_features([], 70.0, 70.0, "One year")
    assert result2["is_monthly_contract"] == 0


def test_extract_features_no_tickets():
    result = extract_features([], 70.0, 70.0, "One year")
    assert result["ticket_freq_7d"] == 0
    assert result["avg_sentiment"] == 0.0
    assert result["avg_days_between_tickets"] == 0.0
