"""
Shared feature engineering module — used by BOTH training (batch) and inference (single record).
This is the critical module that prevents training-serving skew.
"""
from datetime import date, timedelta

from src.features.sentiment import average_sentiment

FEATURE_NAMES = [
    "ticket_freq_7d",
    "ticket_freq_30d",
    "ticket_freq_90d",
    "avg_sentiment",
    "complaint_count",
    "billing_count",
    "technical_count",
    "avg_days_between_tickets",
    "charge_delta",
    "is_monthly_contract",
]


def compute_ticket_frequency(
    tickets: list[dict], windows: list[int] = None, reference_date: date = None
) -> dict:
    """
    Compute ticket count within each time window.
    tickets: list of dicts with key 'date' (date object)
    """
    if windows is None:
        windows = [7, 30, 90]
    ref = reference_date or date.today()
    result = {}
    for w in windows:
        cutoff = ref - timedelta(days=w)
        result[f"ticket_freq_{w}d"] = sum(1 for t in tickets if t["date"] >= cutoff)
    return result


def compute_category_counts(tickets: list[dict]) -> dict:
    """Count tickets by category."""
    counts = {"complaint_count": 0, "billing_count": 0, "technical_count": 0}
    for t in tickets:
        cat = t.get("category", "").strip().lower()
        if cat == "complaint":
            counts["complaint_count"] += 1
        elif cat == "billing":
            counts["billing_count"] += 1
        elif cat == "technical":
            counts["technical_count"] += 1
    return counts


def compute_avg_days_between_tickets(tickets: list[dict]) -> float:
    """Average number of days between consecutive tickets (0 if < 2 tickets)."""
    if len(tickets) < 2:
        return 0.0
    sorted_dates = sorted(t["date"] for t in tickets)
    gaps = [(sorted_dates[i + 1] - sorted_dates[i]).days for i in range(len(sorted_dates) - 1)]
    return sum(gaps) / len(gaps)


def compute_charge_delta(monthly_charges: float, previous_monthly_charges: float) -> float:
    """Absolute change in monthly charges."""
    return monthly_charges - previous_monthly_charges


def extract_features(
    tickets: list[dict],
    monthly_charges: float,
    previous_monthly_charges: float,
    contract: str,
    reference_date: date | None = None,
) -> dict:
    """
    Extract all features from raw inputs. Works for both single-record (inference)
    and batch (training) scenarios when called row-wise.
    """
    features = {}
    features.update(compute_ticket_frequency(tickets, reference_date=reference_date))
    features["avg_sentiment"] = average_sentiment([t.get("text", "") for t in tickets])
    features.update(compute_category_counts(tickets))
    features["avg_days_between_tickets"] = compute_avg_days_between_tickets(tickets)
    features["charge_delta"] = compute_charge_delta(monthly_charges, previous_monthly_charges)
    features["is_monthly_contract"] = 1 if contract.strip().lower() == "month-to-month" else 0
    return {k: features[k] for k in FEATURE_NAMES}
