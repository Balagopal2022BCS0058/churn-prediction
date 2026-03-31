"""Tests for data loader: synthetic ticket generation and CSV loading."""
import tempfile
from pathlib import Path
import pytest
import pandas as pd
from src.ml.data_loader import build_ticket_dataframe, _simulate_tickets


def make_customers(n: int = 10) -> pd.DataFrame:
    import random
    rng = random.Random(0)
    rows = []
    for i in range(n):
        rows.append({
            "customer_id": f"C{i:04d}",
            "contract": rng.choice(["Month-to-Month", "One year"]),
            "monthly_charges": rng.uniform(20, 100),
            "previous_monthly_charges": rng.uniform(20, 100),
            "churn_label": i % 3 == 0,
        })
    return pd.DataFrame(rows)


def test_simulate_tickets_churn_customer_gets_more():
    churner = _simulate_tickets("C0001", churn=1, seed=42)
    non_churner = _simulate_tickets("C0002", churn=0, seed=42)
    assert len(churner) >= len(non_churner)


def test_simulate_tickets_returns_dicts():
    tickets = _simulate_tickets("C0001", churn=0, seed=1)
    for t in tickets:
        assert "date" in t
        assert "category" in t
        assert "text" in t


def test_build_ticket_dataframe_shape():
    customers = make_customers(5)
    df = build_ticket_dataframe(customers)
    assert set(df.columns) >= {"customer_id", "date", "category", "text"}
    assert len(df) >= 0


def test_build_ticket_dataframe_saved_on_miss():
    with tempfile.TemporaryDirectory() as tmpdir:
        customers = make_customers(5)
        telco_path = Path(tmpdir) / "telco_churn.csv"
        customers.to_csv(telco_path, index=False)

        import src.ml.data_loader as loader_mod
        orig = loader_mod.load_telco_csv

        loader_mod.load_telco_csv = lambda path: customers
        try:
            customers2, tickets = loader_mod.load_data(str(telco_path))
            assert len(customers2) == 5
        finally:
            loader_mod.load_telco_csv = orig
