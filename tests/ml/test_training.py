"""Smoke test: train on tiny synthetic dataset, verify artifacts created."""
import json
import tempfile
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import pytest
from src.ml.train import train, build_feature_matrix


def make_synthetic_data(n: int = 60):
    """Create minimal synthetic telco + ticket data for testing."""
    rng = __import__("random").Random(42)
    rows = []
    for i in range(n):
        churn = i % 3 == 0  # ~33% churn
        rows.append({
            "customer_id": f"C{i:04d}",
            "contract": rng.choice(["Month-to-Month", "One year", "Two year"]),
            "monthly_charges": rng.uniform(20, 120),
            "previous_monthly_charges": rng.uniform(20, 120),
            "churn_label": int(churn),
        })
    customers = pd.DataFrame(rows)

    ticket_rows = []
    today = date.today()
    for _, row in customers.iterrows():
        n_tickets = rng.randint(5, 8) if row["churn_label"] else rng.randint(0, 2)
        for j in range(n_tickets):
            ticket_rows.append({
                "customer_id": row["customer_id"],
                "date": today - timedelta(days=rng.randint(1, 90)),
                "category": rng.choice(["complaint", "billing", "technical"]),
                "text": "test ticket text",
            })
    tickets = pd.DataFrame(ticket_rows) if ticket_rows else pd.DataFrame(
        columns=["customer_id", "date", "category", "text"]
    )
    return customers, tickets


def test_build_feature_matrix():
    customers, tickets = make_synthetic_data(20)
    df = build_feature_matrix(customers, tickets)
    assert len(df) == 20
    assert "churn_label" in df.columns


def test_train_creates_artifacts():
    customers, tickets = make_synthetic_data(60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save synthetic data to temp CSVs
        telco_path = Path(tmpdir) / "telco_churn.csv"
        customers.to_csv(telco_path, index=False)
        tickets.to_csv(Path(tmpdir) / "ticket_logs.csv", index=False)

        out_dir = Path(tmpdir) / "model_out"

        # Patch load_data to return our synthetic data
        import src.ml.train as train_module
        orig = train_module.load_data
        train_module.load_data = lambda path: (customers, tickets)

        try:
            train(str(telco_path), str(out_dir))
        finally:
            train_module.load_data = orig

        assert (out_dir / "model.pkl").exists()
        assert (out_dir / "features.json").exists()
        assert (out_dir / "model_info.json").exists()

        info = json.loads((out_dir / "model_info.json").read_text())
        assert "f1_macro" in info
        assert info["f1_macro"] >= 0.0

        features = json.loads((out_dir / "features.json").read_text())
        assert isinstance(features, list)
        assert len(features) > 0
