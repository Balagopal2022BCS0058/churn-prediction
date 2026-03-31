"""
Data loader: load Telco CSV and simulate/load ticket logs, merge on customer_id.
"""
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

TICKET_CATEGORIES = ["complaint", "billing", "technical", "general"]
TICKET_TEXTS = {
    "complaint": [
        "Service is terrible, very unhappy",
        "I want to cancel my subscription",
        "This is unacceptable, fix this now",
        "I have been overcharged again",
    ],
    "billing": [
        "Question about my latest invoice",
        "Unexpected charge on my bill",
        "Please clarify the billing cycle",
    ],
    "technical": [
        "Internet connection is dropping",
        "Router keeps resetting",
        "Speed is much slower than advertised",
    ],
    "general": [
        "When will the outage be resolved?",
        "Please update my address",
        "I need information about plan upgrades",
    ],
}


def _simulate_tickets(customer_id: str, churn: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed + hash(customer_id) % 10000)
    today = date.today()
    n_tickets = rng.randint(6, 12) if churn else rng.randint(0, 4)
    tickets = []
    for _ in range(n_tickets):
        days_ago = rng.randint(1, 90)
        cat = rng.choice(TICKET_CATEGORIES if not churn else ["complaint", "billing", "complaint"])
        text = rng.choice(TICKET_TEXTS[cat])
        tickets.append({
            "customer_id": customer_id,
            "date": today - timedelta(days=days_ago),
            "category": cat,
            "text": text,
        })
    return tickets


def load_telco_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Ensure required columns exist
    required = ["customerid", "churn", "monthlycharges", "contract"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df.rename(columns={"customerid": "customer_id", "monthlycharges": "monthly_charges"})
    df["churn_label"] = (df["churn"].str.lower() == "yes").astype(int)
    # Simulate previous monthly charges (±20%)
    rng = random.Random(42)
    df["previous_monthly_charges"] = df["monthly_charges"].apply(
        lambda x: max(1.0, x * (1 + rng.uniform(-0.2, 0.2)))
    )
    return df


def build_ticket_dataframe(customers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in customers.iterrows():
        tickets = _simulate_tickets(row["customer_id"], row["churn_label"])
        rows.extend(tickets)
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["customer_id", "date", "category", "text"]
    )


def load_data(telco_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (customers_df, tickets_df)."""
    customers = load_telco_csv(telco_path)
    tickets_path = Path(telco_path).parent / "ticket_logs.csv"
    if tickets_path.exists():
        tickets = pd.read_csv(tickets_path, parse_dates=["date"])
        tickets["date"] = pd.to_datetime(tickets["date"]).dt.date
    else:
        tickets = build_ticket_dataframe(customers)
        tickets.to_csv(tickets_path, index=False)
    return customers, tickets
