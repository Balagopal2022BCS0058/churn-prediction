"""
Training script: load data, engineer features, train model, save artifact.

Usage:
    python -m src.ml.train --telco-path data/raw/telco_churn.csv --output-dir models/v1
"""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.ml.data_loader import load_data
from src.ml.evaluate import evaluate_model
from src.features.engineering import extract_features, FEATURE_NAMES


def build_feature_matrix(customers: pd.DataFrame, tickets: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix by computing features per customer."""
    rows = []
    ticket_groups = tickets.groupby("customer_id")
    for _, row in customers.iterrows():
        cid = row["customer_id"]
        cust_tickets = ticket_groups.get_group(cid).to_dict("records") if cid in ticket_groups.groups else []
        feats = extract_features(
            tickets=cust_tickets,
            monthly_charges=float(row["monthly_charges"]),
            previous_monthly_charges=float(row["previous_monthly_charges"]),
            contract=str(row["contract"]),
        )
        feats["customer_id"] = cid
        feats["churn_label"] = int(row["churn_label"])
        rows.append(feats)
    return pd.DataFrame(rows)


def train(telco_path: str, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {telco_path}...")
    customers, tickets = load_data(telco_path)

    print("Engineering features...")
    df = build_feature_matrix(customers, tickets)

    X = df[FEATURE_NAMES].values
    y = df["churn_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    candidates = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=100, scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            random_state=42, eval_metric="logloss", verbosity=0,
        ),
    }

    best_model = None
    best_f1 = -1.0
    best_name = ""

    for name, model in candidates.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, out / name)
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_model = model
            best_name = name

    print(f"\nBest model: {best_name} (F1={best_f1:.4f})")

    # Save best model
    joblib.dump(best_model, out / "model.pkl")
    (out / "features.json").write_text(json.dumps(FEATURE_NAMES, indent=2))
    (out / "model_info.json").write_text(json.dumps({"model_name": best_name, "f1_macro": best_f1}, indent=2))

    print(f"Model saved to {out}/model.pkl")
    return best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument("--telco-path", default="data/raw/telco_churn.csv")
    parser.add_argument("--output-dir", default="models/v1")
    args = parser.parse_args()
    train(args.telco_path, args.output_dir)
