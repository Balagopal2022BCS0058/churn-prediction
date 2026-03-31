import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_recall_curve,
    classification_report, confusion_matrix,
)


def evaluate_model(model, X_test, y_test, output_dir: Path) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred, average="macro")
    roc_auc = roc_auc_score(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    try:
        pr_auc = float(np.trapezoid(recall[::-1], precision[::-1]))
    except AttributeError:
        pr_auc = float(np.trapz(recall[::-1], precision[::-1]))

    metrics = {
        "f1_macro": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\n=== Model Evaluation ===")
    print(f"F1 (macro):  {metrics['f1_macro']}")
    print(f"ROC-AUC:     {metrics['roc_auc']}")
    print(f"PR-AUC:      {metrics['pr_auc']}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return metrics
