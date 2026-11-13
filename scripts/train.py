"""Train and evaluate fraud models on leakage-free features.

Thresholds are chosen on validation at a fixed alert budget (share of transactions
sent to review), then applied unchanged to the test period.

Usage: python scripts/train.py [path/to/transactions_full.csv] [--save]
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import FEATURE_COLUMNS, LABEL_DELAY_DAYS, build_features  # noqa: E402

ALERT_BUDGET = 0.01  # review at most 1% of transactions


def time_split(df):
    max_day = df["TX_TIME_DAYS"].max()
    train_end, val_end = int(max_day * 0.70), int(max_day * 0.85)
    return (
        df[df["TX_TIME_DAYS"] <= train_end],
        df[(df["TX_TIME_DAYS"] > train_end) & (df["TX_TIME_DAYS"] <= val_end)],
        df[df["TX_TIME_DAYS"] > val_end],
    )


def budget_threshold(probs, budget):
    k = max(1, int(budget * len(probs)))
    return float(np.sort(probs)[::-1][k - 1])


def evaluate(y, probs, threshold, scenarios):
    flagged = probs >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    return {
        "recall": round(tp / int((y == 1).sum()), 4),
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "alert_rate": round(float(flagged.mean()), 4),
        "frauds_caught": tp,
        "false_alarms": fp,
        "roc_auc": round(roc_auc_score(y, probs), 4),
        "pr_auc": round(average_precision_score(y, probs), 4),
        "recall_by_scenario": {
            s: round(float(flagged[scenarios == s].mean()), 3) for s in (1, 2, 3)
        },
    }


def main():
    csv = next(
        (a for a in sys.argv[1:] if not a.startswith("--")), "data/raw/transactions_full.csv"
    )
    train, val, test = time_split(build_features(pd.read_csv(csv)))
    X = {
        n: d[FEATURE_COLUMNS].fillna(0) for n, d in (("train", train), ("val", val), ("test", test))
    }
    y = {n: d["TX_FRAUD"].to_numpy() for n, d in (("train", train), ("val", val), ("test", test))}
    spw = (y["train"] == 0).sum() / (y["train"] == 1).sum()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=1000,
            early_stopping_rounds=50,
            eval_metric="aucpr",
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }
    results = {}
    for name, model in models.items():
        fit_args = (
            {"eval_set": [(X["val"], y["val"])], "verbose": False} if name == "XGBoost" else {}
        )
        model.fit(X["train"], y["train"], **fit_args)
        threshold = budget_threshold(model.predict_proba(X["val"])[:, 1], ALERT_BUDGET)
        results[name] = evaluate(
            y["test"],
            model.predict_proba(X["test"])[:, 1],
            threshold,
            test["TX_FRAUD_SCENARIO"].to_numpy(),
        )
        results[name]["threshold"] = threshold
        print(name, json.dumps(results[name]), flush=True)

    if "--save" in sys.argv:
        best = max(results, key=lambda n: results[n]["pr_auc"])
        joblib.dump(
            {
                "model": models[best],
                "threshold": results[best]["threshold"],
                "feature_columns": FEATURE_COLUMNS,
                "model_type": best,
                "trained_at": datetime.now().isoformat(),
                "performance": results[best],
                "alert_budget": ALERT_BUDGET,
                "label_delay_days": LABEL_DELAY_DAYS,
            },
            "artifacts/fraud_detection_model_final.pkl",
        )
        print(f"saved {best}")


if __name__ == "__main__":
    main()
