"""Train and evaluate baseline churn models."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from telco_churn_mlops.config import BASELINE_RESULTS_FILE, RANDOM_STATE
from telco_churn_mlops.data import prepare_processed_data
from telco_churn_mlops.features import build_preprocessor, split_features_target


def evaluate_classifier(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Evaluate a fitted classifier with challenge-relevant metrics."""
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    matrix = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": matrix,
    }


def build_dummy_model(train_df: pd.DataFrame) -> Pipeline:
    """Build a simple majority-class baseline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(train_df)),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]
    )


def build_logistic_regression_model(train_df: pd.DataFrame) -> Pipeline:
    """Build a logistic regression baseline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(train_df)),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            ),
        ]
    )


def run_baselines() -> dict[str, Any]:
    """Prepare data, train baselines, evaluate, and save metrics."""
    train_df, test_df, summary = prepare_processed_data()
    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)

    models = {
        "dummy_most_frequent": build_dummy_model(train_df),
        "logistic_regression_balanced": build_logistic_regression_model(train_df),
    }

    results: dict[str, Any] = {
        "dataset": {
            "rows_after_cleaning": summary.rows,
            "columns": summary.columns,
            "churn_rate": summary.churn_rate,
            "positive_count": summary.positive_count,
            "negative_count": summary.negative_count,
            "train_rows": int(train_df.shape[0]),
            "test_rows": int(test_df.shape[0]),
            "random_state": RANDOM_STATE,
        },
        "models": {},
    }

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        results["models"][model_name] = evaluate_classifier(model, x_test, y_test)

    BASELINE_RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    metrics = run_baselines()
    print(json.dumps(metrics, indent=2))
