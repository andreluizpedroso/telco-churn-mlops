"""Train and persist the baseline model used by the API."""

from __future__ import annotations

import json

import joblib

from telco_churn_mlops.baselines import build_logistic_regression_model, evaluate_classifier
from telco_churn_mlops.config import MODELS_DIR
from telco_churn_mlops.data import prepare_processed_data
from telco_churn_mlops.features import split_features_target
from telco_churn_mlops.inference import MODEL_FILE


def train_and_save_baseline_model() -> dict:
    """Train the logistic regression baseline and save it as an inference artifact."""
    train_df, test_df, _ = prepare_processed_data()
    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)

    # O pipeline inclui preprocessing + classificador, entao um unico arquivo
    # joblib e suficiente para prever novos clientes pela API.
    model = build_logistic_regression_model(train_df)
    model.fit(x_train, y_train)
    metrics = evaluate_classifier(model, x_test, y_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    metrics_file = MODELS_DIR / "baseline_logistic_regression_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {"model_file": str(MODEL_FILE), "metrics": metrics}


if __name__ == "__main__":
    result = train_and_save_baseline_model()
    print(json.dumps(result, indent=2))
