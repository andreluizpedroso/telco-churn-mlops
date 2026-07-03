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
from telco_churn_mlops.mlflow_tracking import log_baseline_results


def evaluate_classifier(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Evaluate a fitted classifier with challenge-relevant metrics."""
    # predict devolve a classe final: 0 para nao churn, 1 para churn.
    y_pred = model.predict(x_test)

    # predict_proba devolve probabilidades por classe. A coluna 1 e a
    # probabilidade de churn, usada para ROC-AUC.
    y_proba = model.predict_proba(x_test)[:, 1]

    # A matriz de confusao ajuda a enxergar falsos positivos e falsos negativos,
    # que serao importantes para a analise de custo de negocio.
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
    # O DummyClassifier nao aprende padroes. Ele serve como linha de base para
    # mostrar o minimo que qualquer modelo util precisa superar.
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(train_df)),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]
    )


def build_logistic_regression_model(train_df: pd.DataFrame) -> Pipeline:
    """Build a logistic regression baseline."""
    # A Regressao Logistica e um baseline forte para dados tabulares porque e
    # simples, rapida e interpretavel.
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(train_df)),
            (
                "classifier",
                # class_weight="balanced" compensa o desbalanceamento entre
                # clientes com churn e sem churn.
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            ),
        ]
    )


def run_baselines() -> dict[str, Any]:
    """Prepare data, train baselines, evaluate, and save metrics."""
    # Primeiro garantimos que os dados processados existem e que o split e
    # sempre reproduzivel.
    train_df, test_df, summary = prepare_processed_data()
    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)

    # Todos os modelos seguem o mesmo contrato: preprocessing + classificador.
    # Isso torna a comparacao justa, pois cada modelo recebe as mesmas features.
    models = {
        "dummy_most_frequent": build_dummy_model(train_df),
        "logistic_regression_balanced": build_logistic_regression_model(train_df),
    }

    # O dicionario final concentra metadados do dataset e metricas dos modelos.
    # Nesta sprint salvamos em JSON; depois o MLflow sera usado para tracking.
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
        # fit treina o pipeline completo: primeiro o preprocessor, depois o
        # classificador.
        model.fit(x_train, y_train)
        results["models"][model_name] = evaluate_classifier(model, x_test, y_test)

    # Persistimos as metricas para documentacao e comparacao com a MLP.
    BASELINE_RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Se MLflow estiver instalado, registramos parametros, metricas e artefatos.
    # Se nao estiver, o helper apenas avisa e o pipeline continua funcionando.
    log_baseline_results(results, artifact_path=BASELINE_RESULTS_FILE)
    return results


if __name__ == "__main__":
    # Permite executar a sprint pelo terminal com:
    # python -m telco_churn_mlops.baselines
    metrics = run_baselines()
    print(json.dumps(metrics, indent=2))
