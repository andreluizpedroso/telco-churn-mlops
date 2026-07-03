"""Optional MLflow tracking helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import mlflow
except ImportError:
    # MLflow e uma dependencia do projeto, mas pode nao estar instalada no
    # ambiente local usado para testes rapidos. O restante do pipeline nao deve
    # quebrar por causa disso.
    mlflow = None


EXPERIMENT_NAME = "telco-churn-mlops"


def is_mlflow_available() -> bool:
    """Return whether MLflow is installed in the active environment."""
    return mlflow is not None


def _log_artifact_if_exists(path: Path | None) -> None:
    """Log an artifact only when a valid path is provided."""
    if path is not None and path.exists():
        mlflow.log_artifact(str(path))


def log_baseline_results(results: dict[str, Any], artifact_path: Path | None = None) -> bool:
    """Log baseline model metrics to MLflow when MLflow is available."""
    if mlflow is None:
        print("MLflow is not installed. Skipping MLflow baseline tracking.")
        return False

    mlflow.set_experiment(EXPERIMENT_NAME)

    for model_name, metrics in results["models"].items():
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", results["dataset"]["random_state"])
            mlflow.log_param("train_rows", results["dataset"]["train_rows"])
            mlflow.log_param("test_rows", results["dataset"]["test_rows"])
            mlflow.log_param("churn_rate", results["dataset"]["churn_rate"])

            for metric_name, metric_value in metrics.items():
                if metric_name == "confusion_matrix":
                    mlflow.log_dict({"confusion_matrix": metric_value}, "confusion_matrix.json")
                else:
                    mlflow.log_metric(metric_name, metric_value)

            _log_artifact_if_exists(artifact_path)

    return True


def log_mlp_results(results: dict[str, Any], metrics_file: Path | None = None) -> bool:
    """Log MLP training results to MLflow when MLflow is available."""
    if mlflow is None:
        print("MLflow is not installed. Skipping MLflow MLP tracking.")
        return False

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="pytorch_mlp"):
        mlflow.log_params(results["config"])
        mlflow.log_param("train_rows", results["dataset"]["train_rows"])
        mlflow.log_param("validation_rows", results["dataset"]["validation_rows"])
        mlflow.log_param("test_rows", results["dataset"]["test_rows"])
        mlflow.log_param("positive_class_weight", results["positive_class_weight"])
        mlflow.log_metric("best_validation_roc_auc", results["best_validation_roc_auc"])
        mlflow.log_metric("epochs_trained", results["epochs_trained"])

        for metric_name, metric_value in results["test_metrics"].items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)

        _log_artifact_if_exists(metrics_file)

    return True
