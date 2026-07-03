from telco_churn_mlops import mlflow_tracking


def test_mlflow_tracking_helpers_skip_when_mlflow_is_missing(monkeypatch):
    # O projeto deve continuar testavel mesmo quando MLflow nao esta instalado
    # no ambiente ativo.
    monkeypatch.setattr(mlflow_tracking, "mlflow", None)

    baseline_logged = mlflow_tracking.log_baseline_results(
        {
            "dataset": {
                "random_state": 42,
                "train_rows": 10,
                "test_rows": 5,
                "churn_rate": 0.2,
            },
            "models": {},
        }
    )
    mlp_logged = mlflow_tracking.log_mlp_results(
        {
            "dataset": {"train_rows": 10, "validation_rows": 2, "test_rows": 5},
            "config": {},
            "positive_class_weight": 1.0,
            "best_validation_roc_auc": 0.5,
            "epochs_trained": 1,
            "test_metrics": {},
        }
    )

    assert baseline_logged is False
    assert mlp_logged is False
