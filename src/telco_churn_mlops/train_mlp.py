"""Train and evaluate the PyTorch MLP churn model."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from telco_churn_mlops import mlp as mlp_runtime
from telco_churn_mlops.config import MODELS_DIR, RANDOM_STATE
from telco_churn_mlops.data import TARGET_COLUMN, prepare_processed_data
from telco_churn_mlops.features import build_preprocessor, split_features_target
from telco_churn_mlops.mlflow_tracking import log_mlp_results
from telco_churn_mlops.mlp import ChurnMLP, ensure_torch_is_available


@dataclass(frozen=True)
class MLPTrainingConfig:
    """Hyperparameters used to train the first MLP baseline."""

    hidden_size: int = 64
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 128
    max_epochs: int = 80
    patience: int = 8
    validation_size: float = 0.2
    threshold: float = 0.5


def set_reproducibility(seed: int = RANDOM_STATE) -> None:
    """Fix random seeds to make the training run more reproducible."""
    random.seed(seed)
    np.random.seed(seed)

    if mlp_runtime.torch is not None:
        mlp_runtime.torch.manual_seed(seed)
        # Deterministic algorithms reduce run-to-run variance. Some operations
        # may still vary by hardware, but this is enough for this project stage.
        mlp_runtime.torch.use_deterministic_algorithms(True, warn_only=True)


def split_train_validation(
    train_df: pd.DataFrame,
    validation_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified train/validation split from the training data."""
    train_part, validation_part = train_test_split(
        train_df,
        test_size=validation_size,
        random_state=RANDOM_STATE,
        stratify=train_df[TARGET_COLUMN],
    )
    return train_part.reset_index(drop=True), validation_part.reset_index(drop=True)


def to_float_tensor(array: np.ndarray):
    """Convert a numpy array to a PyTorch float tensor."""
    ensure_torch_is_available()
    return mlp_runtime.torch.tensor(array, dtype=mlp_runtime.torch.float32)


def make_batches(x_tensor, y_tensor, batch_size: int):
    """Yield shuffled mini-batches for one training epoch."""
    # randperm cria uma ordem aleatoria dos indices a cada epoca.
    indices = mlp_runtime.torch.randperm(x_tensor.shape[0])

    for start in range(0, x_tensor.shape[0], batch_size):
        batch_indices = indices[start : start + batch_size]
        yield x_tensor[batch_indices], y_tensor[batch_indices]


def evaluate_mlp(model, x_tensor, y_true: np.ndarray, threshold: float) -> dict[str, float]:
    """Evaluate the MLP with probabilities and classification metrics."""
    model.eval()

    # no_grad desliga o calculo de gradientes, deixando a avaliacao mais leve.
    with mlp_runtime.torch.no_grad():
        logits = model(x_tensor)
        probabilities = mlp_runtime.torch.sigmoid(logits).cpu().numpy()

    predictions = (probabilities >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
    }


def train_one_epoch(model, x_train, y_train, optimizer, loss_fn, batch_size: int) -> float:
    """Train the MLP for one epoch and return the average loss."""
    model.train()
    losses: list[float] = []

    for batch_x, batch_y in make_batches(x_train, y_train, batch_size):
        # Zera gradientes antigos antes de calcular os gradientes do batch atual.
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses))


def train_mlp(config: MLPTrainingConfig | None = None) -> dict[str, Any]:
    """Train the MLP, apply early stopping, save artifacts, and return metrics."""
    ensure_torch_is_available()
    config = config or MLPTrainingConfig()
    set_reproducibility()

    # Reaproveitamos a limpeza e o split da Sprint 2 para manter consistencia.
    train_df, test_df, dataset_summary = prepare_processed_data()
    fit_df, validation_df = split_train_validation(train_df, config.validation_size)

    x_fit, y_fit = split_features_target(fit_df)
    x_validation, y_validation = split_features_target(validation_df)
    x_test, y_test = split_features_target(test_df)

    # O preprocessor e ajustado apenas no treino interno para evitar vazamento
    # de informacao da validacao ou teste.
    preprocessor = build_preprocessor(fit_df)
    x_fit_array = preprocessor.fit_transform(x_fit)
    x_validation_array = preprocessor.transform(x_validation)
    x_test_array = preprocessor.transform(x_test)

    x_fit_tensor = to_float_tensor(x_fit_array)
    y_fit_tensor = to_float_tensor(y_fit.to_numpy())
    x_validation_tensor = to_float_tensor(x_validation_array)
    x_test_tensor = to_float_tensor(x_test_array)

    model = ChurnMLP(
        input_size=x_fit_array.shape[1],
        hidden_size=config.hidden_size,
        dropout=config.dropout,
    )
    optimizer = mlp_runtime.torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Como ha menos clientes com churn do que sem churn, pos_weight aumenta o
    # custo de errar exemplos positivos durante o treino.
    positive_count = float(y_fit.sum())
    negative_count = float((y_fit == 0).sum())
    pos_weight = mlp_runtime.torch.tensor(
        [negative_count / positive_count],
        dtype=mlp_runtime.torch.float32,
    )
    loss_fn = mlp_runtime.torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_validation_auc = -np.inf
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            x_train=x_fit_tensor,
            y_train=y_fit_tensor,
            optimizer=optimizer,
            loss_fn=loss_fn,
            batch_size=config.batch_size,
        )
        validation_metrics = evaluate_mlp(
            model=model,
            x_tensor=x_validation_tensor,
            y_true=y_validation.to_numpy(),
            threshold=config.threshold,
        )
        validation_auc = validation_metrics["roc_auc"]
        history.append({"epoch": epoch, "train_loss": train_loss, **validation_metrics})

        # Early stopping guarda o melhor modelo e para quando a validacao nao
        # melhora por varias epocas seguidas.
        if validation_auc > best_validation_auc:
            best_validation_auc = validation_auc
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_mlp(
        model=model,
        x_tensor=x_test_tensor,
        y_true=y_test.to_numpy(),
        threshold=config.threshold,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_file = MODELS_DIR / "mlp.pt"
    preprocessor_file = MODELS_DIR / "mlp_preprocessor.joblib"
    metrics_file = MODELS_DIR / "mlp_metrics.json"

    # O arquivo .pt guarda os pesos da rede; o joblib guarda o preprocessor que
    # precisa ser reutilizado na inferencia.
    mlp_runtime.torch.save(model.state_dict(), model_file)
    joblib.dump(preprocessor, preprocessor_file)

    results = {
        "dataset": {
            "rows_after_cleaning": dataset_summary.rows,
            "churn_rate": dataset_summary.churn_rate,
            "train_rows": int(fit_df.shape[0]),
            "validation_rows": int(validation_df.shape[0]),
            "test_rows": int(test_df.shape[0]),
        },
        "config": asdict(config),
        "positive_class_weight": float(pos_weight.item()),
        "best_validation_roc_auc": float(best_validation_auc),
        "epochs_trained": len(history),
        "test_metrics": test_metrics,
        "artifacts": {
            "model_file": str(model_file),
            "preprocessor_file": str(preprocessor_file),
        },
    }
    metrics_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Registra os resultados da MLP quando MLflow esta disponivel.
    log_mlp_results(results, metrics_file=metrics_file)
    return results


if __name__ == "__main__":
    metrics = train_mlp()
    print(json.dumps(metrics, indent=2))
