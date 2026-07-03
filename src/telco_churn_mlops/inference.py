"""Inference helpers used by the FastAPI application."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from telco_churn_mlops.config import MODELS_DIR
from telco_churn_mlops.data import ID_COLUMN
from telco_churn_mlops.schemas import CustomerFeatures

MODEL_NAME = "logistic_regression_balanced"
MODEL_FILE = MODELS_DIR / "baseline_logistic_regression.joblib"
DEFAULT_THRESHOLD = 0.5


def payload_to_dataframe(payload: CustomerFeatures) -> pd.DataFrame:
    """Convert API payload names to the original dataset column names."""
    # A API usa nomes pythonicos em snake_case. O modelo foi treinado com os
    # nomes originais do dataset, entao fazemos o mapeamento antes da predicao.
    row = {
        ID_COLUMN: "api-request",
        "gender": payload.gender,
        "SeniorCitizen": payload.senior_citizen,
        "Partner": payload.partner,
        "Dependents": payload.dependents,
        "tenure": payload.tenure,
        "PhoneService": payload.phone_service,
        "MultipleLines": payload.multiple_lines,
        "InternetService": payload.internet_service,
        "OnlineSecurity": payload.online_security,
        "OnlineBackup": payload.online_backup,
        "DeviceProtection": payload.device_protection,
        "TechSupport": payload.tech_support,
        "StreamingTV": payload.streaming_tv,
        "StreamingMovies": payload.streaming_movies,
        "Contract": payload.contract,
        "PaperlessBilling": payload.paperless_billing,
        "PaymentMethod": payload.payment_method,
        "MonthlyCharges": payload.monthly_charges,
        "TotalCharges": payload.total_charges,
    }
    return pd.DataFrame([row])


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load the serialized model pipeline once and reuse it between requests."""
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado em {MODEL_FILE}. "
            "Execute `python -m telco_churn_mlops.train_baseline_model` antes de subir a API."
        )
    return joblib.load(MODEL_FILE)


def is_model_available() -> bool:
    """Return whether the model artifact exists locally."""
    return MODEL_FILE.exists()


def predict_churn(payload: CustomerFeatures, threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """Run churn prediction for one customer payload."""
    model = load_model()
    features_df = payload_to_dataframe(payload)

    # predict_proba retorna uma matriz [classe_0, classe_1]. A classe 1 e churn.
    probability = float(model.predict_proba(features_df)[0, 1])
    prediction = int(probability >= threshold)

    return {
        "churn_probability": probability,
        "churn_prediction": prediction,
        "threshold": threshold,
        "model_name": MODEL_NAME,
    }
