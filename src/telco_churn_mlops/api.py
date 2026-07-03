"""FastAPI application for churn inference."""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request

from telco_churn_mlops.inference import is_model_available, predict_churn
from telco_churn_mlops.schemas import CustomerFeatures, HealthResponse, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Telco Churn MLOps API",
    description="API de inferencia para previsao de churn em telecomunicacoes.",
    version="0.1.0",
)


@app.middleware("http")
async def log_request_latency(request: Request, call_next):
    """Log method, path, status code, and latency for each request."""
    start_time = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start_time) * 1000

    # Logging estruturado simples para facilitar monitoramento em producao.
    logger.info(
        "request_completed method=%s path=%s status_code=%s latency_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return API health and whether the model artifact is available."""
    return HealthResponse(status="ok", model_loaded=is_model_available())


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures) -> PredictionResponse:
    """Predict churn probability for one customer."""
    result = predict_churn(payload)
    return PredictionResponse(**result)
