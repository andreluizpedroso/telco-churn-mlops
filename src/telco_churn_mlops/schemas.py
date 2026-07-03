"""Pydantic schemas used by the inference API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Input payload for one customer churn prediction."""

    model_config = ConfigDict(extra="forbid")

    gender: str = Field(examples=["Female"])
    senior_citizen: int = Field(ge=0, le=1, examples=[0])
    partner: str = Field(examples=["Yes"])
    dependents: str = Field(examples=["No"])
    tenure: int = Field(ge=0, examples=[12])
    phone_service: str = Field(examples=["Yes"])
    multiple_lines: str = Field(examples=["No"])
    internet_service: str = Field(examples=["Fiber optic"])
    online_security: str = Field(examples=["No"])
    online_backup: str = Field(examples=["Yes"])
    device_protection: str = Field(examples=["No"])
    tech_support: str = Field(examples=["No"])
    streaming_tv: str = Field(examples=["Yes"])
    streaming_movies: str = Field(examples=["Yes"])
    contract: str = Field(examples=["Month-to-month"])
    paperless_billing: str = Field(examples=["Yes"])
    payment_method: str = Field(examples=["Electronic check"])
    monthly_charges: float = Field(gt=0, examples=[89.1])
    total_charges: float = Field(ge=0, examples=[950.5])


class PredictionResponse(BaseModel):
    """Response returned by the prediction endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    churn_probability: float
    churn_prediction: int
    threshold: float
    model_name: str


class HealthResponse(BaseModel):
    """Response returned by the health endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
