from fastapi.testclient import TestClient

from telco_churn_mlops.api import app
from telco_churn_mlops.inference import payload_to_dataframe
from telco_churn_mlops.schemas import CustomerFeatures


def example_payload() -> dict:
    # Payload minimo representando um cliente no formato esperado pela API.
    return {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure": 12,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 89.1,
        "total_charges": 950.5,
    }


def test_health_endpoint_returns_status():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_loaded" in response.json()


def test_payload_to_dataframe_uses_training_column_names():
    payload = CustomerFeatures(**example_payload())

    df = payload_to_dataframe(payload)

    assert df.loc[0, "SeniorCitizen"] == 0
    assert df.loc[0, "MonthlyCharges"] == 89.1
    assert df.loc[0, "TotalCharges"] == 950.5
    assert "senior_citizen" not in df.columns


def test_predict_endpoint_validates_payload_before_model_loading():
    client = TestClient(app)
    invalid_payload = example_payload()
    invalid_payload["monthly_charges"] = -10

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422
