import pandas as pd
import pytest


@pytest.fixture
def sample_churn_dataframe():
    # Dataset artificial reutilizavel em testes de split estratificado.
    return pd.DataFrame(
        {
            "customerID": [f"id-{i}" for i in range(20)],
            "TotalCharges": [float(i) for i in range(20)],
            "Churn": [0] * 10 + [1] * 10,
        }
    )
