import pandas as pd

from telco_churn_mlops.data import clean_telco_data, split_train_test


def test_clean_telco_data_converts_total_charges_and_target():
    df = pd.DataFrame(
        {
            "customerID": ["a", "b", "c"],
            "TotalCharges": ["10.5", " ", "30.0"],
            "Churn": ["Yes", "No", "No"],
        }
    )

    cleaned = clean_telco_data(df)

    assert cleaned.shape[0] == 2
    assert cleaned["TotalCharges"].tolist() == [10.5, 30.0]
    assert cleaned["Churn"].tolist() == [1, 0]


def test_split_train_test_preserves_target_classes():
    df = pd.DataFrame(
        {
            "customerID": [f"id-{i}" for i in range(20)],
            "TotalCharges": [float(i) for i in range(20)],
            "Churn": [0] * 10 + [1] * 10,
        }
    )

    train_df, test_df = split_train_test(df, test_size=0.25, random_state=42)

    assert train_df.shape[0] == 15
    assert test_df.shape[0] == 5
    assert set(train_df["Churn"]) == {0, 1}
    assert set(test_df["Churn"]) == {0, 1}
