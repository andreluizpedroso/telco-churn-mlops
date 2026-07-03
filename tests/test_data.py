import pandas as pd

from telco_churn_mlops.data import clean_telco_data, split_train_test


def test_clean_telco_data_converts_total_charges_and_target():
    # Este dataframe pequeno simula o problema real da base: TotalCharges vem
    # como texto e pode conter valor em branco.
    df = pd.DataFrame(
        {
            "customerID": ["a", "b", "c"],
            "TotalCharges": ["10.5", " ", "30.0"],
            "Churn": ["Yes", "No", "No"],
        }
    )

    cleaned = clean_telco_data(df)

    # A linha com TotalCharges em branco deve sair da base limpa.
    assert cleaned.shape[0] == 2

    # TotalCharges passa a ser numerico e Churn vira alvo binario.
    assert cleaned["TotalCharges"].tolist() == [10.5, 30.0]
    assert cleaned["Churn"].tolist() == [1, 0]


def test_split_train_test_preserves_target_classes(sample_churn_dataframe):
    # Criamos uma base balanceada artificial para confirmar que o split
    # estratificado mantem as duas classes em treino e teste.
    train_df, test_df = split_train_test(sample_churn_dataframe, test_size=0.25, random_state=42)

    assert train_df.shape[0] == 15
    assert test_df.shape[0] == 5

    # Os dois subconjuntos precisam conter exemplos positivos e negativos.
    assert set(train_df["Churn"]) == {0, 1}
    assert set(test_df["Churn"]) == {0, 1}
