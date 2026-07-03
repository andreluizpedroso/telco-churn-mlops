"""Data loading, validation, and train/test split utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from telco_churn_mlops.config import (
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_FILE,
    TEST_DATA_FILE,
    TRAIN_DATA_FILE,
)

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


@dataclass(frozen=True)
class DatasetSummary:
    """Small summary used in docs and logs."""

    # O dataclass agrupa numeros importantes da base sem precisar carregar o
    # dataframe inteiro em documentos ou logs.
    rows: int
    columns: int
    churn_rate: float
    positive_count: int
    negative_count: int


def load_raw_data(path=RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV."""
    # A leitura fica em uma funcao propria para facilitar testes e troca futura
    # da origem dos dados, por exemplo S3, banco ou outro CSV.
    return pd.read_csv(path)


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean known issues from the Telco Customer Churn dataset."""
    # Trabalhamos em uma copia para nao alterar o dataframe original recebido.
    cleaned = df.copy()

    # No CSV original, TotalCharges vem como texto e alguns registros possuem
    # espacos em branco. errors="coerce" transforma valores invalidos em NaN.
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    # As linhas sem TotalCharges representam clientes sem cobranca total
    # registrada. Para o baseline, removemos esses poucos casos.
    cleaned = cleaned.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    # Modelos de classificacao binaria esperam o alvo em formato numerico:
    # 0 para nao churn e 1 para churn.
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype(int)
    return cleaned


def summarize_dataset(df: pd.DataFrame) -> DatasetSummary:
    """Return a compact dataset summary."""
    # Como o alvo ja foi convertido para 0/1, a soma conta os casos positivos.
    positives = int(df[TARGET_COLUMN].sum())
    negatives = int((df[TARGET_COLUMN] == 0).sum())
    return DatasetSummary(
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        churn_rate=float(df[TARGET_COLUMN].mean()),
        positive_count=positives,
        negative_count=negatives,
    )


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified train/test split."""
    # stratify mantem uma proporcao parecida de churn e nao churn no treino e
    # no teste. Isso evita uma avaliacao distorcida em bases desbalanceadas.
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_processed_data() -> tuple[pd.DataFrame, pd.DataFrame, DatasetSummary]:
    """Load, clean, split, and save processed train/test datasets."""
    # Esta funcao orquestra o fluxo completo de dados da Sprint 2.
    raw_df = load_raw_data()
    cleaned_df = clean_telco_data(raw_df)
    train_df, test_df = split_train_test(cleaned_df)

    # Criamos a pasta antes de salvar para que o comando funcione mesmo em um
    # clone novo do repositorio.
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_DATA_FILE, index=False)
    test_df.to_csv(TEST_DATA_FILE, index=False)

    return train_df, test_df, summarize_dataset(cleaned_df)
