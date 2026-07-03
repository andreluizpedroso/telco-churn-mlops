"""Feature engineering utilities for tabular churn models."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telco_churn_mlops.data import ID_COLUMN, TARGET_COLUMN


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into model features and target."""
    # customerID identifica o cliente, mas nao deve virar feature do modelo.
    # Churn e o alvo que queremos prever, entao tambem sai de X.
    x = df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return x, y


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical columns for preprocessing."""
    # O preprocessing depende do tipo da coluna: numeros recebem escala,
    # categorias recebem one-hot encoding.
    feature_df = df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    numeric_features = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = feature_df.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric_features, categorical_features


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder compatible with multiple scikit-learn versions."""
    try:
        # scikit-learn mais novo usa sparse_output.
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # scikit-learn mais antigo usa sparse.
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing for numeric and categorical features."""
    numeric_features, categorical_features = get_feature_columns(df)

    # StandardScaler coloca variaveis numericas em escala comparavel, o que
    # ajuda modelos lineares e redes neurais.
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

    # OneHotEncoder transforma categorias em colunas 0/1. handle_unknown evita
    # erro caso apareca uma categoria nova no teste ou em producao.
    categorical_pipeline = Pipeline(steps=[("onehot", make_one_hot_encoder())])

    # ColumnTransformer aplica cada pipeline somente nas colunas corretas.
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
