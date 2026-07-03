"""Project paths and reproducibility settings."""

from pathlib import Path

# Seed unica do projeto. Usar sempre o mesmo valor ajuda a reproduzir splits,
# treinos e comparacoes de modelos.
RANDOM_STATE = 42

# __file__ aponta para este arquivo. parents[2] sobe de
# src/telco_churn_mlops/config.py para a raiz do repositorio.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Pastas principais do projeto. Centralizar os caminhos aqui evita strings
# repetidas espalhadas pelo codigo.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = PROJECT_ROOT / "models"

# Arquivos usados pela Sprint 2. Os dados brutos e processados ficam fora do
# Git, mas o caminho padronizado permite rodar o pipeline localmente.
RAW_DATA_FILE = RAW_DATA_DIR / "Telco-Customer-Churn.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"
BASELINE_RESULTS_FILE = DOCS_DIR / "baseline-results.json"
