"""Project paths and reproducibility settings."""

from pathlib import Path

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA_FILE = RAW_DATA_DIR / "Telco-Customer-Churn.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"
BASELINE_RESULTS_FILE = DOCS_DIR / "baseline-results.json"
