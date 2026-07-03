"""Download the Telco Customer Churn dataset to the local raw data folder."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


def download_dataset(force: bool = False) -> Path:
    """Download the dataset if it does not exist locally."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists() and not force:
        print(f"Dataset already exists: {OUTPUT_FILE}")
        return OUTPUT_FILE

    print(f"Downloading dataset to: {OUTPUT_FILE}")
    urlretrieve(DATA_URL, OUTPUT_FILE)
    print("Download completed.")
    return OUTPUT_FILE


if __name__ == "__main__":
    download_dataset()
