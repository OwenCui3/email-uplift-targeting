"""Data loading utilities for the Hillstrom e-mail uplift project.

Dataset: Kevin Hillstrom's MineThatData E-Mail Analytics challenge (March 2008).
64,000 customers randomized 1/3 : 1/3 : 1/3 into Mens E-Mail / Womens E-Mail /
No E-Mail (control). Outcomes measured over the following two weeks.
"""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

DATA_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)

# Project root = one level above src/
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "hillstrom.csv"

CONTROL = "No E-Mail"
ARMS = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]

# Pre-treatment covariates (everything measured BEFORE the email was sent).
# 'visit', 'conversion', 'spend' are outcomes and must never appear here.
COVARIATES = ["recency", "history", "mens", "womens", "newbie",
              "history_segment", "zip_code", "channel"]
OUTCOMES = ["visit", "conversion", "spend"]


def download_data(url: str = DATA_URL, dest: Path = RAW_PATH, force: bool = False) -> Path:
    """Download the raw CSV once; skip if it already exists (idempotent)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"Already downloaded: {dest}")
        return dest
    print(f"Downloading {url} -> {dest}")
    urlretrieve(url, dest)
    return dest


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the CSV, normalize column names, set categorical dtypes, sanity-check."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    for col in ["history_segment", "zip_code", "channel", "segment"]:
        df[col] = df[col].astype("category")

    # Fail loudly if the file is not what we expect -- cheap insurance.
    expected_cols = set(COVARIATES + OUTCOMES + ["segment"])
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    if set(df["segment"].cat.categories) != set(ARMS):
        raise ValueError(f"Unexpected treatment arms: {list(df['segment'].cat.categories)}")
    if len(df) != 64_000:
        print(f"WARNING: expected 64,000 rows, got {len(df):,}")
    return df
