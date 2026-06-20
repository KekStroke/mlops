"""
Create ClearML Dataset from UCI SMS Spam Collection.

Usage:
    python create_dataset.py
    python create_dataset.py --version 2.0.0
"""

import argparse
import io
import os
import sys
import urllib.request
import zipfile

import pandas as pd
from clearml import Dataset


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
CSV_FILENAME = "sms_spam.csv"

DATASET_PROJECT = "Course MLOps"
DATASET_NAME = "SMS Spam Dataset"

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00228/smsspamcollection.zip"
)
RAW_FILENAME = "SMSSpamCollection"


def main():
    args = parse_args()

    try:
        df = fetch_and_prepare_data()
    except Exception as e:
        print(f"Error fetching data from UCI repository: {e}")
        sys.exit(1)

    csv_path = save_csv(df)
    print(f"CSV saved to {csv_path}  ({len(df)} rows)")

    dataset_id = create_clearml_dataset(csv_path, version=args.version)

    print("=" * 60)
    print("Dataset created successfully!")
    print(f"  dataset_id  = {dataset_id}")
    print(f"  version     = {args.version}")
    print("=" * 60)
    print("\nCopy dataset_id and pass it to train.py:")
    print(f'  python train.py --dataset-id "{dataset_id}"\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch UCI SMS Spam dataset and upload it to ClearML.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help='ClearML dataset version (default: "1.0.0")',
    )
    return parser.parse_args()


def fetch_and_prepare_data() -> pd.DataFrame:
    """Download UCI SMS Spam Collection and return a DataFrame with label/text columns."""
    print("Downloading SMS Spam Collection from UCI...")
    resp = urllib.request.urlopen(UCI_URL)
    zip_bytes = resp.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        with zf.open(RAW_FILENAME) as f:
            df = pd.read_csv(
                f,
                sep="\t",
                header=None,
                names=["label", "text"],
                encoding="utf-8",
            )

    label_map = {"ham": 0, "spam": 1}
    df["label"] = df["label"].str.strip().str.lower().map(label_map)
    df = df.dropna(subset=["label", "text"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df


def save_csv(df: pd.DataFrame) -> str:
    """Save DataFrame to CSV, creating directories as needed."""
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, CSV_FILENAME)
    df.to_csv(csv_path, index=False)
    return csv_path


def create_clearml_dataset(csv_path: str, version: str) -> str:
    """Create and upload a ClearML Dataset, return its id."""
    print(f"Creating ClearML dataset '{DATASET_NAME}' (version={version})...")

    dataset = Dataset.create(
        dataset_project=DATASET_PROJECT,
        dataset_name=DATASET_NAME,
        dataset_version=version,
    )

    dataset.add_files(path=csv_path)

    try:
        preview = pd.read_csv(csv_path, nrows=10)
        dataset.get_logger().report_table(
            title="Dataset preview",
            series="first_10_rows",
            table_plot=preview,
        )
    except Exception as e:
        print(f"Warning: could not log dataset preview: {e}")

    print("Uploading dataset (this may take a moment)...")
    dataset.upload()
    dataset.finalize()

    return dataset.id


if __name__ == "__main__":
    main()
