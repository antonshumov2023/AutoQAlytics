#!/usr/bin/env python3
"""
dataset_overview.py

Reads a JSONL issues dataset and prints counts of issues
by category, by severity, and a crosstabulation of both.
"""

import argparse
import pandas as pd

def overview(input_path: str):
    # 1. Load JSONL dataset
    df = pd.read_json(input_path, lines=True)

    # 2. Basic sanity
    total = len(df)
    print(f"\nTotal issues loaded: {total}\n")

    # 3. Counts by category
    print("Issues per Category:")
    cat_counts = df['category'].fillna("Unknown").value_counts(dropna=False)
    print(cat_counts.to_string(), "\n")

    # 4. Counts by severity
    print("Issues per Severity:")
    sev_counts = df['severity'].fillna("Unknown").value_counts(dropna=False)
    print(sev_counts.to_string(), "\n")

    # 5. Cross-tabulation (Category × Severity)
    print("Category × Severity Crosstab:")
    crosstab = pd.crosstab(
        df['category'].fillna("Unknown"),
        df['severity'].fillna("Unknown"),
        margins=True,
        dropna=False
    )
    print(crosstab.to_string(), "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overview of issues dataset (counts by category and severity)")
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to issues_dataset.jsonl")
    args = parser.parse_args()

    overview(args.input)

