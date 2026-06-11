"""
Task 1: Dataset Inspection for GSK_dataset.csv

Performs an initial inspection of the dataset:
- dimensions
- column names
- data types
- missing values
- duplicate rows
- first 10 rows

No modeling is performed.
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "GSK_dataset.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "dataset_summary.md"


def main():
    df = pd.read_csv(DATA_PATH)

    lines = []
    lines.append("# Dataset Summary: GSK_dataset.csv\n")

    # 1. Dimensions
    lines.append("## 1. Dataset Dimensions\n")
    lines.append(f"- Rows: {df.shape[0]}\n")
    lines.append(f"- Columns: {df.shape[1]}\n")

    # 2. Column names
    lines.append("\n## 2. Column Names\n")
    for col in df.columns:
        lines.append(f"- {col}\n")

    # 3. Data types
    lines.append("\n## 3. Data Types\n")
    lines.append("| Column | Dtype |\n")
    lines.append("|---|---|\n")
    for col, dtype in df.dtypes.items():
        lines.append(f"| {col} | {dtype} |\n")

    # 4. Missing values
    lines.append("\n## 4. Missing Values\n")
    missing = df.isnull().sum()
    lines.append("| Column | Missing Count |\n")
    lines.append("|---|---|\n")
    for col, count in missing.items():
        lines.append(f"| {col} | {count} |\n")
    lines.append(f"\nTotal missing values: {missing.sum()}\n")

    # 5. Duplicate rows
    lines.append("\n## 5. Duplicate Rows\n")
    n_duplicates = df.duplicated().sum()
    lines.append(f"- Number of fully duplicated rows: {n_duplicates}\n")

    # 6. First 10 rows
    lines.append("\n## 6. First 10 Rows\n")
    head = df.head(10)
    lines.append("| " + " | ".join(str(c) for c in head.columns) + " |\n")
    lines.append("|" + "---|" * len(head.columns) + "\n")
    for _, row in head.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |\n")

    # Print to console as well
    print("=" * 60)
    print("DATASET DIMENSIONS")
    print("=" * 60)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n" + "=" * 60)
    print("COLUMN NAMES")
    print("=" * 60)
    print(list(df.columns))

    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(missing)

    print("\n" + "=" * 60)
    print("DUPLICATE ROWS")
    print("=" * 60)
    print(f"Number of duplicate rows: {n_duplicates}")

    print("\n" + "=" * 60)
    print("FIRST 10 ROWS")
    print("=" * 60)
    print(df.head(10))

    OUTPUT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"\nSummary written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
