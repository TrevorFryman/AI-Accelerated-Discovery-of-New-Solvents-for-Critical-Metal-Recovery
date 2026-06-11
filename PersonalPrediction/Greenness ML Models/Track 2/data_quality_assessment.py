"""
Task 3: Data Quality Assessment for GSK_dataset.csv

Performs:
- Missing value detection
- Duplicate record detection
- Invalid SMILES detection (via RDKit)
- Outlier detection in G-score (IQR method)
- Data cleaning recommendations

Generates Data_Quality_Report.md. Does NOT modify the dataset.
"""

import pandas as pd
from pathlib import Path
from rdkit import Chem

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "GSK_dataset.csv"
REPORT_PATH = BASE_DIR / "Data_Quality_Report.md"

TARGET = "G-score"
SMILES_COL = "solvent_SMILES"


def main():
    df = pd.read_csv(DATA_PATH)

    lines = []
    lines.append("# Data Quality Report: GSK_dataset.csv\n\n")

    # 1. Missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    lines.append("## 1. Missing Values\n\n")
    lines.append("| Column | Missing Count |\n")
    lines.append("|---|---|\n")
    for col, count in missing.items():
        lines.append(f"| {col} | {count} |\n")
    lines.append(f"\nTotal missing values: {total_missing}\n\n")

    # 2. Duplicate records
    n_dup_full = df.duplicated().sum()
    n_dup_smiles = df.duplicated(subset=[SMILES_COL]).sum()
    n_dup_cas = df.duplicated(subset=["CAS Number"]).sum()
    lines.append("## 2. Duplicate Records\n\n")
    lines.append(f"- Fully duplicated rows: {n_dup_full}\n")
    lines.append(f"- Duplicate `{SMILES_COL}` values: {n_dup_smiles}\n")
    lines.append(f"- Duplicate `CAS Number` values: {n_dup_cas}\n\n")

    if n_dup_smiles > 0:
        dup_smiles_rows = df[df.duplicated(subset=[SMILES_COL], keep=False)]
        lines.append("Rows with duplicate SMILES:\n\n")
        lines.append("| Unnamed: 0 | solvent_common_name | solvent_SMILES | CAS Number |\n")
        lines.append("|---|---|---|---|\n")
        for _, row in dup_smiles_rows.iterrows():
            lines.append(
                f"| {row['Unnamed: 0']} | {row['solvent_common_name']} | "
                f"{row[SMILES_COL]} | {row['CAS Number']} |\n"
            )
        lines.append("\n")

    if n_dup_cas > 0:
        dup_cas_rows = df[df.duplicated(subset=["CAS Number"], keep=False)]
        lines.append("Rows with duplicate CAS Numbers:\n\n")
        lines.append("| Unnamed: 0 | solvent_common_name | solvent_SMILES | CAS Number |\n")
        lines.append("|---|---|---|---|\n")
        for _, row in dup_cas_rows.iterrows():
            lines.append(
                f"| {row['Unnamed: 0']} | {row['solvent_common_name']} | "
                f"{row[SMILES_COL]} | {row['CAS Number']} |\n"
            )
        lines.append("\n")

    # 3. Invalid SMILES
    lines.append("## 3. Invalid SMILES\n\n")
    invalid_rows = []
    for idx, row in df.iterrows():
        smiles = row[SMILES_COL]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_rows.append(row)

    if invalid_rows:
        lines.append(f"Found {len(invalid_rows)} invalid SMILES string(s):\n\n")
        lines.append("| Unnamed: 0 | solvent_common_name | solvent_SMILES | CAS Number |\n")
        lines.append("|---|---|---|---|\n")
        for row in invalid_rows:
            lines.append(
                f"| {row['Unnamed: 0']} | {row['solvent_common_name']} | "
                f"{row[SMILES_COL]} | {row['CAS Number']} |\n"
            )
        lines.append("\n")
    else:
        lines.append("All SMILES strings parsed successfully with RDKit. No invalid SMILES found.\n\n")

    # 4. Outliers in G-score (IQR method)
    lines.append("## 4. Outliers in G-score (IQR Method)\n\n")
    g = df[TARGET]
    q1 = g.quantile(0.25)
    q3 = g.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[(g < lower_bound) | (g > upper_bound)]

    lines.append(f"- Q1: {q1:.4f}\n")
    lines.append(f"- Q3: {q3:.4f}\n")
    lines.append(f"- IQR: {iqr:.4f}\n")
    lines.append(f"- Lower bound: {lower_bound:.4f}\n")
    lines.append(f"- Upper bound: {upper_bound:.4f}\n")
    lines.append(f"- Number of outliers: {len(outliers)}\n\n")

    if len(outliers) > 0:
        lines.append("Outlier rows:\n\n")
        lines.append("| Unnamed: 0 | solvent_common_name | G-score |\n")
        lines.append("|---|---|---|\n")
        for _, row in outliers.iterrows():
            lines.append(f"| {row['Unnamed: 0']} | {row['solvent_common_name']} | {row[TARGET]:.4f} |\n")
        lines.append("\n")

    # 5. Cleaning recommendations
    lines.append("## 5. Data Cleaning Recommendations\n\n")
    recs = []

    if total_missing == 0:
        recs.append("No missing values were detected, so no imputation is required.")
    else:
        recs.append("Address missing values via imputation or row removal, depending on column importance.")

    if n_dup_full == 0 and n_dup_smiles == 0 and n_dup_cas == 0:
        recs.append("No duplicate rows, SMILES, or CAS numbers were detected.")
    else:
        if n_dup_full > 0:
            recs.append(f"Remove {n_dup_full} fully duplicated row(s).")
        if n_dup_smiles > 0:
            recs.append(
                f"Investigate {n_dup_smiles} duplicate SMILES entries — these may represent the same "
                f"molecule listed under different names/CAS numbers and could bias the model."
            )
        if n_dup_cas > 0:
            recs.append(
                f"Investigate {n_dup_cas} duplicate CAS Number entries for redundant records."
            )

    if invalid_rows:
        recs.append(
            f"Correct or remove {len(invalid_rows)} row(s) with invalid SMILES before any "
            f"RDKit-based featurization, as these will fail molecule parsing."
        )
    else:
        recs.append("All SMILES are RDKit-valid; no SMILES correction needed.")

    if len(outliers) == 0:
        recs.append("No G-score outliers detected via the IQR method; the target distribution appears clean.")
    else:
        recs.append(
            f"Review the {len(outliers)} G-score outlier(s) flagged by the IQR method. These may be "
            f"legitimate extreme values (e.g., highly green or hazardous solvents) rather than errors, "
            f"so consider domain context before removal."
        )

    recs.append(
        "The leading unnamed index column (`Unnamed: 0`) should be dropped before modeling, as it "
        "carries no predictive information."
    )

    for r in recs:
        lines.append(f"- {r}\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")

    print("=" * 60)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 60)
    print(f"Total missing values: {total_missing}")
    print(f"Fully duplicated rows: {n_dup_full}")
    print(f"Duplicate SMILES: {n_dup_smiles}")
    print(f"Duplicate CAS Numbers: {n_dup_cas}")
    print(f"Invalid SMILES: {len(invalid_rows)}")
    print(f"G-score outliers (IQR): {len(outliers)}")
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
