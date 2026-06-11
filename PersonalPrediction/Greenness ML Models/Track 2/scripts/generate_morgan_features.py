"""
Morgan Fingerprint Pipeline - Feature Generation

Loads GSK_dataset.csv, identifies the SMILES column, and generates
Morgan fingerprints (radius=2, nBits=2048) for each solvent using RDKit.

Outputs:
- descriptors/Morgan_Features.csv (fingerprint bits + identifiers + target)
- reports/Morgan_Features_Summary.md (validation / summary report)

This pipeline is standalone and does not reference RDKit-descriptor
or ChemBERTa pipelines.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR.parent / "GSK_dataset.csv"
DESCRIPTORS_DIR = BASE_DIR / "descriptors"
REPORTS_DIR = BASE_DIR / "reports"

OUTPUT_PATH = DESCRIPTORS_DIR / "Morgan_Features.csv"
REPORT_PATH = REPORTS_DIR / "Morgan_Features_Summary.md"

RADIUS = 2
N_BITS = 2048
TARGET = "G-score"


def main():
    df = pd.read_csv(DATA_PATH)

    # Identify the SMILES column dynamically (do not assume "SMILES")
    smiles_col = None
    for col in df.columns:
        if "smiles" in col.lower():
            smiles_col = col
            break
    if smiles_col is None:
        raise ValueError(
            f"No SMILES column found in dataset. Available columns: {list(df.columns)}"
        )
    print(f"Using SMILES column: '{smiles_col}'")

    # Generate Morgan fingerprints
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)

    fp_rows = []
    invalid_indices = []
    for idx, smiles in df[smiles_col].items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_indices.append(idx)
            fp_rows.append([np.nan] * N_BITS)
            continue
        fp = generator.GetFingerprint(mol)
        fp_rows.append(list(fp))

    fp_array = np.array(fp_rows, dtype=float)
    fp_cols = [f"morgan_bit_{i}" for i in range(N_BITS)]
    fp_df = pd.DataFrame(fp_array, columns=fp_cols, index=df.index)

    # Combine identifiers, target, and fingerprint bits
    id_cols = [c for c in ["solvent_common_name", "CAS Number", smiles_col] if c in df.columns]
    output_df = pd.concat([df[id_cols], df[[TARGET]], fp_df], axis=1)

    if invalid_indices:
        print(f"Warning: {len(invalid_indices)} row(s) had invalid SMILES and were dropped.")
        output_df = output_df.drop(index=invalid_indices)

    DESCRIPTORS_DIR.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    # Validation / summary report
    bit_sums = fp_array[~np.isnan(fp_array).any(axis=1)].sum(axis=0)
    n_active_bits = (bit_sums > 0).sum()
    n_zero_bits = (bit_sums == 0).sum()
    bits_per_molecule = fp_array[~np.isnan(fp_array).any(axis=1)].sum(axis=1)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Morgan Fingerprint Feature Summary\n\n")
    lines.append(f"- SMILES column used: `{smiles_col}`\n")
    lines.append(f"- Radius: {RADIUS}\n")
    lines.append(f"- nBits: {N_BITS}\n")
    lines.append(f"- Total molecules: {len(df)}\n")
    lines.append(f"- Invalid SMILES (dropped): {len(invalid_indices)}\n")
    lines.append(f"- Final feature matrix shape: {output_df.shape[0]} rows x {N_BITS} bit columns\n\n")

    lines.append("## Bit Activity\n\n")
    lines.append(f"- Bits set at least once across the dataset: {n_active_bits} / {N_BITS}\n")
    lines.append(f"- Bits never set (all-zero columns): {n_zero_bits} / {N_BITS}\n\n")

    lines.append("## Bits-per-Molecule Statistics\n\n")
    lines.append("| Statistic | Value |\n")
    lines.append("|---|---|\n")
    lines.append(f"| Mean | {bits_per_molecule.mean():.2f} |\n")
    lines.append(f"| Median | {np.median(bits_per_molecule):.2f} |\n")
    lines.append(f"| Min | {bits_per_molecule.min():.0f} |\n")
    lines.append(f"| Max | {bits_per_molecule.max():.0f} |\n\n")

    if invalid_indices:
        lines.append("## Invalid SMILES Rows (dropped)\n\n")
        for idx in invalid_indices:
            lines.append(f"- Row index {idx}: `{df.loc[idx, smiles_col]}`\n")
        lines.append("\n")

    lines.append("## Output\n\n")
    lines.append(f"- Feature matrix saved to `descriptors/Morgan_Features.csv`\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")

    print("=" * 60)
    print("MORGAN FINGERPRINT GENERATION COMPLETE")
    print("=" * 60)
    print(f"Molecules processed: {len(df)}")
    print(f"Invalid SMILES dropped: {len(invalid_indices)}")
    print(f"Active bits: {n_active_bits} / {N_BITS}")
    print(f"Feature matrix shape: {output_df.shape}")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
