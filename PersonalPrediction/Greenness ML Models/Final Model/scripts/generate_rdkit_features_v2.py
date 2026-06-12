"""
RDKit Descriptor Pipeline - Feature Generation (v2, Final Model)

v2 of the original Track 2 generate_rdkit_features.py: expands the
descriptor set from 7 to 13 named, interpretable RDKit descriptors
(Step 2 of the fine-tuning plan).

Loads GSK_dataset.csv, identifies the SMILES column, and computes RDKit
molecular descriptors for each solvent:
- Molecular Weight (MolWt)
- LogP (Crippen MolLogP)
- TPSA
- Number of Hydrogen Bond Donors (NumHDonors)
- Number of Hydrogen Bond Acceptors (NumHAcceptors)
- Number of Rotatable Bonds (NumRotatableBonds)
- Ring Count (RingCount)
- Number of Aromatic Rings (NumAromaticRings)
- Fraction of sp3 Carbons (FractionCSP3)
- Molar Refractivity (MolMR)
- Heavy Atom Count (HeavyAtomCount)
- Number of Aliphatic Rings (NumAliphaticRings)
- BertzCT (topological complexity index)

Outputs:
- descriptors/RDKit_Features.csv (descriptors + identifiers + target)
- reports/RDKit_Features_Summary.md (validation / summary report)

This pipeline is standalone and does not reference Morgan Fingerprint
or ChemBERTa pipelines.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "GSK_dataset.csv"
DESCRIPTORS_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "results"

OUTPUT_PATH = DESCRIPTORS_DIR / "RDKit_Features.csv"
REPORT_PATH = REPORTS_DIR / "RDKit_Features_Summary.md"

TARGET = "G-score"

DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "LogP": Crippen.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Lipinski.NumHDonors,
    "NumHAcceptors": Lipinski.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
    # Step 2 (fine-tuning plan): additional named, interpretable descriptors
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "FractionCSP3": Descriptors.FractionCSP3,
    "MolMR": Crippen.MolMR,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "NumAliphaticRings": Descriptors.NumAliphaticRings,
    "BertzCT": Descriptors.BertzCT,
}


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

    mols = df[smiles_col].apply(Chem.MolFromSmiles)
    invalid_indices = df.index[mols.isnull()].tolist()

    descriptor_data = {}
    for name, func in DESCRIPTOR_FUNCS.items():
        descriptor_data[name] = mols.apply(lambda m: func(m) if m is not None else np.nan)

    desc_df = pd.DataFrame(descriptor_data, index=df.index)

    id_cols = [c for c in ["solvent_common_name", "CAS Number", smiles_col] if c in df.columns]
    output_df = pd.concat([df[id_cols], df[[TARGET]], desc_df], axis=1)

    if invalid_indices:
        print(f"Warning: {len(invalid_indices)} row(s) had invalid SMILES and were dropped.")
        output_df = output_df.drop(index=invalid_indices)

    DESCRIPTORS_DIR.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    descriptor_cols = list(DESCRIPTOR_FUNCS.keys())
    desc_stats = output_df[descriptor_cols].describe()

    print("=" * 60)
    print("RDKIT DESCRIPTOR SUMMARY STATISTICS")
    print("=" * 60)
    print(desc_stats)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# RDKit Descriptor Feature Summary\n\n")
    lines.append(f"- SMILES column used: `{smiles_col}`\n")
    lines.append(f"- Total molecules: {len(df)}\n")
    lines.append(f"- Invalid SMILES (dropped): {len(invalid_indices)}\n")
    lines.append(f"- Final feature matrix shape: {output_df.shape[0]} rows x {len(descriptor_cols)} descriptor columns\n\n")

    lines.append("## Descriptors Computed\n\n")
    for name in descriptor_cols:
        lines.append(f"- {name}\n")

    lines.append("\n## Descriptor Summary Statistics\n\n")
    lines.append("| Statistic | " + " | ".join(descriptor_cols) + " |\n")
    lines.append("|---" * (len(descriptor_cols) + 1) + "|\n")
    for stat in desc_stats.index:
        row = desc_stats.loc[stat]
        lines.append(f"| {stat} | " + " | ".join(f"{row[c]:.4f}" for c in descriptor_cols) + " |\n")

    lines.append("\n## Missing Values After Computation\n\n")
    missing = output_df[descriptor_cols].isnull().sum()
    lines.append("| Descriptor | Missing Count |\n")
    lines.append("|---|---|\n")
    for col, count in missing.items():
        lines.append(f"| {col} | {count} |\n")

    if invalid_indices:
        lines.append("\n## Invalid SMILES Rows (dropped)\n\n")
        for idx in invalid_indices:
            lines.append(f"- Row index {idx}: `{df.loc[idx, smiles_col]}`\n")

    lines.append("\n## Output\n\n")
    lines.append("- Feature matrix saved to `descriptors/RDKit_Features.csv`\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print("RDKIT DESCRIPTOR GENERATION COMPLETE")
    print("=" * 60)
    print(f"Molecules processed: {len(df)}")
    print(f"Invalid SMILES dropped: {len(invalid_indices)}")
    print(f"Missing values after computation:\n{missing}")
    print(f"Feature matrix shape: {output_df.shape}")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
