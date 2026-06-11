"""
Task 4: Molecular Diversity Analysis for GSK_dataset.csv

Using RDKit:
- Convert all SMILES to molecules
- Calculate Molecular Weight, LogP, TPSA
- Create histograms, pair plots, and a correlation matrix
- Generate a molecular diversity report

Plots are saved to results/EDA/.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "GSK_dataset.csv"
EDA_DIR = BASE_DIR / "results" / "EDA"
REPORT_PATH = BASE_DIR / "molecular_diversity_report.md"

SMILES_COL = "solvent_SMILES"
TARGET = "G-score"


def main():
    df = pd.read_csv(DATA_PATH)

    # Convert SMILES to molecules and compute descriptors
    mols = df[SMILES_COL].apply(Chem.MolFromSmiles)
    df["MolWt"] = mols.apply(Descriptors.MolWt)
    df["LogP"] = mols.apply(Crippen.MolLogP)
    df["TPSA"] = mols.apply(Descriptors.TPSA)

    descriptor_cols = ["MolWt", "LogP", "TPSA"]

    # Histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, descriptor_cols):
        sns.histplot(df[col], bins=20, kde=True, ax=ax, color="slateblue", edgecolor="black")
        ax.set_title(f"Histogram of {col}")
        ax.set_xlabel(col)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "molecular_descriptor_histograms.png", dpi=150)
    plt.close()

    # Pair plot (descriptors + target)
    pairplot_cols = descriptor_cols + [TARGET]
    pair_grid = sns.pairplot(df[pairplot_cols], diag_kind="kde", corner=True)
    pair_grid.fig.suptitle("Pair Plot: Molecular Descriptors and G-score", y=1.02)
    pair_grid.savefig(EDA_DIR / "molecular_descriptor_pairplot.png", dpi=150)
    plt.close("all")

    # Correlation matrix
    corr = df[pairplot_cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Correlation Matrix: Descriptors and G-score")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "molecular_descriptor_correlation.png", dpi=150)
    plt.close()

    # Summary statistics for descriptors
    desc_stats = df[descriptor_cols].describe()

    print("=" * 60)
    print("MOLECULAR DESCRIPTOR SUMMARY STATISTICS")
    print("=" * 60)
    print(desc_stats)
    print("\nCorrelation matrix:")
    print(corr)

    # Build report
    lines = []
    lines.append("# Molecular Diversity Report\n\n")
    lines.append(
        "All SMILES strings in GSK_dataset.csv were converted to RDKit molecule objects, "
        "and the following descriptors were calculated for each solvent: Molecular Weight "
        "(MolWt), Octanol-Water Partition Coefficient (LogP), and Topological Polar Surface "
        "Area (TPSA).\n\n"
    )

    lines.append("## Descriptor Summary Statistics\n\n")
    lines.append("| Statistic | MolWt | LogP | TPSA |\n")
    lines.append("|---|---|---|---|\n")
    for stat in desc_stats.index:
        row = desc_stats.loc[stat]
        lines.append(f"| {stat} | {row['MolWt']:.4f} | {row['LogP']:.4f} | {row['TPSA']:.4f} |\n")

    lines.append("\n## Correlation with G-score\n\n")
    lines.append("| Descriptor | Correlation with G-score |\n")
    lines.append("|---|---|\n")
    for col in descriptor_cols:
        lines.append(f"| {col} | {corr.loc[col, TARGET]:.4f} |\n")

    lines.append("\n## Plots\n\n")
    lines.append("- Histograms: `results/EDA/molecular_descriptor_histograms.png`\n")
    lines.append("- Pair plot: `results/EDA/molecular_descriptor_pairplot.png`\n")
    lines.append("- Correlation matrix: `results/EDA/molecular_descriptor_correlation.png`\n")

    lines.append("\n## Diversity Observations\n\n")
    lines.append(
        f"- Molecular Weight ranges from {df['MolWt'].min():.2f} to {df['MolWt'].max():.2f} "
        f"(mean {df['MolWt'].mean():.2f}), indicating the dataset spans both small molecules "
        f"(e.g., water) and larger solvents.\n"
    )
    lines.append(
        f"- LogP ranges from {df['LogP'].min():.2f} to {df['LogP'].max():.2f} "
        f"(mean {df['LogP'].mean():.2f}), reflecting a mix of hydrophilic and lipophilic solvents.\n"
    )
    lines.append(
        f"- TPSA ranges from {df['TPSA'].min():.2f} to {df['TPSA'].max():.2f} "
        f"(mean {df['TPSA'].mean():.2f}), reflecting variation in polar functional groups "
        f"across the solvent set.\n"
    )

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")
    print(f"Plots saved to {EDA_DIR}")


if __name__ == "__main__":
    main()
