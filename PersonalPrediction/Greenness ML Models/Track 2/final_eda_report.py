"""
Task 5: Final EDA Report for GSK_dataset.csv

Combines all prior EDA analyses (dataset overview, data quality, G-score
distribution, molecular diversity) into a single report, EDA_Report_Final.md.

Also generates publication-quality figures (higher resolution, consistent
styling) saved to results/EDA/.
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
REPORT_PATH = BASE_DIR / "EDA_Report_Final.md"

TARGET = "G-score"
SMILES_COL = "solvent_SMILES"

# Publication-quality plot style
sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
PUB_DPI = 300


def make_publication_figures(df):
    descriptor_cols = ["MolWt", "LogP", "TPSA"]

    # Combined G-score distribution figure (histogram, density, boxplot)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.histplot(df[TARGET], bins=20, kde=False, ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title("Histogram")
    axes[0].set_xlabel("G-score")

    sns.kdeplot(df[TARGET], fill=True, ax=axes[1], color="seagreen")
    axes[1].set_title("Density")
    axes[1].set_xlabel("G-score")

    sns.boxplot(y=df[TARGET], ax=axes[2], color="lightcoral")
    axes[2].set_title("Boxplot")
    axes[2].set_ylabel("G-score")

    fig.suptitle("G-score Distribution Overview", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(EDA_DIR / "publication_gscore_overview.png", dpi=PUB_DPI)
    plt.close(fig)

    # Combined molecular descriptor histograms
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, descriptor_cols):
        sns.histplot(df[col], bins=20, kde=True, ax=ax, color="slateblue", edgecolor="black")
        ax.set_title(col)
    fig.suptitle("Molecular Descriptor Distributions", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(EDA_DIR / "publication_descriptor_overview.png", dpi=PUB_DPI)
    plt.close(fig)

    # Publication correlation matrix
    corr_cols = descriptor_cols + [TARGET]
    corr = df[corr_cols].corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True,
                cbar_kws={"label": "Pearson correlation"})
    plt.title("Correlation Matrix: Descriptors and G-score")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "publication_correlation_matrix.png", dpi=PUB_DPI)
    plt.close()

    return corr


def main():
    df = pd.read_csv(DATA_PATH)

    # Recompute descriptors (needed for correlation + recommendations)
    mols = df[SMILES_COL].apply(Chem.MolFromSmiles)
    df["MolWt"] = mols.apply(Descriptors.MolWt)
    df["LogP"] = mols.apply(Crippen.MolLogP)
    df["TPSA"] = mols.apply(Descriptors.TPSA)

    corr = make_publication_figures(df)

    g = df[TARGET]
    n_dup_smiles = df.duplicated(subset=[SMILES_COL]).sum()
    invalid_smiles = sum(1 for m in df[SMILES_COL].apply(Chem.MolFromSmiles) if m is None)

    q1, q3 = g.quantile(0.25), g.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = ((g < lower) | (g > upper)).sum()

    lines = []
    lines.append("# Final EDA Report: GSK_dataset.csv\n\n")

    # Dataset overview
    lines.append("## 1. Dataset Overview\n\n")
    lines.append(f"- **Rows:** {df.shape[0]}\n")
    lines.append(f"- **Columns:** {df.shape[1]}\n")
    lines.append(
        "- **Columns:** `Unnamed: 0` (row index), `Classification`, `solvent_common_name`, "
        "`IPUAC name`, `solvent_SMILES`, `CAS Number`, `G-score`\n"
    )
    lines.append(
        f"- **Classifications represented:** {df['Classification'].nunique()} "
        f"({', '.join(sorted(df['Classification'].unique()))})\n\n"
    )

    # Data quality findings
    lines.append("## 2. Data Quality Findings\n\n")
    lines.append(f"- Missing values: {df.isnull().sum().sum()}\n")
    lines.append(f"- Fully duplicated rows: {df.duplicated().sum()}\n")
    lines.append(
        f"- Duplicate `solvent_SMILES` values: {n_dup_smiles} "
        "(Ethanol/IMS denatured ethanol, and 2-Methylpentane/Petroleum spirit)\n"
    )
    lines.append(f"- Invalid SMILES (RDKit): {invalid_smiles}\n")
    lines.append(f"- G-score outliers (IQR method): {n_outliers}\n\n")
    lines.append("Full details: see `Data_Quality_Report.md`.\n\n")

    # G-score distribution
    lines.append("## 3. G-score Distribution\n\n")
    lines.append("| Statistic | Value |\n")
    lines.append("|---|---|\n")
    lines.append(f"| Mean | {g.mean():.4f} |\n")
    lines.append(f"| Median | {g.median():.4f} |\n")
    lines.append(f"| Standard Deviation | {g.std():.4f} |\n")
    lines.append(f"| Min | {g.min():.4f} |\n")
    lines.append(f"| Max | {g.max():.4f} |\n")
    lines.append(f"| Skewness | {g.skew():.4f} |\n\n")
    lines.append(
        f"The G-score distribution is approximately symmetric (skewness = {g.skew():.4f}), "
        f"with mean ({g.mean():.4f}) and median ({g.median():.4f}) close together and no "
        f"outliers detected via the IQR method. Full details: see `target_variable_report.md`.\n\n"
    )

    # Molecular diversity analysis
    lines.append("## 4. Molecular Diversity Analysis\n\n")
    lines.append("| Descriptor | Min | Mean | Max | Correlation with G-score |\n")
    lines.append("|---|---|---|---|---|\n")
    for col in ["MolWt", "LogP", "TPSA"]:
        lines.append(
            f"| {col} | {df[col].min():.2f} | {df[col].mean():.2f} | {df[col].max():.2f} | "
            f"{corr.loc[col, TARGET]:.4f} |\n"
        )
    lines.append(
        "\nTPSA shows the strongest correlation with G-score (positive), suggesting that "
        "more polar solvents tend to have higher (greener) G-scores. LogP is moderately "
        "negatively correlated, while MolWt shows essentially no correlation. Full details: "
        "see `molecular_diversity_report.md`.\n\n"
    )

    # Recommendations
    lines.append("## 5. Recommendations for Machine Learning\n\n")
    lines.append("- Drop the unnamed index column (`Unnamed: 0`) before modeling.\n")
    lines.append(
        "- Review the duplicate-SMILES pairs (Ethanol/IMS, 2-Methylpentane/Petroleum spirit) — "
        "consider whether to merge, deduplicate, or retain as distinct samples, since they "
        "represent the same molecule under different commercial names.\n"
    )
    lines.append(
        "- No missing values, invalid SMILES, or G-score outliers were found, so the dataset "
        "is largely ready for featurization without imputation or filtering.\n"
    )
    lines.append(
        "- Molecular descriptors (MolWt, LogP, TPSA) provide a reasonable starting feature set; "
        "TPSA in particular shows a meaningful linear relationship with G-score and should be "
        "retained as a feature.\n"
    )
    lines.append(
        "- Given the small dataset size (154 samples), favor simple, low-variance models "
        "(e.g., linear/regularized regression, small tree ensembles) and use cross-validation "
        "rather than a large held-out test set.\n"
    )
    lines.append(
        "- Consider expanding the feature set with additional RDKit descriptors or "
        "fingerprints, since MolWt/LogP/TPSA alone show moderate correlations with G-score.\n\n"
    )

    # Figures
    lines.append("## 6. Figures\n\n")
    lines.append("Publication-quality figures (saved to `results/EDA/`):\n\n")
    lines.append("- `publication_gscore_overview.png` — G-score histogram, density, and boxplot\n")
    lines.append("- `publication_descriptor_overview.png` — MolWt, LogP, TPSA histograms\n")
    lines.append("- `publication_correlation_matrix.png` — Correlation matrix of descriptors and G-score\n\n")
    lines.append("Additional individual EDA figures generated in earlier tasks are also available in `results/EDA/`.\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")

    print("=" * 60)
    print("FINAL EDA REPORT GENERATED")
    print("=" * 60)
    print(f"Report written to {REPORT_PATH}")
    print(f"Publication figures saved to {EDA_DIR}")


if __name__ == "__main__":
    main()
