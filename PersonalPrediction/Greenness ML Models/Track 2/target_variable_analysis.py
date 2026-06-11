"""
Task 2: Target Variable Analysis for G-score (GSK_dataset.csv)

Generates:
- Histogram
- Density plot
- Boxplot
- Summary statistics (mean, median, std, min, max, skewness)
- Markdown report describing the distribution

Plots are saved to results/EDA/.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "GSK_dataset.csv"
EDA_DIR = BASE_DIR / "results" / "EDA"
REPORT_PATH = BASE_DIR / "target_variable_report.md"

TARGET = "G-score"


def main():
    df = pd.read_csv(DATA_PATH)
    g = df[TARGET]

    # Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(g, bins=20, kde=False, color="steelblue", edgecolor="black")
    plt.title("Histogram of G-score")
    plt.xlabel("G-score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "g_score_histogram.png", dpi=150)
    plt.close()

    # Density plot
    plt.figure(figsize=(8, 5))
    sns.kdeplot(g, fill=True, color="seagreen")
    plt.title("Density Plot of G-score")
    plt.xlabel("G-score")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "g_score_density.png", dpi=150)
    plt.close()

    # Boxplot
    plt.figure(figsize=(6, 5))
    sns.boxplot(y=g, color="lightcoral")
    plt.title("Boxplot of G-score")
    plt.ylabel("G-score")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "g_score_boxplot.png", dpi=150)
    plt.close()

    # Summary statistics
    stats = {
        "Mean": g.mean(),
        "Median": g.median(),
        "Standard Deviation": g.std(),
        "Min": g.min(),
        "Max": g.max(),
        "Skewness": g.skew(),
    }

    print("=" * 60)
    print("G-SCORE SUMMARY STATISTICS")
    print("=" * 60)
    for k, v in stats.items():
        print(f"{k}: {v:.6f}")

    # Distribution interpretation
    skew_val = stats["Skewness"]
    if skew_val > 0.5:
        skew_desc = "right-skewed (positively skewed), with a longer tail toward higher G-score values"
    elif skew_val < -0.5:
        skew_desc = "left-skewed (negatively skewed), with a longer tail toward lower G-score values"
    else:
        skew_desc = "approximately symmetric"

    lines = []
    lines.append("# Target Variable Analysis: G-score\n\n")
    lines.append("## Summary Statistics\n\n")
    lines.append("| Statistic | Value |\n")
    lines.append("|---|---|\n")
    for k, v in stats.items():
        lines.append(f"| {k} | {v:.4f} |\n")

    lines.append("\n## Distribution Description\n\n")
    lines.append(
        f"The G-score distribution has a mean of {stats['Mean']:.4f} and a median of "
        f"{stats['Median']:.4f}. The standard deviation is {stats['Standard Deviation']:.4f}, "
        f"with values ranging from a minimum of {stats['Min']:.4f} to a maximum of "
        f"{stats['Max']:.4f}.\n\n"
    )
    lines.append(
        f"The skewness value is {skew_val:.4f}, indicating that the distribution is "
        f"{skew_desc}. "
    )
    if abs(stats["Mean"] - stats["Median"]) > 0.01:
        lines.append(
            f"This is consistent with the difference observed between the mean "
            f"({stats['Mean']:.4f}) and median ({stats['Median']:.4f}).\n\n"
        )
    else:
        lines.append(
            f"The mean and median are close ({stats['Mean']:.4f} vs {stats['Median']:.4f}), "
            f"which is consistent with a roughly symmetric distribution.\n\n"
        )

    lines.append("## Plots\n\n")
    lines.append("- Histogram: `results/EDA/g_score_histogram.png`\n")
    lines.append("- Density Plot: `results/EDA/g_score_density.png`\n")
    lines.append("- Boxplot: `results/EDA/g_score_boxplot.png`\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")
    print(f"Plots saved to {EDA_DIR}")


if __name__ == "__main__":
    main()
