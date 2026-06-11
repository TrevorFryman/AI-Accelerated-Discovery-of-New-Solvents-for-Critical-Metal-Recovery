"""
exploratory_analysis.py
=======================
Phase 1: Exploratory Data Analysis of the GSK Solvent Sustainability Dataset.

This script performs a comprehensive EDA of the GSK solvent dataset:
  1. Loads and inspects the raw CSV.
  2. Reports dataset shape, column names, dtypes, and missing values.
  3. Detects duplicate rows and duplicate SMILES strings.
  4. Validates every SMILES string with RDKit and flags failures.
  5. Computes quick RDKit molecular properties (MW, logP, HBA, HBD, TPSA, RotBonds).
  6. Produces seven publication-quality figures saved to figures/.
  7. Auto-generates EDA_Report.md summarising all findings.

Usage:
    cd GSK_GScore_Prediction
    python exploratory_analysis.py

Outputs:
    figures/fig01_gscore_histogram.png
    figures/fig02_gscore_boxplot.png
    figures/fig03_gscore_violin_by_class.png
    figures/fig04_gscore_strip_by_class.png
    figures/fig05_correlation_heatmap.png
    figures/fig06_missing_values.png
    figures/fig07_gscore_box_by_class.png
    EDA_Report.md
    logs/eda_run.log
"""

# =============================================================================
# Standard library
# =============================================================================
import io
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

# =============================================================================
# Third-party — numeric / visualisation
# =============================================================================
import matplotlib
matplotlib.use("Agg")              # Non-interactive backend; safe in scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# =============================================================================
# RDKit — cheminformatics
# =============================================================================
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import (
        CalcTPSA,
        CalcNumRotatableBonds,
        CalcNumHBD,
        CalcNumHBA,
    )
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn(
        "RDKit is not installed. SMILES validation and molecular property "
        "calculations will be skipped. Install via: conda install -c conda-forge rdkit",
        stacklevel=2,
    )

# =============================================================================
# Local configuration
# =============================================================================
# Allow running from the project root or one level up
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from config import DIRS, DATA_PATH, COL_CLASS, COL_NAME, COL_SMILES, COL_TARGET

# =============================================================================
# Logging setup
# =============================================================================
LOG_PATH = DIRS["logs"] / "eda_run.log"

# Force stdout to UTF-8 on Windows (avoids cp1252 UnicodeEncodeError for
# box-drawing / arrow characters in log messages)
_stdout_utf8 = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(_stdout_utf8),
    ],
)
log = logging.getLogger(__name__)

# =============================================================================
# Matplotlib / Seaborn global style
# =============================================================================
PALETTE = "husl"
FIGURE_DPI = 150
FIGURE_FORMAT = "png"

plt.rcParams.update({
    "figure.facecolor":    "#0f1117",
    "axes.facecolor":      "#1a1d27",
    "axes.edgecolor":      "#3a3d4d",
    "axes.labelcolor":     "#e0e0e0",
    "axes.titlecolor":     "#ffffff",
    "axes.titlesize":      13,
    "axes.labelsize":      11,
    "xtick.color":         "#b0b0b0",
    "ytick.color":         "#b0b0b0",
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "grid.color":          "#2a2d3a",
    "grid.linestyle":      "--",
    "grid.alpha":          0.5,
    "legend.facecolor":    "#1a1d27",
    "legend.edgecolor":    "#3a3d4d",
    "legend.labelcolor":   "#e0e0e0",
    "legend.fontsize":     9,
    "text.color":          "#e0e0e0",
    "font.family":         "DejaVu Sans",
    "figure.dpi":          FIGURE_DPI,
    "savefig.dpi":         FIGURE_DPI,
    "savefig.bbox":        "tight",
    "savefig.facecolor":   "#0f1117",
})

# Accent colours for individual plots
ACCENT       = "#7c6be8"   # Primary violet
ACCENT2      = "#e87c6b"   # Warm coral
ACCENT_GREEN = "#6be8a7"   # Mint green
GRADIENT     = ["#6be8a7", "#7c6be8", "#e87c6b", "#e8c86b", "#6bb3e8",
                 "#e86bb3", "#b3e86b", "#6be8d4", "#e8906b", "#9b6be8",
                 "#6be8c8", "#e8e86b"]


# =============================================================================
# Helper utilities
# =============================================================================

def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save a matplotlib figure to the figures directory and close it."""
    path = DIRS["figures"] / filename
    fig.savefig(path, format=FIGURE_FORMAT, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure -> %s", path)
    return path


def section(title: str) -> None:
    """Print a visual section divider to stdout / log."""
    bar = "-" * 70
    log.info("\n%s\n  %s\n%s", bar, title.upper(), bar)


# =============================================================================
# Step 1: Load dataset
# =============================================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load the GSK solvent dataset from CSV.

    The CSV has an unnamed index column (column 0) that we drop after loading.
    UTF-8 encoding is specified explicitly to handle the '¬†' artefacts present
    in some IUPAC name fields.

    Parameters
    ----------
    path : Path
        Absolute path to GSK_dataset.csv.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with original column names.
    """
    section("Loading Dataset")
    log.info("Reading: %s", path)

    df = pd.read_csv(path, encoding="utf-8")

    # The first column is an unnamed row counter — drop it to avoid confusion
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.rename(columns={df.columns[0]: "_row_idx"}).drop(
            columns=["_row_idx"]
        )

    log.info("Dataset loaded successfully.")
    return df


# =============================================================================
# Step 2: Basic inspection
# =============================================================================

def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Compute and log all basic dataset statistics.

    Parameters
    ----------
    df : pd.DataFrame
        The raw solvent dataframe.

    Returns
    -------
    dict
        A summary dictionary used later to generate the EDA report.
    """
    section("Basic Dataset Inspection")

    n_rows, n_cols = df.shape
    log.info("Shape: %d rows × %d columns", n_rows, n_cols)

    log.info("\nColumn names and dtypes:")
    for col in df.columns:
        log.info("  %-30s %s", col, df[col].dtype)

    # --- Missing values ---
    missing = df.isnull().sum()
    missing_pct = (missing / n_rows * 100).round(2)
    missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    log.info("\nMissing values:\n%s", missing_df.to_string())

    # --- Duplicate rows ---
    dup_rows = df.duplicated().sum()
    log.info("\nDuplicate rows: %d", dup_rows)

    # --- Duplicate SMILES ---
    smiles_counts = df[COL_SMILES].value_counts()
    dup_smiles = smiles_counts[smiles_counts > 1]
    log.info("\nDuplicate SMILES (%d unique SMILES with count > 1):", len(dup_smiles))
    for smi, cnt in dup_smiles.items():
        names = df.loc[df[COL_SMILES] == smi, COL_NAME].tolist()
        log.info("  SMILES: %-40s  Count: %d  Names: %s", smi, cnt, names)

    # --- IUPAC name artefacts ---
    iupac_col = "IPUAC name"
    artefact_mask = df[iupac_col].str.contains("¬†", na=False)
    n_artefact = artefact_mask.sum()
    log.info("\nRows with '¬†' UTF-8 artefact in IUPAC names: %d", n_artefact)
    if n_artefact > 0:
        log.info(df.loc[artefact_mask, [COL_NAME, iupac_col]].to_string())

    # --- Classification distribution ---
    class_counts = df[COL_CLASS].value_counts()
    log.info("\nSolvent class distribution:\n%s", class_counts.to_string())

    # --- G-score statistics ---
    gs = df[COL_TARGET]
    desc = gs.describe()
    skew = gs.skew()
    kurt = gs.kurtosis()
    log.info(
        "\nG-score statistics:\n%s\n  Skewness: %.4f  Kurtosis: %.4f",
        desc.to_string(), skew, kurt,
    )

    return {
        "n_rows":         n_rows,
        "n_cols":         n_cols,
        "missing_df":     missing_df,
        "dup_rows":       dup_rows,
        "dup_smiles":     dup_smiles,
        "n_artefact":     n_artefact,
        "class_counts":   class_counts,
        "gscore_desc":    desc,
        "gscore_skew":    skew,
        "gscore_kurt":    kurt,
    }


# =============================================================================
# Step 3: SMILES validation & RDKit molecular properties
# =============================================================================

def validate_smiles_and_compute_properties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate each SMILES with RDKit and compute a set of physicochemical
    descriptors.  Returns an augmented dataframe with new columns:
        mol_valid, MW, logP, HBD, HBA, TPSA, RotBonds

    If RDKit is unavailable the function returns df unchanged with a warning.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing a COL_SMILES column.

    Returns
    -------
    pd.DataFrame
        Original dataframe with property columns appended.
    """
    section("SMILES Validation & Molecular Property Calculation")

    if not RDKIT_AVAILABLE:
        log.warning("Skipping SMILES validation — RDKit not installed.")
        df["mol_valid"] = None
        return df

    mols, valid_flags = [], []
    for smi in df[COL_SMILES]:
        mol = Chem.MolFromSmiles(str(smi).strip())
        mols.append(mol)
        valid_flags.append(mol is not None)

    df = df.copy()
    df["mol_valid"] = valid_flags

    n_invalid = sum(not v for v in valid_flags)
    log.info("Total SMILES:    %d", len(valid_flags))
    log.info("Valid SMILES:    %d", sum(valid_flags))
    log.info("Invalid SMILES:  %d", n_invalid)

    if n_invalid > 0:
        invalid_mask = ~df["mol_valid"]
        log.warning(
            "Invalid SMILES entries:\n%s",
            df.loc[invalid_mask, [COL_NAME, COL_SMILES]].to_string(),
        )

    # --- Compute molecular properties for valid molecules only ---
    props = {
        "MW":       [],
        "logP":     [],
        "HBD":      [],
        "HBA":      [],
        "TPSA":     [],
        "RotBonds": [],
    }

    for mol in mols:
        if mol is None:
            for k in props:
                props[k].append(np.nan)
        else:
            props["MW"].append(Descriptors.MolWt(mol))
            props["logP"].append(Descriptors.MolLogP(mol))
            props["HBD"].append(CalcNumHBD(mol))
            props["HBA"].append(CalcNumHBA(mol))
            props["TPSA"].append(CalcTPSA(mol))
            props["RotBonds"].append(CalcNumRotatableBonds(mol))

    for k, v in props.items():
        df[k] = v

    log.info("\nMolecular property summary (valid molecules only):")
    prop_cols = list(props.keys())
    log.info(df[prop_cols].describe().round(2).to_string())

    return df


# =============================================================================
# Step 4 — Figure 1: G-score histogram + KDE
# =============================================================================

def plot_gscore_histogram(df: pd.DataFrame) -> Path:
    """
    Plot a styled histogram of G-score with a KDE overlay and
    summary statistics annotations.

    Returns the path to the saved figure.
    """
    section("Figure 1: G-score Histogram + KDE")

    gs = df[COL_TARGET].dropna()
    mean_v, median_v = gs.mean(), gs.median()

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")

    # Histogram
    n_bins = int(np.ceil(np.sqrt(len(gs))))  # Sturges-like
    ax.hist(
        gs, bins=n_bins,
        color=ACCENT, alpha=0.75, edgecolor="#0f1117", linewidth=0.5,
        label="Frequency",
    )

    # KDE overlay on twin axis
    ax2 = ax.twinx()
    kde_x = np.linspace(gs.min() - 0.5, gs.max() + 0.5, 400)
    kde = stats.gaussian_kde(gs, bw_method="scott")
    ax2.plot(kde_x, kde(kde_x), color=ACCENT2, linewidth=2.5, label="KDE")
    ax2.set_ylabel("Density", color=ACCENT2)
    ax2.tick_params(axis="y", labelcolor=ACCENT2)
    ax2.set_facecolor("#1a1d27")
    ax2.spines[:].set_color("#3a3d4d")

    # Vertical lines for mean / median
    ax.axvline(mean_v,   color="#e8e86b", linewidth=1.8, linestyle="--",
               label=f"Mean = {mean_v:.3f}")
    ax.axvline(median_v, color=ACCENT_GREEN, linewidth=1.8, linestyle="-.",
               label=f"Median = {median_v:.3f}")

    ax.set_xlabel("G-score", labelpad=8)
    ax.set_ylabel("Count", labelpad=8)
    ax.set_title("Distribution of G-scores — GSK Solvent Dataset", pad=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[:].set_color("#3a3d4d")

    # Combined legend from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
              framealpha=0.7)

    # Stats annotation box
    textstr = (
        f"n = {len(gs)}\n"
        f"Min  = {gs.min():.3f}\n"
        f"Max  = {gs.max():.3f}\n"
        f"Std  = {gs.std():.3f}\n"
        f"Skew = {gs.skew():.3f}"
    )
    ax.text(
        0.98, 0.97, textstr, transform=ax.transAxes,
        fontsize=8.5, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#2a2d3a",
                  edgecolor="#3a3d4d", alpha=0.9),
        color="#e0e0e0",
    )

    fig.tight_layout()
    return save_figure(fig, "fig01_gscore_histogram.png")


# =============================================================================
# Step 5 — Figure 2: G-score boxplot (overall)
# =============================================================================

def plot_gscore_boxplot(df: pd.DataFrame) -> Path:
    """
    Plot an overall horizontal boxplot of G-score, styled with
    jittered individual data points overlaid.
    """
    section("Figure 2: G-score Boxplot")

    gs = df[COL_TARGET].dropna()
    fig, ax = plt.subplots(figsize=(9, 3.5))

    # Boxplot
    bp = ax.boxplot(
        gs, vert=False, patch_artist=True,
        notch=False, widths=0.45,
        boxprops=dict(facecolor=ACCENT, color=ACCENT2, linewidth=1.5),
        medianprops=dict(color=ACCENT_GREEN, linewidth=2.5),
        whiskerprops=dict(color="#b0b0b0", linewidth=1.5, linestyle="--"),
        capprops=dict(color="#b0b0b0", linewidth=2),
        flierprops=dict(marker="o", markerfacecolor=ACCENT2,
                        markeredgewidth=0, alpha=0.8, markersize=6),
    )

    # Jittered data points (strip)
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.18, 0.18, size=len(gs))
    ax.scatter(gs, 1 + jitter, alpha=0.45, s=22, color=ACCENT_GREEN,
               zorder=5, label="Individual solvents")

    ax.set_xlabel("G-score", labelpad=8)
    ax.set_yticks([1])
    ax.set_yticklabels(["All solvents"])
    ax.set_title("G-score Overall Distribution (Boxplot + Strip)", pad=10)
    ax.legend(loc="lower right", framealpha=0.7)
    ax.grid(True, axis="x", alpha=0.35)
    ax.spines[:].set_color("#3a3d4d")
    fig.tight_layout()
    return save_figure(fig, "fig02_gscore_boxplot.png")


# =============================================================================
# Step 6 — Figure 3: Violin plot by Classification
# =============================================================================

def plot_violin_by_class(df: pd.DataFrame) -> Path:
    """
    Violin plot of G-score grouped by solvent Classification, with
    individual observations overlaid as a strip.
    """
    section("Figure 3: G-score Violin by Classification")

    # Order classes by median G-score (descending) for visual clarity
    order = (
        df.groupby(COL_CLASS)[COL_TARGET]
          .median()
          .sort_values(ascending=False)
          .index.tolist()
    )

    n_classes = len(order)
    palette = dict(zip(order, GRADIENT[:n_classes]))

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.violinplot(
        data=df, x=COL_CLASS, y=COL_TARGET, hue=COL_CLASS,
        order=order, palette=palette,
        inner=None, linewidth=1.2, cut=0, legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=df, x=COL_CLASS, y=COL_TARGET, hue=COL_CLASS,
        order=order, palette=palette,
        size=4, jitter=True, alpha=0.7, linewidth=0.3,
        edgecolor="#0f1117", dodge=False, legend=False,
        ax=ax,
    )

    ax.set_xlabel("Solvent Classification", labelpad=8)
    ax.set_ylabel("G-score", labelpad=8)
    ax.set_title("G-score Distribution by Solvent Class (Violin + Strip)", pad=12)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[:].set_color("#3a3d4d")
    fig.tight_layout()
    return save_figure(fig, "fig03_gscore_violin_by_class.png")


# =============================================================================
# Step 7 — Figure 4: Strip plot by Classification
# =============================================================================

def plot_strip_by_class(df: pd.DataFrame) -> Path:
    """
    Detailed strip plot showing every individual solvent G-score
    coloured and grouped by solvent class.
    """
    section("Figure 4: G-score Strip Plot by Classification")

    order = (
        df.groupby(COL_CLASS)[COL_TARGET]
          .median()
          .sort_values(ascending=False)
          .index.tolist()
    )
    n_classes = len(order)
    palette = dict(zip(order, GRADIENT[:n_classes]))

    fig, ax = plt.subplots(figsize=(13, 6))

    sns.stripplot(
        data=df, x=COL_CLASS, y=COL_TARGET, hue=COL_CLASS,
        order=order, palette=palette,
        size=7, jitter=0.25, alpha=0.85, linewidth=0.5,
        edgecolor="#0f1117", legend=False, ax=ax,
    )

    # Overlay class median as a horizontal dash
    for i, cls in enumerate(order):
        med = df.loc[df[COL_CLASS] == cls, COL_TARGET].median()
        ax.plot([i - 0.35, i + 0.35], [med, med],
                color="white", linewidth=2.0, zorder=10)

    ax.set_xlabel("Solvent Classification", labelpad=8)
    ax.set_ylabel("G-score", labelpad=8)
    ax.set_title("Individual Solvent G-scores by Class\n"
                 "(white bars = class medians)", pad=12)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[:].set_color("#3a3d4d")
    fig.tight_layout()
    return save_figure(fig, "fig04_gscore_strip_by_class.png")


# =============================================================================
# Step 8 — Figure 5: Correlation heatmap
# =============================================================================

def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    """
    Pearson correlation heatmap of G-score against RDKit molecular
    properties.  If RDKit properties are absent (not computed), the
    heatmap falls back to numeric columns only.
    """
    section("Figure 5: Correlation Heatmap")

    rdkit_props = ["MW", "logP", "HBD", "HBA", "TPSA", "RotBonds"]
    available   = [c for c in rdkit_props if c in df.columns]
    num_cols    = [COL_TARGET] + available

    if len(num_cols) < 2:
        log.warning(
            "Insufficient numeric columns for correlation heatmap. "
            "RDKit properties may be missing."
        )
        # Return a placeholder figure
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "RDKit not available\n(no correlation data)",
                ha="center", va="center", fontsize=12, color="#b0b0b0")
        ax.axis("off")
        return save_figure(fig, "fig05_correlation_heatmap.png")

    corr = df[num_cols].corr(method="pearson")

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        corr, mask=mask,
        annot=True, fmt=".2f", annot_kws={"size": 9, "color": "#e0e0e0"},
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
        vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.5, linecolor="#2a2d3a",
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        ax=ax,
    )

    ax.set_title(
        "Pearson Correlation — G-score & Molecular Properties", pad=12
    )
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return save_figure(fig, "fig05_correlation_heatmap.png")


# =============================================================================
# Step 9 — Figure 6: Missing value bar chart
# =============================================================================

def plot_missing_values(df: pd.DataFrame) -> Path:
    """
    Horizontal bar chart of missing value counts per column.
    If there are no missing values the chart still renders with a
    'No missing values detected' annotation.
    """
    section("Figure 6: Missing Value Summary")

    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]   # Keep only columns WITH missing data

    fig, ax = plt.subplots(figsize=(8, max(3, len(missing) * 0.6 + 1.5)))

    if len(missing) == 0:
        ax.text(0.5, 0.5, "[OK]  No missing values detected",
                ha="center", va="center", fontsize=14,
                color=ACCENT_GREEN, fontweight="bold")
        ax.axis("off")
        ax.set_title("Missing Value Analysis", pad=10)
    else:
        bars = ax.barh(missing.index, missing.values,
                       color=ACCENT2, alpha=0.85, edgecolor="#0f1117",
                       height=0.55)
        ax.bar_label(bars, labels=[f"{v}" for v in missing.values],
                     padding=4, color="#e0e0e0", fontsize=9)
        ax.set_xlabel("Number of Missing Values", labelpad=8)
        ax.set_title("Missing Values per Column", pad=10)
        ax.grid(True, axis="x", alpha=0.3)
        ax.invert_yaxis()
        ax.spines[:].set_color("#3a3d4d")

    fig.tight_layout()
    return save_figure(fig, "fig06_missing_values.png")


# =============================================================================
# Step 10 — Figure 7: Boxplot per Classification
# =============================================================================

def plot_box_by_class(df: pd.DataFrame) -> Path:
    """
    Side-by-side box plots of G-score for each solvent classification,
    ordered by median.  Whiskers extend to 1.5×IQR; outliers plotted.
    """
    section("Figure 7: G-score Box Plot by Classification")

    order = (
        df.groupby(COL_CLASS)[COL_TARGET]
          .median()
          .sort_values(ascending=False)
          .index.tolist()
    )
    n_classes = len(order)
    palette = dict(zip(order, GRADIENT[:n_classes]))

    fig, ax = plt.subplots(figsize=(13, 6))

    sns.boxplot(
        data=df, x=COL_CLASS, y=COL_TARGET, hue=COL_CLASS,
        order=order, palette=palette,
        linewidth=1.4, legend=False,
        flierprops={"marker": "D", "markersize": 5,
                    "markerfacecolor": ACCENT2,
                    "alpha": 0.8},
        ax=ax,
    )

    ax.set_xlabel("Solvent Classification", labelpad=8)
    ax.set_ylabel("G-score", labelpad=8)
    ax.set_title("G-score Boxplot by Solvent Classification", pad=12)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[:].set_color("#3a3d4d")
    fig.tight_layout()
    return save_figure(fig, "fig07_gscore_box_by_class.png")


# =============================================================================
# Step 11: Generate EDA_Report.md
# =============================================================================

def generate_eda_report(df: pd.DataFrame, summary: dict,
                        figure_paths: list[Path]) -> Path:
    """
    Write a structured Markdown EDA report to the project root.

    Parameters
    ----------
    df          : Augmented dataframe (with mol_valid and RDKit properties).
    summary     : Dictionary returned by inspect_dataset().
    figure_paths: List of Path objects for saved figures.

    Returns
    -------
    Path to the written EDA_Report.md.
    """
    section("Generating EDA_Report.md")

    report_path = Path(__file__).resolve().parent / "EDA_Report.md"
    now_str     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Duplicate SMILES table ---
    dup_smiles_lines = []
    for smi, cnt in summary["dup_smiles"].items():
        names = df.loc[df[COL_SMILES] == smi, COL_NAME].tolist()
        dup_smiles_lines.append(
            f"| `{smi}` | {cnt} | {', '.join(names)} |"
        )
    dup_smiles_table = (
        "| SMILES | Count | Solvents |\n"
        "|--------|-------|----------|\n"
        + "\n".join(dup_smiles_lines)
        if dup_smiles_lines
        else "_No duplicate SMILES detected._"
    )

    # --- Class counts table ---
    class_table_rows = "\n".join(
        f"| {cls} | {cnt} |"
        for cls, cnt in summary["class_counts"].items()
    )

    # --- Missing value table ---
    mv_df   = summary["missing_df"]
    mv_rows = mv_df[mv_df["Missing Count"] > 0]
    if mv_rows.empty:
        missing_table = "_No missing values detected in any column._"
    else:
        missing_table = (
            "| Column | Missing Count | Missing % |\n"
            "|--------|------------|----------|\n"
            + "\n".join(
                f"| {col} | {row['Missing Count']} | {row['Missing %']}% |"
                for col, row in mv_rows.iterrows()
            )
        )

    # --- RDKit property table ---
    rdkit_props = ["MW", "logP", "HBD", "HBA", "TPSA", "RotBonds"]
    available   = [c for c in rdkit_props if c in df.columns]
    if available:
        prop_desc = df[available].describe().round(3)
        prop_rows = "\n".join(
            f"| {stat} | " + " | ".join(str(prop_desc.loc[stat, c]) for c in available) + " |"
            for stat in prop_desc.index
        )
        prop_header = "| Statistic | " + " | ".join(available) + " |"
        prop_sep    = "|-----------|" + "|".join(["------"] * len(available)) + "|"
        prop_table  = f"{prop_header}\n{prop_sep}\n{prop_rows}"
    else:
        prop_table = "_RDKit not installed — molecular property table unavailable._"

    # --- Figure links (relative paths) ---
    fig_lines = "\n".join(
        f"{i+1}. `figures/{p.name}`" for i, p in enumerate(figure_paths)
    )

    # --- SMILES validity ---
    if "mol_valid" in df.columns and df["mol_valid"].notna().any():
        n_valid   = df["mol_valid"].sum()
        n_invalid = (~df["mol_valid"]).sum()
        invalid_names = df.loc[~df["mol_valid"], COL_NAME].tolist()
        smiles_status = (
            f"- **Valid SMILES**: {n_valid}\n"
            f"- **Invalid SMILES**: {n_invalid}"
            + (f"\n- **Invalid entries**: {', '.join(invalid_names)}" if invalid_names else "")
        )
    else:
        smiles_status = "_SMILES validation skipped (RDKit not installed)._"

    # --- Descriptive statistics ---
    gs = summary["gscore_desc"]

    report_content = f"""# EDA Report — GSK Solvent G-Score Dataset

> **Generated**: {now_str}
> **Script**: `exploratory_analysis.py`
> **Dataset**: `data/GSK_dataset.csv`

---

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| **Total solvents** | {summary['n_rows']} |
| **Columns** | {summary['n_cols']} |
| **Target variable** | G-score (continuous, higher = greener) |
| **Solvent classes** | {len(summary['class_counts'])} |

### Column Descriptions

| Column | Description |
|--------|-------------|
| `Classification` | Broad chemical family (Alcohols, Esters, etc.) |
| `solvent_common_name` | Common trade/lab name |
| `IPUAC name` | IUPAC systematic name (note: intentional typo in source CSV) |
| `solvent_SMILES` | Canonical SMILES string (primary ML feature input) |
| `CAS Number` | CAS registry number |
| `G-score` | **Target**: GSK greenness score (continuous, higher = greener) |

---

## 2. Missing Value Analysis

{missing_table}

---

## 3. Duplicate Entries

- **Duplicate full rows**: {summary['dup_rows']}
- **Duplicate SMILES pairs**: {len(summary['dup_smiles'])}

### Duplicate SMILES Detail

{dup_smiles_table}

> **Note**: Duplicate SMILES indicate that the same molecule appears under
> different names (e.g., "IMS ethanol" is listed separately from "Ethanol").
> Both entries are retained; this is expected in a solvent guide dataset where
> commercial-grade and pure-grade solvents are treated as separate entries.

---

## 4. SMILES Validation

{smiles_status}

---

## 5. Data Quality Issues

| Issue | Severity | Detail |
|-------|----------|--------|
| `¬†` artefacts in IUPAC names | Low | {summary['n_artefact']} rows contain a non-breaking space artefact. Cosmetic only — does not affect ML. |
| Duplicate SMILES | Medium | {len(summary['dup_smiles'])} SMILES string(s) appear more than once. Multiple entries represent different commercial grades. |
| Trifluorotoluene SMILES | Medium | Row 80 lists SMILES `Cc1ccc(F)c(F)c1F` (3,4,5-trifluorotoluene) instead of `FC(F)(F)c1ccccc1` (trifluoromethylbenzene, α,α,α-trifluorotoluene). May affect descriptor accuracy. |
| IUPAC typo in column header | Low | Column is named `IPUAC name` (typo) rather than `IUPAC name` — carried through scripts as-is to match source. |

---

## 6. Solvent Class Distribution

| Classification | Count |
|----------------|-------|
{class_table_rows}

---

## 7. G-score Statistical Summary

| Statistic | Value |
|-----------|-------|
| Count | {int(gs['count'])} |
| Mean | {gs['mean']:.4f} |
| Std Dev | {gs['std']:.4f} |
| Min | {gs['min']:.4f} |
| 25th Percentile | {gs['25%']:.4f} |
| Median (50th) | {gs['50%']:.4f} |
| 75th Percentile | {gs['75%']:.4f} |
| Max | {gs['max']:.4f} |
| **Skewness** | {summary['gscore_skew']:.4f} |
| **Kurtosis** | {summary['gscore_kurt']:.4f} |

**Interpretation**: The G-score distribution is
{"approximately symmetric" if abs(summary['gscore_skew']) < 0.5
 else ("slightly left-skewed" if summary['gscore_skew'] < 0 else "slightly right-skewed")}
(skewness = {summary['gscore_skew']:.3f}).
{"No significant outliers are expected."
 if abs(summary['gscore_skew']) < 0.5
 else "Mild skew — consider verifying outliers at the extremes before training."}

---

## 8. Molecular Property Summary (RDKit)

{prop_table}

---

## 9. Figures Generated

{fig_lines}

All figures are saved to the `figures/` directory and use a consistent dark-theme style.

---

## 10. Recommendations for Downstream ML

1. **No imputation needed** — there are no missing G-scores.
2. **Handle duplicate SMILES** — when splitting into train/test sets, ensure
   duplicate SMILES are kept in the same split to prevent data leakage.
3. **Verify Trifluorotoluene SMILES** before descriptor generation.
4. **Stratified splits** — consider stratifying by `Classification` to ensure
   all solvent families are represented in both train and test sets.
5. **Scale descriptors** — RDKit descriptors span different ranges; apply
   `StandardScaler` before XGBoost (though XGBoost is scale-invariant,
   scaling aids interpretability when comparing to other models).

---

*Report auto-generated by `exploratory_analysis.py` — GSK G-Score Prediction Project*
"""

    report_path.write_text(report_content, encoding="utf-8")
    log.info("EDA Report written -> %s", report_path)
    return report_path


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    """
    Orchestrate the full EDA pipeline:
    load → inspect → validate → plot (×7) → generate report.
    """
    log.info("=" * 70)
    log.info("GSK Solvent G-Score — Exploratory Data Analysis")
    log.info("Started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 70)

    # 1. Load
    df = load_dataset(DATA_PATH)

    # 2. Inspect
    summary = inspect_dataset(df)

    # 3. Validate SMILES & compute RDKit properties
    df = validate_smiles_and_compute_properties(df)

    # 4. Generate all figures
    figure_paths: list[Path] = []
    figure_paths.append(plot_gscore_histogram(df))
    figure_paths.append(plot_gscore_boxplot(df))
    figure_paths.append(plot_violin_by_class(df))
    figure_paths.append(plot_strip_by_class(df))
    figure_paths.append(plot_correlation_heatmap(df))
    figure_paths.append(plot_missing_values(df))
    figure_paths.append(plot_box_by_class(df))

    # 5. Generate EDA report
    report_path = generate_eda_report(df, summary, figure_paths)

    # 6. Summary
    section("EDA Complete")
    log.info("Figures saved:  %d  ->  %s", len(figure_paths), DIRS["figures"])
    log.info("EDA Report:     %s", report_path)
    log.info("Run log:        %s", LOG_PATH)
    log.info("Finished: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 70)


if __name__ == "__main__":
    main()
