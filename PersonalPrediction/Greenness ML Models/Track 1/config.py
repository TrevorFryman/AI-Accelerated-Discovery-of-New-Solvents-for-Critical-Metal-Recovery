"""
config.py
=========
Central configuration for the GSK Solvent G-Score Prediction project.

All scripts import from here to ensure consistent paths, seeds, and column
names across the entire pipeline (EDA → descriptor generation → training →
comparison).

Usage:
    from config import CONFIG
    df = pd.read_csv(CONFIG["DATA_PATH"])
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — all paths are relative to this file's location
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
DIRS = {
    "data":        PROJECT_ROOT / "data",
    "descriptors": PROJECT_ROOT / "descriptors",
    "models":      PROJECT_ROOT / "models",
    "results":     PROJECT_ROOT / "results",
    "figures":     PROJECT_ROOT / "figures",
    "notebooks":   PROJECT_ROOT / "notebooks",
    "logs":        PROJECT_ROOT / "logs",
}

# Ensure every directory exists (safe to call multiple times)
for _dir in DIRS.values():
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATA_PATH    = DIRS["data"] / "GSK_dataset.csv"

# Column names as they appear in the CSV
COL_INDEX    = ""                    # Unnamed first column (row index)
COL_CLASS    = "Classification"
COL_NAME     = "solvent_common_name"
COL_IUPAC    = "IPUAC name"          # Note: intentional typo in source CSV
COL_SMILES   = "solvent_SMILES"
COL_CAS      = "CAS Number"
COL_TARGET   = "G-score"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Morgan fingerprint settings
# ---------------------------------------------------------------------------
MORGAN_RADIUS   = 2    # Equivalent to ECFP4
MORGAN_N_BITS   = 2048

# ---------------------------------------------------------------------------
# XGBoost hyperparameters (identical across all three models)
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "n_estimators":      500,
    "learning_rate":     0.05,
    "max_depth":         4,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  3,
    "reg_alpha":         0.1,   # L1 regularisation
    "reg_lambda":        1.0,   # L2 regularisation
    "random_state":      SEED,
    "n_jobs":            -1,
    "objective":         "reg:squarederror",
    "eval_metric":       "rmse",
}

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
CV_FOLDS   = 5
TEST_SIZE  = 0.2   # 80/20 train-test split

# ---------------------------------------------------------------------------
# ChemBERTa settings
# ---------------------------------------------------------------------------
CHEMBERTA_MODEL  = "seyonec/ChemBERTa-zinc-base-v1"
CHEMBERTA_DEVICE = "cuda"  # GPU confirmed available
CHEMBERTA_BATCH  = 32      # Molecules per inference batch (GPU can handle larger batches)

# ---------------------------------------------------------------------------
# Convenience dictionary (optional — import CONFIG["KEY"] style)
# ---------------------------------------------------------------------------
CONFIG = {
    "PROJECT_ROOT":      PROJECT_ROOT,
    "DIRS":              DIRS,
    "DATA_PATH":         DATA_PATH,
    "COL_CLASS":         COL_CLASS,
    "COL_NAME":          COL_NAME,
    "COL_IUPAC":         COL_IUPAC,
    "COL_SMILES":        COL_SMILES,
    "COL_CAS":           COL_CAS,
    "COL_TARGET":        COL_TARGET,
    "SEED":              SEED,
    "MORGAN_RADIUS":     MORGAN_RADIUS,
    "MORGAN_N_BITS":     MORGAN_N_BITS,
    "XGB_PARAMS":        XGB_PARAMS,
    "CV_FOLDS":          CV_FOLDS,
    "TEST_SIZE":         TEST_SIZE,
    "CHEMBERTA_MODEL":   CHEMBERTA_MODEL,
    "CHEMBERTA_DEVICE":  CHEMBERTA_DEVICE,
    "CHEMBERTA_BATCH":   CHEMBERTA_BATCH,
}
