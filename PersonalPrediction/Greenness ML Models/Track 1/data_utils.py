"""
data_utils.py
=============
Shared data loading, grouping, and train/test splitting utilities.

CRITICAL: All three pipelines (Morgan, RDKit, ChemBERTa) import this module
to guarantee bit-for-bit identical train/test splits. The split is computed
once on first call and cached to data/train_test_split.json so all scripts
share the same partition regardless of execution order.

Split strategy
--------------
- StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
- Fold 0's test indices form the global held-out test set (~20%, ~31 samples)
- Groups: molecules with identical SMILES get the same group ID
  → duplicate SMILES always land in the SAME partition (no data leakage)
- Stratification: by Classification column
  → all 11 solvent families represented in both train and test
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# ---------------------------------------------------------------------------
# Path bootstrap — allow running from any working directory
# ---------------------------------------------------------------------------
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from config import (
    DATA_PATH, DIRS, SEED,
    COL_SMILES, COL_CLASS, COL_TARGET,
)

log = logging.getLogger(__name__)

# Cached split file — shared across all pipelines
SPLIT_FILE = DIRS["data"] / "train_test_split.json"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """
    Load the GSK solvent dataset from CSV.

    Drops the unnamed index column produced by pandas when the CSV has a
    leading comma, resets the integer index, and returns a clean DataFrame.

    Returns
    -------
    pd.DataFrame
        154 rows × 6 columns (Classification, solvent_common_name,
        IPUAC name, solvent_SMILES, CAS Number, G-score)
    """
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    # Drop the unnamed first column (row counter in source CSV)
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.drop(columns=[df.columns[0]])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Group assignment
# ---------------------------------------------------------------------------

def assign_groups(df: pd.DataFrame) -> np.ndarray:
    """
    Assign an integer group ID to each row based on canonical SMILES.

    Molecules sharing the same SMILES (duplicate entries in the dataset)
    receive the same group ID, so StratifiedGroupKFold will always keep
    them in the same train/test partition — preventing data leakage from
    near-identical molecules appearing on both sides of the split.

    Parameters
    ----------
    df : pd.DataFrame  Full or subset dataset with COL_SMILES column.

    Returns
    -------
    np.ndarray of int, shape (n_rows,)
        Group IDs (integers in [0, n_unique_smiles)).
    """
    smiles_to_id = {smi: i for i, smi in enumerate(df[COL_SMILES].unique())}
    return df[COL_SMILES].map(smiles_to_id).values


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def make_split(
    df: pd.DataFrame,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (train_idx, test_idx) for the global train/test split.

    On first call the split is computed and written to SPLIT_FILE.
    On all subsequent calls the cached split is loaded, guaranteeing
    identical indices for all three pipelines even if run independently.

    Parameters
    ----------
    df    : pd.DataFrame  Full 154-row dataset.
    force : bool          If True, recompute even if cache exists.

    Returns
    -------
    train_idx, test_idx : np.ndarray  Row indices into df.
    """
    if SPLIT_FILE.exists() and not force:
        with open(SPLIT_FILE) as f:
            cached = json.load(f)
        train_idx = np.array(cached["train_idx"], dtype=int)
        test_idx  = np.array(cached["test_idx"],  dtype=int)
        log.info(
            "Loaded cached split: %d train / %d test  (seed=%s, strategy=%s)",
            len(train_idx), len(test_idx),
            cached.get("seed"), cached.get("strategy"),
        )
        return train_idx, test_idx

    groups  = assign_groups(df)
    strata  = df[COL_CLASS].values

    # StratifiedGroupKFold: fold 0 test set = ~20% of data
    sgkf   = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(df, strata, groups))
    train_idx, test_idx = splits[0]

    # Verify: no SMILES overlaps between train and test
    train_smiles = set(df.iloc[train_idx][COL_SMILES])
    test_smiles  = set(df.iloc[test_idx][COL_SMILES])
    overlap      = train_smiles & test_smiles
    leak_passed  = len(overlap) == 0

    cache = {
        "train_idx":            train_idx.tolist(),
        "test_idx":             test_idx.tolist(),
        "seed":                 SEED,
        "n_train":              int(len(train_idx)),
        "n_test":               int(len(test_idx)),
        "strategy":             "StratifiedGroupKFold(n_splits=5, fold=0, shuffle=True)",
        "leakage_check_passed": leak_passed,
        "smiles_overlap":       list(overlap) if overlap else [],
    }
    with open(SPLIT_FILE, "w") as f:
        json.dump(cache, f, indent=2)

    if leak_passed:
        log.info("SMILES leakage check: PASSED — no overlap between train and test SMILES.")
    else:
        log.warning("SMILES leakage DETECTED: %s", overlap)

    log.info(
        "Split created and cached: %d train / %d test -> %s",
        len(train_idx), len(test_idx), SPLIT_FILE,
    )
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Cross-validation splitter
# ---------------------------------------------------------------------------

def get_cv_splitter(n_splits: int = 5) -> StratifiedGroupKFold:
    """
    Return a configured StratifiedGroupKFold splitter for inner CV.

    Use this in all training scripts to ensure consistent CV strategy.

    Parameters
    ----------
    n_splits : int  Number of CV folds (default 5).
    """
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
