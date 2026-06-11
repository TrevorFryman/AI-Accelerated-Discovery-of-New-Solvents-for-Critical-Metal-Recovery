"""Hyperparameter optimization utilities for XGBoost models.

Provides a RandomizedSearchCV-based tuner that respects StratifiedGroupKFold
splits from `data_utils.get_cv_splitter` by passing explicit (train, test)
fold tuples to scikit-learn search objects.
"""
from pathlib import Path
import json
import numpy as np
from typing import Dict, Any, List, Tuple

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from config import XGB_PARAMS, SEED
from data_utils import get_cv_splitter


def build_cv_splits(X, strata, groups, n_splits: int = 5):
    """Return list of (train_idx, val_idx) tuples using StratifiedGroupKFold.

    Parameters
    - X: feature matrix (unused but kept for API parity)
    - strata: array-like of stratification labels (categorical)
    - groups: group labels for StratifiedGroupKFold
    """
    sgkf = get_cv_splitter(n_splits)
    return list(sgkf.split(X, strata, groups))


def tune_xgb(
    X,
    y,
    groups,
    param_distributions: Dict[str, List[Any]],
    n_iter: int = 50,
    n_splits: int = 5,
    strata=None,
    random_state: int = SEED,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = -1,
):
    """Run a randomized search for XGB hyperparameters.

    Returns the fitted RandomizedSearchCV object and the list of cv splits used.
    """
    if strata is None:
        raise ValueError("tune_xgb requires stratification labels via 'strata' argument")
    cv_splits = build_cv_splits(X, strata, groups, n_splits=n_splits)

    # Base estimator with fixed params as starting point
    estimator = XGBRegressor(**XGB_PARAMS)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv_splits,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
    )

    search.fit(X, y)
    return search, cv_splits


def default_param_space():
    """Return a reasonable default search space for the requested XGBoost params."""
    return {
        "n_estimators": [100, 200, 300, 500, 800, 1000],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.001, 0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
    }


if __name__ == "__main__":
    print("This module exposes `tune_xgb` for import. Run from `compare_models.py`.")
