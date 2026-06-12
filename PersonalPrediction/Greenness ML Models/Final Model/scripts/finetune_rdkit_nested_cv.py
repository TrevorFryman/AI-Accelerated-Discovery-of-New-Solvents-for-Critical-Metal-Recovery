"""
RDKit Descriptor Pipeline - Nested Cross-Validation (Step 4)

Fine-tuning pipeline so far:
- Step 1: StratifiedGroupKFold split (grouped by solvent_SMILES, stratified
  by Classification) instead of the old random 80/20 + plain KFold.
- Step 2: expanded the descriptor set from 7 to 13 named, interpretable
  RDKit descriptors. Kept hyperparameters:
  n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.7,
  colsample_bytree=0.8, min_child_weight=1.
  Single-split results: Train R2=0.8893, Test R2=0.3105, OOF R2=0.5123,
  gap=0.3770.
- Step 3 (regularized re-tune): rejected, kept Step 2 config.

Step 4 (this script): instead of tuning on one train split and reporting on
one held-out test split, run a nested cross-validation. An outer 5-fold
StratifiedGroupKFold (grouped by solvent_SMILES, stratified by
Classification, same as Step 1) loops over test folds; for each outer fold,
hyperparameters are tuned via RandomizedSearchCV using an inner 4-fold
StratifiedGroupKFold over that fold's training data only. The outer fold's
test score is then computed from a model trained with those
fold-specific best hyperparameters.

This produces 5 independent test-set estimates (RMSE, MAE, R2), from which a
mean +/- 95% CI (t-distribution, df=4) is reported -- the same statistical
framework used in Track 2's Final_Descriptor_Comparison_Report.md.

This step does not "improve" the model; it produces the headline number(s)
that should be reported as the model's performance, and serves as a sanity
check on the single-split Step 1/2 test metrics.

Outputs:
- results/RDKit_Nested_CV_Report.md
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "data" / "RDKit_Features.csv"
DATASET_PATH = BASE_DIR / "data" / "GSK_dataset.csv"
REPORTS_DIR = BASE_DIR / "results"
REPORT_PATH = REPORTS_DIR / "RDKit_Nested_CV_Report.md"

TARGET = "G-score"
NON_FEATURE_COLS = ["solvent_common_name", "CAS Number", "solvent_SMILES", "Classification", TARGET]

# Single-split results to compare against (Step 1/2, current best config)
SINGLE_SPLIT = {
    "rmse_test": 1.0625, "mae_test": 0.8568, "r2_test": 0.3105,
    "rmse_oof": 0.8881, "mae_oof": 0.6778, "r2_oof": 0.5123,
}

PARAM_DISTRIBUTIONS = {
    "n_estimators": [50, 100, 200, 300, 500],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 2, 3, 5, 7, 10, 15],
    "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0, 2.0],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    "gamma": [0, 0.05, 0.1, 0.5, 1.0],
}


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def assign_groups(smiles_series):
    smiles_to_id = {smi: i for i, smi in enumerate(smiles_series.unique())}
    return smiles_series.map(smiles_to_id).to_numpy(dtype=int)


def mean_ci(values, confidence=0.95):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = arr.mean()
    sem = arr.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * sem
    return mean, mean - margin, mean + margin


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_PATH)

    class_df = pd.read_csv(DATASET_PATH)[["solvent_SMILES", "Classification"]]
    df = df.merge(class_df, on="solvent_SMILES", how="left")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].values
    y = df[TARGET].values
    groups = assign_groups(df["solvent_SMILES"])
    strata = df["Classification"].astype(str).to_numpy()

    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fold_results = []
    fold_params = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, strata, groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, strata_train = groups[train_idx], strata[train_idx]

        inner_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
        inner_splits = list(inner_cv.split(X_train, strata_train, groups_train))

        base_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=PARAM_DISTRIBUTIONS,
            n_iter=150,
            scoring="neg_root_mean_squared_error",
            cv=inner_splits,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        rmse, mae, r2 = evaluate(y_test, y_pred)

        fold_results.append({"fold": fold_idx, "rmse": rmse, "mae": mae, "r2": r2, "n_test": len(test_idx)})
        fold_params.append(search.best_params_)

        print(f"Outer fold {fold_idx}: n_test={len(test_idx)}, RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        print(f"  best params: {search.best_params_}")

    rmse_vals = [r["rmse"] for r in fold_results]
    mae_vals = [r["mae"] for r in fold_results]
    r2_vals = [r["r2"] for r in fold_results]

    rmse_mean, rmse_lo, rmse_hi = mean_ci(rmse_vals)
    mae_mean, mae_lo, mae_hi = mean_ci(mae_vals)
    r2_mean, r2_lo, r2_hi = mean_ci(r2_vals)

    print("\n" + "=" * 60)
    print("NESTED CV RESULTS (5 outer folds, mean +/- 95% CI, t-dist df=4)")
    print("=" * 60)
    print(f"RMSE: {rmse_mean:.4f}  [{rmse_lo:.4f}, {rmse_hi:.4f}]")
    print(f"MAE:  {mae_mean:.4f}  [{mae_lo:.4f}, {mae_hi:.4f}]")
    print(f"R2:   {r2_mean:.4f}  [{r2_lo:.4f}, {r2_hi:.4f}]")

    print("\nComparison to single-split test metrics (Step 1/2 config):")
    print(f"  Single-split Test R2: {SINGLE_SPLIT['r2_test']:.4f}")
    print(f"  Single-split OOF R2:  {SINGLE_SPLIT['r2_oof']:.4f}")
    print(f"  Nested CV mean R2:    {r2_mean:.4f}")

    # --- Success check: is the single-split test metric within the nested CV CI? ---
    single_split_within_ci = r2_lo <= SINGLE_SPLIT["r2_test"] <= r2_hi
    oof_within_ci = r2_lo <= SINGLE_SPLIT["r2_oof"] <= r2_hi

    # --- Report ---
    lines = []
    lines.append("# RDKit Descriptor Model - Nested Cross-Validation Report (Step 4)\n\n")
    lines.append(
        "Step 4 of the fine-tuning plan: nested cross-validation. An outer "
        "5-fold `StratifiedGroupKFold` (grouped by `solvent_SMILES`, "
        "stratified by `Classification`, same as Step 1) provides 5 "
        "independent test folds. For each outer fold, hyperparameters are "
        "tuned via `RandomizedSearchCV` (n_iter=150) using an inner 4-fold "
        "`StratifiedGroupKFold` over that fold's training data only, then "
        "the outer fold's test score is computed from a model trained with "
        "those fold-specific best hyperparameters. The reported mean +/- 95% "
        "CI (t-distribution, df=4) is the headline performance estimate for "
        "this model.\n\n"
    )
    lines.append(f"Feature columns ({len(feature_cols)}): {', '.join(feature_cols)}\n\n")

    lines.append("## Per-Fold Results\n\n")
    lines.append("| Outer Fold | n_test | RMSE | MAE | R2 |\n")
    lines.append("|---|---|---|---|---|\n")
    for r in fold_results:
        lines.append(f"| {r['fold']} | {r['n_test']} | {r['rmse']:.4f} | {r['mae']:.4f} | {r['r2']:.4f} |\n")

    lines.append("\n## Per-Fold Best Hyperparameters\n\n")
    param_names = list(PARAM_DISTRIBUTIONS.keys())
    lines.append("| Outer Fold | " + " | ".join(param_names) + " |\n")
    lines.append("|---" * (len(param_names) + 1) + "|\n")
    for r, params in zip(fold_results, fold_params):
        lines.append(f"| {r['fold']} | " + " | ".join(str(params.get(p)) for p in param_names) + " |\n")

    lines.append("\n## Nested CV Summary (Mean +/- 95% CI, t-distribution df=4)\n\n")
    lines.append("| Metric | Mean | 95% CI Lower | 95% CI Upper |\n")
    lines.append("|---|---|---|---|\n")
    lines.append(f"| RMSE | {rmse_mean:.4f} | {rmse_lo:.4f} | {rmse_hi:.4f} |\n")
    lines.append(f"| MAE | {mae_mean:.4f} | {mae_lo:.4f} | {mae_hi:.4f} |\n")
    lines.append(f"| R2 | {r2_mean:.4f} | {r2_lo:.4f} | {r2_hi:.4f} |\n\n")

    lines.append("## Comparison to Single-Split Metrics (Step 1/2 Config)\n\n")
    lines.append("| Metric | Single-Split Test | Single-Split OOF | Nested CV Mean |\n")
    lines.append("|---|---|---|---|\n")
    lines.append(f"| RMSE | {SINGLE_SPLIT['rmse_test']:.4f} | {SINGLE_SPLIT['rmse_oof']:.4f} | {rmse_mean:.4f} |\n")
    lines.append(f"| MAE | {SINGLE_SPLIT['mae_test']:.4f} | {SINGLE_SPLIT['mae_oof']:.4f} | {mae_mean:.4f} |\n")
    lines.append(f"| R2 | {SINGLE_SPLIT['r2_test']:.4f} | {SINGLE_SPLIT['r2_oof']:.4f} | {r2_mean:.4f} |\n\n")

    lines.append("## Success Check\n\n")
    lines.append(
        f"- Single-split Test R2 ({SINGLE_SPLIT['r2_test']:.4f}) within nested CV 95% CI "
        f"[{r2_lo:.4f}, {r2_hi:.4f}]: {single_split_within_ci}\n"
    )
    lines.append(
        f"- Single-split OOF R2 ({SINGLE_SPLIT['r2_oof']:.4f}) within nested CV 95% CI "
        f"[{r2_lo:.4f}, {r2_hi:.4f}]: {oof_within_ci}\n\n"
    )
    if single_split_within_ci and oof_within_ci:
        lines.append(
            "Both single-split estimates fall within the nested CV confidence "
            "interval, so the original single-split test set was reasonably "
            "representative. The nested CV mean +/- 95% CI is recommended as "
            "the headline reported metric for the RDKit descriptor model "
            f"(R2 = {r2_mean:.4f} [{r2_lo:.4f}, {r2_hi:.4f}], "
            f"RMSE = {rmse_mean:.4f} [{rmse_lo:.4f}, {rmse_hi:.4f}]).\n"
        )
    else:
        lines.append(
            "At least one single-split estimate falls outside the nested CV "
            "confidence interval, meaning the single 80/20-style split was "
            "not fully representative (high variance from the small "
            "held-out set, as expected with n~31 per fold). The nested CV "
            "mean +/- 95% CI should be trusted and reported instead of the "
            "single-split numbers as the headline metric for the RDKit "
            "descriptor model "
            f"(R2 = {r2_mean:.4f} [{r2_lo:.4f}, {r2_hi:.4f}], "
            f"RMSE = {rmse_mean:.4f} [{rmse_lo:.4f}, {rmse_hi:.4f}]).\n"
        )

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
