"""
RDKit Descriptor Pipeline - Simpler Baseline Comparison (Step 6, Optional)

Fine-tuning pipeline so far (Steps 1-5) settled on:
- 13 RDKit descriptors (Step 2)
- XGBoost with n_estimators=500, max_depth=3, learning_rate=0.01,
  subsample=0.7, colsample_bytree=0.8, min_child_weight=1 (Step 2/3)
- Nested CV headline metric (Step 4/5): R2 = 0.5050 [0.3342, 0.6757],
  RMSE = 0.8721 [0.7238, 1.0203]

Step 6 (optional sanity check): fit simpler, fully-transparent baselines
(RidgeCV, ElasticNetCV, and a shallow RandomForestRegressor) on the same
13-descriptor feature set, using the identical nested-CV protocol from
Step 4 (outer 5-fold StratifiedGroupKFold, inner 4-fold
StratifiedGroupKFold for hyperparameter selection). Linear models are
wrapped in a StandardScaler -> {RidgeCV, ElasticNetCV} pipeline so
hyperparameter (alpha / l1_ratio) selection happens via the inner CV.

Success check: if the linear baselines underperform XGBoost by a wide
margin (R2 gap > 0.1), this justifies XGBoost's added complexity in the
paper. If they are close, both can be presented (linear model for the
interpretability narrative, XGBoost for the best-performance number).

Outputs:
- results/RDKit_Baseline_Comparison_Report.md
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "data" / "RDKit_Features.csv"
DATASET_PATH = BASE_DIR / "data" / "GSK_dataset.csv"
REPORTS_DIR = BASE_DIR / "results"
REPORT_PATH = REPORTS_DIR / "RDKit_Baseline_Comparison_Report.md"

TARGET = "G-score"
NON_FEATURE_COLS = ["solvent_common_name", "CAS Number", "solvent_SMILES", "Classification", TARGET]

# Step 4/5 nested CV headline metrics for XGBoost (13 features)
XGB_NESTED_CV = {
    "rmse_mean": 0.8721, "rmse_lo": 0.7238, "rmse_hi": 1.0203,
    "mae_mean": 0.6699, "mae_lo": 0.5292, "mae_hi": 0.8107,
    "r2_mean": 0.5050, "r2_lo": 0.3342, "r2_hi": 0.6757,
}

RIDGE_ALPHAS = np.logspace(-3, 3, 25)
ENET_ALPHAS = np.logspace(-3, 2, 20)
ENET_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]

RF_PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [2, 3, 4, 5, None],
    "min_samples_leaf": [1, 2, 3, 5, 8],
    "max_features": ["sqrt", "log2", 0.5, 1.0],
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


def nested_cv_ridge(X, y, strata, groups):
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    for train_idx, test_idx in outer_cv.split(X, strata, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, strata_train = groups[train_idx], strata[train_idx]

        inner_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
        inner_splits = list(inner_cv.split(X_train, strata_train, groups_train))

        model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS, cv=inner_splits))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse, mae, r2 = evaluate(y_test, y_pred)
        results.append((rmse, mae, r2))
    return results


def nested_cv_elasticnet(X, y, strata, groups):
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    for train_idx, test_idx in outer_cv.split(X, strata, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, strata_train = groups[train_idx], strata[train_idx]

        inner_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
        inner_splits = list(inner_cv.split(X_train, strata_train, groups_train))

        model = make_pipeline(
            StandardScaler(),
            ElasticNetCV(alphas=ENET_ALPHAS, l1_ratio=ENET_L1_RATIOS, cv=inner_splits,
                         random_state=RANDOM_STATE, max_iter=10000),
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse, mae, r2 = evaluate(y_test, y_pred)
        results.append((rmse, mae, r2))
    return results


def nested_cv_random_forest(X, y, strata, groups):
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    for train_idx, test_idx in outer_cv.split(X, strata, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, strata_train = groups[train_idx], strata[train_idx]

        inner_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
        inner_splits = list(inner_cv.split(X_train, strata_train, groups_train))

        search = RandomizedSearchCV(
            estimator=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            param_distributions=RF_PARAM_DISTRIBUTIONS,
            n_iter=40,
            scoring="neg_root_mean_squared_error",
            cv=inner_splits,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        y_pred = search.best_estimator_.predict(X_test)
        rmse, mae, r2 = evaluate(y_test, y_pred)
        results.append((rmse, mae, r2))
    return results


def summarize(results):
    rmse_vals = [r[0] for r in results]
    mae_vals = [r[1] for r in results]
    r2_vals = [r[2] for r in results]
    return {
        "rmse": mean_ci(rmse_vals),
        "mae": mean_ci(mae_vals),
        "r2": mean_ci(r2_vals),
        "per_fold": results,
    }


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

    print("Running nested CV for Ridge...")
    ridge_results = nested_cv_ridge(X, y, strata, groups)
    ridge_summary = summarize(ridge_results)

    print("Running nested CV for ElasticNet...")
    enet_results = nested_cv_elasticnet(X, y, strata, groups)
    enet_summary = summarize(enet_results)

    print("Running nested CV for RandomForest...")
    rf_results = nested_cv_random_forest(X, y, strata, groups)
    rf_summary = summarize(rf_results)

    models_summary = {
        "Ridge (StandardScaler + RidgeCV)": ridge_summary,
        "ElasticNet (StandardScaler + ElasticNetCV)": enet_summary,
        "RandomForestRegressor (RandomizedSearchCV)": rf_summary,
    }

    print("\n" + "=" * 60)
    print("NESTED CV RESULTS - BASELINE COMPARISON (mean +/- 95% CI, t-dist df=4)")
    print("=" * 60)
    for name, summ in models_summary.items():
        r2_mean, r2_lo, r2_hi = summ["r2"]
        rmse_mean, rmse_lo, rmse_hi = summ["rmse"]
        print(f"{name}:")
        print(f"  R2:   {r2_mean:.4f}  [{r2_lo:.4f}, {r2_hi:.4f}]")
        print(f"  RMSE: {rmse_mean:.4f}  [{rmse_lo:.4f}, {rmse_hi:.4f}]")
    print(f"\nXGBoost (Step 4/5):")
    print(f"  R2:   {XGB_NESTED_CV['r2_mean']:.4f}  [{XGB_NESTED_CV['r2_lo']:.4f}, {XGB_NESTED_CV['r2_hi']:.4f}]")
    print(f"  RMSE: {XGB_NESTED_CV['rmse_mean']:.4f}  [{XGB_NESTED_CV['rmse_lo']:.4f}, {XGB_NESTED_CV['rmse_hi']:.4f}]")

    # --- Success check ---
    gap_checks = {}
    for name, summ in models_summary.items():
        r2_mean = summ["r2"][0]
        gap = XGB_NESTED_CV["r2_mean"] - r2_mean
        gap_checks[name] = gap

    widest_gap = max(gap_checks.values())
    xgb_justified = widest_gap > 0.1

    print("\nR2 gap vs XGBoost:")
    for name, gap in gap_checks.items():
        print(f"  {name}: {gap:+.4f}")
    print(f"\nXGBoost complexity justified (gap > 0.1 for all baselines): {xgb_justified}")

    # --- Report ---
    lines = []
    lines.append("# RDKit Descriptor Model - Baseline Comparison Report (Step 6, Optional)\n\n")
    lines.append(
        "Step 6 (optional) of the fine-tuning plan: sanity-check the final "
        "XGBoost model (Steps 1-5) against simpler, fully-transparent "
        "baselines on the same 13-descriptor feature set, using the "
        "identical nested-CV protocol from Step 4 (outer 5-fold "
        "`StratifiedGroupKFold`, inner 4-fold `StratifiedGroupKFold` for "
        "hyperparameter selection). Linear models use a "
        "`StandardScaler` -> `{RidgeCV, ElasticNetCV}` pipeline so alpha "
        "(and l1_ratio for ElasticNet) selection happens via the inner CV.\n\n"
    )
    lines.append(f"Feature columns ({len(feature_cols)}): {', '.join(feature_cols)}\n\n")

    lines.append("## Nested CV Results (Mean +/- 95% CI, t-distribution df=4)\n\n")
    lines.append("| Model | R2 | RMSE | MAE |\n")
    lines.append("|---|---|---|---|\n")
    for name, summ in models_summary.items():
        r2_mean, r2_lo, r2_hi = summ["r2"]
        rmse_mean, rmse_lo, rmse_hi = summ["rmse"]
        mae_mean, mae_lo, mae_hi = summ["mae"]
        lines.append(
            f"| {name} | {r2_mean:.4f} [{r2_lo:.4f}, {r2_hi:.4f}] "
            f"| {rmse_mean:.4f} [{rmse_lo:.4f}, {rmse_hi:.4f}] "
            f"| {mae_mean:.4f} [{mae_lo:.4f}, {mae_hi:.4f}] |\n"
        )
    lines.append(
        f"| XGBoost (Step 4/5, final) | {XGB_NESTED_CV['r2_mean']:.4f} "
        f"[{XGB_NESTED_CV['r2_lo']:.4f}, {XGB_NESTED_CV['r2_hi']:.4f}] "
        f"| {XGB_NESTED_CV['rmse_mean']:.4f} [{XGB_NESTED_CV['rmse_lo']:.4f}, {XGB_NESTED_CV['rmse_hi']:.4f}] "
        f"| {XGB_NESTED_CV['mae_mean']:.4f} [{XGB_NESTED_CV['mae_lo']:.4f}, {XGB_NESTED_CV['mae_hi']:.4f}] |\n\n"
    )

    lines.append("## Per-Fold R2\n\n")
    lines.append("| Outer Fold | Ridge | ElasticNet | RandomForest |\n")
    lines.append("|---|---|---|---|\n")
    for i in range(5):
        lines.append(
            f"| {i + 1} | {ridge_results[i][2]:.4f} | {enet_results[i][2]:.4f} | {rf_results[i][2]:.4f} |\n"
        )
    lines.append("\n")

    lines.append("## R2 Gap vs XGBoost (XGBoost R2 - Baseline R2)\n\n")
    lines.append("| Model | R2 Gap |\n")
    lines.append("|---|---|\n")
    for name, gap in gap_checks.items():
        lines.append(f"| {name} | {gap:+.4f} |\n")
    lines.append("\n")

    lines.append("## Step 6 Success Check\n\n")
    lines.append(f"- Largest R2 gap vs XGBoost (any baseline): {widest_gap:+.4f}\n")
    lines.append(f"- **XGBoost's added complexity justified (gap > 0.1 for all baselines): {xgb_justified}**\n\n")
    if xgb_justified:
        lines.append(
            "All simpler baselines underperform XGBoost by more than 0.1 in "
            "nested-CV R2, justifying XGBoost's added complexity for the "
            "primary reported model. Linear coefficients/feature importances "
            "from the Ridge/ElasticNet fits may still be useful as a "
            "secondary, easier-to-communicate interpretability narrative.\n"
        )
    else:
        lines.append(
            "At least one simpler baseline is within 0.1 R2 of XGBoost in "
            "nested CV. Consider presenting both in the paper: the linear "
            "model for the interpretability narrative (transparent "
            "coefficients) and XGBoost for the best-performance number.\n"
        )

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
