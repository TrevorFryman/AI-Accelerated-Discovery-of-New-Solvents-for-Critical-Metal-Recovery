"""
RDKit Descriptor Pipeline - GPR and Feature-Pruning Check (Step 7, Optional)

Steps 1-6 settled on a 13-descriptor XGBoost model with nested-CV
R2 = 0.5050 [0.3342, 0.6757] (results/RDKit_Nested_CV_Report.md), and showed
that further hyperparameter retuning/regularization (Step 3) and simpler
linear/RF baselines (Step 6) do not close the train/CV gap. With n=154
samples grouped into a small number of solvent classes, the gap looks like a
data-size ceiling rather than a tuning problem.

Step 7 (this script) tries two cheap, untried levers on the existing data:

1. Gaussian Process Regression (GPR) - a model family well-suited to small
   n that also gives calibrated uncertainty. Evaluated on the full
   13-descriptor set using the same nested-CV protocol as Step 4/6
   (outer 5-fold StratifiedGroupKFold, inner 4-fold StratifiedGroupKFold for
   kernel/alpha selection via GridSearchCV).

2. Feature pruning - several descriptors have near-zero gain/SHAP importance
   in the final model report (RingCount, NumAliphaticRings, NumAromaticRings,
   HeavyAtomCount, MolWt, BertzCT). Combine the gain-based and SHAP rankings
   from results/RDKit_Final_Model_Report.md into one score and evaluate
   XGBoost (with the Step 4 nested hyperparameter search) and GPR on the
   top-6 and top-8 feature subsets.

Outputs:
- results/RDKit_GPR_Pruning_Report.md
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "data" / "RDKit_Features.csv"
DATASET_PATH = BASE_DIR / "data" / "GSK_dataset.csv"
REPORTS_DIR = BASE_DIR / "results"
REPORT_PATH = REPORTS_DIR / "RDKit_GPR_Pruning_Report.md"

TARGET = "G-score"
NON_FEATURE_COLS = ["solvent_common_name", "CAS Number", "solvent_SMILES", "Classification", TARGET]

# Step 4/5 nested CV headline metrics for XGBoost (13 features) - reference
XGB13_NESTED_CV = {
    "rmse_mean": 0.8721, "rmse_lo": 0.7238, "rmse_hi": 1.0203,
    "mae_mean": 0.6699, "mae_lo": 0.5292, "mae_hi": 0.8107,
    "r2_mean": 0.5050, "r2_lo": 0.3342, "r2_hi": 0.6757,
}

# Combined gain + SHAP ranks from results/RDKit_Final_Model_Report.md (lower = more important)
GAIN_RANK = {
    "TPSA": 1, "NumHAcceptors": 2, "NumHDonors": 3, "NumRotatableBonds": 4,
    "MolMR": 5, "FractionCSP3": 6, "HeavyAtomCount": 7, "NumAromaticRings": 8,
    "LogP": 9, "BertzCT": 10, "MolWt": 11, "NumAliphaticRings": 12, "RingCount": 13,
}
SHAP_RANK = {
    "TPSA": 1, "MolMR": 2, "LogP": 3, "MolWt": 4, "BertzCT": 5,
    "NumHAcceptors": 6, "FractionCSP3": 7, "NumHDonors": 8, "NumRotatableBonds": 9,
    "RingCount": 10, "HeavyAtomCount": 11, "NumAromaticRings": 12, "NumAliphaticRings": 13,
}

XGB_PARAM_DISTRIBUTIONS = {
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

GPR_KERNELS = {
    "RBF": C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
           + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e1)),
    "Matern1.5": C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
                 + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e1)),
    "Matern2.5": C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
                 + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e1)),
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


def nested_cv_xgb(X, y, strata, groups, n_iter=100):
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    for train_idx, test_idx in outer_cv.split(X, strata, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, strata_train = groups[train_idx], strata[train_idx]

        inner_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
        inner_splits = list(inner_cv.split(X_train, strata_train, groups_train))

        search = RandomizedSearchCV(
            estimator=XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1),
            param_distributions=XGB_PARAM_DISTRIBUTIONS,
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=inner_splits,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        y_pred = search.best_estimator_.predict(X_test)
        results.append(evaluate(y_test, y_pred))
    return results


def nested_cv_gpr(X, y, strata, groups):
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    chosen_kernels = []
    for train_idx, test_idx in outer_cv.split(X, strata, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, strata_train = groups[train_idx], strata[train_idx]

        inner_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
        inner_splits = list(inner_cv.split(X_train, strata_train, groups_train))

        pipe = make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=5, random_state=RANDOM_STATE),
        )
        param_grid = {"gaussianprocessregressor__kernel": list(GPR_KERNELS.values())}
        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=inner_splits,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        y_pred = search.best_estimator_.predict(X_test)
        results.append(evaluate(y_test, y_pred))

        best_kernel = search.best_params_["gaussianprocessregressor__kernel"]
        kernel_name = next((k for k, v in GPR_KERNELS.items() if v == best_kernel), str(best_kernel))
        chosen_kernels.append(kernel_name)
    return results, chosen_kernels


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_PATH)
    class_df = pd.read_csv(DATASET_PATH)[["solvent_SMILES", "Classification"]]
    df = df.merge(class_df, on="solvent_SMILES", how="left")

    all_feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    y = df[TARGET].values
    groups = assign_groups(df["solvent_SMILES"])
    strata = df["Classification"].astype(str).to_numpy()

    # Combined rank -> sorted feature order (tiebreak: gain rank, then name)
    combined = sorted(
        all_feature_cols,
        key=lambda c: ((GAIN_RANK[c] + SHAP_RANK[c]) / 2, GAIN_RANK[c], c),
    )
    top6 = combined[:6]
    top8 = combined[:8]

    feature_sets = {
        "13 (full)": all_feature_cols,
        "8 (pruned)": top8,
        "6 (pruned)": top6,
    }

    print("Feature subsets:")
    for name, cols in feature_sets.items():
        print(f"  {name}: {cols}")

    xgb_summaries = {}
    for name, cols in feature_sets.items():
        print(f"\nRunning nested CV (XGBoost) for feature set: {name} ...")
        X = df[cols].values
        results = nested_cv_xgb(X, y, strata, groups, n_iter=100)
        xgb_summaries[name] = summarize(results)

    gpr_summaries = {}
    gpr_kernels = {}
    for name, cols in feature_sets.items():
        print(f"\nRunning nested CV (GPR) for feature set: {name} ...")
        X = df[cols].values
        results, kernels = nested_cv_gpr(X, y, strata, groups)
        gpr_summaries[name] = summarize(results)
        gpr_kernels[name] = kernels

    # --- Console summary ---
    print("\n" + "=" * 70)
    print("NESTED CV RESULTS (mean +/- 95% CI, t-dist df=4)")
    print("=" * 70)
    print(f"XGBoost 13-feature (Step 4/5 reference): R2 = {XGB13_NESTED_CV['r2_mean']:.4f} "
          f"[{XGB13_NESTED_CV['r2_lo']:.4f}, {XGB13_NESTED_CV['r2_hi']:.4f}]")
    for name, summ in xgb_summaries.items():
        r2_mean, r2_lo, r2_hi = summ["r2"]
        print(f"XGBoost {name}: R2 = {r2_mean:.4f} [{r2_lo:.4f}, {r2_hi:.4f}]")
    for name, summ in gpr_summaries.items():
        r2_mean, r2_lo, r2_hi = summ["r2"]
        print(f"GPR     {name}: R2 = {r2_mean:.4f} [{r2_lo:.4f}, {r2_hi:.4f}]  kernels={gpr_kernels[name]}")

    # --- Report ---
    lines = []
    lines.append("# RDKit Descriptor Model - GPR and Feature-Pruning Check (Step 7, Optional)\n\n")
    lines.append(
        "Step 7 (optional) of the fine-tuning plan: evaluate Gaussian Process "
        "Regression (untried model family, well-suited to small n) and "
        "feature-pruned subsets (top-8 and top-6 descriptors by combined "
        "gain+SHAP rank from `results/RDKit_Final_Model_Report.md`), using "
        "the same nested-CV protocol as Step 4/6 (outer 5-fold "
        "`StratifiedGroupKFold`, inner 4-fold `StratifiedGroupKFold` for "
        "hyperparameter/kernel selection).\n\n"
    )
    lines.append("## Feature Subsets\n\n")
    for name, cols in feature_sets.items():
        lines.append(f"- **{name}**: {', '.join(cols)}\n")
    lines.append("\n")

    lines.append("## Nested CV Results (Mean +/- 95% CI, t-distribution df=4)\n\n")
    lines.append("| Model | Feature Set | R2 | RMSE | MAE |\n")
    lines.append("|---|---|---|---|---|\n")
    lines.append(
        f"| XGBoost (Step 4/5, reference) | 13 (full) | "
        f"{XGB13_NESTED_CV['r2_mean']:.4f} [{XGB13_NESTED_CV['r2_lo']:.4f}, {XGB13_NESTED_CV['r2_hi']:.4f}] | "
        f"{XGB13_NESTED_CV['rmse_mean']:.4f} [{XGB13_NESTED_CV['rmse_lo']:.4f}, {XGB13_NESTED_CV['rmse_hi']:.4f}] | "
        f"{XGB13_NESTED_CV['mae_mean']:.4f} [{XGB13_NESTED_CV['mae_lo']:.4f}, {XGB13_NESTED_CV['mae_hi']:.4f}] |\n"
    )
    for name, summ in xgb_summaries.items():
        r2_mean, r2_lo, r2_hi = summ["r2"]
        rmse_mean, rmse_lo, rmse_hi = summ["rmse"]
        mae_mean, mae_lo, mae_hi = summ["mae"]
        lines.append(
            f"| XGBoost (Step 7 re-search) | {name} | "
            f"{r2_mean:.4f} [{r2_lo:.4f}, {r2_hi:.4f}] | "
            f"{rmse_mean:.4f} [{rmse_lo:.4f}, {rmse_hi:.4f}] | "
            f"{mae_mean:.4f} [{mae_lo:.4f}, {mae_hi:.4f}] |\n"
        )
    for name, summ in gpr_summaries.items():
        r2_mean, r2_lo, r2_hi = summ["r2"]
        rmse_mean, rmse_lo, rmse_hi = summ["rmse"]
        mae_mean, mae_lo, mae_hi = summ["mae"]
        lines.append(
            f"| GPR | {name} | "
            f"{r2_mean:.4f} [{r2_lo:.4f}, {r2_hi:.4f}] | "
            f"{rmse_mean:.4f} [{rmse_lo:.4f}, {rmse_hi:.4f}] | "
            f"{mae_mean:.4f} [{mae_lo:.4f}, {mae_hi:.4f}] |\n"
        )
    lines.append("\n")

    lines.append("## Per-Fold R2\n\n")
    lines.append("| Outer Fold | XGB 13 (Step 7) | XGB 8 | XGB 6 | GPR 13 | GPR 8 | GPR 6 |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for i in range(5):
        lines.append(
            f"| {i + 1} "
            f"| {xgb_summaries['13 (full)']['per_fold'][i][2]:.4f} "
            f"| {xgb_summaries['8 (pruned)']['per_fold'][i][2]:.4f} "
            f"| {xgb_summaries['6 (pruned)']['per_fold'][i][2]:.4f} "
            f"| {gpr_summaries['13 (full)']['per_fold'][i][2]:.4f} "
            f"| {gpr_summaries['8 (pruned)']['per_fold'][i][2]:.4f} "
            f"| {gpr_summaries['6 (pruned)']['per_fold'][i][2]:.4f} |\n"
        )
    lines.append("\n")

    lines.append("## GPR Kernel Selection Per Fold\n\n")
    lines.append("| Outer Fold | 13 (full) | 8 (pruned) | 6 (pruned) |\n")
    lines.append("|---|---|---|---|\n")
    for i in range(5):
        lines.append(
            f"| {i + 1} | {gpr_kernels['13 (full)'][i]} | {gpr_kernels['8 (pruned)'][i]} | "
            f"{gpr_kernels['6 (pruned)'][i]} |\n"
        )
    lines.append("\n")

    # --- Success check: any candidate beat the reference XGBoost 13-feature R2 mean? ---
    best_name, best_r2 = None, XGB13_NESTED_CV["r2_mean"]
    for name, summ in xgb_summaries.items():
        if name == "13 (full)":
            continue
        if summ["r2"][0] > best_r2:
            best_name, best_r2 = f"XGBoost {name}", summ["r2"][0]
    for name, summ in gpr_summaries.items():
        if summ["r2"][0] > best_r2:
            best_name, best_r2 = f"GPR {name}", summ["r2"][0]

    lines.append("## Step 7 Success Check\n\n")
    lines.append(
        f"- Reference (XGBoost, 13 features, Step 4/5): "
        f"R2 = {XGB13_NESTED_CV['r2_mean']:.4f} [{XGB13_NESTED_CV['r2_lo']:.4f}, {XGB13_NESTED_CV['r2_hi']:.4f}]\n"
    )
    if best_name is not None:
        lines.append(
            f"- **Best alternative: {best_name}, R2 = {best_r2:.4f} "
            f"(improvement of {best_r2 - XGB13_NESTED_CV['r2_mean']:+.4f})**\n\n"
        )
        lines.append(
            "Note that with only 5 outer folds, differences smaller than the "
            "width of the confidence intervals are not statistically "
            "meaningful; treat this as a directional signal rather than a "
            "confirmed improvement.\n"
        )
    else:
        lines.append(
            "- **No alternative (GPR or pruned-feature XGBoost) improved on "
            "the reference R2 mean.**\n\n"
        )
        lines.append(
            "This reinforces the conclusion from Steps 3 and 6: the "
            "train/CV gap for this model is most likely a data-size ceiling "
            "(n=154 across a small number of solvent groups) rather than a "
            "model-family, hyperparameter, or feature-count problem. Closing "
            "the gap further would most likely require additional labeled "
            "solvents.\n"
        )

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
