"""
RDKit Descriptor Pipeline - Regularization Tuning (v2, Step 3)

v2 of the original Track 2 optimize_rdkit_model.py, evolved through the
Final Model fine-tuning plan (Steps 1-3).

Fine-tuning pipeline so far:
- Step 1: StratifiedGroupKFold split (grouped by solvent_SMILES, stratified
  by Classification) instead of the old random 80/20 + plain KFold.
- Step 2: expanded the descriptor set from 7 to 13 named, interpretable
  RDKit descriptors (added NumAromaticRings, FractionCSP3, MolMR,
  HeavyAtomCount, NumAliphaticRings, BertzCT). OOF R2 improved
  (0.4801 -> 0.5123), kept hyperparameters:
  n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.7,
  colsample_bytree=0.8, min_child_weight=1 (no L1/L2 penalty, no gamma).
  Train R2=0.8893, OOF R2=0.5123, gap=0.3770.
- Step 3 (a first attempt at re-tuning with the original search space) was
  REJECTED: it produced strictly worse Train/Test/OOF R2 and a wider gap,
  likely due to noisy inner StratifiedGroupKFold(5) CV folds on only 126
  training samples across 11 classes.

Step 3 (this script, regularization-focused): extend the search space with
explicit regularization terms (reg_alpha, reg_lambda, gamma) that were not
in the original grid, cap max_depth to [2, 3, 4], and bias min_child_weight
toward higher values. Goal: shrink the train-CV gap by discouraging the
model from fitting individual training points, while holding OOF R2 steady
or improving it slightly.

Outputs:
- model/rdkit_xgboost_model.pkl (refit on training data with best params)
- model/rdkit_best_params.json
- results/RDKit/feature_importance.png
- results/RDKit/shap_summary.png
- results/RDKit/shap_importance_bar.png
- results/RDKit_Regularization_Check.md
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "data" / "RDKit_Features.csv"
DATASET_PATH = BASE_DIR / "data" / "GSK_dataset.csv"
MODELS_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results" / "RDKit"
REPORTS_DIR = BASE_DIR / "results"

TARGET = "G-score"
NON_FEATURE_COLS = ["solvent_common_name", "CAS Number", "solvent_SMILES", "Classification", TARGET]
MODEL_PATH = MODELS_DIR / "rdkit_xgboost_model.pkl"
BEST_PARAMS_PATH = MODELS_DIR / "rdkit_best_params.json"
REPORT_PATH = REPORTS_DIR / "RDKit_Regularization_Check.md"

# Current best (Step 2, kept): 13 features, no regularization
CURRENT_BEST = {
    "params": {
        "subsample": 0.7, "n_estimators": 500, "min_child_weight": 1,
        "max_depth": 3, "learning_rate": 0.01, "colsample_bytree": 0.8,
    },
    "rmse_train": 0.4198, "mae_train": 0.3266, "r2_train": 0.8893,
    "rmse_test": 1.0625, "mae_test": 0.8568, "r2_test": 0.3105,
    "rmse_oof": 0.8881, "mae_oof": 0.6778, "r2_oof": 0.5123,
    "gap": 0.3770,
}


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def assign_groups(smiles_series):
    smiles_to_id = {smi: i for i, smi in enumerate(smiles_series.unique())}
    return smiles_series.map(smiles_to_id).to_numpy(dtype=int)


def main():
    sns.set_theme(style="whitegrid")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_PATH)

    class_df = pd.read_csv(DATASET_PATH)[["solvent_SMILES", "Classification"]]
    df = df.merge(class_df, on="solvent_SMILES", how="left")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].values
    y = df[TARGET].values
    groups = assign_groups(df["solvent_SMILES"])
    strata = df["Classification"].astype(str).to_numpy()

    # Step 1 split: fold 0 is the held-out test set, remaining 4 folds for OOF CV.
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(sgkf.split(X, strata, groups))
    train_idx, test_idx = splits[0]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train, strata_train = groups[train_idx], strata[train_idx]

    # --- Hyperparameter search on the training split only ---
    search_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search_splits = list(search_cv.split(X_train, strata_train, groups_train))

    param_distributions = {
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

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=250,
        scoring="neg_root_mean_squared_error",
        cv=search_splits,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_model: XGBRegressor = search.best_estimator_  # type: ignore[assignment]
    best_params = search.best_params_
    best_cv_rmse = -search.best_score_

    print("=" * 60)
    print("REGULARIZED SEARCH RESULTS (13 features, StratifiedGroupKFold)")
    print("=" * 60)
    print(f"Best CV RMSE (train split, 5-fold SGKF): {best_cv_rmse:.4f}")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    rmse_train, mae_train, r2_train = evaluate(y_train, y_pred_train)
    rmse_test, mae_test, r2_test = evaluate(y_test, y_pred_test)

    print("\n" + "=" * 60)
    print("REGULARIZED MODEL - TRAIN/TEST RESULTS")
    print("=" * 60)
    print(f"Train: RMSE={rmse_train:.4f}, MAE={mae_train:.4f}, R2={r2_train:.4f}")
    print(f"Test:  RMSE={rmse_test:.4f}, MAE={mae_test:.4f}, R2={r2_test:.4f}")

    # OOF predictions on full dataset using StratifiedGroupKFold with best params
    oof_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best_params,
    )
    y_oof = cross_val_predict(oof_model, X, y, cv=list(oof_sgkf.split(X, strata, groups)))
    rmse_oof, mae_oof, r2_oof = evaluate(y, y_oof)

    gap = r2_train - r2_oof

    print("\n" + "=" * 60)
    print("REGULARIZED MODEL - 5-FOLD CV (OOF) ON FULL DATASET (StratifiedGroupKFold)")
    print("=" * 60)
    print(f"OOF: RMSE={rmse_oof:.4f}, MAE={mae_oof:.4f}, R2={r2_oof:.4f}")
    print(f"\nTrain-CV R2 gap (Step 3, regularized): {gap:.4f}")
    print(f"Train-CV R2 gap (current best): {CURRENT_BEST['gap']:.4f}")
    print(f"OOF R2: {r2_oof:.4f} (current best: {CURRENT_BEST['r2_oof']:.4f})")

    # --- Success check ---
    train_r2_dropped_toward_target = r2_train < CURRENT_BEST["r2_train"]
    oof_stable_or_improved = r2_oof >= (CURRENT_BEST["r2_oof"] - 0.01)
    gap_shrunk = gap < CURRENT_BEST["gap"]
    success = train_r2_dropped_toward_target and oof_stable_or_improved and gap_shrunk

    print(f"\nTrain R2 dropped vs current best: {train_r2_dropped_toward_target} ({r2_train:.4f} vs {CURRENT_BEST['r2_train']:.4f})")
    print(f"OOF R2 stable or improved (>= best - 0.01): {oof_stable_or_improved} ({r2_oof:.4f} vs {CURRENT_BEST['r2_oof']:.4f})")
    print(f"Gap shrunk vs current best: {gap_shrunk} ({gap:.4f} vs {CURRENT_BEST['gap']:.4f})")
    print(f"Overall success: {success}")

    # --- Save best model + params only if successful, else keep current best ---
    final_model: XGBRegressor
    if success:
        joblib.dump(best_model, MODEL_PATH)
        with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)
        print(f"\nSUCCESS: new model saved to {MODEL_PATH}")
        print(f"New best params saved to {BEST_PARAMS_PATH}")
        final_model = best_model
    else:
        print("\nFAIL: keeping current best model/params unchanged.")
        final_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **CURRENT_BEST["params"],
        )
        final_model.fit(X_train, y_train)

    # --- Feature importance plot ---
    importances = final_model.feature_importances_
    importance_order = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in importance_order]
    sorted_importances = importances[importance_order]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, color="slateblue")
    plt.xlabel("Feature Importance (XGBoost gain-based)")
    plt.ylabel("Descriptor")
    plt.title("Feature Importance - RDKit Descriptor Model (13 features)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close()

    # --- SHAP analysis ---
    explainer = shap.TreeExplainer(final_model)
    X_all_df = pd.DataFrame(X, columns=feature_cols)
    shap_values = explainer(X_all_df)

    plt.figure()
    shap.summary_plot(shap_values, X_all_df, show=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_all_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_order = np.argsort(mean_abs_shap)[::-1]
    shap_ranked_features = [feature_cols[i] for i in shap_order]
    shap_ranked_values = mean_abs_shap[shap_order]

    # --- Report ---
    lines = []
    lines.append("# RDKit Descriptor Model - Regularization Check (Step 3)\n\n")
    lines.append(
        "Step 3 of the fine-tuning plan: extended the hyperparameter search "
        "space with explicit regularization terms (`reg_alpha`, `reg_lambda`, "
        "`gamma`), capped `max_depth` to [2, 3, 4], and biased "
        "`min_child_weight` toward higher values, using `RandomizedSearchCV` "
        "(n_iter=250) on the 13-descriptor feature set under the Step 1 "
        "`StratifiedGroupKFold` split (fold 0 = held-out test set). The search "
        "CV is a separate `StratifiedGroupKFold(5)` over the training portion "
        "only.\n\n"
    )
    lines.append(f"Feature columns ({len(feature_cols)}): {', '.join(feature_cols)}\n\n")

    lines.append("## Hyperparameters\n\n")
    lines.append("| Hyperparameter | Search Space | Best Value (Step 3) | Current Best |\n")
    lines.append("|---|---|---|---|\n")
    for param, space in param_distributions.items():
        lines.append(f"| {param} | {space} | {best_params.get(param)} | {CURRENT_BEST['params'].get(param, '-')} |\n")
    lines.append(f"\nBest training-CV RMSE during search: {best_cv_rmse:.4f}\n\n")

    lines.append("## Results Comparison\n\n")
    lines.append("| Config | Set | RMSE | MAE | R2 |\n")
    lines.append("|---|---|---|---|---|\n")
    lines.append(f"| Current best (Step 2 params) | Train | {CURRENT_BEST['rmse_train']:.4f} | {CURRENT_BEST['mae_train']:.4f} | {CURRENT_BEST['r2_train']:.4f} |\n")
    lines.append(f"| Current best (Step 2 params) | Test | {CURRENT_BEST['rmse_test']:.4f} | {CURRENT_BEST['mae_test']:.4f} | {CURRENT_BEST['r2_test']:.4f} |\n")
    lines.append(f"| Current best (Step 2 params) | OOF | {CURRENT_BEST['rmse_oof']:.4f} | {CURRENT_BEST['mae_oof']:.4f} | {CURRENT_BEST['r2_oof']:.4f} |\n")
    lines.append(f"| Step 3 (regularized) | Train | {rmse_train:.4f} | {mae_train:.4f} | {r2_train:.4f} |\n")
    lines.append(f"| Step 3 (regularized) | Test | {rmse_test:.4f} | {mae_test:.4f} | {r2_test:.4f} |\n")
    lines.append(f"| Step 3 (regularized) | OOF | {rmse_oof:.4f} | {mae_oof:.4f} | {r2_oof:.4f} |\n\n")

    lines.append("## Train-CV R2 Gap\n\n")
    lines.append(f"- Current best gap: {CURRENT_BEST['gap']:.4f}\n")
    lines.append(f"- Step 3 gap (regularized): {gap:.4f}\n")
    lines.append(f"- Change: {gap - CURRENT_BEST['gap']:+.4f}\n\n")

    lines.append("## Step 3 Success Check\n\n")
    lines.append(f"- Train R2 dropped vs current best: {train_r2_dropped_toward_target} ({r2_train:.4f} vs {CURRENT_BEST['r2_train']:.4f})\n")
    lines.append(f"- OOF R2 stable or improved (>= best - 0.01): {oof_stable_or_improved} ({r2_oof:.4f} vs {CURRENT_BEST['r2_oof']:.4f})\n")
    lines.append(f"- Train-CV gap shrunk vs current best: {gap_shrunk} ({gap:.4f} vs {CURRENT_BEST['gap']:.4f})\n")
    lines.append(f"- **Overall: {'PASS' if success else 'FAIL'}**\n\n")

    if success:
        lines.append("Step 3 hyperparameters were **accepted**. `model/rdkit_xgboost_model.pkl` "
                      "and `model/rdkit_best_params.json` have been updated to the regularized "
                      "configuration above.\n\n")
    else:
        lines.append("Step 3 hyperparameters were **rejected**. `model/rdkit_xgboost_model.pkl` "
                      "and `model/rdkit_best_params.json` remain at the current best "
                      "(Step 2) configuration, refit on the Step 1 training split.\n\n")

    lines.append("## Feature Importance (XGBoost Gain-Based, current model)\n\n")
    lines.append("| Rank | Descriptor | Importance |\n")
    lines.append("|---|---|---|\n")
    for rank, (feat, imp) in enumerate(zip(sorted_features, sorted_importances), start=1):
        lines.append(f"| {rank} | {feat} | {imp:.4f} |\n")
    lines.append("\n![Feature Importance](RDKit/feature_importance.png)\n\n")

    lines.append("## SHAP Ranking (Mean |SHAP value|, current model)\n\n")
    lines.append("| Rank | Descriptor | Mean |SHAP value| |\n")
    lines.append("|---|---|---|\n")
    for rank, (feat, val) in enumerate(zip(shap_ranked_features, shap_ranked_values), start=1):
        lines.append(f"| {rank} | {feat} | {val:.4f} |\n")
    lines.append("\n![SHAP Summary](RDKit/shap_summary.png)\n\n")
    lines.append("![SHAP Importance Bar](RDKit/shap_importance_bar.png)\n\n")

    lines.append("## Output Files\n\n")
    lines.append("- Current model: `model/rdkit_xgboost_model.pkl`\n")
    lines.append("- Current hyperparameters: `model/rdkit_best_params.json`\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
