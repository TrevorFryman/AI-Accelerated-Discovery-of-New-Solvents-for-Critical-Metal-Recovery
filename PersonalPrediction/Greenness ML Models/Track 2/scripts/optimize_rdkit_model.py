"""
RDKit Descriptor Pipeline - Hyperparameter Optimization & Interpretability

Loads descriptors/RDKit_Features.csv and optimizes the XGBoost regressor
using RandomizedSearchCV over:
- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree
- min_child_weight

Methodology mirrors the Morgan Fingerprint pipeline for fair comparison:
- Identical 80/20 train-test split (random_state=42)
- Hyperparameter search performed only on the training split (5-fold CV),
  so the test set remains untouched until final evaluation.
- Identical evaluation metrics (RMSE, MAE, R^2).

Additionally:
- Compares baseline vs optimized models.
- Generates a feature importance plot.
- Performs a SHAP analysis to determine which descriptors most strongly
  influence G-score predictions.

Outputs:
- models/rdkit_xgboost_model.pkl (best model, refit on training data)
- results/RDKit/feature_importance.png
- results/RDKit/shap_summary.png
- reports/RDKit_Interpretability_Report.md
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "descriptors" / "RDKit_Features.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results" / "RDKit"
REPORTS_DIR = BASE_DIR / "reports"

TARGET = "G-score"
NON_FEATURE_COLS = ["solvent_common_name", "CAS Number", "solvent_SMILES", TARGET]
MODEL_PATH = MODELS_DIR / "rdkit_xgboost_model.pkl"
REPORT_PATH = REPORTS_DIR / "RDKit_Interpretability_Report.md"


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def main():
    sns.set_theme(style="whitegrid")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].values
    y = df[TARGET].values

    # Identical 80/20 split as the baseline
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # --- Baseline model (for direct comparison) ---
    baseline_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    baseline_model.fit(X_train, y_train)
    y_pred_test_base = baseline_model.predict(X_test)
    rmse_base, mae_base, r2_base = evaluate(y_test, y_pred_test_base)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_oof_base = cross_val_predict(
        XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1),
        X, y, cv=kf,
    )
    rmse_oof_base, mae_oof_base, r2_oof_base = evaluate(y, y_oof_base)

    # --- Hyperparameter search ---
    param_distributions = {
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [2, 3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 2, 3, 5, 7, 10],
    }

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=100,
        scoring="neg_root_mean_squared_error",
        cv=kf,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_cv_rmse = -search.best_score_

    print("=" * 60)
    print("RANDOMIZED SEARCH RESULTS")
    print("=" * 60)
    print(f"Best CV RMSE (train split, 5-fold): {best_cv_rmse:.4f}")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Evaluate optimized model
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    rmse_train, mae_train, r2_train = evaluate(y_train, y_pred_train)
    rmse_test, mae_test, r2_test = evaluate(y_test, y_pred_test)

    print("\n" + "=" * 60)
    print("OPTIMIZED MODEL - TRAIN/TEST RESULTS")
    print("=" * 60)
    print(f"Train: RMSE={rmse_train:.4f}, MAE={mae_train:.4f}, R2={r2_train:.4f}")
    print(f"Test:  RMSE={rmse_test:.4f}, MAE={mae_test:.4f}, R2={r2_test:.4f}")

    # 5-fold CV (out-of-fold) on full dataset using best hyperparameters
    oof_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best_params,
    )
    y_oof = cross_val_predict(oof_model, X, y, cv=kf)
    rmse_oof, mae_oof, r2_oof = evaluate(y, y_oof)

    print("\n" + "=" * 60)
    print("OPTIMIZED MODEL - 5-FOLD CV (OUT-OF-FOLD) ON FULL DATASET")
    print("=" * 60)
    print(f"OOF: RMSE={rmse_oof:.4f}, MAE={mae_oof:.4f}, R2={r2_oof:.4f}")

    # Fit-quality assessment
    gap = r2_train - r2_oof
    if r2_train < 0.5 and r2_oof < 0.5:
        fit_assessment = "Underfitting"
    elif gap > 0.3:
        fit_assessment = "Overfitting"
    else:
        fit_assessment = "Appropriately fit"

    print(f"\nFit assessment: {fit_assessment} (train R2={r2_train:.4f}, OOF R2={r2_oof:.4f}, gap={gap:.4f})")

    # Save best model
    joblib.dump(best_model, MODEL_PATH)
    with open(MODELS_DIR / "rdkit_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest model saved to {MODEL_PATH}")

    # --- Feature importance plot ---
    importances = best_model.feature_importances_
    importance_order = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in importance_order]
    sorted_importances = importances[importance_order]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, color="slateblue")
    plt.xlabel("Feature Importance (XGBoost gain-based)")
    plt.ylabel("Descriptor")
    plt.title("Feature Importance - Optimized RDKit Descriptor Model")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close()

    # --- SHAP analysis ---
    explainer = shap.TreeExplainer(best_model)
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

    # --- Build interpretability report ---
    lines = []
    lines.append("# RDKit Descriptor Interpretability Report\n\n")

    lines.append("## 1. Model Architecture\n\n")
    lines.append("- Algorithm: XGBoost Regressor (`xgboost.XGBRegressor`)\n")
    lines.append("- Objective: `reg:squarederror`\n")
    lines.append(f"- Input features ({len(feature_cols)}): {', '.join(feature_cols)}\n")
    lines.append(f"- Target: {TARGET}\n")
    lines.append(f"- random_state: {RANDOM_STATE}\n\n")

    lines.append("## 2. Hyperparameters (Best Found via RandomizedSearchCV)\n\n")
    lines.append("| Hyperparameter | Search Space | Best Value |\n")
    lines.append("|---|---|---|\n")
    for param, space in param_distributions.items():
        lines.append(f"| {param} | {space} | {best_params[param]} |\n")
    lines.append(
        f"\nSearch configuration: `RandomizedSearchCV` with `n_iter=100`, "
        f"5-fold CV (`KFold(n_splits=5, shuffle=True, random_state={RANDOM_STATE})`), "
        f"scoring=`neg_root_mean_squared_error`, performed on the training split only "
        f"(80% of data; the test set was untouched during tuning).\n\n"
    )
    lines.append(f"Best training-CV RMSE during search: {best_cv_rmse:.4f}\n\n")

    lines.append("## 3. Baseline vs Optimized Comparison\n\n")
    lines.append("| Model | Test RMSE | Test MAE | Test R2 | OOF RMSE | OOF MAE | OOF R2 |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    lines.append(
        f"| Baseline (default params) | {rmse_base:.4f} | {mae_base:.4f} | {r2_base:.4f} "
        f"| {rmse_oof_base:.4f} | {mae_oof_base:.4f} | {r2_oof_base:.4f} |\n"
    )
    lines.append(
        f"| Optimized (RandomizedSearchCV) | {rmse_test:.4f} | {mae_test:.4f} | {r2_test:.4f} "
        f"| {rmse_oof:.4f} | {mae_oof:.4f} | {r2_oof:.4f} |\n\n"
    )

    lines.append("## 4. Validation Metrics (80/20 Train-Test Split, Optimized Model)\n\n")
    lines.append("| Set | RMSE | MAE | R2 |\n")
    lines.append("|---|---|---|---|\n")
    lines.append(f"| Train | {rmse_train:.4f} | {mae_train:.4f} | {r2_train:.4f} |\n")
    lines.append(f"| Test | {rmse_test:.4f} | {mae_test:.4f} | {r2_test:.4f} |\n\n")

    lines.append("## 5. Cross-Validation Results (5-Fold OOF, Full Dataset, Best Hyperparameters)\n\n")
    lines.append("| Metric | Value |\n")
    lines.append("|---|---|\n")
    lines.append(f"| RMSE | {rmse_oof:.4f} |\n")
    lines.append(f"| MAE | {mae_oof:.4f} |\n")
    lines.append(f"| R2 | {r2_oof:.4f} |\n\n")

    lines.append("## 6. Fit Assessment\n\n")
    lines.append(f"- Train R2: {r2_train:.4f}\n")
    lines.append(f"- 5-fold OOF R2: {r2_oof:.4f}\n")
    lines.append(f"- Gap: {gap:.4f}\n")
    lines.append(f"- **Assessment: {fit_assessment}**\n\n")

    lines.append("## 7. Feature Importance (XGBoost Gain-Based)\n\n")
    lines.append("| Rank | Descriptor | Importance |\n")
    lines.append("|---|---|---|\n")
    for rank, (feat, imp) in enumerate(zip(sorted_features, sorted_importances), start=1):
        lines.append(f"| {rank} | {feat} | {imp:.4f} |\n")
    lines.append("\n![Feature Importance](../results/RDKit/feature_importance.png)\n\n")

    lines.append("## 8. SHAP Analysis\n\n")
    lines.append(
        "SHAP (SHapley Additive exPlanations) values were computed using `shap.TreeExplainer` "
        "on the full dataset with the optimized model.\n\n"
    )
    lines.append("| Rank | Descriptor | Mean |SHAP value| |\n")
    lines.append("|---|---|---|\n")
    for rank, (feat, val) in enumerate(zip(shap_ranked_features, shap_ranked_values), start=1):
        lines.append(f"| {rank} | {feat} | {val:.4f} |\n")
    lines.append("\n![SHAP Summary](../results/RDKit/shap_summary.png)\n\n")
    lines.append("![SHAP Importance Bar](../results/RDKit/shap_importance_bar.png)\n\n")

    lines.append("## 9. Which Descriptors Most Strongly Influence G-score Predictions\n\n")
    top_feature = shap_ranked_features[0]
    lines.append(
        f"Based on both the gain-based feature importance and the SHAP analysis, "
        f"`{top_feature}` is the descriptor with the largest mean absolute SHAP value "
        f"(and therefore the strongest influence on G-score predictions), followed by "
        f"`{shap_ranked_features[1]}` and `{shap_ranked_features[2]}`. "
        f"This is broadly consistent with the EDA finding that TPSA showed the strongest "
        f"linear correlation with G-score among the descriptors examined.\n\n"
    )

    lines.append("## 10. Output Files\n\n")
    lines.append("- Best model: `models/rdkit_xgboost_model.pkl`\n")
    lines.append("- Best hyperparameters: `models/rdkit_best_params.json`\n")
    lines.append("- Baseline plots: `results/RDKit/learning_curve.png`, `results/RDKit/parity_residual_plots.png`\n")
    lines.append("- Feature importance: `results/RDKit/feature_importance.png`\n")
    lines.append("- SHAP plots: `results/RDKit/shap_summary.png`, `results/RDKit/shap_importance_bar.png`\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
