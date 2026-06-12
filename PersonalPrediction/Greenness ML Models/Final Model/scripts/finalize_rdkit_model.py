"""
RDKit Descriptor Pipeline - Final Model Interpretability (Step 5)

Fine-tuning pipeline so far:
- Step 1: StratifiedGroupKFold split (grouped by solvent_SMILES, stratified
  by Classification).
- Step 2: expanded the descriptor set from 7 to 13 named, interpretable
  RDKit descriptors.
- Step 3 (regularized re-tune): rejected, kept Step 2 hyperparameters
  (n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.7,
  colsample_bytree=0.8, min_child_weight=1).
- Step 4: nested cross-validation gave the headline generalization metric:
  R2 = 0.5050 [0.3342, 0.6757] (95% CI), RMSE = 0.8721 [0.7238, 1.0203].

Step 5 (this script): with the final hyperparameter set (model/rdkit_best_params.json)
and feature set (13 descriptors) settled, refit the model on the FULL
dataset (all 154 samples) for deployment/reporting, and regenerate all
interpretability and diagnostic plots against this final model:
- feature_importance.png (gain-based, full-data model)
- shap_summary.png, shap_importance_bar.png (full-data model)
- parity_residual_plots.png (OOF predictions via StratifiedGroupKFold(5)
  with the final hyperparameters, since the full-data model has no
  held-out test set)
- learning_curve.png (StratifiedGroupKFold(5) CV)

The nested-CV scores from Step 4 remain the headline generalization metric
reported for this model; the metrics computed here (train-set fit, OOF via
CV) are diagnostic only and are not reported as generalization estimates.

Outputs:
- model/rdkit_xgboost_model.pkl (refit on full 154-sample dataset)
- results/RDKit/feature_importance.png
- results/RDKit/shap_summary.png
- results/RDKit/shap_importance_bar.png
- results/RDKit/parity_residual_plots.png
- results/RDKit/learning_curve.png
- results/RDKit_Final_Model_Report.md
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, learning_curve
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
REPORT_PATH = REPORTS_DIR / "RDKit_Final_Model_Report.md"

# Step 4 nested CV headline metrics
NESTED_CV = {
    "rmse_mean": 0.8721, "rmse_lo": 0.7238, "rmse_hi": 1.0203,
    "mae_mean": 0.6699, "mae_lo": 0.5292, "mae_hi": 0.8107,
    "r2_mean": 0.5050, "r2_lo": 0.3342, "r2_hi": 0.6757,
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

    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        best_params = json.load(f)

    # --- Refit final model on the FULL dataset ---
    final_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best_params,
    )
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_PATH)

    y_pred_full = final_model.predict(X)
    rmse_full, mae_full, r2_full = evaluate(y, y_pred_full)
    print("=" * 60)
    print("FINAL MODEL - FULL-DATASET FIT (diagnostic only, not generalization)")
    print("=" * 60)
    print(f"Full data: RMSE={rmse_full:.4f}, MAE={mae_full:.4f}, R2={r2_full:.4f}")

    # --- OOF predictions via StratifiedGroupKFold(5) with final hyperparameters ---
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best_params,
    )
    y_oof = cross_val_predict(oof_model, X, y, cv=list(sgkf.split(X, strata, groups)))
    rmse_oof, mae_oof, r2_oof = evaluate(y, y_oof)
    print(f"OOF (SGKF5):  RMSE={rmse_oof:.4f}, MAE={mae_oof:.4f}, R2={r2_oof:.4f}")

    # --- Feature importance plot (final, full-data model) ---
    importances = final_model.feature_importances_
    importance_order = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in importance_order]
    sorted_importances = importances[importance_order]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, color="slateblue")
    plt.xlabel("Feature Importance (XGBoost gain-based)")
    plt.ylabel("Descriptor")
    plt.title("Feature Importance - Final RDKit Descriptor Model (13 features)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close()

    # --- SHAP analysis (final, full-data model) ---
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

    # --- Parity and residual plots (full-data fit + OOF) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].scatter(y, y_pred_full, color="steelblue", edgecolor="black", alpha=0.7)
    lims_full = [min(y.min(), y_pred_full.min()), max(y.max(), y_pred_full.max())]
    axes[0, 0].plot(lims_full, lims_full, "k--", label="Ideal")
    axes[0, 0].set_xlabel("Actual G-score")
    axes[0, 0].set_ylabel("Predicted G-score")
    axes[0, 0].set_title(f"Parity Plot - Full-Data Fit (R2={r2_full:.3f}, diagnostic only)")
    axes[0, 0].legend()

    residuals_full = y - y_pred_full
    axes[0, 1].scatter(y_pred_full, residuals_full, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0, 1].axhline(0, color="k", linestyle="--")
    axes[0, 1].set_xlabel("Predicted G-score")
    axes[0, 1].set_ylabel("Residual (Actual - Predicted)")
    axes[0, 1].set_title("Residual Plot - Full-Data Fit")

    axes[1, 0].scatter(y, y_oof, color="seagreen", edgecolor="black", alpha=0.7)
    lims_oof = [min(y.min(), y_oof.min()), max(y.max(), y_oof.max())]
    axes[1, 0].plot(lims_oof, lims_oof, "k--", label="Ideal")
    axes[1, 0].set_xlabel("Actual G-score")
    axes[1, 0].set_ylabel("OOF Predicted G-score")
    axes[1, 0].set_title(f"Parity Plot - 5-Fold OOF (SGKF, R2={r2_oof:.3f})")
    axes[1, 0].legend()

    residuals_oof = y - y_oof
    axes[1, 1].scatter(y_oof, residuals_oof, color="seagreen", edgecolor="black", alpha=0.7)
    axes[1, 1].axhline(0, color="k", linestyle="--")
    axes[1, 1].set_xlabel("OOF Predicted G-score")
    axes[1, 1].set_ylabel("Residual (Actual - Predicted)")
    axes[1, 1].set_title("Residual Plot - 5-Fold OOF (SGKF)")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "parity_residual_plots.png", dpi=150)
    plt.close()

    # --- Learning curve (StratifiedGroupKFold(5)) ---
    lc_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best_params,
    )
    lc_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, val_scores = learning_curve(
        lc_model, X, y,
        cv=list(lc_cv.split(X, strata, groups)),
        scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 8),
        random_state=RANDOM_STATE,
    )
    train_scores_mean = train_scores.mean(axis=1)
    val_scores_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, "o-", color="steelblue", label="Training score (R2)")
    plt.plot(train_sizes, val_scores_mean, "o-", color="darkorange", label="Cross-validation score (R2)")
    plt.xlabel("Training set size")
    plt.ylabel("R2 score")
    plt.title("Learning Curve - Final RDKit Descriptor Model (13 features)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curve.png", dpi=150)
    plt.close()

    # --- Success check: TPSA should remain a top-2 driver ---
    tpsa_importance_rank = sorted_features.index("TPSA") + 1
    tpsa_shap_rank = shap_ranked_features.index("TPSA") + 1
    tpsa_top2 = tpsa_importance_rank <= 2 or tpsa_shap_rank <= 2

    print(f"\nTPSA importance rank: {tpsa_importance_rank}, TPSA SHAP rank: {tpsa_shap_rank}")
    print(f"TPSA remains a top-2 driver: {tpsa_top2}")

    # --- Report ---
    lines = []
    lines.append("# RDKit Descriptor Model - Final Model Report (Step 5)\n\n")
    lines.append(
        "Step 5 of the fine-tuning plan: with the final hyperparameter set "
        "(`model/rdkit_best_params.json`, from Step 2/3) and the 13-descriptor "
        "feature set (Step 2) settled, the model was refit on the full "
        "154-sample dataset for deployment/reporting, and all interpretability "
        "and diagnostic plots were regenerated against this final model. The "
        "Step 4 nested-CV scores remain the headline generalization metric "
        "for this model; the full-data fit and OOF metrics below are "
        "diagnostic only.\n\n"
    )
    lines.append(f"Feature columns ({len(feature_cols)}): {', '.join(feature_cols)}\n\n")
    lines.append(f"Final hyperparameters: `{best_params}`\n\n")

    lines.append("## Headline Generalization Metric (Step 4, Nested CV)\n\n")
    lines.append("| Metric | Mean | 95% CI Lower | 95% CI Upper |\n")
    lines.append("|---|---|---|---|\n")
    lines.append(f"| RMSE | {NESTED_CV['rmse_mean']:.4f} | {NESTED_CV['rmse_lo']:.4f} | {NESTED_CV['rmse_hi']:.4f} |\n")
    lines.append(f"| MAE | {NESTED_CV['mae_mean']:.4f} | {NESTED_CV['mae_lo']:.4f} | {NESTED_CV['mae_hi']:.4f} |\n")
    lines.append(f"| R2 | {NESTED_CV['r2_mean']:.4f} | {NESTED_CV['r2_lo']:.4f} | {NESTED_CV['r2_hi']:.4f} |\n\n")

    lines.append("## Diagnostic Metrics (Final Model, Not Generalization Estimates)\n\n")
    lines.append("| Set | RMSE | MAE | R2 |\n")
    lines.append("|---|---|---|---|\n")
    lines.append(f"| Full-data fit (refit model) | {rmse_full:.4f} | {mae_full:.4f} | {r2_full:.4f} |\n")
    lines.append(f"| OOF (5-fold SGKF, final params) | {rmse_oof:.4f} | {mae_oof:.4f} | {r2_oof:.4f} |\n\n")

    lines.append("## Feature Importance (XGBoost Gain-Based, Final Model)\n\n")
    lines.append("| Rank | Descriptor | Importance |\n")
    lines.append("|---|---|---|\n")
    for rank, (feat, imp) in enumerate(zip(sorted_features, sorted_importances), start=1):
        lines.append(f"| {rank} | {feat} | {imp:.4f} |\n")
    lines.append("\n![Feature Importance](RDKit/feature_importance.png)\n\n")

    lines.append("## SHAP Ranking (Mean |SHAP value|, Final Model)\n\n")
    lines.append("| Rank | Descriptor | Mean |SHAP value| |\n")
    lines.append("|---|---|---|\n")
    for rank, (feat, val) in enumerate(zip(shap_ranked_features, shap_ranked_values), start=1):
        lines.append(f"| {rank} | {feat} | {val:.4f} |\n")
    lines.append("\n![SHAP Summary](RDKit/shap_summary.png)\n\n")
    lines.append("![SHAP Importance Bar](RDKit/shap_importance_bar.png)\n\n")

    lines.append("## Step 5 Success Check\n\n")
    lines.append(f"- TPSA feature-importance rank: {tpsa_importance_rank} (of {len(feature_cols)})\n")
    lines.append(f"- TPSA SHAP rank: {tpsa_shap_rank} (of {len(feature_cols)})\n")
    lines.append(f"- **TPSA remains a top-2 driver: {tpsa_top2}**\n\n")

    lines.append("## Diagnostic Plots\n\n")
    lines.append("- `results/RDKit/parity_residual_plots.png`\n")
    lines.append("- `results/RDKit/learning_curve.png`\n\n")

    lines.append("## Output Files\n\n")
    lines.append("- Final model (refit on all 154 samples): `model/rdkit_xgboost_model.pkl`\n")
    lines.append("- Final hyperparameters: `model/rdkit_best_params.json`\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
