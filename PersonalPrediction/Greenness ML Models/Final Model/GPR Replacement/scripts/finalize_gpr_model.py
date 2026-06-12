"""
RDKit Descriptor Pipeline - GPR Final Model (Step 7 Replacement)

Step 7 (results/RDKit_GPR_Pruning_Report.md, in the parent Final Model
folder) showed that a Gaussian Process Regressor (GPR) with the same
13-descriptor feature set as the XGBoost final model outperforms it in
nested CV:

  XGBoost (Step 4/5):  R2 = 0.5050 [0.3342, 0.6757], RMSE = 0.8721
  GPR (Step 7):        R2 = 0.5798 [0.5067, 0.6530], RMSE = 0.8111

Every outer fold selected a Matern(nu=1.5) + WhiteKernel kernel, so that is
fixed as the final kernel here.

This script (the GPR analogue of finalize_rdkit_model.py /
generate_parity_residual_with_stats.py):
- Refits StandardScaler -> GaussianProcessRegressor(Matern(1.5) + White,
  normalize_y=True) on the full 154-sample dataset for deployment/reporting.
- Computes OOF predictions via 5-fold StratifiedGroupKFold (same protocol as
  the XGBoost final model) for diagnostic parity/residual plots.
- Generates SHAP summary + bar plots via KernelExplainer (model-agnostic,
  since GPR has no TreeExplainer).
- Generates a learning curve.
- Writes results/GPR_Final_Model_Report.md.

The Step 7 nested-CV scores remain the headline generalization metric; the
full-data fit and OOF metrics here are diagnostic only.

Outputs:
- model/gpr_model.pkl
- results/GPR/parity_residual_plots_with_stats.png
- results/GPR/shap_summary.png
- results/GPR/shap_importance_bar.png
- results/GPR/learning_curve.png
- results/GPR_Final_Model_Report.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from pathlib import Path
from scipy.stats import pearsonr

from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "data" / "RDKit_Features.csv"
DATASET_PATH = BASE_DIR / "data" / "GSK_dataset.csv"
MODELS_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results" / "GPR"
REPORTS_DIR = BASE_DIR / "results"

TARGET = "G-score"
NON_FEATURE_COLS = ["solvent_common_name", "CAS Number", "solvent_SMILES", "Classification", TARGET]
MODEL_PATH = MODELS_DIR / "gpr_model.pkl"
REPORT_PATH = REPORTS_DIR / "GPR_Final_Model_Report.md"

# Step 7 nested CV headline metrics (GPR, 13 features, Matern1.5 + White kernel)
NESTED_CV = {
    "rmse_mean": 0.8111, "rmse_lo": 0.7194, "rmse_hi": 0.9029,
    "mae_mean": 0.6176, "mae_lo": 0.5584, "mae_hi": 0.6768,
    "r2_mean": 0.5798, "r2_lo": 0.5067, "r2_hi": 0.6530,
}

# Step 4/5 XGBoost reference, for comparison in the report
XGB_NESTED_CV = {
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


def make_gpr_pipeline():
    kernel = (
        C(1.0, (1e-2, 1e2))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e1))
    )
    return make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=RANDOM_STATE),
    )


def main():
    sns.set_theme(style="whitegrid")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_PATH)
    class_df = pd.read_csv(DATASET_PATH)[["solvent_SMILES", "Classification"]]
    df = df.merge(class_df, on="solvent_SMILES", how="left")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X_df = df[feature_cols]
    X = X_df.values
    y = df[TARGET].values
    groups = assign_groups(df["solvent_SMILES"])
    strata = df["Classification"].astype(str).to_numpy()

    # --- Refit final GPR model on the FULL dataset ---
    final_model = make_gpr_pipeline()
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_PATH)

    y_pred_full, std_full = final_model.predict(X, return_std=True)
    rmse_full, mae_full, r2_full = evaluate(y, y_pred_full)
    print("=" * 60)
    print("GPR FINAL MODEL - FULL-DATASET FIT (diagnostic only, not generalization)")
    print("=" * 60)
    print(f"Full data: RMSE={rmse_full:.4f}, MAE={mae_full:.4f}, R2={r2_full:.4f}")
    print(f"Mean predictive std (full-data fit): {std_full.mean():.4f}")
    kernel_ = final_model.named_steps["gaussianprocessregressor"].kernel_
    print(f"Fitted kernel: {kernel_}")

    # --- OOF predictions via StratifiedGroupKFold(5) ---
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_model = make_gpr_pipeline()
    y_oof = cross_val_predict(oof_model, X, y, cv=list(sgkf.split(X, strata, groups)))
    rmse_oof, mae_oof, r2_oof = evaluate(y, y_oof)
    print(f"OOF (SGKF5):  RMSE={rmse_oof:.4f}, MAE={mae_oof:.4f}, R2={r2_oof:.4f}")

    pearson_full, _ = pearsonr(y, y_pred_full)
    pearson_oof, _ = pearsonr(y, y_oof)

    # --- Parity/residual plots with RMSE and Pearson r annotations ---
    stats_box = dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].scatter(y, y_pred_full, color="steelblue", edgecolor="black", alpha=0.7)
    lims_full = [min(y.min(), y_pred_full.min()), max(y.max(), y_pred_full.max())]
    axes[0, 0].plot(lims_full, lims_full, "k--", label="Ideal")
    axes[0, 0].set_xlabel("Actual G-score")
    axes[0, 0].set_ylabel("Predicted G-score")
    axes[0, 0].set_title(f"Parity Plot - Full-Data Fit (R2={r2_full:.3f}, diagnostic only)")
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].text(
        0.05, 0.95,
        f"RMSE: {rmse_full:.2f}\nPearson r: {pearson_full:.2f}",
        transform=axes[0, 0].transAxes, va="top", ha="left",
        fontsize=11, bbox=stats_box,
    )

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
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].text(
        0.05, 0.95,
        f"RMSE: {rmse_oof:.2f}\nPearson r: {pearson_oof:.2f}",
        transform=axes[1, 0].transAxes, va="top", ha="left",
        fontsize=11, bbox=stats_box,
    )

    residuals_oof = y - y_oof
    axes[1, 1].scatter(y_oof, residuals_oof, color="seagreen", edgecolor="black", alpha=0.7)
    axes[1, 1].axhline(0, color="k", linestyle="--")
    axes[1, 1].set_xlabel("OOF Predicted G-score")
    axes[1, 1].set_ylabel("Residual (Actual - Predicted)")
    axes[1, 1].set_title("Residual Plot - 5-Fold OOF (SGKF)")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "parity_residual_plots_with_stats.png", dpi=150)
    plt.close()
    print(f"\nSaved {RESULTS_DIR / 'parity_residual_plots_with_stats.png'}")

    # --- SHAP analysis (model-agnostic KernelExplainer, GPR has no TreeExplainer) ---
    print("\nComputing SHAP values (KernelExplainer, this may take a minute)...")
    background = shap.kmeans(X_df, 25)
    explainer = shap.KernelExplainer(final_model.predict, background)
    shap_values = explainer.shap_values(X_df, nsamples=100)

    plt.figure()
    shap.summary_plot(shap_values, X_df, show=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_order = np.argsort(mean_abs_shap)[::-1]
    shap_ranked_features = [feature_cols[i] for i in shap_order]
    shap_ranked_values = mean_abs_shap[shap_order]
    print(f"Saved SHAP plots to {RESULTS_DIR}")

    # --- Learning curve (StratifiedGroupKFold(5)) ---
    lc_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, val_scores = learning_curve(
        make_gpr_pipeline(), X, y,
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
    plt.title("Learning Curve - GPR Final Model (13 features, Matern1.5 + White)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curve.png", dpi=150)
    plt.close()
    print(f"Saved {RESULTS_DIR / 'learning_curve.png'}")

    # --- Report ---
    lines = []
    lines.append("# GPR Descriptor Model - Final Model Report (Step 7 Replacement)\n\n")
    lines.append(
        "Step 7 (`results/RDKit_GPR_Pruning_Report.md` in the parent `Final "
        "Model` folder) showed that a Gaussian Process Regressor (GPR, "
        "Matern(nu=1.5) + WhiteKernel, StandardScaler-normalized) on the same "
        "13-descriptor feature set as the XGBoost final model improves the "
        "nested-CV generalization estimate from R2=0.5050 [0.3342, 0.6757] "
        "(XGBoost, Steps 4/5) to R2=0.5798 [0.5067, 0.6530] (GPR), with a "
        "substantially tighter confidence interval. Every outer fold in the "
        "nested CV selected the Matern(1.5)+White kernel, so it is fixed as "
        "the final kernel here. The model was refit on the full 154-sample "
        "dataset for deployment/reporting, and all interpretability and "
        "diagnostic plots were regenerated against this final model. The "
        "Step 7 nested-CV scores remain the headline generalization metric "
        "for this model; the full-data fit and OOF metrics below are "
        "diagnostic only.\n\n"
    )
    lines.append(f"Feature columns ({len(feature_cols)}): {', '.join(feature_cols)}\n\n")
    lines.append(f"Final kernel (fitted): `{kernel_}`\n\n")
    lines.append("Pipeline: `StandardScaler -> GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=5)`\n\n")

    lines.append("## Headline Generalization Metric (Step 7, Nested CV)\n\n")
    lines.append("| Metric | Model | Mean | 95% CI Lower | 95% CI Upper |\n")
    lines.append("|---|---|---|---|---|\n")
    lines.append(f"| RMSE | GPR (final) | {NESTED_CV['rmse_mean']:.4f} | {NESTED_CV['rmse_lo']:.4f} | {NESTED_CV['rmse_hi']:.4f} |\n")
    lines.append(f"| MAE | GPR (final) | {NESTED_CV['mae_mean']:.4f} | {NESTED_CV['mae_lo']:.4f} | {NESTED_CV['mae_hi']:.4f} |\n")
    lines.append(f"| R2 | GPR (final) | {NESTED_CV['r2_mean']:.4f} | {NESTED_CV['r2_lo']:.4f} | {NESTED_CV['r2_hi']:.4f} |\n")
    lines.append(f"| RMSE | XGBoost (Step 4/5, prior model) | {XGB_NESTED_CV['rmse_mean']:.4f} | {XGB_NESTED_CV['rmse_lo']:.4f} | {XGB_NESTED_CV['rmse_hi']:.4f} |\n")
    lines.append(f"| MAE | XGBoost (Step 4/5, prior model) | {XGB_NESTED_CV['mae_mean']:.4f} | {XGB_NESTED_CV['mae_lo']:.4f} | {XGB_NESTED_CV['mae_hi']:.4f} |\n")
    lines.append(f"| R2 | XGBoost (Step 4/5, prior model) | {XGB_NESTED_CV['r2_mean']:.4f} | {XGB_NESTED_CV['r2_lo']:.4f} | {XGB_NESTED_CV['r2_hi']:.4f} |\n\n")

    lines.append("## Diagnostic Metrics (Final Model, Not Generalization Estimates)\n\n")
    lines.append("| Set | RMSE | MAE | R2 | Pearson r |\n")
    lines.append("|---|---|---|---|---|\n")
    lines.append(f"| Full-data fit (refit model) | {rmse_full:.4f} | {mae_full:.4f} | {r2_full:.4f} | {pearson_full:.4f} |\n")
    lines.append(f"| OOF (5-fold SGKF, final kernel) | {rmse_oof:.4f} | {mae_oof:.4f} | {r2_oof:.4f} | {pearson_oof:.4f} |\n\n")
    lines.append(f"Mean predictive standard deviation (full-data fit): {std_full.mean():.4f}\n\n")

    lines.append("## SHAP Ranking (Mean |SHAP value|, Final Model, KernelExplainer)\n\n")
    lines.append("| Rank | Descriptor | Mean |SHAP value| |\n")
    lines.append("|---|---|---|\n")
    for rank, (feat, val) in enumerate(zip(shap_ranked_features, shap_ranked_values), start=1):
        lines.append(f"| {rank} | {feat} | {val:.4f} |\n")
    lines.append("\n![SHAP Summary](GPR/shap_summary.png)\n\n")
    lines.append("![SHAP Importance Bar](GPR/shap_importance_bar.png)\n\n")

    lines.append("## Diagnostic Plots\n\n")
    lines.append("![Parity and Residual Plots](GPR/parity_residual_plots_with_stats.png)\n\n")
    lines.append("![Learning Curve](GPR/learning_curve.png)\n\n")

    lines.append("## Output Files\n\n")
    lines.append("- Final model (refit on all 154 samples): `model/gpr_model.pkl`\n")
    lines.append("- Parity/residual plot: `results/GPR/parity_residual_plots_with_stats.png`\n")
    lines.append("- SHAP plots: `results/GPR/shap_summary.png`, `results/GPR/shap_importance_bar.png`\n")
    lines.append("- Learning curve: `results/GPR/learning_curve.png`\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
