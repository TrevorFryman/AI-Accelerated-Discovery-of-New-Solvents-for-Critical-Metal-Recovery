"""
RDKit Descriptor Pipeline - Baseline XGBoost Model

Loads descriptors/RDKit_Features.csv and trains a baseline XGBoost
regression model to predict G-score from RDKit molecular descriptors.

Steps:
1. 80/20 train-test split (random_state=42) -> RMSE, MAE, R^2 on test set
2. 5-fold cross-validation with out-of-fold (OOF) predictions
3. Learning curve analysis (train vs CV score across training sizes)
4. Underfitting / overfitting assessment
5. Parity plots and residual plots

This pipeline is standalone (RDKit descriptors only) and uses an
identical train/test split methodology (80/20, random_state=42) and
evaluation metrics (RMSE, MAE, R^2) to allow fair comparison against
the Morgan Fingerprint and ChemBERTa pipelines.

Outputs are saved to results/RDKit/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_predict, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "descriptors" / "RDKit_Features.csv"
RESULTS_DIR = BASE_DIR / "results" / "RDKit"
REPORTS_DIR = BASE_DIR / "reports"

TARGET = "G-score"
NON_FEATURE_COLS = ["solvent_common_name", "CAS Number", "solvent_SMILES", TARGET]


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def main():
    sns.set_theme(style="whitegrid")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_PATH)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].values
    y = df[TARGET].values

    # 1. 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    rmse_test, mae_test, r2_test = evaluate(y_test, y_pred_test)

    y_pred_train = model.predict(X_train)
    rmse_train, mae_train, r2_train = evaluate(y_train, y_pred_train)

    print("=" * 60)
    print("80/20 TRAIN-TEST SPLIT RESULTS")
    print("=" * 60)
    print(f"Train: RMSE={rmse_train:.4f}, MAE={mae_train:.4f}, R2={r2_train:.4f}")
    print(f"Test:  RMSE={rmse_test:.4f}, MAE={mae_test:.4f}, R2={r2_test:.4f}")

    # 2. 5-fold cross-validation with out-of-fold predictions (on full dataset)
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    y_oof = cross_val_predict(cv_model, X, y, cv=kf)
    rmse_oof, mae_oof, r2_oof = evaluate(y, y_oof)

    print("\n" + "=" * 60)
    print("5-FOLD CROSS-VALIDATION (OUT-OF-FOLD) RESULTS")
    print("=" * 60)
    print(f"OOF: RMSE={rmse_oof:.4f}, MAE={mae_oof:.4f}, R2={r2_oof:.4f}")

    # 3. Learning curve analysis
    lc_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    train_sizes, train_scores, val_scores = learning_curve(
        lc_model, X, y,
        cv=kf,
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
    plt.title("Learning Curve - RDKit Descriptor XGBoost Baseline")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curve.png", dpi=150)
    plt.close()

    # 4. Underfitting / overfitting assessment
    final_train_r2 = train_scores_mean[-1]
    final_val_r2 = val_scores_mean[-1]
    gap = final_train_r2 - final_val_r2

    if final_train_r2 < 0.5 and final_val_r2 < 0.5:
        fit_assessment = "Underfitting"
        fit_explanation = (
            f"Both the training R2 ({final_train_r2:.3f}) and cross-validation R2 "
            f"({final_val_r2:.3f}) are low, suggesting the model is not capturing "
            f"enough signal from the RDKit descriptor features."
        )
    elif gap > 0.3:
        fit_assessment = "Overfitting"
        fit_explanation = (
            f"The training R2 ({final_train_r2:.3f}) is substantially higher than the "
            f"cross-validation R2 ({final_val_r2:.3f}), a gap of {gap:.3f}, indicating "
            f"the model fits the training data much better than unseen data."
        )
    else:
        fit_assessment = "Appropriately fit"
        fit_explanation = (
            f"The training R2 ({final_train_r2:.3f}) and cross-validation R2 "
            f"({final_val_r2:.3f}) are reasonably close (gap = {gap:.3f}), suggesting "
            f"the model generalizes reasonably well given the available data."
        )

    print("\n" + "=" * 60)
    print("MODEL FIT ASSESSMENT")
    print("=" * 60)
    print(f"Assessment: {fit_assessment}")
    print(fit_explanation)

    # 5. Parity plots and residual plots (test set + OOF)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Parity plot - test set
    axes[0, 0].scatter(y_test, y_pred_test, color="steelblue", edgecolor="black", alpha=0.7)
    lims = [min(y.min(), y_pred_test.min()), max(y.max(), y_pred_test.max())]
    axes[0, 0].plot(lims, lims, "k--", label="Ideal")
    axes[0, 0].set_xlabel("Actual G-score")
    axes[0, 0].set_ylabel("Predicted G-score")
    axes[0, 0].set_title(f"Parity Plot - Test Set (R2={r2_test:.3f})")
    axes[0, 0].legend()

    # Residual plot - test set
    residuals_test = y_test - y_pred_test
    axes[0, 1].scatter(y_pred_test, residuals_test, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0, 1].axhline(0, color="k", linestyle="--")
    axes[0, 1].set_xlabel("Predicted G-score")
    axes[0, 1].set_ylabel("Residual (Actual - Predicted)")
    axes[0, 1].set_title("Residual Plot - Test Set")

    # Parity plot - OOF
    axes[1, 0].scatter(y, y_oof, color="seagreen", edgecolor="black", alpha=0.7)
    lims_oof = [min(y.min(), y_oof.min()), max(y.max(), y_oof.max())]
    axes[1, 0].plot(lims_oof, lims_oof, "k--", label="Ideal")
    axes[1, 0].set_xlabel("Actual G-score")
    axes[1, 0].set_ylabel("OOF Predicted G-score")
    axes[1, 0].set_title(f"Parity Plot - 5-Fold OOF (R2={r2_oof:.3f})")
    axes[1, 0].legend()

    # Residual plot - OOF
    residuals_oof = y - y_oof
    axes[1, 1].scatter(y_oof, residuals_oof, color="seagreen", edgecolor="black", alpha=0.7)
    axes[1, 1].axhline(0, color="k", linestyle="--")
    axes[1, 1].set_xlabel("OOF Predicted G-score")
    axes[1, 1].set_ylabel("Residual (Actual - Predicted)")
    axes[1, 1].set_title("Residual Plot - 5-Fold OOF")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "parity_residual_plots.png", dpi=150)
    plt.close()

    # Save summary report
    report_lines = []
    report_lines.append("# RDKit Descriptor Baseline Model - Results\n\n")
    report_lines.append("## Model\n\n")
    report_lines.append("- Algorithm: XGBoost Regressor (default hyperparameters)\n")
    report_lines.append(f"- random_state: {RANDOM_STATE}\n")
    report_lines.append(f"- Features: {len(feature_cols)} RDKit descriptors ({', '.join(feature_cols)})\n")
    report_lines.append(f"- Target: {TARGET}\n\n")

    report_lines.append("## 80/20 Train-Test Split\n\n")
    report_lines.append("| Set | RMSE | MAE | R2 |\n")
    report_lines.append("|---|---|---|---|\n")
    report_lines.append(f"| Train | {rmse_train:.4f} | {mae_train:.4f} | {r2_train:.4f} |\n")
    report_lines.append(f"| Test | {rmse_test:.4f} | {mae_test:.4f} | {r2_test:.4f} |\n\n")

    report_lines.append("## 5-Fold Cross-Validation (Out-of-Fold)\n\n")
    report_lines.append("| Metric | Value |\n")
    report_lines.append("|---|---|\n")
    report_lines.append(f"| RMSE | {rmse_oof:.4f} |\n")
    report_lines.append(f"| MAE | {mae_oof:.4f} |\n")
    report_lines.append(f"| R2 | {r2_oof:.4f} |\n\n")

    report_lines.append("## Learning Curve / Fit Assessment\n\n")
    report_lines.append(f"- Final training R2: {final_train_r2:.4f}\n")
    report_lines.append(f"- Final cross-validation R2: {final_val_r2:.4f}\n")
    report_lines.append(f"- Gap: {gap:.4f}\n")
    report_lines.append(f"- **Assessment: {fit_assessment}**\n\n")
    report_lines.append(f"{fit_explanation}\n\n")

    report_lines.append("## Plots\n\n")
    report_lines.append("- `results/RDKit/learning_curve.png`\n")
    report_lines.append("- `results/RDKit/parity_residual_plots.png`\n")

    (REPORTS_DIR / "RDKit_Baseline_Results.md").write_text("".join(report_lines), encoding="utf-8")

    print(f"\nResults saved to {RESULTS_DIR}")
    print(f"Report saved to {REPORTS_DIR / 'RDKit_Baseline_Results.md'}")


if __name__ == "__main__":
    main()
