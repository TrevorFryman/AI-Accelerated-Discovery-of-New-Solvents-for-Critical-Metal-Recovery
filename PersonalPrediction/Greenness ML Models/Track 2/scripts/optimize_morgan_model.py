"""
Morgan Fingerprint Pipeline - Hyperparameter Optimization

Loads descriptors/Morgan_Features.csv and optimizes the XGBoost regressor
using RandomizedSearchCV over:
- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree
- min_child_weight

Methodology mirrors train_morgan_baseline.py for fair comparison against
other descriptor pipelines:
- Identical 80/20 train-test split (random_state=42)
- Hyperparameter search performed only on the training split (5-fold CV),
  so the test set remains untouched until final evaluation.
- Identical evaluation metrics (RMSE, MAE, R^2).

Outputs:
- models/morgan_xgboost_model.pkl (best model, refit on training data)
- reports/Morgan_Model_Report.md (complete model report)
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "descriptors" / "Morgan_Features.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

TARGET = "G-score"
NON_FEATURE_COLS = ["solvent_common_name", "CAS Number", "solvent_SMILES", TARGET]
MODEL_PATH = MODELS_DIR / "morgan_xgboost_model.pkl"
REPORT_PATH = REPORTS_DIR / "Morgan_Model_Report.md"


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].values
    y = df[TARGET].values

    # Identical 80/20 split as the baseline
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Hyperparameter search space
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

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=100,
        scoring="neg_root_mean_squared_error",
        cv=cv,
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

    # Evaluate on held-out test set
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
    y_oof = cross_val_predict(oof_model, X, y, cv=cv)
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

    # Save best model (refit on training split, as evaluated above)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nBest model saved to {MODEL_PATH}")

    # Build complete model report
    lines = []
    lines.append("# Morgan Fingerprint Model Report\n\n")

    lines.append("## 1. Model Architecture\n\n")
    lines.append("- Algorithm: XGBoost Regressor (`xgboost.XGBRegressor`)\n")
    lines.append("- Objective: `reg:squarederror`\n")
    lines.append(f"- Input features: {len(feature_cols)} Morgan fingerprint bits (radius=2, nBits=2048)\n")
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

    lines.append("## 3. Validation Metrics (80/20 Train-Test Split)\n\n")
    lines.append("| Set | RMSE | MAE | R2 |\n")
    lines.append("|---|---|---|---|\n")
    lines.append(f"| Train | {rmse_train:.4f} | {mae_train:.4f} | {r2_train:.4f} |\n")
    lines.append(f"| Test | {rmse_test:.4f} | {mae_test:.4f} | {r2_test:.4f} |\n\n")

    lines.append("## 4. Cross-Validation Results (5-Fold OOF, Full Dataset, Best Hyperparameters)\n\n")
    lines.append("| Metric | Value |\n")
    lines.append("|---|---|\n")
    lines.append(f"| RMSE | {rmse_oof:.4f} |\n")
    lines.append(f"| MAE | {mae_oof:.4f} |\n")
    lines.append(f"| R2 | {r2_oof:.4f} |\n\n")

    lines.append("## 5. Fit Assessment\n\n")
    lines.append(f"- Train R2: {r2_train:.4f}\n")
    lines.append(f"- 5-fold OOF R2: {r2_oof:.4f}\n")
    lines.append(f"- Gap: {gap:.4f}\n")
    lines.append(f"- **Assessment: {fit_assessment}**\n\n")

    lines.append("## 6. Comparison to Baseline (Default Hyperparameters)\n\n")
    lines.append("| Model | Test RMSE | Test MAE | Test R2 | OOF RMSE | OOF MAE | OOF R2 |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    lines.append("| Baseline (default params) | 0.9207 | 0.7488 | 0.5294 | 0.9830 | 0.7533 | 0.4047 |\n")
    lines.append(
        f"| Optimized (RandomizedSearchCV) | {rmse_test:.4f} | {mae_test:.4f} | {r2_test:.4f} "
        f"| {rmse_oof:.4f} | {mae_oof:.4f} | {r2_oof:.4f} |\n\n"
    )

    lines.append("## 7. Strengths\n\n")
    strengths = []
    if r2_oof > 0.4047:
        strengths.append(
            f"Hyperparameter tuning improved 5-fold OOF R2 from 0.4047 (baseline) to "
            f"{r2_oof:.4f}, indicating better generalization than the default-parameter model."
        )
    strengths.append(
        "Morgan fingerprints require no manual descriptor engineering and capture "
        "local substructure information directly from SMILES."
    )
    strengths.append(
        "The pipeline is fully reproducible (fixed random_state=42 for splitting, "
        "cross-validation, and model fitting)."
    )
    for s in strengths:
        lines.append(f"- {s}\n")

    lines.append("\n## 8. Weaknesses\n\n")
    weaknesses = []
    if gap > 0.2:
        weaknesses.append(
            f"A train/CV R2 gap of {gap:.4f} remains, suggesting some residual overfitting "
            f"despite tuning — likely driven by the high-dimensional (2048-bit), sparse "
            f"feature space relative to the small sample size (154 molecules)."
        )
    weaknesses.append(
        "Many of the 2048 fingerprint bits are constant (zero) across the dataset and "
        "contribute no information, increasing the effective dimensionality without benefit."
    )
    weaknesses.append(
        "The dataset is small (154 samples), so both the test-set metrics and the "
        "RandomizedSearchCV results carry meaningful variance; results should be "
        "interpreted with that uncertainty in mind."
    )
    for w in weaknesses:
        lines.append(f"- {w}\n")

    lines.append("\n## 9. Output Files\n\n")
    lines.append(f"- Best model: `models/morgan_xgboost_model.pkl`\n")
    lines.append(f"- Baseline plots: `results/Morgan/learning_curve.png`, `results/Morgan/parity_residual_plots.png`\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")

    # Also save best params as JSON for downstream comparison
    with open(MODELS_DIR / "morgan_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
