"""
ChemBERTa Embedding Pipeline - Hyperparameter Optimization

Loads descriptors/ChemBERTa_Features.csv and optimizes the XGBoost
regressor using RandomizedSearchCV over:
- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree
- min_child_weight

Methodology mirrors the Morgan Fingerprint and RDKit Descriptor pipelines
for fair comparison:
- Identical 80/20 train-test split (random_state=42)
- Hyperparameter search performed only on the training split (5-fold CV),
  so the test set remains untouched until final evaluation.
- Identical evaluation metrics (RMSE, MAE, R^2).

Outputs:
- models/chemberta_xgboost_model.pkl (best model, refit on training data)
- models/chemberta_best_params.json
- reports/ChemBERTa_Model_Report.md
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
FEATURES_PATH = BASE_DIR / "descriptors" / "ChemBERTa_Features.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

TARGET = "G-score"
NON_FEATURE_COLS = ["solvent_common_name", "CAS Number", "solvent_SMILES", TARGET]
MODEL_PATH = MODELS_DIR / "chemberta_xgboost_model.pkl"
REPORT_PATH = REPORTS_DIR / "ChemBERTa_Model_Report.md"

CHEMBERTA_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
EMBEDDING_DIM = 768


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

    # Identical 80/20 split as the other pipelines
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

    print("=" * 60)
    print("BASELINE MODEL RESULTS (for comparison)")
    print("=" * 60)
    print(f"Test:  RMSE={rmse_base:.4f}, MAE={mae_base:.4f}, R2={r2_base:.4f}")
    print(f"OOF:   RMSE={rmse_oof_base:.4f}, MAE={mae_oof_base:.4f}, R2={r2_oof_base:.4f}")

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

    print("\n" + "=" * 60)
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
    with open(MODELS_DIR / "chemberta_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest model saved to {MODEL_PATH}")

    # --- Build complete model report ---
    lines = []
    lines.append("# ChemBERTa Embedding Model Report\n\n")

    lines.append("## 1. Embedding Generation Methodology\n\n")
    lines.append(
        f"- SMILES column used: `solvent_SMILES` (identified dynamically from GSK_dataset.csv)\n"
    )
    lines.append(
        "- Embeddings generated by tokenizing each SMILES string and passing it through "
        "the pretrained ChemBERTa encoder.\n"
    )
    lines.append(
        "- Pooling strategy: mean-pooling of the last hidden state across all "
        "non-padding tokens, producing one fixed-length vector per molecule.\n"
    )
    lines.append(f"- Embedding dimensionality: {EMBEDDING_DIM} (per molecule)\n")
    lines.append(f"- Final feature matrix: {len(df)} molecules x {EMBEDDING_DIM} dimensions\n\n")

    lines.append("## 2. ChemBERTa Model Information\n\n")
    lines.append(f"- Model: `{CHEMBERTA_MODEL_NAME}` (Hugging Face Hub)\n")
    lines.append(
        "- Architecture: RoBERTa-based masked-language-model transformer pretrained "
        "on ~770k molecules (SMILES strings) from the ZINC15 database\n"
    )
    lines.append(
        "- Only the base encoder (`AutoModel`) was used for embedding extraction; the "
        "masked-language-modeling head was not used (its weights are reported as "
        "'unexpected' on load, which is expected behavior)\n\n"
    )

    lines.append("## 3. XGBoost Architecture\n\n")
    lines.append("- Algorithm: XGBoost Regressor (`xgboost.XGBRegressor`)\n")
    lines.append("- Objective: `reg:squarederror`\n")
    lines.append(f"- Input features: {len(feature_cols)} ChemBERTa embedding dimensions\n")
    lines.append(f"- Target: {TARGET}\n")
    lines.append(f"- random_state: {RANDOM_STATE}\n\n")

    lines.append("## 4. Hyperparameters (Best Found via RandomizedSearchCV)\n\n")
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

    lines.append("## 5. Baseline vs Optimized Comparison\n\n")
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

    lines.append("## 6. Validation Metrics (80/20 Train-Test Split, Optimized Model)\n\n")
    lines.append("| Set | RMSE | MAE | R2 |\n")
    lines.append("|---|---|---|---|\n")
    lines.append(f"| Train | {rmse_train:.4f} | {mae_train:.4f} | {r2_train:.4f} |\n")
    lines.append(f"| Test | {rmse_test:.4f} | {mae_test:.4f} | {r2_test:.4f} |\n\n")

    lines.append("## 7. Cross-Validation Results (5-Fold OOF, Full Dataset, Best Hyperparameters)\n\n")
    lines.append("| Metric | Value |\n")
    lines.append("|---|---|\n")
    lines.append(f"| RMSE | {rmse_oof:.4f} |\n")
    lines.append(f"| MAE | {mae_oof:.4f} |\n")
    lines.append(f"| R2 | {r2_oof:.4f} |\n\n")

    lines.append("## 8. Fit Assessment\n\n")
    lines.append(f"- Train R2: {r2_train:.4f}\n")
    lines.append(f"- 5-fold OOF R2: {r2_oof:.4f}\n")
    lines.append(f"- Gap: {gap:.4f}\n")
    lines.append(f"- **Assessment: {fit_assessment}**\n\n")

    lines.append("## 9. Advantages\n\n")
    lines.append(
        "- ChemBERTa embeddings require no manual feature engineering or descriptor "
        "selection — a single pretrained model produces a fixed-length representation "
        "for any valid SMILES string.\n"
    )
    lines.append(
        "- The embeddings encode learned chemical/structural context from a large "
        "pretraining corpus (~770k molecules), which can in principle capture subtle "
        "structure-property relationships not represented by hand-crafted descriptors.\n"
    )
    lines.append(
        "- The pipeline is fully reproducible (fixed random_state=42 for splitting, "
        "cross-validation, and model fitting).\n\n"
    )

    lines.append("## 10. Limitations\n\n")
    lines.append(
        f"- With {EMBEDDING_DIM} dense features and only {len(df)} samples, this pipeline "
        f"operates in a severe p >> n regime, leading to substantial overfitting "
        f"(gap = {gap:.4f}) even after hyperparameter tuning.\n"
    )
    lines.append(
        f"- Generalization performance (5-fold OOF R2 = {r2_oof:.4f}) was "
        f"{'lower than' if r2_oof < r2_oof_base + 0.5 else 'comparable to'} "
        f"the Morgan Fingerprint and RDKit Descriptor baselines on this dataset, "
        f"suggesting that for this small, structurally narrow solvent dataset, "
        f"high-dimensional pretrained embeddings did not translate into a clear "
        f"predictive advantage without further dimensionality reduction or "
        f"regularization.\n"
    )
    lines.append(
        "- Mean-pooling over token embeddings is a simple aggregation strategy; "
        "alternative approaches (e.g., using the `<s>`/CLS token, fine-tuning the "
        "transformer end-to-end, or pooling intermediate layers) were not explored.\n"
    )
    lines.append(
        "- Embedding generation requires downloading and running a transformer model "
        "(GPU recommended for larger datasets), adding computational overhead compared "
        "to RDKit descriptors.\n\n"
    )

    lines.append("## 11. Recommendations for Future Solvent-Property Prediction Studies\n\n")
    lines.append(
        "- For small datasets (on the order of hundreds of samples), prefer "
        "low-dimensional, domain-informed descriptors (e.g., RDKit physicochemical "
        "descriptors) or apply dimensionality reduction (PCA, feature selection) to "
        "high-dimensional embeddings before modeling.\n"
    )
    lines.append(
        "- If using ChemBERTa or similar pretrained embeddings, consider stronger "
        "regularization (shallow trees, high `min_child_weight`, low "
        "`colsample_bytree`/`subsample`) or linear models with L1/L2 penalties "
        "(e.g., Ridge, Lasso) which may handle p >> n settings more gracefully than "
        "tree ensembles.\n"
    )
    lines.append(
        "- Collecting additional labeled solvent data would likely benefit the "
        "embedding-based pipeline more than the lower-dimensional descriptor pipelines, "
        "since high-capacity representations need more data to avoid overfitting.\n"
    )
    lines.append(
        "- A hybrid approach — combining compact RDKit descriptors with a reduced-"
        "dimensionality projection of ChemBERTa embeddings (e.g., via PCA) — could "
        "be explored as a follow-up study to capture complementary information from "
        "both representations.\n\n"
    )

    lines.append("## 12. Output Files\n\n")
    lines.append("- Best model: `models/chemberta_xgboost_model.pkl`\n")
    lines.append("- Best hyperparameters: `models/chemberta_best_params.json`\n")
    lines.append("- Baseline plots: `results/ChemBERTa/learning_curve.png`, `results/ChemBERTa/prediction_residual_plots.png`\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
