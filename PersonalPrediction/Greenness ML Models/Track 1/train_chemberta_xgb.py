"""
train_chemberta_xgb.py
======================
Trains an XGBoost regression model using 768-dimensional ChemBERTa embeddings as features.
Applies standard scaling within cross-validation to prevent data leakage.
Generates 8 evaluation plots and saves results.
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).resolve().parent / "logs" / "train_chemberta_xgb.log", mode="w", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

# Path bootstrap
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from config import DIRS, SEED, XGB_PARAMS, COL_CLASS, COL_SMILES
from data_utils import load_dataset, make_split, assign_groups, get_cv_splitter
import plot_utils

# Seeds for stability analysis
STABILITY_SEEDS = [42, 123, 456, 789, 1337]


def run_pipeline():
    log.info("Starting ChemBERTa Embeddings + XGBoost pipeline...")
    
    # 1. Load dataset & features
    df = load_dataset()
    features_path = DIRS["descriptors"] / "chemberta_embeddings.npz"
    if not features_path.exists():
        raise FileNotFoundError(f"ChemBERTa features not found at {features_path}. Run generate_chemberta_features.py first.")
        
    data = np.load(features_path, allow_pickle=True)
    X = data["features"]
    y = data["targets"]
    log.info("Loaded ChemBERTa embeddings shape: %s", X.shape)
    
    # Define feature names
    feature_names = [f"dim_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # 2. Get global train/test split
    train_idx, test_idx = make_split(df)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    groups = assign_groups(df)
    groups_train = groups[train_idx]
    strata_train = df.iloc[train_idx][COL_CLASS].values
    
    # Preprocessing: StandardScaler
    preprocessor = StandardScaler()
    
    # 3. 5-Fold Stratified Group CV (for seed=SEED)
    sgkf = get_cv_splitter()
    
    oof_preds = np.zeros(len(train_idx))
    cv_rmse_scores = []
    cv_r2_scores = []
    cv_mae_scores = []
    
    for fold, (fold_train_idx, fold_val_idx) in enumerate(sgkf.split(X_train, strata_train, groups_train)):
        X_fold_train, y_fold_train = X_train[fold_train_idx], y_train[fold_train_idx]
        X_fold_val, y_fold_val = X_train[fold_val_idx], y_train[fold_val_idx]
        
        # Fit scaler on fold train, transform both
        X_fold_train_prep = preprocessor.fit_transform(X_fold_train)
        X_fold_val_prep = preprocessor.transform(X_fold_val)
        
        # Fit model
        fold_model = xgb.XGBRegressor(**XGB_PARAMS)
        fold_model.fit(X_fold_train_prep, y_fold_train)
        
        # Predict
        val_preds = fold_model.predict(X_fold_val_prep)
        oof_preds[fold_val_idx] = val_preds
        
        # Calculate fold metrics
        fold_rmse = np.sqrt(np.mean((y_fold_val - val_preds) ** 2))
        fold_r2 = 1.0 - (np.sum((y_fold_val - val_preds) ** 2) / np.sum((y_fold_val - y_fold_val.mean()) ** 2))
        fold_mae = np.mean(np.abs(y_fold_val - val_preds))
        
        cv_rmse_scores.append(float(fold_rmse))
        cv_r2_scores.append(float(fold_r2))
        cv_mae_scores.append(float(fold_mae))
        
        log.info("Fold %d: RMSE = %.3f, R² = %.3f, MAE = %.3f", fold + 1, fold_rmse, fold_r2, fold_mae)
        
    cv_metrics = {
        "rmse": cv_rmse_scores,
        "r2": cv_r2_scores,
        "mae": cv_mae_scores,
        "mean_rmse": float(np.mean(cv_rmse_scores)),
        "std_rmse": float(np.std(cv_rmse_scores)),
        "mean_r2": float(np.mean(cv_r2_scores)),
        "std_r2": float(np.std(cv_r2_scores)),
        "mean_mae": float(np.mean(cv_mae_scores)),
        "std_mae": float(np.std(cv_mae_scores))
    }
    
    log.info("CV Summary: Mean RMSE = %.3f, Mean R² = %.3f", cv_metrics["mean_rmse"], cv_metrics["mean_r2"])
    
    # Save CV metrics
    with open(DIRS["results"] / "chemberta" / "cv_metrics.json", "w") as f:
        json.dump(cv_metrics, f, indent=2)
        
    # Save OOF predictions
    oof_df = pd.DataFrame({
        "solvent_SMILES": df.iloc[train_idx][COL_SMILES],
        "Actual_G_Score": y_train,
        "Predicted_G_Score_OOF": oof_preds
    })
    oof_df.to_csv(DIRS["results"] / "chemberta" / "oof_predictions.csv", index=False)
    
    # 4. Fit final scaler and train on full training set
    log.info("Fitting final scaler and training on full training set...")
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    # Save scaler
    scaler_path = DIRS["models"] / "chemberta_scaler.joblib"
    joblib.dump(preprocessor, scaler_path)
    log.info("Saved final scaler to: %s", scaler_path)
    
    # Fit final model
    final_model = xgb.XGBRegressor(**XGB_PARAMS)
    final_model.fit(X_train_prep, y_train)
    
    # Save model
    model_path = DIRS["models"] / "chemberta_xgb.json"
    final_model.save_model(str(model_path))
    log.info("Saved final model to: %s", model_path)
    
    # 5. Evaluate on held-out test set
    train_preds = final_model.predict(X_train_prep)
    test_preds = final_model.predict(X_test_prep)
    
    test_rmse = float(np.sqrt(np.mean((y_test - test_preds) ** 2)))
    test_r2 = float(1.0 - (np.sum((y_test - test_preds) ** 2) / np.sum((y_test - y_test.mean()) ** 2)))
    test_mae = float(np.mean(np.abs(y_test - test_preds)))
    
    train_rmse = float(np.sqrt(np.mean((y_train - train_preds) ** 2)))
    train_r2 = float(1.0 - (np.sum((y_train - train_preds) ** 2) / np.sum((y_train - y_train.mean()) ** 2)))
    train_mae = float(np.mean(np.abs(y_train - train_preds)))
    
    metrics = {
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "train_mae": train_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "test_mae": test_mae
    }
    log.info("Test Set Results: RMSE = %.3f, R² = %.3f, MAE = %.3f", test_rmse, test_r2, test_mae)
    
    # Save final metrics
    with open(DIRS["results"] / "chemberta" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Save test predictions
    test_df = pd.DataFrame({
        "solvent_SMILES": df.iloc[test_idx][COL_SMILES],
        "Actual_G_Score": y_test,
        "Predicted_G_Score_Test": test_preds
    })
    test_df.to_csv(DIRS["results"] / "chemberta" / "test_predictions.csv", index=False)
    
    # 6. Stability Analysis (across 5 random seeds)
    log.info("Running stability analysis across seeds: %s", STABILITY_SEEDS)
    stability_data = []
    
    for seed in STABILITY_SEEDS:
        seed_params = XGB_PARAMS.copy()
        seed_params["random_state"] = seed
        
        seed_sgkf = get_cv_splitter()
        seed_sgkf.random_state = seed
        
        seed_cv_rmse = []
        seed_cv_r2 = []
        
        for fold_t_idx, fold_v_idx in seed_sgkf.split(X_train, strata_train, groups_train):
            X_f_train, y_f_train = X_train[fold_t_idx], y_train[fold_t_idx]
            X_f_val, y_f_val = X_train[fold_v_idx], y_train[fold_v_idx]
            
            seed_scaler = StandardScaler()
            X_f_train_prep = seed_scaler.fit_transform(X_f_train)
            X_f_val_prep = seed_scaler.transform(X_f_val)
            
            s_model = xgb.XGBRegressor(**seed_params)
            s_model.fit(X_f_train_prep, y_f_train)
            
            f_preds = s_model.predict(X_f_val_prep)
            seed_cv_rmse.append(np.sqrt(np.mean((y_f_val - f_preds) ** 2)))
            seed_cv_r2.append(1.0 - (np.sum((y_f_val - f_preds) ** 2) / np.sum((y_f_val - y_f_val.mean()) ** 2)))
            
        final_seed_scaler = StandardScaler()
        X_train_s_prep = final_seed_scaler.fit_transform(X_train)
        X_test_s_prep = final_seed_scaler.transform(X_test)
        
        final_s_model = xgb.XGBRegressor(**seed_params)
        final_s_model.fit(X_train_s_prep, y_train)
        s_test_preds = final_s_model.predict(X_test_s_prep)
        s_test_rmse = np.sqrt(np.mean((y_test - s_test_preds) ** 2))
        s_test_r2 = 1.0 - (np.sum((y_test - s_test_preds) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
        
        stability_data.append({
            "seed": seed,
            "cv_rmse_mean": float(np.mean(seed_cv_rmse)),
            "cv_r2_mean": float(np.mean(seed_cv_r2)),
            "test_rmse": float(s_test_rmse),
            "test_r2": float(s_test_r2)
        })
        
    with open(DIRS["results"] / "chemberta" / "stability_metrics.json", "w") as f:
        json.dump(stability_data, f, indent=2)
        
    # 7. Generate Plots
    log.info("Generating evaluation plots...")
    fig_dir = DIRS["figures"] / "chemberta"
    
    # 7.1 Parity Plot
    plot_utils.plot_parity(y_train, train_preds, y_test, test_preds, fig_dir / "parity.png")
    
    # 7.2 Residuals Plot
    plot_utils.plot_residuals(y_train, train_preds, y_test, test_preds, fig_dir / "residuals.png")
    
    # 7.3 Feature Importance (Gain-based)
    plot_utils.plot_feature_importance_xgb(final_model, feature_names, fig_dir / "feature_importance.png", max_features=20)
    
    # 7.4 SHAP Summary Plot
    log.info("Calculating SHAP values...")
    explainer = shap.TreeExplainer(final_model)
    shap_vals = explainer.shap_values(X_train_prep)
    X_train_prep_df = pd.DataFrame(X_train_prep, columns=feature_names)
    plot_utils.plot_shap_importance(shap_vals, X_train_prep_df, fig_dir / "shap_importance.png", max_display=20)
    
    # 7.5 Out-of-Fold Predictions
    plot_utils.plot_oof(y_train, oof_preds, fig_dir / "oof_parity.png")
    
    # 7.6 CV Fold Scores
    plot_utils.plot_cv_fold_scores(cv_rmse_scores, "RMSE", fig_dir / "cv_fold_scores.png")
    
    # 7.7 Learning Curves
    log.info("Calculating learning curves...")
    lc_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', xgb.XGBRegressor(**XGB_PARAMS))
    ])
    cv_splits = list(sgkf.split(X_train, strata_train, groups_train))
    plot_utils.plot_learning_curves(lc_pipeline, X_train, y_train, cv_splits, fig_dir / "learning_curves.png")
    
    # 7.8 Stability Plot
    stability_rmse = [s["test_rmse"] for s in stability_data]
    stability_r2 = [s["test_r2"] for s in stability_data]
    plot_utils.plot_stability({"RMSE": stability_rmse, "R²": stability_r2}, fig_dir / "stability.png")
    
    log.info("Pipeline completed successfully! All ChemBERTa outputs saved.")


if __name__ == "__main__":
    run_pipeline()
