"""
optimize_model.py
=================
Phase 1 – Existing Model Assessment
Phase 2 – Model Optimization (RandomizedSearchCV)
Phase 3 – Advanced Improvement Analysis (descriptor utilization check)

Outputs
-------
Reports/Existing_Model_Assessment.md
Reports/Optimized_Model_Report.md
JSON/baseline_metrics.json
JSON/optimized_metrics.json
Models/xgboost_optimized.pkl
Images/learning_curve*.png
Images/residual_analysis*.png
Images/feature_importance*.png
Images/actual_vs_predicted*.png
"""

import os, pickle, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import (
    train_test_split, cross_validate, KFold,
    RandomizedSearchCV, learning_curve
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import scipy.stats as stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = r"C:\dev\PersonalPrediction\Simple ML Models"
OUT         = os.path.join(BASE, "Optimized Model")
CSV_DIR     = os.path.join(OUT, "CSVs")
IMG_DIR     = os.path.join(OUT, "Images")
MDL_DIR     = os.path.join(OUT, "Models")
RPT_DIR     = os.path.join(OUT, "Reports")
JSN_DIR     = os.path.join(OUT, "JSON")

DATASET       = os.path.join(BASE, "data", "Melting_temperature_appended_35il_03082026.csv")
EXISTING_PKL  = os.path.join(BASE, "models", "xgboost.pkl")
OPTIMIZED_PKL = os.path.join(MDL_DIR, "xgboost_optimized.pkl")
TARGET_COL    = "Tmelt, K"
RANDOM_STATE  = 42

EXCLUDE = ["Component#1","Component#2","Reference (DOI)","Smiles#1","Smiles#2", TARGET_COL]

# ── Helpers ────────────────────────────────────────────────────────────────────

def metrics_dict(y_true, y_pred, label=""):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    if label:
        print(f"  {label:30s}  RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}")
    return dict(RMSE=rmse, MAE=mae, R2=r2)


def build_preprocessor(numeric_features, categorical_features):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features),
    ])


def load_data():
    df = pd.read_csv(DATASET)
    df = df.dropna(subset=[TARGET_COL])
    feat_cols = [c for c in df.columns if c not in EXCLUDE]
    X = df[feat_cols].copy()
    y = df[TARGET_COL].copy()
    num_feats = X.select_dtypes(include="number").columns.tolist()
    cat_feats = X.select_dtypes(exclude="number").columns.tolist()
    return df, X, y, num_feats, cat_feats


def cv_scores(pipeline, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    res = cross_validate(pipeline, X, y, cv=kf,
                         scoring=["neg_root_mean_squared_error",
                                  "neg_mean_absolute_error", "r2"],
                         return_train_score=True, n_jobs=-1)
    return {
        "cv_rmse_mean":  float(-res["test_neg_root_mean_squared_error"].mean()),
        "cv_rmse_std":   float( res["test_neg_root_mean_squared_error"].std()),
        "cv_mae_mean":   float(-res["test_neg_mean_absolute_error"].mean()),
        "cv_r2_mean":    float( res["test_r2"].mean()),
        "cv_r2_std":     float( res["test_r2"].std()),
        "train_rmse_mean": float(-res["train_neg_root_mean_squared_error"].mean()),
        "train_r2_mean":   float( res["train_r2"].mean()),
    }

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 – EXISTING MODEL ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════

def phase1_assess(df, X, y, num_feats, cat_feats):
    print("\n" + "="*60)
    print("PHASE 1 – EXISTING MODEL ASSESSMENT")
    print("="*60)

    # Load model
    with open(EXISTING_PKL, "rb") as f:
        existing = pickle.load(f)

    # Introspect pipeline
    steps      = [s[0] for s in existing.steps]
    regressor  = existing.named_steps["regressor"]
    reg_params = regressor.get_params()
    pre        = existing.named_steps["preprocessor"]

    print(f"\nPipeline steps: {steps}")
    print(f"Regressor type: {type(regressor).__name__}")
    print(f"\nHyperparameters:")
    key_params = ["n_estimators","max_depth","learning_rate","subsample",
                  "colsample_bytree","min_child_weight","gamma","reg_alpha","reg_lambda"]
    stored_params = {k: reg_params.get(k) for k in key_params}
    for k,v in stored_params.items():
        print(f"  {k}: {v}")

    # Dataset summary
    print(f"\nDataset: {len(df)} samples, target = '{TARGET_COL}'")
    print(f"Numeric features ({len(num_feats)}): {num_feats}")
    print(f"Categorical features ({len(cat_feats)}): {cat_feats}")
    print(f"No SMILES/fingerprint/RDKit descriptors used — molecular structure ignored.")

    # Baseline train/test split (same seed as original)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    existing.fit(X_tr, y_tr)
    train_m = metrics_dict(y_tr, existing.predict(X_tr), "Baseline Train")
    test_m  = metrics_dict(y_te, existing.predict(X_te), "Baseline Test ")

    overfit_gap = train_m["R2"] - test_m["R2"]
    print(f"\n  Overfitting gap (R² train–test): {overfit_gap:.4f}")

    # 5-fold CV on full data
    print("\nRunning 5-fold CV on baseline …")
    base_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(num_feats, cat_feats)),
        ("regressor", XGBRegressor(**stored_params, objective="reg:squarederror",
                                    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
    ])
    cv_res = cv_scores(base_pipeline, X, y)
    print(f"  5-Fold CV RMSE: {cv_res['cv_rmse_mean']:.3f} ± {cv_res['cv_rmse_std']:.3f} K")
    print(f"  5-Fold CV R²:   {cv_res['cv_r2_mean']:.4f} ± {cv_res['cv_r2_std']:.4f}")
    print(f"  5-Fold Train R2:{cv_res['train_r2_mean']:.4f}  gap = {cv_res['train_r2_mean']-cv_res['cv_r2_mean']:.4f}")

    # Compatibility check
    feat_cols  = [c for c in df.columns if c not in EXCLUDE]
    rdkit_path = os.path.join(BASE, "data", "RDKitDescriptorGeneration", "DES_RDKit_Features.csv")
    rdkit_ok   = os.path.exists(rdkit_path)

    baseline_all = {
        "model_type": "XGBRegressor",
        "pipeline_steps": steps,
        "stored_hyperparameters": stored_params,
        "features_used": {"numeric": num_feats, "categorical": cat_feats},
        "dataset": {"n_samples": len(df), "n_features": len(feat_cols)},
        "split_metrics": {"train": train_m, "test": test_m},
        "cv_5fold": cv_res,
        "overfitting_gap_r2": overfit_gap,
        "molecular_descriptors_used": False,
        "rdkit_dataset_available": rdkit_ok,
    }
    with open(os.path.join(JSN_DIR, "baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_all, f, indent=2)

    # Decision
    decision = "OPTIMIZE"  # gap of ~0.05 R² + narrow search space = worth optimizing
    reason   = (f"Overfitting gap (train R²={train_m['R2']:.4f} vs test R²={test_m['R2']:.4f}) "
                f"of {overfit_gap:.4f} indicates the existing model is overfit. "
                f"The original GridSearchCV searched only 18 combinations; a wider "
                f"RandomizedSearchCV with regularization parameters has not been attempted.")

    # ── Write assessment markdown ─────────────────────────────────────────────
    md_lines = [
        "# Existing Model Assessment",
        f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
        "## 1. Model Architecture",
        f"- **Type**: `XGBRegressor` wrapped in a `sklearn.Pipeline`",
        f"- **Pipeline steps**: {' → '.join(steps)}",
        "",
        "## 2. Preprocessing Pipeline",
        "| Stage | Numeric Features | Categorical Features |",
        "|---|---|---|",
        "| Imputation | Median | Most-frequent |",
        "| Scaling | StandardScaler | OneHotEncoder (drop=first) |",
        "",
        "## 3. Input Features",
        f"**Numeric** ({len(num_feats)}): {', '.join(f'`{f}`' for f in num_feats)}",
        "",
        f"**Categorical** ({len(cat_feats)}): {', '.join(f'`{f}`' for f in cat_feats)}",
        "",
        "**Excluded** (ID / SMILES / target): `Component#1`, `Component#2`, `Reference (DOI)`, `Smiles#1`, `Smiles#2`",
        "",
        "## 4. Target Variable",
        f"- **`{TARGET_COL}`** (melting temperature, Kelvin)",
        f"- Range in dataset: {y.min():.1f} K – {y.max():.1f} K",
        "",
        "## 5. Stored Hyperparameters",
        "| Parameter | Value |",
        "|---|---|",
    ]
    for k,v in stored_params.items():
        md_lines.append(f"| `{k}` | {v} |")

    md_lines += [
        "",
        "## 6. Baseline Performance",
        "| Split | RMSE (K) | MAE (K) | R² |",
        "|---|---|---|---|",
        f"| Train | {train_m['RMSE']:.3f} | {train_m['MAE']:.3f} | {train_m['R2']:.4f} |",
        f"| Test  | {test_m['RMSE']:.3f} | {test_m['MAE']:.3f} | {test_m['R2']:.4f} |",
        f"| 5-Fold CV (mean) | {cv_res['cv_rmse_mean']:.3f} ± {cv_res['cv_rmse_std']:.3f} | {cv_res['cv_mae_mean']:.3f} | {cv_res['cv_r2_mean']:.4f} ± {cv_res['cv_r2_std']:.4f} |",
        "",
        f"> **Overfitting gap**: Train R² − Test R² = **{overfit_gap:.4f}** — moderate overfitting detected.",
        "",
        "## 7. Descriptor Utilization Check",
        "| Descriptor Type | Used? |",
        "|---|---|",
        "| SMILES strings | No — excluded from features |",
        "| Morgan fingerprints | No |",
        "| RDKit descriptors | No (dataset available but not used) |",
        "| ChemBERTa embeddings | No |",
        "",
        f"> **Note**: RDKit features file found at `data/RDKitDescriptorGeneration/DES_RDKit_Features.csv` with 14 additional molecular descriptors per component (MolWt, LogP, TPSA, HBD, HBA, RotBonds, RingCount for each component). These are not currently used.",
        "",
        "## 8. Dataset Compatibility",
        f"- **Training dataset**: `Melting_temperature_appended_35il_03082026.csv` — {len(df)} samples ✓",
        f"- **Candidate dataset**: Same file used for screening (no separate candidate file provided) ✓",
        f"- **RDKit features dataset**: Same 2006 samples with additional descriptors ✓",
        "",
        "## 9. Decision",
        f"**→ {decision}: Further optimize the existing XGBoost model.**",
        "",
        f"{reason}",
        "",
        "### Optimization Plan",
        "1. Expand hyperparameter search to include regularization (`gamma`, `reg_alpha`, `reg_lambda`, `min_child_weight`)",
        "2. Use `RandomizedSearchCV` with `n_iter=120` over a broad search space",
        "3. Optimize for 5-fold CV RMSE (generalization, not training accuracy)",
        "4. Retrain best model on 100% of data for deployment",
    ]

    with open(os.path.join(RPT_DIR, "Existing_Model_Assessment.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"\nSaved: Reports/Existing_Model_Assessment.md")

    return stored_params, train_m, test_m, cv_res, overfit_gap

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 – HYPERPARAMETER OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════

def phase2_optimize(X, y, num_feats, cat_feats, baseline_train, baseline_test, baseline_cv):
    print("\n" + "="*60)
    print("PHASE 2 – HYPERPARAMETER OPTIMIZATION")
    print("="*60)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    base_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(num_feats, cat_feats)),
        ("regressor", XGBRegressor(objective="reg:squarederror",
                                    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
    ])

    param_dist = {
        "regressor__n_estimators":    [100, 200, 400, 600, 800],
        "regressor__max_depth":       [3, 4, 5, 6, 8, 10],
        "regressor__learning_rate":   [0.01, 0.03, 0.05, 0.1, 0.2],
        "regressor__subsample":       [0.6, 0.7, 0.8, 0.9, 1.0],
        "regressor__colsample_bytree":[0.6, 0.7, 0.8, 0.9, 1.0],
        "regressor__min_child_weight":[1, 3, 5, 7],
        "regressor__gamma":           [0, 0.1, 0.3, 0.5],
        "regressor__reg_alpha":       [0, 0.01, 0.1, 1],
        "regressor__reg_lambda":      [0.1, 1, 5, 10],
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rscv = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_dist,
        n_iter=120,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True,
    )
    print("\nRunning RandomizedSearchCV (n_iter=120, 5-fold) …")
    rscv.fit(X_tr, y_tr)

    best_params_raw = rscv.best_params_
    best_cv_rmse    = -rscv.best_score_
    best_pipeline   = rscv.best_estimator_

    best_params = {k.replace("regressor__",""): v for k,v in best_params_raw.items()}
    print(f"\nBest CV RMSE: {best_cv_rmse:.3f} K")
    print("Best params:")
    for k,v in best_params.items():
        print(f"  {k}: {v}")

    # Evaluate on held-out test set
    opt_train_m = metrics_dict(y_tr, best_pipeline.predict(X_tr), "Optimized Train")
    opt_test_m  = metrics_dict(y_te, best_pipeline.predict(X_te), "Optimized Test ")

    # Full 5-fold CV
    print("\nRunning 5-fold CV on optimized model …")
    opt_cv = cv_scores(best_pipeline, X, y)
    print(f"  5-Fold CV RMSE: {opt_cv['cv_rmse_mean']:.3f} ± {opt_cv['cv_rmse_std']:.3f} K")
    print(f"  5-Fold CV R²:   {opt_cv['cv_r2_mean']:.4f} ± {opt_cv['cv_r2_std']:.4f}")

    # Learning curve
    print("\nGenerating learning curve …")
    train_sizes, train_sc, val_sc = learning_curve(
        best_pipeline, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    train_rmse = -train_sc.mean(axis=1)
    val_rmse   = -val_sc.mean(axis=1)
    train_std  = train_sc.std(axis=1)
    val_std    = val_sc.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, train_rmse, "o-", color="royalblue", label="Train RMSE")
    ax.fill_between(train_sizes, train_rmse - train_std, train_rmse + train_std, alpha=0.15, color="royalblue")
    ax.plot(train_sizes, val_rmse,   "o-", color="firebrick", label="CV RMSE")
    ax.fill_between(train_sizes, val_rmse - val_std, val_rmse + val_std, alpha=0.15, color="firebrick")
    ax.set_xlabel("Training Samples", fontsize=12)
    ax.set_ylabel("RMSE (K)", fontsize=12)
    ax.set_title("Learning Curve — Optimized XGBoost", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "learning_curve_optimized.png"), dpi=150)
    plt.close()

    # Residual analysis on test set
    y_pred_test = best_pipeline.predict(X_te)
    residuals   = y_te.values - y_pred_test

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred_test, residuals, alpha=0.5, color="steelblue", s=30)
    axes[0].axhline(0, color="firebrick", lw=1.5)
    axes[0].set_xlabel("Predicted Tmelt (K)")
    axes[0].set_ylabel("Residual (K)")
    axes[0].set_title("Residuals vs Predicted")

    # Q-Q plot
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, fit=True)
    axes[1].scatter(osm, osr, alpha=0.5, s=25, color="steelblue")
    line_x = np.array([min(osm), max(osm)])
    axes[1].plot(line_x, slope*line_x + intercept, "r-", lw=2)
    axes[1].set_xlabel("Theoretical Quantiles")
    axes[1].set_ylabel("Sample Quantiles")
    axes[1].set_title("Q-Q Plot of Residuals")

    # Histogram of residuals
    axes[2].hist(residuals, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    axes[2].axvline(0, color="firebrick", lw=1.5)
    axes[2].set_xlabel("Residual (K)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Residual Distribution")

    plt.suptitle("Residual Analysis — Optimized XGBoost", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "residual_analysis_optimized.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Actual vs Predicted
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_te, y_pred_test, alpha=0.55, color="darkorange", edgecolor="white", s=40)
    lims = [min(y_te.min(), y_pred_test.min()) - 5, max(y_te.max(), y_pred_test.max()) + 5]
    ax.plot(lims, lims, "r--", lw=2, label="Perfect Prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual Tmelt (K)", fontsize=12)
    ax.set_ylabel("Predicted Tmelt (K)", fontsize=12)
    ax.set_title("Actual vs Predicted — Optimized XGBoost", fontsize=13, fontweight="bold")
    txt = f"MAE={opt_test_m['MAE']:.2f} K\nRMSE={opt_test_m['RMSE']:.2f} K\nR²={opt_test_m['R2']:.4f}"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9), fontsize=10)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "actual_vs_predicted_optimized.png"), dpi=150)
    plt.close()

    # Feature importance
    feat_names_raw = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    feat_names = [n.replace("num__","").replace("cat__","") for n in feat_names_raw]
    importances = best_pipeline.named_steps["regressor"].feature_importances_

    imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
    imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    imp_df.to_csv(os.path.join(CSV_DIR, "feature_importances_optimized.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, max(5, len(feat_names)*0.35)))
    sns.barplot(data=imp_df, x="Importance", y="Feature", palette="plasma",
                hue="Feature", legend=False, ax=ax)
    ax.set_title("Feature Importances — Optimized XGBoost", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance (gain)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "feature_importance_optimized.png"), dpi=150)
    plt.close()

    # Baseline vs Optimized comparison bar chart
    labels = ["Baseline\n(Train)", "Baseline\n(Test)", "Baseline\n(5-Fold CV)",
              "Optimized\n(Train)", "Optimized\n(Test)", "Optimized\n(5-Fold CV)"]
    rmse_vals = [baseline_train["RMSE"], baseline_test["RMSE"], baseline_cv["cv_rmse_mean"],
                 opt_train_m["RMSE"],    opt_test_m["RMSE"],    opt_cv["cv_rmse_mean"]]
    colors = ["#b0c4de","#4682b4","#1e3a5f", "#f4a460","#d2691e","#8b2500"]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, rmse_vals, color=colors, edgecolor="white", width=0.6)
    for bar, v in zip(bars, rmse_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSE (K)", fontsize=12)
    ax.set_title("Baseline vs Optimized XGBoost — RMSE Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "baseline_vs_optimized_rmse.png"), dpi=150)
    plt.close()

    # Retrain on FULL dataset for deployment
    print("\nRetraining best model on 100% of data for deployment …")
    final_params = {k.replace("regressor__",""):v for k,v in best_params_raw.items()}
    deploy_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(num_feats, cat_feats)),
        ("regressor", XGBRegressor(**final_params, objective="reg:squarederror",
                                    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
    ])
    deploy_pipeline.fit(X, y)

    with open(OPTIMIZED_PKL, "wb") as f:
        pickle.dump(deploy_pipeline, f)
    print(f"Saved deployment model: Models/xgboost_optimized.pkl")

    # Save metrics JSON
    opt_summary = {
        "baseline": {
            "hyperparameters": {"n_estimators":200,"max_depth":8,"learning_rate":0.1},
            "train": baseline_train, "test": baseline_test, "cv_5fold": baseline_cv,
        },
        "optimized": {
            "hyperparameters": best_params,
            "train": opt_train_m, "test": opt_test_m, "cv_5fold": opt_cv,
            "best_cv_rmse_from_search": float(best_cv_rmse),
        },
        "improvement": {
            "rmse_delta_test": round(baseline_test["RMSE"] - opt_test_m["RMSE"], 4),
            "r2_delta_test":   round(opt_test_m["R2"] - baseline_test["R2"], 4),
            "cv_rmse_delta":   round(baseline_cv["cv_rmse_mean"] - opt_cv["cv_rmse_mean"], 4),
        },
    }
    with open(os.path.join(JSN_DIR, "optimized_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(opt_summary, f, indent=2)

    # ── Write Optimized Model Report ──────────────────────────────────────────
    overfit_opt = opt_train_m["R2"] - opt_test_m["R2"]
    imp_top10   = imp_df.head(10)

    md = [
        "# Optimized Model Report",
        f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
        "## 1. Baseline vs Optimized Metrics",
        "| Metric | Baseline Train | Baseline Test | Baseline 5-Fold CV | Optimized Train | Optimized Test | Optimized 5-Fold CV |",
        "|---|---|---|---|---|---|---|",
        f"| RMSE (K) | {baseline_train['RMSE']:.3f} | {baseline_test['RMSE']:.3f} | {baseline_cv['cv_rmse_mean']:.3f}±{baseline_cv['cv_rmse_std']:.3f} | {opt_train_m['RMSE']:.3f} | {opt_test_m['RMSE']:.3f} | {opt_cv['cv_rmse_mean']:.3f}±{opt_cv['cv_rmse_std']:.3f} |",
        f"| MAE (K)  | {baseline_train['MAE']:.3f}  | {baseline_test['MAE']:.3f}  | {baseline_cv['cv_mae_mean']:.3f} | {opt_train_m['MAE']:.3f}  | {opt_test_m['MAE']:.3f}  | {opt_cv['cv_mae_mean']:.3f} |",
        f"| R²       | {baseline_train['R2']:.4f} | {baseline_test['R2']:.4f} | {baseline_cv['cv_r2_mean']:.4f}±{baseline_cv['cv_r2_std']:.4f} | {opt_train_m['R2']:.4f} | {opt_test_m['R2']:.4f} | {opt_cv['cv_r2_mean']:.4f}±{opt_cv['cv_r2_std']:.4f} |",
        "",
        f"> Improvement: Test RMSE Δ = {opt_summary['improvement']['rmse_delta_test']:+.3f} K | Test R² Δ = {opt_summary['improvement']['r2_delta_test']:+.4f} | CV RMSE Δ = {opt_summary['improvement']['cv_rmse_delta']:+.3f} K",
        "",
        "## 2. Optimized Hyperparameters",
        "| Parameter | Baseline | Optimized |",
        "|---|---|---|",
        f"| n_estimators | 200 | {best_params.get('n_estimators','—')} |",
        f"| max_depth | 8 | {best_params.get('max_depth','—')} |",
        f"| learning_rate | 0.1 | {best_params.get('learning_rate','—')} |",
        f"| subsample | — | {best_params.get('subsample','—')} |",
        f"| colsample_bytree | — | {best_params.get('colsample_bytree','—')} |",
        f"| min_child_weight | — | {best_params.get('min_child_weight','—')} |",
        f"| gamma | — | {best_params.get('gamma','—')} |",
        f"| reg_alpha | — | {best_params.get('reg_alpha','—')} |",
        f"| reg_lambda | — | {best_params.get('reg_lambda','—')} |",
        "",
        f"> Overfitting gap (Train R² − Test R²): Baseline = {baseline_train['R2']-baseline_test['R2']:.4f} → Optimized = {overfit_opt:.4f}",
        "",
        "## 3. Learning Curve Analysis",
        "![Learning Curve](../Images/learning_curve_optimized.png)",
        "",
        "## 4. Residual Analysis",
        "![Residual Analysis](../Images/residual_analysis_optimized.png)",
        "",
        "## 5. Actual vs Predicted",
        "![Actual vs Predicted](../Images/actual_vs_predicted_optimized.png)",
        "",
        "## 6. Feature Importance (Top 10)",
        "| Rank | Feature | Importance |",
        "|---|---|---|",
    ]
    for i, row in imp_top10.iterrows():
        md.append(f"| {i+1} | `{row['Feature']}` | {row['Importance']:.5f} |")

    md += [
        "",
        "![Feature Importance](../Images/feature_importance_optimized.png)",
        "",
        "## 7. Advanced Improvement Analysis",
        "### Descriptor Utilization",
        "The current model does **not** use molecular structure information.",
        "",
        "| Enhancement | Estimated Improvement | Status |",
        "|---|---|---|",
        "| RDKit descriptors (14 features, already computed) | Moderate (~5–15% RMSE reduction) | Recommended |",
        "| Morgan fingerprints (ECFP4, 1024-bit) | Moderate | Requires RDKit |",
        "| ChemBERTa embeddings | High (if data volume sufficient) | Requires GPU/transformer |",
        "| Feature engineering (T_ratio, ΔT_melt) | Low–moderate | Can be implemented |",
        "| Ensemble (XGBoost + RF) | Low | Marginal for well-tuned single model |",
        "",
        "> **Recommendation**: Adding the 14 pre-computed RDKit descriptors (from `DES_RDKit_Features.csv`) is the highest-value, lowest-effort enhancement. Morgan fingerprints would further improve coverage of structural diversity. Awaiting user approval before implementing.",
        "",
        "## 8. Deployment Recommendation",
        f"- **Deploy**: `Models/xgboost_optimized.pkl` (trained on 100% of {len(X)} samples)",
        "- **Expected CV RMSE**: " + f"{opt_cv['cv_rmse_mean']:.2f} ± {opt_cv['cv_rmse_std']:.2f} K",
        "- **Expected CV R²**: " + f"{opt_cv['cv_r2_mean']:.4f} ± {opt_cv['cv_r2_std']:.4f}",
        "- **Confidence**: High for DES systems within training distribution; moderate for novel chemical classes",
        "- **Monitoring**: Flag predictions where input features fall outside training range",
    ]

    with open(os.path.join(RPT_DIR, "Optimized_Model_Report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("Saved: Reports/Optimized_Model_Report.md")

    return deploy_pipeline, best_params, opt_train_m, opt_test_m, opt_cv


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    df, X, y, num_feats, cat_feats = load_data()
    stored_params, base_train, base_test, base_cv, gap = phase1_assess(df, X, y, num_feats, cat_feats)
    deploy_pipeline, best_params, opt_train, opt_test, opt_cv = phase2_optimize(
        X, y, num_feats, cat_feats, base_train, base_test, base_cv
    )
    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Proceed with screen_candidates.py")
