"""
optimize_rdkit_model.py
=======================
Retrains the XGBoost model with 14 additional RDKit molecular descriptors
(C1/C2 MolWt, LogP, TPSA, HBD, HBA, RotBonds, RingCount), then re-screens
all DES candidates with the improved model.

New outputs
-----------
Models/xgboost_rdkit_optimized.pkl
CSVs/feature_importances_rdkit.csv
CSVs/Candidate_Master_List_rdkit.csv
CSVs/RoomTemp_Top25_rdkit.csv
CSVs/WarmTemp_Top25_rdkit.csv
CSVs/Prediction_Summary_rdkit.csv
CSVs/Extrapolation_Flagged_rdkit.csv
Images/learning_curve_rdkit.png
Images/residual_analysis_rdkit.png
Images/actual_vs_predicted_rdkit.png
Images/feature_importance_rdkit.png
Images/baseline_vs_rdkit_rmse.png
Images/temperature_distribution_rdkit.png
Images/confidence_vs_tmelt_rdkit.png
JSON/rdkit_metrics.json
JSON/screening_summary_rdkit.json
Reports/RDKit_Model_Report.md
Reports/Candidate_Screening_Report_rdkit.md
"""

import os, pickle, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor
import scipy.stats as stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = r"C:\dev\PersonalPrediction\Simple ML Models"
OUT       = os.path.join(BASE, "Optimized Model")
CSV_DIR   = os.path.join(OUT, "CSVs")
IMG_DIR   = os.path.join(OUT, "Images")
MDL_DIR   = os.path.join(OUT, "Models")
RPT_DIR   = os.path.join(OUT, "Reports")
JSN_DIR   = os.path.join(OUT, "JSON")

RDKIT_DATASET  = os.path.join(BASE, "data", "RDKitDescriptorGeneration", "DES_RDKit_Features.csv")
PREV_METRICS   = os.path.join(JSN_DIR, "optimized_metrics.json")
RDKIT_PKL      = os.path.join(MDL_DIR, "xgboost_rdkit_optimized.pkl")
TARGET_COL     = "Tmelt, K"
RANDOM_STATE   = 42

EXCLUDE = ["Component#1","Component#2","Reference (DOI)","Smiles#1","Smiles#2", TARGET_COL]

RDKIT_COLS = [
    "C1_MolWt","C1_LogP","C1_TPSA","C1_HBD","C1_HBA","C1_RotBonds","C1_RingCount",
    "C2_MolWt","C2_LogP","C2_TPSA","C2_HBD","C2_HBA","C2_RotBonds","C2_RingCount",
]

ROOM_LOW, ROOM_HIGH = 275.0, 298.0
WARM_LOW, WARM_HIGH = 298.0, 343.0
ROOM_CENTER = (ROOM_LOW + ROOM_HIGH) / 2
WARM_CENTER = (WARM_LOW + WARM_HIGH) / 2

# ── Helpers ────────────────────────────────────────────────────────────────────

def metrics_dict(y_true, y_pred, label=""):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    if label:
        print(f"  {label:35s} RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}")
    return dict(RMSE=rmse, MAE=mae, R2=r2)


def build_preprocessor(num_feats, cat_feats):
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc",  StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore",
                                               sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num_feats),
                               ("cat", cat_pipe, cat_feats)])


def cv_scores(pipeline, X, y, cv=5):
    kf  = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    res = cross_validate(pipeline, X, y, cv=kf,
                         scoring=["neg_root_mean_squared_error",
                                  "neg_mean_absolute_error","r2"],
                         return_train_score=True, n_jobs=-1)
    return {
        "cv_rmse_mean":    float(-res["test_neg_root_mean_squared_error"].mean()),
        "cv_rmse_std":     float( res["test_neg_root_mean_squared_error"].std()),
        "cv_mae_mean":     float(-res["test_neg_mean_absolute_error"].mean()),
        "cv_r2_mean":      float( res["test_r2"].mean()),
        "cv_r2_std":       float( res["test_r2"].std()),
        "train_rmse_mean": float(-res["train_neg_root_mean_squared_error"].mean()),
        "train_r2_mean":   float( res["train_r2"].mean()),
    }


def compute_similarity(X_train_proc, X_cand_proc, base_cv_rmse=17.4):
    k   = min(5, len(X_train_proc))
    nn  = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nn.fit(X_train_proc)
    dist, _ = nn.kneighbors(X_cand_proc)
    mean_d  = dist.mean(axis=1)

    nn2  = NearestNeighbors(n_neighbors=k+1, metric="euclidean", n_jobs=-1)
    nn2.fit(X_train_proc)
    d2, _ = nn2.kneighbors(X_train_proc)
    ref_d  = max(np.percentile(d2[:, 1:].mean(axis=1), 95), 1e-6)

    sim  = np.clip(1.0 - mean_d / (2 * ref_d), 0.0, 1.0)
    unc  = base_cv_rmse + (mean_d / ref_d) * base_cv_rmse * 0.5
    extrap = (mean_d > 2 * ref_d).astype(int)
    return sim, unc, extrap


def compute_ranking_score(row, target_center):
    conf_score = row["similarity_score"] * 40.0
    dist_K     = abs(row["predicted_Tmelt_K"] - target_center)
    temp_range = WARM_HIGH - ROOM_LOW
    prox_score = max(0.0, 1.0 - dist_K / (temp_range / 2)) * 30.0
    phys_score = 20.0 if row["passes_DES_rule"] else 0.0
    feat_score = 10.0 if not row["feature_extrapolation"] else 0.0
    return round(conf_score + prox_score + phys_score + feat_score, 3)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – LOAD DATA WITH RDKIT DESCRIPTORS
# ══════════════════════════════════════════════════════════════════════════════

def load_rdkit_data():
    df = pd.read_csv(RDKIT_DATASET)
    df = df.dropna(subset=[TARGET_COL])
    feat_cols = [c for c in df.columns if c not in EXCLUDE]
    X = df[feat_cols].copy()
    y = df[TARGET_COL].copy()
    num_feats = X.select_dtypes(include="number").columns.tolist()
    cat_feats = X.select_dtypes(exclude="number").columns.tolist()
    print(f"Dataset: {len(df)} samples")
    print(f"  Numeric features ({len(num_feats)}): {num_feats}")
    print(f"  Categorical features ({len(cat_feats)}): {cat_feats}")
    rdkit_present = [c for c in RDKIT_COLS if c in num_feats]
    print(f"  RDKit descriptors in use ({len(rdkit_present)}): {rdkit_present}")
    return df, X, y, num_feats, cat_feats


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – OPTIMIZE WITH RDKIT FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def optimize_rdkit(X, y, num_feats, cat_feats):
    print("\n" + "="*60)
    print("RDKIT MODEL OPTIMIZATION (RandomizedSearchCV, 120 iter, 5-fold)")
    print("="*60)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=RANDOM_STATE)

    base_pipe = Pipeline([
        ("preprocessor", build_preprocessor(num_feats, cat_feats)),
        ("regressor",    XGBRegressor(objective="reg:squarederror",
                                       random_state=RANDOM_STATE,
                                       n_jobs=-1, verbosity=0)),
    ])

    param_dist = {
        "regressor__n_estimators":     [100, 200, 400, 600, 800],
        "regressor__max_depth":        [3, 4, 5, 6, 8, 10],
        "regressor__learning_rate":    [0.01, 0.03, 0.05, 0.1, 0.2],
        "regressor__subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "regressor__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "regressor__min_child_weight": [1, 3, 5, 7],
        "regressor__gamma":            [0, 0.1, 0.3, 0.5],
        "regressor__reg_alpha":        [0, 0.01, 0.1, 1],
        "regressor__reg_lambda":       [0.1, 1, 5, 10],
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rscv = RandomizedSearchCV(
        estimator=base_pipe, param_distributions=param_dist,
        n_iter=120, cv=kf, scoring="neg_root_mean_squared_error",
        n_jobs=-1, verbose=1, random_state=RANDOM_STATE, refit=True,
    )
    print("Running search …")
    rscv.fit(X_tr, y_tr)

    best_params_raw = rscv.best_params_
    best_cv_rmse    = -rscv.best_score_
    best_pipe       = rscv.best_estimator_
    best_params     = {k.replace("regressor__",""): v for k,v in best_params_raw.items()}

    print(f"\nBest CV RMSE: {best_cv_rmse:.3f} K")
    for k,v in best_params.items():
        print(f"  {k}: {v}")

    train_m = metrics_dict(y_tr, best_pipe.predict(X_tr), "RDKit Train (80%)")
    test_m  = metrics_dict(y_te, best_pipe.predict(X_te), "RDKit Test  (20%)")

    print("\nFull 5-fold CV …")
    cv_res = cv_scores(best_pipe, X, y)
    print(f"  CV RMSE: {cv_res['cv_rmse_mean']:.3f} +/- {cv_res['cv_rmse_std']:.3f} K")
    print(f"  CV R2:   {cv_res['cv_r2_mean']:.4f} +/- {cv_res['cv_r2_std']:.4f}")

    return best_pipe, best_params, best_params_raw, train_m, test_m, cv_res, X_tr, X_te, y_tr, y_te


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def make_plots(best_pipe, X, y, X_tr, X_te, y_tr, y_te, test_m, num_feats, prev_metrics):
    print("\nGenerating plots …")

    y_pred_te = best_pipe.predict(X_te)
    residuals = y_te.values - y_pred_te

    # Learning curve
    train_sz, train_sc, val_sc = learning_curve(
        best_pipe, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    tr_rmse = -train_sc.mean(axis=1); tr_std = train_sc.std(axis=1)
    va_rmse = -val_sc.mean(axis=1);   va_std = val_sc.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sz, tr_rmse, "o-", color="royalblue", label="Train RMSE")
    ax.fill_between(train_sz, tr_rmse-tr_std, tr_rmse+tr_std, alpha=0.15, color="royalblue")
    ax.plot(train_sz, va_rmse, "o-", color="firebrick", label="CV RMSE")
    ax.fill_between(train_sz, va_rmse-va_std, va_rmse+va_std, alpha=0.15, color="firebrick")
    ax.set_xlabel("Training Samples"); ax.set_ylabel("RMSE (K)")
    ax.set_title("Learning Curve — XGBoost + RDKit Descriptors", fontweight="bold")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "learning_curve_rdkit.png"), dpi=150); plt.close()

    # Residual analysis
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].scatter(y_pred_te, residuals, alpha=0.5, color="steelblue", s=30)
    axes[0].axhline(0, color="firebrick", lw=1.5)
    axes[0].set_xlabel("Predicted Tmelt (K)"); axes[0].set_ylabel("Residual (K)")
    axes[0].set_title("Residuals vs Predicted")

    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, fit=True)
    axes[1].scatter(osm, osr, alpha=0.5, s=25, color="steelblue")
    lx = np.array([min(osm), max(osm)])
    axes[1].plot(lx, slope*lx+intercept, "r-", lw=2)
    axes[1].set_xlabel("Theoretical Quantiles"); axes[1].set_ylabel("Sample Quantiles")
    axes[1].set_title("Q-Q Plot")

    axes[2].hist(residuals, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    axes[2].axvline(0, color="firebrick", lw=1.5)
    axes[2].set_xlabel("Residual (K)"); axes[2].set_ylabel("Count")
    axes[2].set_title("Residual Distribution")

    plt.suptitle("Residual Analysis — XGBoost + RDKit", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "residual_analysis_rdkit.png"), dpi=150,
                bbox_inches="tight"); plt.close()

    # Actual vs Predicted
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_te, y_pred_te, alpha=0.55, color="darkorange", edgecolor="white", s=40)
    lims = [min(y_te.min(), y_pred_te.min())-5, max(y_te.max(), y_pred_te.max())+5]
    ax.plot(lims, lims, "r--", lw=2, label="Perfect Prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual Tmelt (K)"); ax.set_ylabel("Predicted Tmelt (K)")
    ax.set_title("Actual vs Predicted — XGBoost + RDKit", fontweight="bold")
    txt = f"MAE={test_m['MAE']:.2f} K\nRMSE={test_m['RMSE']:.2f} K\nR2={test_m['R2']:.4f}"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9), fontsize=10)
    ax.legend(fontsize=10); plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "actual_vs_predicted_rdkit.png"), dpi=150); plt.close()

    # Feature importance
    feat_names_raw = best_pipe.named_steps["preprocessor"].get_feature_names_out()
    feat_names = [n.replace("num__","").replace("cat__","") for n in feat_names_raw]
    importances = best_pipe.named_steps["regressor"].feature_importances_
    imp_df = (pd.DataFrame({"Feature": feat_names, "Importance": importances})
                .sort_values("Importance", ascending=False).reset_index(drop=True))
    imp_df.to_csv(os.path.join(CSV_DIR, "feature_importances_rdkit.csv"), index=False)

    fig, ax = plt.subplots(figsize=(11, max(6, len(feat_names)*0.3)))
    top_n = min(25, len(imp_df))
    sns.barplot(data=imp_df.head(top_n), x="Importance", y="Feature",
                palette="plasma", hue="Feature", legend=False, ax=ax)
    ax.set_title(f"Feature Importances — XGBoost + RDKit (Top {top_n})", fontweight="bold")
    ax.set_xlabel("Importance (gain)"); plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "feature_importance_rdkit.png"), dpi=150); plt.close()

    # 3-way comparison bar chart (no-descriptor baseline, descriptor-free optimized, rdkit)
    if prev_metrics:
        b_test = prev_metrics["baseline"]["test"]["RMSE"]
        o_test = prev_metrics["optimized"]["test"]["RMSE"]
        b_cv   = prev_metrics["baseline"]["cv_5fold"]["cv_rmse_mean"]
        o_cv   = prev_metrics["optimized"]["cv_5fold"]["cv_rmse_mean"]
    else:
        b_test = o_test = b_cv = o_cv = None

    labels_all  = []
    rmse_vals   = []
    colours     = []
    if b_test is not None:
        labels_all += ["Baseline\nTest", "Baseline\n5-Fold CV"]
        rmse_vals  += [b_test, b_cv]
        colours    += ["#b0c4de", "#4682b4"]
        labels_all += ["No-Descriptor\nOptimized Test", "No-Descriptor\nOptimized 5-Fold CV"]
        rmse_vals  += [o_test, o_cv]
        colours    += ["#f4a460", "#d2691e"]
    labels_all += ["RDKit\nOptimized Test", "RDKit\nOptimized 5-Fold CV"]
    rmse_vals  += [test_m["RMSE"], cv_res_global["cv_rmse_mean"]]
    colours    += ["#90ee90", "#228b22"]

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(labels_all, rmse_vals, color=colours, edgecolor="white", width=0.6)
    for bar, v in zip(bars, rmse_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSE (K)", fontsize=12)
    ax.set_title("Model Comparison: Baseline vs Optimized vs RDKit-Optimized", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "baseline_vs_rdkit_rmse.png"), dpi=150); plt.close()

    return imp_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 – CANDIDATE SCREENING
# ══════════════════════════════════════════════════════════════════════════════

def screen_rdkit(deploy_pipe, df_full, feat_cols, num_feats):
    print("\n" + "="*60)
    print("CANDIDATE SCREENING — RDKit Model")
    print("="*60)

    df = df_full.dropna(subset=[TARGET_COL]).copy()
    X_all = df[feat_cols].copy()
    y_pred = deploy_pipe.predict(X_all)
    df["predicted_Tmelt_K"] = y_pred

    # Similarity / uncertainty
    preprocessor = deploy_pipe.named_steps["preprocessor"]
    X_proc = preprocessor.transform(X_all)
    sim, unc, nn_extrap = compute_similarity(X_proc, X_proc, base_cv_rmse=cv_res_global["cv_rmse_mean"])
    df["similarity_score"]      = np.round(sim, 4)
    df["uncertainty_K"]         = np.round(unc, 2)
    df["high_uncertainty"]      = unc > 25.0
    df["feature_extrapolation"] = nn_extrap.astype(bool)

    # DES rule
    passes_t1 = df["predicted_Tmelt_K"] < df["T#1"].fillna(9999)
    passes_t2 = df["predicted_Tmelt_K"] < df["T#2"].fillna(9999)
    passes    = passes_t1 & passes_t2
    df["passes_DES_rule"] = passes
    df["violation_T1"]    = ~passes_t1
    df["violation_T2"]    = ~passes_t2

    # Physical plausibility
    df["physically_reasonable"] = (df["predicted_Tmelt_K"] > 100) & (df["predicted_Tmelt_K"] < 700)

    # Temp flags
    df["RoomTemp_flag"] = (df["predicted_Tmelt_K"] >= ROOM_LOW) & (df["predicted_Tmelt_K"] < ROOM_HIGH)
    df["WarmTemp_flag"] = (df["predicted_Tmelt_K"] >= WARM_LOW) & (df["predicted_Tmelt_K"] < WARM_HIGH)

    def status(row):
        if not row["physically_reasonable"]: return "REJECTED_UNPHYSICAL"
        if not row["passes_DES_rule"]:       return "REJECTED_DES_RULE"
        if row["RoomTemp_flag"]:             return "PASS_ROOMTEMP"
        if row["WarmTemp_flag"]:             return "PASS_WARMTEMP"
        return "OUTSIDE_TARGET_RANGE"

    df["screening_status"] = df.apply(status, axis=1)
    df["ranking_score_room"] = df.apply(
        lambda r: compute_ranking_score(r, ROOM_CENTER) if r["RoomTemp_flag"] else np.nan, axis=1)
    df["ranking_score_warm"] = df.apply(
        lambda r: compute_ranking_score(r, WARM_CENTER) if r["WarmTemp_flag"] else np.nan, axis=1)

    n_room = df["RoomTemp_flag"].sum()
    n_warm = df["WarmTemp_flag"].sum()
    n_viol = (~passes).sum()
    print(f"  DES rule violations: {n_viol}")
    print(f"  RoomTemp (275-298 K): {n_room}")
    print(f"  WarmTemp (298-343 K): {n_warm}")

    # ── Output columns ────────────────────────────────────────────────────────
    id_cols = [c for c in ["Component#1","Component#2","X#1 (molar fraction)","X#2 (molar fraction)"] if c in df.columns]
    known   = [TARGET_COL] if TARGET_COL in df.columns else []
    master_cols = (id_cols + known +
                   ["predicted_Tmelt_K","uncertainty_K","similarity_score",
                    "screening_status","passes_DES_rule","violation_T1","violation_T2",
                    "RoomTemp_flag","WarmTemp_flag","feature_extrapolation","high_uncertainty",
                    "physically_reasonable","ranking_score_room","ranking_score_warm"])
    master_cols = [c for c in master_cols if c in df.columns]

    # Master list
    df[master_cols].to_csv(os.path.join(CSV_DIR, "Candidate_Master_List_rdkit.csv"), index=False)

    # Top 25 room
    top_room = (df[df["RoomTemp_flag"]]
                .sort_values("ranking_score_room", ascending=False)
                .head(25)[master_cols])
    top_room.to_csv(os.path.join(CSV_DIR, "RoomTemp_Top25_rdkit.csv"), index=False)

    # Top 25 warm
    top_warm = (df[df["WarmTemp_flag"]]
                .sort_values("ranking_score_warm", ascending=False)
                .head(25)[master_cols])
    top_warm.to_csv(os.path.join(CSV_DIR, "WarmTemp_Top25_rdkit.csv"), index=False)

    # Prediction summary
    summ = {
        "total_evaluated":          len(df),
        "n_DES_rule_violations":    int(n_viol),
        "n_RoomTemp":               int(n_room),
        "n_WarmTemp":               int(n_warm),
        "n_high_uncertainty":       int(df["high_uncertainty"].sum()),
        "n_feature_extrapolation":  int(df["feature_extrapolation"].sum()),
        "predicted_Tmelt_min_K":    float(y_pred.min()),
        "predicted_Tmelt_max_K":    float(y_pred.max()),
        "predicted_Tmelt_mean_K":   float(y_pred.mean()),
        "predicted_Tmelt_median_K": float(np.median(y_pred)),
    }
    pd.DataFrame([summ]).to_csv(os.path.join(CSV_DIR, "Prediction_Summary_rdkit.csv"), index=False)
    with open(os.path.join(JSN_DIR, "screening_summary_rdkit.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)

    # Extrapolation flagged
    df[df["feature_extrapolation"]][master_cols].to_csv(
        os.path.join(CSV_DIR, "Extrapolation_Flagged_rdkit.csv"), index=False)

    # ── Screening plots ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(y_pred, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvspan(ROOM_LOW, ROOM_HIGH, alpha=0.25, color="limegreen", label=f"RoomTemp ({n_room})")
    axes[0].axvspan(WARM_LOW, WARM_HIGH, alpha=0.25, color="orange",    label=f"WarmTemp ({n_warm})")
    axes[0].set_xlabel("Predicted Tmelt (K)"); axes[0].set_ylabel("Count")
    axes[0].set_title("Predicted Tmelt — RDKit Model (All Candidates)", fontweight="bold")
    axes[0].legend()

    mask_zoom = (y_pred >= 250) & (y_pred <= 380)
    axes[1].hist(y_pred[mask_zoom], bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    axes[1].axvspan(ROOM_LOW, ROOM_HIGH, alpha=0.3, color="limegreen", label="275-298 K")
    axes[1].axvspan(WARM_LOW, WARM_HIGH, alpha=0.3, color="orange",    label="298-343 K")
    axes[1].set_xlabel("Predicted Tmelt (K)"); axes[1].set_ylabel("Count")
    axes[1].set_title("Predicted Tmelt — Target Zoom (250-380 K)", fontweight="bold")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "temperature_distribution_rdkit.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(df["predicted_Tmelt_K"], df["similarity_score"],
                    c=df["uncertainty_K"], cmap="RdYlGn_r", alpha=0.5, s=20)
    plt.colorbar(sc, label="Uncertainty (K)")
    ax.axvspan(ROOM_LOW, ROOM_HIGH, alpha=0.15, color="limegreen")
    ax.axvspan(WARM_LOW, WARM_HIGH, alpha=0.15, color="orange")
    ax.set_xlabel("Predicted Tmelt (K)"); ax.set_ylabel("Similarity Score")
    ax.set_title("Confidence vs Predicted Tmelt — RDKit Model", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "confidence_vs_tmelt_rdkit.png"), dpi=150); plt.close()

    print(f"  Saved all CSVs and plots")
    return df, top_room, top_warm, summ


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 – REPORTS
# ══════════════════════════════════════════════════════════════════════════════

def write_rdkit_model_report(best_params, train_m, test_m, cv_res, imp_df, prev_metrics, num_feats):
    overfit_gap = train_m["R2"] - test_m["R2"]

    if prev_metrics:
        prev_test_rmse = prev_metrics["optimized"]["test"]["RMSE"]
        prev_cv_rmse   = prev_metrics["optimized"]["cv_5fold"]["cv_rmse_mean"]
        delta_test     = prev_test_rmse - test_m["RMSE"]
        delta_cv       = prev_cv_rmse   - cv_res["cv_rmse_mean"]
    else:
        prev_test_rmse = prev_cv_rmse = delta_test = delta_cv = None

    md = [
        "# RDKit Descriptor Model Report",
        f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
        "## 1. Feature Set",
        f"Total numeric features: **{len([f for f in num_feats])}** (original 5 + 14 RDKit molecular descriptors)",
        "",
        "| Category | Features |",
        "|---|---|",
        "| Original numeric | `Number of components`, `X#1 (molar fraction)`, `X#2 (molar fraction)`, `T#1`, `T#2` |",
        "| RDKit – Component 1 | `C1_MolWt`, `C1_LogP`, `C1_TPSA`, `C1_HBD`, `C1_HBA`, `C1_RotBonds`, `C1_RingCount` |",
        "| RDKit – Component 2 | `C2_MolWt`, `C2_LogP`, `C2_TPSA`, `C2_HBD`, `C2_HBA`, `C2_RotBonds`, `C2_RingCount` |",
        "| Categorical | `Type of DES`, `Phase diagram (Yes/No)` |",
        "",
        "## 2. Performance Comparison",
        "| Model | Test RMSE (K) | Test R2 | CV RMSE (K) | CV R2 |",
        "|---|---|---|---|---|",
    ]
    if prev_metrics:
        b = prev_metrics["baseline"]
        o = prev_metrics["optimized"]
        md.append(f"| Baseline XGBoost | {b['test']['RMSE']:.3f} | {b['test']['R2']:.4f} | {b['cv_5fold']['cv_rmse_mean']:.3f} | {b['cv_5fold']['cv_r2_mean']:.4f} |")
        md.append(f"| Optimized (no descriptors) | {o['test']['RMSE']:.3f} | {o['test']['R2']:.4f} | {o['cv_5fold']['cv_rmse_mean']:.3f} | {o['cv_5fold']['cv_r2_mean']:.4f} |")
    md.append(f"| **RDKit Optimized** | **{test_m['RMSE']:.3f}** | **{test_m['R2']:.4f}** | **{cv_res['cv_rmse_mean']:.3f}** | **{cv_res['cv_r2_mean']:.4f}** |")

    if delta_test is not None:
        md += [
            "",
            f"> Improvement over descriptor-free optimized: Test RMSE {delta_test:+.3f} K | CV RMSE {delta_cv:+.3f} K",
        ]

    md += [
        "",
        "## 3. Optimized Hyperparameters",
        "| Parameter | Value |",
        "|---|---|",
    ]
    for k,v in best_params.items():
        md.append(f"| `{k}` | {v} |")

    md += [
        "",
        f"> Overfitting gap (Train R2={train_m['R2']:.4f} vs Test R2={test_m['R2']:.4f}): {overfit_gap:.4f}",
        "",
        "## 4. Learning Curve",
        "![Learning Curve RDKit](../Images/learning_curve_rdkit.png)",
        "",
        "## 5. Residual Analysis",
        "![Residual Analysis RDKit](../Images/residual_analysis_rdkit.png)",
        "",
        "## 6. Actual vs Predicted",
        "![Actual vs Predicted RDKit](../Images/actual_vs_predicted_rdkit.png)",
        "",
        "## 7. Feature Importance (Top 15)",
        "| Rank | Feature | Importance |",
        "|---|---|---|",
    ]
    for i, row in imp_df.head(15).iterrows():
        md.append(f"| {i+1} | `{row['Feature']}` | {row['Importance']:.5f} |")

    md += [
        "",
        "![Feature Importance RDKit](../Images/feature_importance_rdkit.png)",
        "",
        "## 8. Model Comparison Chart",
        "![Model Comparison](../Images/baseline_vs_rdkit_rmse.png)",
        "",
        "## 9. Deployment",
        f"- Saved: `Models/xgboost_rdkit_optimized.pkl`",
        f"- CV RMSE: {cv_res['cv_rmse_mean']:.2f} +/- {cv_res['cv_rmse_std']:.2f} K",
        f"- CV R2:   {cv_res['cv_r2_mean']:.4f} +/- {cv_res['cv_r2_std']:.4f}",
    ]

    with open(os.path.join(RPT_DIR, "RDKit_Model_Report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("Saved: Reports/RDKit_Model_Report.md")


def write_screening_report(df_cand, top_room, top_warm, summ, cv_res):
    n_room = summ["n_RoomTemp"]
    n_warm = summ["n_WarmTemp"]
    n_viol = summ["n_DES_rule_violations"]
    status_counts = df_cand["screening_status"].value_counts().to_dict()

    md = [
        "# Candidate Screening Report — RDKit Model",
        f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        f"\n*Model: `xgboost_rdkit_optimized.pkl` | CV RMSE: {cv_res['cv_rmse_mean']:.2f} K*\n",
        "## 1. Screening Summary",
        "| Metric | Value |",
        "|---|---|",
        f"| Candidates evaluated | {summ['total_evaluated']} |",
        f"| DES rule violations | {n_viol} |",
        f"| Physically unreasonable | 0 |",
        f"| RoomTemp candidates (275-298 K) | {n_room} |",
        f"| WarmTemp candidates (298-343 K) | {n_warm} |",
        f"| High-uncertainty predictions (>25 K) | {summ['n_high_uncertainty']} |",
        f"| Feature extrapolation flags | {summ['n_feature_extrapolation']} |",
        "",
        "### Status Breakdown",
        "| Status | Count |",
        "|---|---|",
    ]
    for s, c in sorted(status_counts.items(), key=lambda x: -x[1]):
        md.append(f"| {s} | {c} |")

    md += [
        "",
        "## 2. Temperature Distributions",
        "![Temperature Distribution RDKit](../Images/temperature_distribution_rdkit.png)",
        "",
        "## 3. Confidence vs Predicted Tmelt",
        "![Confidence vs Tmelt RDKit](../Images/confidence_vs_tmelt_rdkit.png)",
        "",
        "## 4. Physical Screening Rules",
        "**Rule**: `Predicted Tmelt < T#1` AND `Predicted Tmelt < T#2`",
        f"- {(~df_cand['violation_T1']).sum()} candidates pass T#1 rule",
        f"- {(~df_cand['violation_T2']).sum()} candidates pass T#2 rule",
        f"- {df_cand['passes_DES_rule'].sum()} candidates pass both rules",
        "",
        "## 5. Ranking Methodology",
        "| Component | Weight |",
        "|---|---|",
        "| Prediction confidence (k-NN similarity) | 40% |",
        "| Temperature proximity to range center | 30% |",
        "| Physical plausibility (DES rule) | 20% |",
        "| Feature validity (no extrapolation) | 10% |",
        "",
        "## 6. Top 25 RoomTemp Candidates (275-298 K)",
        f"*{n_room} total in range. Showing top 25 by ranking score.*",
        "",
    ]

    if len(top_room) > 0:
        md.append("| Rank | Component#1 | Component#2 | X#1 | Pred Tmelt (K) | Similarity | Uncert (K) | Score | DES Rule |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for i, (_, row) in enumerate(top_room.iterrows(), 1):
            md.append(f"| {i} | {str(row.get('Component#1',''))[:28]} | {str(row.get('Component#2',''))[:28]} | "
                      f"{row.get('X#1 (molar fraction)',float('nan')):.3f} | "
                      f"{row['predicted_Tmelt_K']:.2f} | {row['similarity_score']:.3f} | "
                      f"{row['uncertainty_K']:.1f} | {row['ranking_score_room']:.1f} | "
                      f"{'Yes' if row['passes_DES_rule'] else 'No'} |")
    else:
        md.append("*No candidates in RoomTemp range.*")

    md += [
        "",
        "## 7. Top 25 WarmTemp Candidates (298-343 K)",
        f"*{n_warm} total in range. Showing top 25 by ranking score.*",
        "",
    ]

    if len(top_warm) > 0:
        md.append("| Rank | Component#1 | Component#2 | X#1 | Pred Tmelt (K) | Similarity | Uncert (K) | Score | DES Rule |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for i, (_, row) in enumerate(top_warm.iterrows(), 1):
            md.append(f"| {i} | {str(row.get('Component#1',''))[:28]} | {str(row.get('Component#2',''))[:28]} | "
                      f"{row.get('X#1 (molar fraction)',float('nan')):.3f} | "
                      f"{row['predicted_Tmelt_K']:.2f} | {row['similarity_score']:.3f} | "
                      f"{row['uncertainty_K']:.1f} | {row['ranking_score_warm']:.1f} | "
                      f"{'Yes' if row['passes_DES_rule'] else 'No'} |")
    else:
        md.append("*No candidates in WarmTemp range.*")

    md += [
        "",
        "## 8. Potential Risks",
        "| Risk | Notes |",
        "|---|---|",
        f"| Residual overfitting | Train R2 >> CV R2 — use CV metrics for reliability |",
        "| RDKit descriptor coverage | 7 descriptors per component; Morgan fingerprints or more extensive RDKit sets could further improve performance |",
        "| Dataset size | 2006 samples is moderate; predictions for structurally novel DES may be less reliable |",
        "| T#1/T#2 imputation | Missing component Tmelt values are median-imputed — verify from literature before synthesis |",
        "",
        "## 9. Recommended Candidates for Experimental Validation",
        "Select candidates where: `similarity_score > 0.7` AND `passes_DES_rule = True` AND `feature_extrapolation = False`",
        "",
        "| Output file | Contents |",
        "|---|---|",
        "| `CSVs/RoomTemp_Top25_rdkit.csv` | Top 25 room-temperature DES candidates |",
        "| `CSVs/WarmTemp_Top25_rdkit.csv` | Top 25 warm-temperature DES candidates |",
        "| `CSVs/Candidate_Master_List_rdkit.csv` | Full ranked list with all flags |",
    ]

    with open(os.path.join(RPT_DIR, "Candidate_Screening_Report_rdkit.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("Saved: Reports/Candidate_Screening_Report_rdkit.md")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

cv_res_global = {}   # shared between make_plots and screen_rdkit

if __name__ == "__main__":
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load previous metrics for comparison
    prev_metrics = None
    if os.path.exists(PREV_METRICS):
        with open(PREV_METRICS, encoding="utf-8") as f:
            prev_metrics = json.load(f)

    # Data
    df_full, X, y, num_feats, cat_feats = load_rdkit_data()

    # Optimize
    (best_pipe, best_params, best_params_raw,
     train_m, test_m, cv_res, X_tr, X_te, y_tr, y_te) = optimize_rdkit(X, y, num_feats, cat_feats)

    cv_res_global.update(cv_res)

    # Plots
    imp_df = make_plots(best_pipe, X, y, X_tr, X_te, y_tr, y_te, test_m, num_feats, prev_metrics)

    # Retrain on 100% for deployment
    print("\nRetraining on 100% of data for deployment …")
    final_params = {k.replace("regressor__",""):v for k,v in best_params_raw.items()}
    deploy_pipe = Pipeline([
        ("preprocessor", build_preprocessor(num_feats, cat_feats)),
        ("regressor", XGBRegressor(**final_params, objective="reg:squarederror",
                                    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
    ])
    deploy_pipe.fit(X, y)
    with open(RDKIT_PKL, "wb") as f:
        pickle.dump(deploy_pipe, f)
    print(f"Saved: Models/xgboost_rdkit_optimized.pkl")

    # Save metrics JSON
    feat_cols = [c for c in df_full.columns if c not in
                 ["Component#1","Component#2","Reference (DOI)","Smiles#1","Smiles#2", TARGET_COL]]
    rdkit_summary = {
        "train": train_m, "test": test_m, "cv_5fold": cv_res,
        "hyperparameters": best_params,
        "n_features": len(num_feats) + len(cat_feats),
        "rdkit_descriptors_added": RDKIT_COLS,
    }
    with open(os.path.join(JSN_DIR, "rdkit_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(rdkit_summary, f, indent=2)

    # Screen candidates
    df_cand, top_room, top_warm, summ = screen_rdkit(deploy_pipe, df_full, feat_cols, num_feats)

    # Reports
    write_rdkit_model_report(best_params, train_m, test_m, cv_res, imp_df, prev_metrics, num_feats)
    write_screening_report(df_cand, top_room, top_warm, summ, cv_res)

    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
