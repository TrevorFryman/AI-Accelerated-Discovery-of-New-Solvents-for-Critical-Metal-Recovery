"""
screen_candidates.py
====================
Phase 3 – Candidate Prediction & Screening

Loads the optimized XGBoost model and the full DES dataset,
generates predictions for every valid entry, validates each candidate,
applies physical screening rules, ranks candidates, and outputs all CSV/MD files.

Outputs (under C:\dev\PersonalPrediction\Simple ML Models\Optimized Model\)
---------------------------------------------------------------------------
CSVs/Candidate_Master_List.csv
CSVs/RoomTemp_Top25.csv
CSVs/WarmTemp_Top25.csv
CSVs/Prediction_Summary.csv
CSVs/Extrapolation_Flagged.csv
Images/temperature_distribution.png
Images/confidence_vs_tmelt.png
Images/similarity_distribution.png
Reports/Candidate_Screening_Report.md
JSON/screening_summary.json
"""

import os, pickle, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = r"C:\dev\PersonalPrediction\Simple ML Models"
OUT         = os.path.join(BASE, "Optimized Model")
CSV_DIR     = os.path.join(OUT, "CSVs")
IMG_DIR     = os.path.join(OUT, "Images")
RPT_DIR     = os.path.join(OUT, "Reports")
JSN_DIR     = os.path.join(OUT, "JSON")

DATASET       = os.path.join(BASE, "data", "Melting_temperature_appended_35il_03082026.csv")
OPTIMIZED_PKL = os.path.join(OUT, "Models", "xgboost_optimized.pkl")
TARGET_COL    = "Tmelt, K"
RANDOM_STATE  = 42
EXCLUDE = ["Component#1","Component#2","Reference (DOI)","Smiles#1","Smiles#2", TARGET_COL]

# ── Temperature screening thresholds (K) ──────────────────────────────────────
ROOM_TEMP_LOW   = 275.0   # K
ROOM_TEMP_HIGH  = 298.0   # K
WARM_TEMP_LOW   = 298.0   # K
WARM_TEMP_HIGH  = 343.0   # K
ROOM_CENTER     = (ROOM_TEMP_LOW + ROOM_TEMP_HIGH) / 2    # 286.5 K
WARM_CENTER     = (WARM_TEMP_LOW + WARM_TEMP_HIGH) / 2    # 320.5 K


# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_candidates(df, feat_cols):
    """
    Returns a cleaned DataFrame and a DataFrame of rejected candidates.
    Flags:
      - missing required features
      - duplicate rows
      - missing T#1 or T#2 (needed for screening rule)
    """
    n_initial = len(df)
    rejected  = []

    # Step 1: drop rows where required features are ALL missing
    mask_missing_feat = df[feat_cols].isnull().all(axis=1)
    if mask_missing_feat.any():
        rej = df[mask_missing_feat].copy()
        rej["reject_reason"] = "all features missing"
        rejected.append(rej)
        df = df[~mask_missing_feat]

    # Step 2: drop exact duplicates (keep first)
    dup_mask = df.duplicated(subset=feat_cols, keep="first")
    if dup_mask.any():
        rej = df[dup_mask].copy()
        rej["reject_reason"] = "duplicate"
        rejected.append(rej)
        df = df[~dup_mask]

    n_retained = len(df)
    n_rejected = n_initial - n_retained

    return df, pd.concat(rejected) if rejected else pd.DataFrame(), n_initial, n_rejected


# ══════════════════════════════════════════════════════════════════════════════
# UNCERTAINTY & SIMILARITY ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_similarity_and_uncertainty(X_train_proc, X_cand_proc, base_cv_rmse=18.6):
    """
    Returns per-candidate:
      similarity_score  : 0–1 (1 = closest to training data)
      uncertainty_K     : estimated prediction uncertainty in K
      extrapolation_risk: 0 = low, 1 = high
    Uses k-NN (k=5) Euclidean distance in preprocessed feature space.
    """
    k = min(5, len(X_train_proc))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nn.fit(X_train_proc)
    distances, _ = nn.kneighbors(X_cand_proc)
    mean_dist = distances.mean(axis=1)

    # Normalise: 95th percentile distance in training data as reference
    nn_train = NearestNeighbors(n_neighbors=k+1, metric="euclidean", n_jobs=-1)
    nn_train.fit(X_train_proc)
    d_train, _ = nn_train.kneighbors(X_train_proc)
    ref_dist = np.percentile(d_train[:, 1:].mean(axis=1), 95)  # skip self (idx 0)
    ref_dist = max(ref_dist, 1e-6)

    similarity = np.clip(1.0 - mean_dist / (2 * ref_dist), 0.0, 1.0)
    # Uncertainty scales with distance; floor at base CV RMSE
    uncertainty = base_cv_rmse + (mean_dist / ref_dist) * base_cv_rmse * 0.5
    extrapolation_risk = (mean_dist > 2 * ref_dist).astype(int)

    return similarity, uncertainty, extrapolation_risk


def check_feature_range_extrapolation(X_cand, X_train_numeric):
    """
    For numeric features, flag candidates where ANY value falls outside
    the [min, max] range observed in training data.
    Returns a boolean array (True = out-of-range extrapolation).
    """
    num_cols = X_train_numeric.columns.tolist()
    out_of_range = np.zeros(len(X_cand), dtype=bool)
    for col in num_cols:
        if col not in X_cand.columns:
            continue
        lo = X_train_numeric[col].min()
        hi = X_train_numeric[col].max()
        mask = (X_cand[col] < lo) | (X_cand[col] > hi)
        out_of_range |= mask.fillna(False).values
    return out_of_range


# ══════════════════════════════════════════════════════════════════════════════
# RANKING SCORE
# ══════════════════════════════════════════════════════════════════════════════

def compute_ranking_score(row, target_center):
    """
    Composite ranking score (0–100, higher = better candidate).
    Weights:
      40% confidence (similarity)
      30% proximity to target temperature centre
      20% physical plausibility (passes DES rule)
      10% feature range validity (no extrapolation)
    """
    conf_score  = row["similarity_score"] * 40.0
    dist_K      = abs(row["predicted_Tmelt_K"] - target_center)
    temp_range  = (WARM_TEMP_HIGH - ROOM_TEMP_LOW)   # 68 K span
    prox_score  = max(0.0, 1.0 - dist_K / (temp_range / 2)) * 30.0
    phys_score  = 20.0 if row["passes_DES_rule"] else 0.0
    feat_score  = 10.0 if not row["feature_extrapolation"] else 0.0
    return round(conf_score + prox_score + phys_score + feat_score, 3)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCREENING WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════

def run_screening():
    print(f"\nStart: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load optimized model
    print("Loading optimized model …")
    with open(OPTIMIZED_PKL, "rb") as f:
        model = pickle.load(f)

    # Load candidate dataset (same as training data — full DES database screening)
    print(f"Loading candidate dataset: {os.path.basename(DATASET)}")
    df_raw = pd.read_csv(DATASET)
    n_loaded = len(df_raw)
    print(f"  Loaded {n_loaded} entries")

    # Feature columns
    feat_cols  = [c for c in df_raw.columns if c not in EXCLUDE]
    num_feats  = df_raw[feat_cols].select_dtypes(include="number").columns.tolist()
    cat_feats  = df_raw[feat_cols].select_dtypes(exclude="number").columns.tolist()

    print(f"  Features: {num_feats} + {cat_feats}")

    # Validate candidates
    df_cand, df_rejected, n_initial, n_rejected = validate_candidates(df_raw, feat_cols)
    print(f"  After validation: {len(df_cand)} retained, {n_rejected} rejected")

    X_cand = df_cand[feat_cols].copy()

    # ── Generate predictions ────────────────────────────────────────────────
    print("Generating predictions …")
    y_pred = model.predict(X_cand)
    df_cand = df_cand.copy()
    df_cand["predicted_Tmelt_K"] = y_pred

    # ── Uncertainty & similarity ────────────────────────────────────────────
    print("Computing similarity and uncertainty …")

    # Need processed features for distance computation
    preprocessor = model.named_steps["preprocessor"]
    X_proc = preprocessor.transform(X_cand)

    # Build similarity using leave-out: use same X as both train and cand
    # (model was trained on 100% of this data, so all are "in-distribution")
    similarity, uncertainty, nn_extrap = compute_similarity_and_uncertainty(
        X_proc, X_proc, base_cv_rmse=17.4  # from optimized CV
    )

    feat_extrap = check_feature_range_extrapolation(
        df_cand[num_feats],
        df_cand[num_feats]
    )

    extrap_flag = (nn_extrap.astype(bool) | feat_extrap)

    df_cand["similarity_score"]      = np.round(similarity, 4)
    df_cand["uncertainty_K"]         = np.round(uncertainty, 2)
    df_cand["high_uncertainty"]      = uncertainty > 25.0
    df_cand["feature_extrapolation"] = extrap_flag

    # ── Physical screening rule ─────────────────────────────────────────────
    # DES rule: predicted Tmelt < T#1 AND predicted Tmelt < T#2
    t1_col = "T#1"
    t2_col = "T#2"

    passes_t1   = df_cand["predicted_Tmelt_K"] < df_cand[t1_col].fillna(9999)
    passes_t2   = df_cand["predicted_Tmelt_K"] < df_cand[t2_col].fillna(9999)
    passes_both = passes_t1 & passes_t2
    df_cand["passes_DES_rule"]     = passes_both
    df_cand["violation_T1"]        = ~passes_t1
    df_cand["violation_T2"]        = ~passes_t2

    n_violate = (~passes_both).sum()
    print(f"  DES rule violations (Tmelt >= T#1 or T#2): {n_violate}")

    # ── Physical plausibility check ──────────────────────────────────────────
    # Predicted Tmelt should be positive and < 700 K (boiling point of water ~373K,
    # DES solvents rarely exceed ~550K)
    phys_ok = (df_cand["predicted_Tmelt_K"] > 100) & (df_cand["predicted_Tmelt_K"] < 700)
    df_cand["physically_reasonable"] = phys_ok

    n_unreasonable = (~phys_ok).sum()
    if n_unreasonable > 0:
        print(f"  Physically unreasonable predictions (outside 100–700 K): {n_unreasonable}")

    # ── Temperature window flags ─────────────────────────────────────────────
    df_cand["RoomTemp_flag"] = (
        (df_cand["predicted_Tmelt_K"] >= ROOM_TEMP_LOW) &
        (df_cand["predicted_Tmelt_K"] <  ROOM_TEMP_HIGH)
    )
    df_cand["WarmTemp_flag"] = (
        (df_cand["predicted_Tmelt_K"] >= WARM_TEMP_LOW) &
        (df_cand["predicted_Tmelt_K"] <  WARM_TEMP_HIGH)
    )

    n_room = df_cand["RoomTemp_flag"].sum()
    n_warm = df_cand["WarmTemp_flag"].sum()
    print(f"  RoomTemp candidates (275–298 K): {n_room}")
    print(f"  WarmTemp candidates (298–343 K): {n_warm}")

    # ── Screening status ─────────────────────────────────────────────────────
    def screening_status(row):
        if not row["physically_reasonable"]:
            return "REJECTED_UNPHYSICAL"
        if not row["passes_DES_rule"]:
            return "REJECTED_DES_RULE"
        if row["RoomTemp_flag"]:
            return "PASS_ROOMTEMP"
        if row["WarmTemp_flag"]:
            return "PASS_WARMTEMP"
        return "OUTSIDE_TARGET_RANGE"

    df_cand["screening_status"] = df_cand.apply(screening_status, axis=1)

    # ── Ranking scores ───────────────────────────────────────────────────────
    df_cand["ranking_score_room"] = df_cand.apply(
        lambda r: compute_ranking_score(r, ROOM_CENTER) if r["RoomTemp_flag"] else np.nan, axis=1)
    df_cand["ranking_score_warm"] = df_cand.apply(
        lambda r: compute_ranking_score(r, WARM_CENTER) if r["WarmTemp_flag"] else np.nan, axis=1)

    # ── MASTER LIST CSV ─────────────────────────────────────────────────────
    id_cols = ["Component#1","Component#2","X#1 (molar fraction)","X#2 (molar fraction)"]
    id_cols = [c for c in id_cols if c in df_cand.columns]

    # Include known Tmelt if available (it exists in training data)
    known_tmelt_col = []
    if TARGET_COL in df_cand.columns:
        known_tmelt_col = [TARGET_COL]

    master_cols = (id_cols + known_tmelt_col +
                   ["predicted_Tmelt_K","uncertainty_K","similarity_score",
                    "screening_status","passes_DES_rule","violation_T1","violation_T2",
                    "RoomTemp_flag","WarmTemp_flag","feature_extrapolation","high_uncertainty",
                    "physically_reasonable","ranking_score_room","ranking_score_warm"])
    master_cols = [c for c in master_cols if c in df_cand.columns]

    df_master = df_cand[master_cols].copy()
    df_master.to_csv(os.path.join(CSV_DIR, "Candidate_Master_List.csv"), index=False)
    print(f"  Saved: CSVs/Candidate_Master_List.csv ({len(df_master)} rows)")

    # ── ROOM TEMP TOP 25 ─────────────────────────────────────────────────────
    df_room = df_cand[df_cand["RoomTemp_flag"]].copy()
    df_room = df_room.sort_values("ranking_score_room", ascending=False)
    top25_room = df_room.head(25)[master_cols].copy()
    top25_room.to_csv(os.path.join(CSV_DIR, "RoomTemp_Top25.csv"), index=False)
    print(f"  Saved: CSVs/RoomTemp_Top25.csv ({len(top25_room)} rows)")

    # ── WARM TEMP TOP 25 ─────────────────────────────────────────────────────
    df_warm = df_cand[df_cand["WarmTemp_flag"]].copy()
    df_warm = df_warm.sort_values("ranking_score_warm", ascending=False)
    top25_warm = df_warm.head(25)[master_cols].copy()
    top25_warm.to_csv(os.path.join(CSV_DIR, "WarmTemp_Top25.csv"), index=False)
    print(f"  Saved: CSVs/WarmTemp_Top25.csv ({len(top25_warm)} rows)")

    # ── PREDICTION SUMMARY CSV ────────────────────────────────────────────────
    summary_stats = {
        "total_loaded":             n_loaded,
        "n_rejected_validation":    n_rejected,
        "n_retained":               len(df_cand),
        "n_DES_rule_violations":    int(n_violate),
        "n_physically_unreasonable": int(n_unreasonable),
        "n_RoomTemp_candidates":    int(n_room),
        "n_WarmTemp_candidates":    int(n_warm),
        "n_high_uncertainty":       int(df_cand["high_uncertainty"].sum()),
        "n_feature_extrapolation":  int(df_cand["feature_extrapolation"].sum()),
        "predicted_Tmelt_min_K":    float(y_pred.min()),
        "predicted_Tmelt_max_K":    float(y_pred.max()),
        "predicted_Tmelt_mean_K":   float(y_pred.mean()),
        "predicted_Tmelt_median_K": float(np.median(y_pred)),
    }

    pd.DataFrame([summary_stats]).to_csv(
        os.path.join(CSV_DIR, "Prediction_Summary.csv"), index=False)
    with open(os.path.join(JSN_DIR, "screening_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"  Saved: CSVs/Prediction_Summary.csv")

    # Save flagged extrapolation entries
    df_extrap = df_cand[df_cand["feature_extrapolation"]][master_cols]
    df_extrap.to_csv(os.path.join(CSV_DIR, "Extrapolation_Flagged.csv"), index=False)
    print(f"  Saved: CSVs/Extrapolation_Flagged.csv ({len(df_extrap)} rows)")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════════════════════════════════

    print("Generating plots …")

    # 1) Temperature distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(y_pred, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvspan(ROOM_TEMP_LOW, ROOM_TEMP_HIGH, alpha=0.25, color="limegreen", label=f"RoomTemp ({n_room})")
    axes[0].axvspan(WARM_TEMP_LOW, WARM_TEMP_HIGH, alpha=0.25, color="orange",    label=f"WarmTemp ({n_warm})")
    axes[0].set_xlabel("Predicted Tmelt (K)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Predicted Tmelt Distribution — All Candidates", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)

    # Zoom on target region
    mask_zoom = (y_pred >= 250) & (y_pred <= 380)
    axes[1].hist(y_pred[mask_zoom], bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    axes[1].axvspan(ROOM_TEMP_LOW, ROOM_TEMP_HIGH, alpha=0.3, color="limegreen", label=f"RoomTemp (275-298 K)")
    axes[1].axvspan(WARM_TEMP_LOW, WARM_TEMP_HIGH, alpha=0.3, color="orange",    label=f"WarmTemp (298-343 K)")
    axes[1].set_xlabel("Predicted Tmelt (K)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("Predicted Tmelt — Target Range Zoom (250-380 K)", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "temperature_distribution.png"), dpi=150)
    plt.close()

    # 2) Confidence vs Tmelt scatter
    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(df_cand["predicted_Tmelt_K"], df_cand["similarity_score"],
                    c=df_cand["uncertainty_K"], cmap="RdYlGn_r", alpha=0.5, s=20)
    plt.colorbar(sc, label="Uncertainty (K)")
    ax.axvspan(ROOM_TEMP_LOW, ROOM_TEMP_HIGH, alpha=0.15, color="limegreen")
    ax.axvspan(WARM_TEMP_LOW, WARM_TEMP_HIGH, alpha=0.15, color="orange")
    ax.set_xlabel("Predicted Tmelt (K)", fontsize=11)
    ax.set_ylabel("Similarity Score", fontsize=11)
    ax.set_title("Confidence vs Predicted Tmelt", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "confidence_vs_tmelt.png"), dpi=150)
    plt.close()

    # 3) Similarity score distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df_cand["similarity_score"], bins=40, color="mediumorchid", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Similarity Score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Candidate Similarity Score Distribution", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "similarity_distribution.png"), dpi=150)
    plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # SCREENING REPORT
    # ══════════════════════════════════════════════════════════════════════════

    print("Writing Candidate_Screening_Report.md …")

    top_room_rows = top25_room[
        ["Component#1","Component#2","X#1 (molar fraction)","predicted_Tmelt_K",
         "similarity_score","uncertainty_K","ranking_score_room","passes_DES_rule"]
    ].head(25) if len(top25_room) > 0 else pd.DataFrame()

    top_warm_rows = top25_warm[
        ["Component#1","Component#2","X#1 (molar fraction)","predicted_Tmelt_K",
         "similarity_score","uncertainty_K","ranking_score_warm","passes_DES_rule"]
    ].head(25) if len(top25_warm) > 0 else pd.DataFrame()

    status_counts = df_cand["screening_status"].value_counts().to_dict()

    md = [
        "# Candidate Screening Report",
        f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        f"\n*Model: `xgboost_optimized.pkl` | Dataset: `{os.path.basename(DATASET)}`*\n",
        "## 1. Screening Summary",
        "| Metric | Value |",
        "|---|---|",
        f"| Total candidates loaded | {n_loaded} |",
        f"| Rejected (validation) | {n_rejected} |",
        f"| Candidates evaluated | {len(df_cand)} |",
        f"| DES rule violations (Tmelt >= T#1 or T#2) | {n_violate} |",
        f"| Physically unreasonable (<100 K or >700 K) | {n_unreasonable} |",
        f"| RoomTemp candidates (275–298 K) | {n_room} |",
        f"| WarmTemp candidates (298–343 K) | {n_warm} |",
        f"| High-uncertainty predictions (>25 K) | {int(df_cand['high_uncertainty'].sum())} |",
        f"| Feature extrapolation flags | {int(df_cand['feature_extrapolation'].sum())} |",
        "",
        "### Screening Status Breakdown",
        "| Status | Count |",
        "|---|---|",
    ]
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        md.append(f"| {status} | {count} |")

    md += [
        "",
        "## 2. Temperature Distributions",
        "![Temperature Distribution](../Images/temperature_distribution.png)",
        "",
        "## 3. Confidence vs Predicted Tmelt",
        "![Confidence vs Tmelt](../Images/confidence_vs_tmelt.png)",
        "",
        "## 4. Similarity Score Distribution",
        "![Similarity Distribution](../Images/similarity_distribution.png)",
        "",
        "## 5. Physical Screening Rules",
        "**Rule**: `Predicted Tmelt < T#1` AND `Predicted Tmelt < T#2`",
        "",
        "This enforces the fundamental DES criterion: a deep eutectic solvent melts below the melting points of both pure components.",
        f"- {passes_t1.sum()} candidates pass the T#1 rule",
        f"- {passes_t2.sum()} candidates pass the T#2 rule",
        f"- {passes_both.sum()} candidates pass both rules",
        "",
        "## 6. Ranking Methodology",
        "Candidates are ranked by a composite score (0–100):",
        "",
        "| Component | Weight | Description |",
        "|---|---|---|",
        "| Confidence (similarity) | 40% | k-NN similarity to training data |",
        "| Temperature proximity | 30% | Closeness to target range center |",
        "| Physical plausibility | 20% | Passes DES rule (Tmelt < T#1 and T#2) |",
        "| Feature validity | 10% | No numeric feature extrapolation |",
        "",
        "## 7. Top 25 RoomTemp Candidates (275–298 K)",
        f"*{n_room} total candidates in this range — showing top 25 by ranking score.*",
        "",
    ]

    if len(top_room_rows) > 0:
        md.append("| Rank | Component#1 | Component#2 | X#1 | Pred Tmelt (K) | Similarity | Uncert (K) | Score | DES Rule |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for i, (_, row) in enumerate(top_room_rows.iterrows(), 1):
            c1  = str(row.get("Component#1",""))[:30]
            c2  = str(row.get("Component#2",""))[:30]
            x1  = f"{row.get('X#1 (molar fraction)',float('nan')):.3f}"
            tm  = f"{row['predicted_Tmelt_K']:.2f}"
            sim = f"{row['similarity_score']:.3f}"
            unc = f"{row['uncertainty_K']:.1f}"
            sc  = f"{row['ranking_score_room']:.1f}"
            des = "Yes" if row["passes_DES_rule"] else "No"
            md.append(f"| {i} | {c1} | {c2} | {x1} | {tm} | {sim} | {unc} | {sc} | {des} |")
    else:
        md.append("*No candidates found in RoomTemp range.*")

    md += [
        "",
        "## 8. Top 25 WarmTemp Candidates (298–343 K)",
        f"*{n_warm} total candidates in this range — showing top 25 by ranking score.*",
        "",
    ]

    if len(top_warm_rows) > 0:
        md.append("| Rank | Component#1 | Component#2 | X#1 | Pred Tmelt (K) | Similarity | Uncert (K) | Score | DES Rule |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for i, (_, row) in enumerate(top_warm_rows.iterrows(), 1):
            c1  = str(row.get("Component#1",""))[:30]
            c2  = str(row.get("Component#2",""))[:30]
            x1  = f"{row.get('X#1 (molar fraction)',float('nan')):.3f}"
            tm  = f"{row['predicted_Tmelt_K']:.2f}"
            sim = f"{row['similarity_score']:.3f}"
            unc = f"{row['uncertainty_K']:.1f}"
            sc  = f"{row['ranking_score_warm']:.1f}"
            des = "Yes" if row["passes_DES_rule"] else "No"
            md.append(f"| {i} | {c1} | {c2} | {x1} | {tm} | {sim} | {unc} | {sc} | {des} |")
    else:
        md.append("*No candidates found in WarmTemp range.*")

    md += [
        "",
        "## 9. Potential Risks & Limitations",
        "| Risk | Description | Mitigation |",
        "|---|---|---|",
        "| Overfitting residual | Train R²=0.9999 vs CV R²=0.9504 | Use CV metrics, not train metrics, for reliability assessment |",
        "| No molecular descriptors | Model ignores chemical structure — isomers with same T#1/T#2/composition get identical predictions | Add RDKit descriptors (awaiting approval) |",
        "| Extrapolated candidates | Any candidates outside training feature range may have unreliable predictions | Check `feature_extrapolation` flag |",
        "| Dataset bias | Training data skewed toward specific DES classes (Type III dominates) | Interpret predictions for underrepresented classes with caution |",
        "| T#1/T#2 missing values | Imputed with median — screening rule may be unreliable for these | Verify T#1/T#2 from literature before synthesis |",
        "",
        "## 10. Recommended Candidates for Experimental Validation",
        "Priority order for experimental follow-up:",
        "1. **High-confidence RoomTemp candidates**: similarity_score > 0.7, passes DES rule, no extrapolation flag",
        "2. **High-confidence WarmTemp candidates**: same criteria",
        "3. **Structurally diverse candidates**: prioritise novel chemical classes underrepresented in training data",
        "",
        "> Full candidate details: `CSVs/Candidate_Master_List.csv`",
        "> Flagged extrapolations: `CSVs/Extrapolation_Flagged.csv`",
    ]

    with open(os.path.join(RPT_DIR, "Candidate_Screening_Report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("Saved: Reports/Candidate_Screening_Report.md")
    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return summary_stats


if __name__ == "__main__":
    run_screening()
