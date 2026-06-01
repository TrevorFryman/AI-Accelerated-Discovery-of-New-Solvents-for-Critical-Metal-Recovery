#!/usr/bin/env python3
"""
End-to-end DES melting point prediction pipeline.

What it does:
1) Read CSV with columns:
   - Smiles#1, Smiles#2, X#1 (molar fraction), T#1, T#2, Tmelt, K
2) Canonicalize SMILES before any embedding
3) Compute embeddings with one of:
   - chemberta
   - morgan
   - gnn (end-to-end, no precompute)
4) Physics learning (Siamese net predicting thermodynamic params)
5) Stack residuals with XGBoost (regression)
6) Evaluate MAE under two split protocols:
   A) random_rows  : random split of rows (ratios can overlap between train/test for same pair)
   B) strict_pairs : group split so a solvent pair (orderless) never appears in both train and test
7) Save plots + a metrics.json summary

Run:
    python pipeline.py --config config.yaml

Tip:
- For small datasets, prefer strict_pairs for a realistic generalization estimate.
- For model selection / tuning, use K-fold on strict_pairs and keep a final strict_pairs holdout.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import xgboost as xgb

from embedding import (
    ChemBERTaParams,
    MorganParams,
    batch_graphs,
    canonicalize_series,
    canonicalize_smiles,
    embed_chemberta_unique,
    embed_morgan_unique,
    mol_to_graph,
    MolGNNEncoder,
)
from physics_learning import (
    PhysicsTrainConfig,
    ParamSiameseNet,
    GraphPhysicsModel,
    seed_all,
    train_physics_siamese,
    predict_physics_siamese,
    train_graph_physics,
    predict_graph_physics,
    load_physics_siamese_checkpoint,
    load_graph_physics_checkpoint,
)
from model_interpretation import plot_loss, parity_plot, residual_hist, xgb_feature_importance, try_shap_summary


# ----------------------------
# Helpers
# ----------------------------
def load_yaml(path: str) -> Dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_pair_id(s1: str, s2: str) -> str:
    # orderless pair id for strict split
    a, b = sorted([s1, s2])
    return f"{a}||{b}"


def make_boost_features(
    emb1: np.ndarray,
    emb2: np.ndarray,
    x1: np.ndarray,
    T1: np.ndarray,
    T2: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    W: np.ndarray,
    include_params: bool = True,
) -> np.ndarray:
    feats = [
        emb1, emb2,
        np.asarray(x1).reshape(-1, 1),
        np.asarray(T1).reshape(-1, 1),
        np.asarray(T2).reshape(-1, 1),
    ]
    if include_params:
        feats += [
            np.asarray(d1).reshape(-1, 1),
            np.asarray(d2).reshape(-1, 1),
            np.asarray(W).reshape(-1, 1),
        ]
    return np.concatenate(feats, axis=1).astype(np.float32)


def fit_xgb_residual_model(X_train: np.ndarray, y_resid: np.ndarray, params: Dict, seed: int) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=int(params.get("n_estimators", 1200)),
        max_depth=int(params.get("max_depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.03)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        min_child_weight=float(params.get("min_child_weight", 1.0)),
        gamma=float(params.get("gamma", 0.0)),
        objective="reg:squarederror",
        tree_method=params.get("tree_method", "hist"),
        random_state=seed,
        n_jobs=int(params.get("n_jobs", -1)),
    )
    model.fit(X_train, y_resid)
    return model


# ----------------------------
# Embedding preparation
# ----------------------------
def prepare_fixed_embeddings(df: pd.DataFrame, cfg: Dict, device: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Returns per-row embeddings X1, X2 and embedding dim.
    """
    emb_cfg = cfg["embedding"]
    method = emb_cfg["method"]

    # Canonicalize SMILES and store
    df["can_smiles1"] = canonicalize_series(df["Smiles#1"].tolist())
    df["can_smiles2"] = canonicalize_series(df["Smiles#2"].tolist())

    all_smiles = pd.concat([df["can_smiles1"], df["can_smiles2"]]).tolist()

    if method == "morgan":
        mp = MorganParams(
            radius=int(emb_cfg.get("radius", 2)),
            n_bits=int(emb_cfg.get("n_bits", 2048)),
            use_counts=bool(emb_cfg.get("use_counts", False)),
            use_chirality=bool(emb_cfg.get("use_chirality", True)),
        )
        feats, idx = embed_morgan_unique(all_smiles, mp)

    elif method == "chemberta":
        cp = ChemBERTaParams(
            model_name=str(emb_cfg.get("model_name", "DeepChem/ChemBERTa-77M-MTR")),
            pooling=str(emb_cfg.get("pooling", "mean")),
            max_length=int(emb_cfg.get("max_length", 256)),
            batch_size=int(emb_cfg.get("batch_size", 64)),
        )
        feats, idx = embed_chemberta_unique(all_smiles, cp, device=device)

    else:
        raise ValueError(f"prepare_fixed_embeddings only supports morgan/chemberta, got {method}")

    X1 = feats[df["can_smiles1"].map(idx).to_numpy()]
    X2 = feats[df["can_smiles2"].map(idx).to_numpy()]
    return X1, X2, int(feats.shape[1])


def prepare_graph_batches(df: pd.DataFrame) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], int]:
    """
    Canonicalize + build padded graph batches for gnn method.
    Returns (g1_batch, g2_batch, atom_feature_dim)
    """
    df["can_smiles1"] = canonicalize_series(df["Smiles#1"].tolist())
    df["can_smiles2"] = canonicalize_series(df["Smiles#2"].tolist())
    g1 = [mol_to_graph(s) for s in df["can_smiles1"].tolist()]
    g2 = [mol_to_graph(s) for s in df["can_smiles2"].tolist()]
    X1b, A1b, m1b = batch_graphs(g1)
    X2b, A2b, m2b = batch_graphs(g2)
    in_dim = int(X1b.shape[-1])
    return (X1b, A1b, m1b), (X2b, A2b, m2b), in_dim


# ----------------------------
# Evaluation loops
# ----------------------------
def run_one_split_fixed(
    X1: np.ndarray, X2: np.ndarray,
    T1: np.ndarray, T2: np.ndarray, x1: np.ndarray, Tm: np.ndarray,
    train_idx: np.ndarray, test_idx: np.ndarray,
    phys_cfg: Dict, xgb_cfg: Dict,
    seed: int,
    out_dir: str,
    tag: str,
) -> Dict[str, float]:

    X1_tr, X1_te = X1[train_idx], X1[test_idx]
    X2_tr, X2_te = X2[train_idx], X2[test_idx]
    T1_tr, T1_te = T1[train_idx], T1[test_idx]
    T2_tr, T2_te = T2[train_idx], T2[test_idx]
    x1_tr, x1_te = x1[train_idx], x1[test_idx]
    Tm_tr, Tm_te = Tm[train_idx], Tm[test_idx]

    os.makedirs(out_dir, exist_ok=True)

    # Physics
    p_cfg = PhysicsTrainConfig(
        epochs=int(phys_cfg.get("epochs", 1500)),
        lr=float(phys_cfg.get("lr", 3e-3)),
        weight_decay=float(phys_cfg.get("weight_decay", 1e-5)),
        clip_grad_norm=float(phys_cfg.get("clip_grad_norm", 1.0)),
        anchor_w_lambda=float(phys_cfg.get("anchor_w_lambda", 1e-10)),
        device=str(phys_cfg.get("device", "cuda")),
        verbose_every=int(phys_cfg.get("verbose_every", 100)),
        checkpoint_path=os.path.join(out_dir, "physics_best.pt"),
        save_best=True,
        save_last=True,
    )
    model, hist = train_physics_siamese(
        X1_tr, X2_tr, T1_tr, T2_tr, x1_tr, Tm_tr,
        X1_val=X1_te, X2_val=X2_te, T1_val=T1_te, T2_val=T2_te, x1_val=x1_te, Tm_val=Tm_te,
        emb_dim=int(X1.shape[1]),
        mlp_width=int(phys_cfg.get("mlp_width", 128)),
        mlp_depth=int(phys_cfg.get("mlp_depth", 2)),
        dropout=float(phys_cfg.get("dropout", 0.2)),
        cfg=p_cfg,
        show_progress=bool(phys_cfg.get("show_progress", False)),
    )

    Tm_phys_tr, d1_tr, d2_tr, W_tr = predict_physics_siamese(model, X1_tr, X2_tr, T1_tr, T2_tr, x1_tr, device=p_cfg.device)
    Tm_phys_te, d1_te, d2_te, W_te = predict_physics_siamese(model, X1_te, X2_te, T1_te, T2_te, x1_te, device=p_cfg.device)

    mae_phys_tr = mean_absolute_error(Tm_tr, Tm_phys_tr)
    mae_phys_te = mean_absolute_error(Tm_te, Tm_phys_te)

    # XGB on residuals
    resid_tr = (Tm_tr - Tm_phys_tr).astype(np.float32)
    Xb_tr = make_boost_features(X1_tr, X2_tr, x1_tr, T1_tr, T2_tr, d1_tr, d2_tr, W_tr, include_params=True)
    Xb_te = make_boost_features(X1_te, X2_te, x1_te, T1_te, T2_te, d1_te, d2_te, W_te, include_params=True)

    xgb_model = fit_xgb_residual_model(Xb_tr, resid_tr, xgb_cfg, seed=seed)
    xgb_model.save_model(os.path.join(out_dir, "xgb.json"))

    Tm_final_tr = Tm_phys_tr + xgb_model.predict(Xb_tr)
    Tm_final_te = Tm_phys_te + xgb_model.predict(Xb_te)
    mae_final_tr = mean_absolute_error(Tm_tr, Tm_final_tr)
    mae_final_te = mean_absolute_error(Tm_te, Tm_final_te)

    # Plots for this split
    os.makedirs(out_dir, exist_ok=True)
    plot_loss(hist, out_png=os.path.join(out_dir, f"loss_{tag}.png"), title=f"Physics loss ({tag})")
    parity_plot(Tm_te, Tm_phys_te, out_png=os.path.join(out_dir, f"parity_phys_{tag}.png"), title=f"Physics parity ({tag})")
    parity_plot(Tm_te, Tm_final_te, out_png=os.path.join(out_dir, f"parity_final_{tag}.png"), title=f"Final parity ({tag})")
    residual_hist(Tm_te, Tm_final_te, out_png=os.path.join(out_dir, f"residual_{tag}.png"), title=f"Residuals ({tag})")
    xgb_feature_importance(xgb_model, out_png=os.path.join(out_dir, f"xgb_importance_{tag}.png"))
    try_shap_summary(xgb_model, Xb_te, out_png=os.path.join(out_dir, f"shap_{tag}.png"))

    return {
        "mae_physics_train": float(mae_phys_tr),
        "mae_physics_test": float(mae_phys_te),
        "mae_final_train": float(mae_final_tr),
        "mae_final_test": float(mae_final_te),
    }


def run_one_split_gnn(
    g1_batch, g2_batch, in_dim: int,
    T1: np.ndarray, T2: np.ndarray, x1: np.ndarray, Tm: np.ndarray,
    train_idx: np.ndarray, test_idx: np.ndarray,
    emb_cfg: Dict, phys_cfg: Dict, xgb_cfg: Dict,
    seed: int,
    out_dir: str,
    tag: str,
) -> Dict[str, float]:

    # slice graph batches by indices
    def slice_batch(batch, idx):
        Xb, Ab, mb = batch
        return (Xb[idx], Ab[idx], mb[idx])

    g1_tr = slice_batch(g1_batch, train_idx)
    g2_tr = slice_batch(g2_batch, train_idx)
    g1_te = slice_batch(g1_batch, test_idx)
    g2_te = slice_batch(g2_batch, test_idx)

    T1_tr, T1_te = T1[train_idx], T1[test_idx]
    T2_tr, T2_te = T2[train_idx], T2[test_idx]
    x1_tr, x1_te = x1[train_idx], x1[test_idx]
    Tm_tr, Tm_te = Tm[train_idx], Tm[test_idx]

    # Build GNN encoder + physics model
    gnn_hidden = int(emb_cfg.get("gnn_hidden_dim", 128))
    gnn_out = int(emb_cfg.get("gnn_out_dim", 256))
    gnn_layers = int(emb_cfg.get("gnn_layers", 3))
    gnn_dropout = float(emb_cfg.get("gnn_dropout", 0.1))
    encoder = MolGNNEncoder(in_dim=in_dim, hidden_dim=gnn_hidden, out_dim=gnn_out, layers=gnn_layers, dropout=gnn_dropout)

    model = GraphPhysicsModel(
        gnn_encoder=encoder,
        head_width=int(phys_cfg.get("mlp_width", 128)),
        head_depth=int(phys_cfg.get("mlp_depth", 2)),
        dropout=float(phys_cfg.get("dropout", 0.2)),
    )

    p_cfg = PhysicsTrainConfig(
        epochs=int(phys_cfg.get("epochs", 1500)),
        lr=float(phys_cfg.get("lr", 3e-3)),
        weight_decay=float(phys_cfg.get("weight_decay", 1e-5)),
        clip_grad_norm=float(phys_cfg.get("clip_grad_norm", 1.0)),
        anchor_w_lambda=float(phys_cfg.get("anchor_w_lambda", 1e-10)),
        device=str(phys_cfg.get("device", "cuda")),
        verbose_every=int(phys_cfg.get("verbose_every", 100)),
        checkpoint_path=os.path.join(out_dir, "physics_best.pt"),
        save_best=True,
        save_last=True,
    )

    model, hist = train_graph_physics(model, g1_tr, g2_tr, T1_tr, T2_tr, x1_tr, Tm_tr,
        g1_val=g1_te, g2_val=g2_te, T1_val=T1_te, T2_val=T2_te, x1_val=x1_te, Tm_val=Tm_te,
        cfg=p_cfg, show_progress=bool(phys_cfg.get("show_progress", False)))

    Tm_phys_te, d1_te, d2_te, W_te = predict_graph_physics(model, g1_te, g2_te, T1_te, T2_te, x1_te, device=p_cfg.device)
    Tm_phys_tr, d1_tr, d2_tr, W_tr = predict_graph_physics(model, g1_tr, g2_tr, T1_tr, T2_tr, x1_tr, device=p_cfg.device)

    mae_phys_tr = mean_absolute_error(Tm_tr, Tm_phys_tr)
    mae_phys_te = mean_absolute_error(Tm_te, Tm_phys_te)

    # Use learned graph embeddings too for boosting:
    # To keep implementation simple, reuse encoder outputs as features by running once.
    import torch
    from physics_learning import get_device
    dev = get_device(p_cfg.device)
    model = model.to(dev).eval()
    with torch.no_grad():
        X1te = torch.tensor(g1_te[0], dtype=torch.float32, device=dev)
        A1te = torch.tensor(g1_te[1], dtype=torch.float32, device=dev)
        m1te = torch.tensor(g1_te[2], dtype=torch.float32, device=dev)
        X2te = torch.tensor(g2_te[0], dtype=torch.float32, device=dev)
        A2te = torch.tensor(g2_te[1], dtype=torch.float32, device=dev)
        m2te = torch.tensor(g2_te[2], dtype=torch.float32, device=dev)
        e1_te = model.encoder(X1te, A1te, m1te).cpu().numpy()
        e2_te = model.encoder(X2te, A2te, m2te).cpu().numpy()

        X1tr = torch.tensor(g1_tr[0], dtype=torch.float32, device=dev)
        A1tr = torch.tensor(g1_tr[1], dtype=torch.float32, device=dev)
        m1tr = torch.tensor(g1_tr[2], dtype=torch.float32, device=dev)
        X2tr = torch.tensor(g2_tr[0], dtype=torch.float32, device=dev)
        A2tr = torch.tensor(g2_tr[1], dtype=torch.float32, device=dev)
        m2tr = torch.tensor(g2_tr[2], dtype=torch.float32, device=dev)
        e1_tr = model.encoder(X1tr, A1tr, m1tr).cpu().numpy()
        e2_tr = model.encoder(X2tr, A2tr, m2tr).cpu().numpy()

    resid_tr = (Tm_tr - Tm_phys_tr).astype(np.float32)
    Xb_tr = make_boost_features(e1_tr, e2_tr, x1_tr, T1_tr, T2_tr, d1_tr, d2_tr, W_tr, include_params=True)
    Xb_te = make_boost_features(e1_te, e2_te, x1_te, T1_te, T2_te, d1_te, d2_te, W_te, include_params=True)

    xgb_model = fit_xgb_residual_model(Xb_tr, resid_tr, xgb_cfg, seed=seed)
    xgb_model.save_model(os.path.join(out_dir, "xgb.json"))
    Tm_final_tr = Tm_phys_tr + xgb_model.predict(Xb_tr)
    Tm_final_te = Tm_phys_te + xgb_model.predict(Xb_te)
    mae_final_tr = mean_absolute_error(Tm_tr, Tm_final_tr)
    mae_final_te = mean_absolute_error(Tm_te, Tm_final_te)

    # Plots
    os.makedirs(out_dir, exist_ok=True)
    plot_loss(hist, out_png=os.path.join(out_dir, f"loss_{tag}.png"), title=f"Physics loss ({tag})")
    parity_plot(Tm_te, Tm_phys_te, out_png=os.path.join(out_dir, f"parity_phys_{tag}.png"), title=f"Physics parity ({tag})")
    parity_plot(Tm_te, Tm_final_te, out_png=os.path.join(out_dir, f"parity_final_{tag}.png"), title=f"Final parity ({tag})")
    residual_hist(Tm_te, Tm_final_te, out_png=os.path.join(out_dir, f"residual_{tag}.png"), title=f"Residuals ({tag})")
    xgb_feature_importance(xgb_model, out_png=os.path.join(out_dir, f"xgb_importance_{tag}.png"))
    try_shap_summary(xgb_model, Xb_te, out_png=os.path.join(out_dir, f"shap_{tag}.png"))

    return {
        "mae_physics_train": float(mae_phys_tr),
        "mae_physics_test": float(mae_phys_te),
        "mae_final_train": float(mae_final_tr),
        "mae_final_test": float(mae_final_te),
    }



def predict_from_artifacts(model_dir: str, input_csv: str, output_csv: str, config_path: str) -> None:
    """Load trained artifacts (physics model + XGBoost residual model) and predict on a new CSV."""
    cfg = load_yaml(config_path)
    emb_cfg = cfg["embedding"]
    method = emb_cfg["method"]
    device = str(cfg.get("device", "cuda"))
    df = pd.read_csv(input_csv)

    # Columns required to generate predictions (do not drop rows based on unrelated columns).
    required = ["Smiles#1", "Smiles#2", "X#1 (molar fraction)", "T#1", "T#2"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns for predict: {miss}")

    # Only drop rows missing the required predictor columns. External CSVs often have empty targets/metadata.
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(
            f"After dropping rows with NaNs in required columns {required}, the input has 0 rows. "
            "Check your CSV for missing values in these columns."
        )

    x1 = df["X#1 (molar fraction)"].to_numpy(dtype=np.float32)
    T1 = df["T#1"].to_numpy(dtype=np.float32)
    T2 = df["T#2"].to_numpy(dtype=np.float32)

    # embeddings / graphs
    if method in ("chemberta", "morgan"):
        X1_emb, X2_emb, _ = prepare_fixed_embeddings(df, cfg, device=device)
        phys_path = os.path.join(model_dir, "physics_best.pt")
        model = load_physics_siamese_checkpoint(phys_path, device=device)
        Tm_phys, d1, d2, W = predict_physics_siamese(model, X1_emb, X2_emb, T1, T2, x1, device=device)

        Xb = make_boost_features(X1_emb, X2_emb, x1, T1, T2, d1, d2, W, include_params=True)

    elif method == "gnn":
        g1_batch, g2_batch, in_dim = prepare_graph_batches(df)
        phys_path = os.path.join(model_dir, "physics_best.pt")
        model = load_graph_physics_checkpoint(
            phys_path,
            in_dim=in_dim,
            gnn_hidden_dim=int(emb_cfg.get("gnn_hidden_dim", 128)),
            gnn_out_dim=int(emb_cfg.get("gnn_out_dim", 256)),
            gnn_layers=int(emb_cfg.get("gnn_layers", 3)),
            gnn_dropout=float(emb_cfg.get("gnn_dropout", 0.1)),
            head_width=int(cfg["physics_learning"].get("mlp_width", 128)),
            head_depth=int(cfg["physics_learning"].get("mlp_depth", 2)),
            head_dropout=float(cfg["physics_learning"].get("dropout", 0.2)),
            device=device,
        )
        Tm_phys, d1, d2, W = predict_graph_physics(model, g1_batch, g2_batch, T1, T2, x1, device=device)

        # also get embeddings for residual model
        import torch
        from physics_learning import get_device as _gd
        dev = _gd(device)
        model = model.to(dev).eval()
        with torch.no_grad():
            X1t = torch.tensor(g1_batch[0], dtype=torch.float32, device=dev)
            A1t = torch.tensor(g1_batch[1], dtype=torch.float32, device=dev)
            m1t = torch.tensor(g1_batch[2], dtype=torch.float32, device=dev)
            X2t = torch.tensor(g2_batch[0], dtype=torch.float32, device=dev)
            A2t = torch.tensor(g2_batch[1], dtype=torch.float32, device=dev)
            m2t = torch.tensor(g2_batch[2], dtype=torch.float32, device=dev)
            e1 = model.encoder(X1t, A1t, m1t).cpu().numpy()
            e2 = model.encoder(X2t, A2t, m2t).cpu().numpy()

        Xb = make_boost_features(e1, e2, x1, T1, T2, d1, d2, W, include_params=True)

    else:
        raise ValueError(f"Unknown embedding method: {method}")

    # load XGB residual model
    xgb_path = os.path.join(model_dir, "xgb.json")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_path)

    resid = xgb_model.predict(Xb).astype(np.float32)
    Tm_final = (Tm_phys + resid).astype(np.float32)
    # Optional: compute MAE if ground truth is present in the input CSV
    y_col_candidates = ["Tmelt, K", "Tmelt", "Tm", "Tmelt_K", "Tmelt(K)"]
    y_col = next((c for c in y_col_candidates if c in df.columns), None)
    metrics = None
    if y_col is not None:
        y_true = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=np.float32)
        mask = np.isfinite(y_true)
        if mask.sum() > 0:
            mae_phys = float(mean_absolute_error(y_true[mask], Tm_phys[mask]))
            mae_final = float(mean_absolute_error(y_true[mask], Tm_final[mask]))
            metrics = {
                "label_column": y_col,
                "n_with_labels": int(mask.sum()),
                "mae_physics": mae_phys,
                "mae_final": mae_final,
            }
            print(f"[predict] MAE physics ({y_col}) = {mae_phys:.4f} K over n={int(mask.sum())}")
            print(f"[predict] MAE final   ({y_col}) = {mae_final:.4f} K over n={int(mask.sum())}")


    df_out = df.copy()
    df_out["Tm_phys_pred"] = Tm_phys
    df_out["Tm_final_pred"] = Tm_final
    df_out.to_csv(output_csv, index=False)
    if metrics is not None:
        metrics_path = os.path.splitext(output_csv)[0] + ".metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved prediction metrics to: {metrics_path}")

    print(f"Saved predictions to: {output_csv}")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--mode", type=str, default="train", choices=["train", "predict"])
    ap.add_argument("--model_dir", type=str, default=None, help="For predict: path to an artifacts directory created during training")
    ap.add_argument("--input_csv", type=str, default=None, help="For predict: CSV to run inference on")
    ap.add_argument("--output_csv", type=str, default="predictions.csv", help="For predict: where to write predictions")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    if args.mode == "predict":
        if not args.model_dir or not args.input_csv:
            raise ValueError("For --mode predict, you must pass --model_dir and --input_csv")
        predict_from_artifacts(args.model_dir, args.input_csv, args.output_csv, args.config)
        return

    seed = int(cfg.get("seed", 42))
    seed_all(seed)

    data_path = cfg["data"]["path"]
    out_dir = cfg.get("output_dir", "runs")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path).dropna().reset_index(drop=True)

    required = ["Smiles#1", "Smiles#2", "X#1 (molar fraction)", "T#1", "T#2", "Tmelt, K"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    x1 = df["X#1 (molar fraction)"].to_numpy(dtype=np.float32)
    T1 = df["T#1"].to_numpy(dtype=np.float32)
    T2 = df["T#2"].to_numpy(dtype=np.float32)
    Tm = df["Tmelt, K"].to_numpy(dtype=np.float32)

    emb_cfg = cfg["embedding"]
    method = emb_cfg["method"]
    device = str(cfg.get("device", "cuda"))

    # prepare embeddings / graphs
    if method in ("chemberta", "morgan"):
        X1_emb, X2_emb, emb_dim = prepare_fixed_embeddings(df, cfg, device=device)
    elif method == "gnn":
        g1_batch, g2_batch, in_dim = prepare_graph_batches(df)
        X1_emb = X2_emb = None
    else:
        raise ValueError(f"Unknown embedding method: {method}")

    # Create pair groups for strict split (uses canonical smiles computed above)
    if "can_smiles1" not in df.columns:
        df["can_smiles1"] = canonicalize_series(df["Smiles#1"].tolist())
        df["can_smiles2"] = canonicalize_series(df["Smiles#2"].tolist())
    groups = df.apply(lambda r: make_pair_id(r["can_smiles1"], r["can_smiles2"]), axis=1).to_numpy()

    split_cfg = cfg["split"]
    test_size = float(split_cfg.get("test_size", 0.2))
    n_runs = int(split_cfg.get("runs", 10))
    if n_runs < 1:
        raise ValueError(f"split.runs must be >= 1, got {n_runs}")

    phys_cfg = cfg["physics_learning"]
    xgb_cfg = cfg["xgboost"]

    metrics = {
        "embedding_method": method,
        "split": {},
        "notes": {
            "random_rows": "random split of rows; a pair can appear in both train/test with different ratios",
            "strict_pairs": "group split by orderless solvent pair; no pair overlap between train/test",
        },
    }

    def do_protocol(protocol: str) -> Dict[str, Dict[str, float]]:
        maes_phys_tr, maes_phys_te = [], []
        maes_final_tr, maes_final_te = [], []
        for i in range(n_runs):
            split_seed = seed + i * 17
            n = len(df)
            idx = np.arange(n)

            if protocol == "random_rows":
                tr, te = train_test_split(idx, test_size=test_size, random_state=split_seed, shuffle=True)
            elif protocol == "strict_pairs":
                gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=split_seed)
                tr, te = next(gss.split(idx, groups=groups))
            else:
                raise ValueError(protocol)

            tag = f"{protocol}_run{i+1}"
            if method in ("chemberta", "morgan"):
                out = run_one_split_fixed(
                    X1_emb, X2_emb, T1, T2, x1, Tm,
                    train_idx=tr, test_idx=te,
                    phys_cfg=phys_cfg, xgb_cfg=xgb_cfg,
                    seed=split_seed,
                    out_dir=os.path.join(out_dir, protocol, tag),
                    tag=tag,
                )
            else:
                out = run_one_split_gnn(
                    g1_batch, g2_batch, in_dim,
                    T1, T2, x1, Tm,
                    train_idx=tr, test_idx=te,
                    emb_cfg=emb_cfg, phys_cfg=phys_cfg, xgb_cfg=xgb_cfg,
                    seed=split_seed,
                    out_dir=os.path.join(out_dir, protocol, tag),
                    tag=tag,
                )

            maes_phys_tr.append(out["mae_physics_train"])
            maes_phys_te.append(out["mae_physics_test"])
            maes_final_tr.append(out["mae_final_train"])
            maes_final_te.append(out["mae_final_test"])
            print(
                f"[{tag}] MAE physics train={out['mae_physics_train']:.2f} K, test={out['mae_physics_test']:.2f} K | "
                f"MAE final train={out['mae_final_train']:.2f} K, test={out['mae_final_test']:.2f} K"
            )

        return {
            "mae_physics_train_mean": float(np.mean(maes_phys_tr)),
            "mae_physics_train_std": float(np.std(maes_phys_tr)),
            "mae_physics_test_mean": float(np.mean(maes_phys_te)),
            "mae_physics_test_std": float(np.std(maes_phys_te)),
            "mae_final_train_mean": float(np.mean(maes_final_tr)),
            "mae_final_train_std": float(np.std(maes_final_tr)),
            "mae_final_test_mean": float(np.mean(maes_final_te)),
            "mae_final_test_std": float(np.std(maes_final_te)),
        }

    # Run both protocols requested
    metrics["split"]["random_rows"] = do_protocol("random_rows")
    rr = metrics["split"]["random_rows"]
    print(f"\n[random_rows] Average MAE physics train = {rr['mae_physics_train_mean']:.2f} ± {rr['mae_physics_train_std']:.2f} K")
    print(f"[random_rows] Average MAE physics test  = {rr['mae_physics_test_mean']:.2f} ± {rr['mae_physics_test_std']:.2f} K")
    print(f"[random_rows] Average MAE final train   = {rr['mae_final_train_mean']:.2f} ± {rr['mae_final_train_std']:.2f} K")
    print(f"[random_rows] Average MAE final test    = {rr['mae_final_test_mean']:.2f} ± {rr['mae_final_test_std']:.2f} K")

    metrics["split"]["strict_pairs"] = do_protocol("strict_pairs")
    sp = metrics["split"]["strict_pairs"]
    print(f"\n[strict_pairs] Average MAE physics train = {sp['mae_physics_train_mean']:.2f} ± {sp['mae_physics_train_std']:.2f} K")
    print(f"[strict_pairs] Average MAE physics test  = {sp['mae_physics_test_mean']:.2f} ± {sp['mae_physics_test_std']:.2f} K")
    print(f"[strict_pairs] Average MAE final train   = {sp['mae_final_train_mean']:.2f} ± {sp['mae_final_train_std']:.2f} K")
    print(f"[strict_pairs] Average MAE final test    = {sp['mae_final_test_mean']:.2f} ± {sp['mae_final_test_std']:.2f} K")

    # Save
    out_json = os.path.join(out_dir, "metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {out_json}")


if __name__ == "__main__":
    main()
