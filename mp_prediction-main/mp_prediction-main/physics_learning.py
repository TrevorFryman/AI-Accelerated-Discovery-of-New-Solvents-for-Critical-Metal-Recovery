#!/usr/bin/env python3
"""
physics_learning_v2.py

Physics-learning module for DES melting point prediction.

Implements:
- ParamSiameseNet: MLP Siamese head mapping (emb1, emb2) -> (d1, d2, W)
- GraphPhysicsModel: wraps a GNN encoder + ParamSiameseNet for end-to-end training
- train/predict helpers for both fixed-embedding and graph-embedding modes
- checkpoint save/load helpers

The physics forward model is a Schröder–van Laar style equation with a simple
1-parameter Margules activity model.

This module is intentionally lightweight and has no dataset/IO code; pipeline.py
handles data preparation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------
# Repro / device
# ----------------------------
def seed_all(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "cuda") -> torch.device:
    if prefer.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")



def _ensure_parent_dir(path: str) -> None:
    """Create parent directory for a file path if needed."""
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

# ----------------------------
# Physics core
# ----------------------------
def physics_forward(
    d1: torch.Tensor,
    d2: torch.Tensor,
    W: torch.Tensor,
    T1: torch.Tensor,
    T2: torch.Tensor,
    x1: torch.Tensor,
    R: float = 8.314,
) -> torch.Tensor:
    """
    Schröder–van Laar with a simple 1-parameter Margules (symmetric) activity model:
        ln(gamma1) = (W/(R*Tref))*(1-x1)^2
        ln(gamma2) = (W/(R*Tref))*(x1)^2

    Args:
        d1, d2, W: (N,) learned parameters
        T1, T2: (N,) component melting points (K)
        x1: (N,) molar fraction of component 1

    Returns:
        Tm_pred: (N,)
    """
    eps = 1e-8
    x1c = torch.clamp(x1, min=eps, max=1.0 - eps)
    Tref = (T1 + T2) / 2.0

    ln_gamma1 = (W / (R * Tref)) * (1 - x1c) ** 2
    ln_gamma2 = (W / (R * Tref)) * (x1c) ** 2

    ln_a1 = torch.log(x1c) + ln_gamma1
    ln_a2 = torch.log(1 - x1c) + ln_gamma2

    # keep denominators away from 0 to avoid explosions
    denom1 = torch.clamp(1.0 - (R / d1) * ln_a1, min=0.1)
    denom2 = torch.clamp(1.0 - (R / d2) * ln_a2, min=0.1)

    return torch.max(T1 / denom1, T2 / denom2)


def l1_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_pred - y_true))


# ----------------------------
# Models
# ----------------------------
class ParamSiameseNet(nn.Module):
    """Siamese head mapping two embeddings -> thermodynamic parameters (d1, d2, W)."""

    def __init__(self, emb_dim: int, width: int = 128, depth: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.width = int(width)
        self.depth = int(depth)
        self.dropout = float(dropout)

        layers: List[nn.Module] = []
        in_dim = self.emb_dim * 2
        for i in range(self.depth):
            layers.append(nn.Linear(in_dim if i == 0 else self.width, self.width))
            layers.append(nn.ReLU())
            if self.dropout and self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
        self.trunk = nn.Sequential(*layers)

        # 3 outputs: d1, d2, W
        self.out = nn.Linear(self.width, 3)

        # initialize to avoid extreme early outputs
        nn.init.zeros_(self.out.bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.cat([x1, x2], dim=-1)
        h = self.trunk(h)
        y = self.out(h)

        # constrain to reasonable ranges:
        # d1,d2 must be positive; W can be any but regularized in training
        d1 = torch.nn.functional.softplus(y[:, 0]) + 1.0
        d2 = torch.nn.functional.softplus(y[:, 1]) + 1.0
        W = y[:, 2]
        return d1, d2, W


class GraphPhysicsModel(nn.Module):
    """GNN encoder + ParamSiameseNet head for end-to-end training."""

    def __init__(
        self,
        gnn_encoder: nn.Module,
        *,
        head_width: int = 128,
        head_depth: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = gnn_encoder
        self._head: Optional[ParamSiameseNet] = None
        self._head_width = int(head_width)
        self._head_depth = int(head_depth)
        self._head_dropout = float(dropout)

    def build_head(self, emb_dim: int) -> None:
        self._head = ParamSiameseNet(
            emb_dim=int(emb_dim),
            width=self._head_width,
            depth=self._head_depth,
            dropout=self._head_dropout,
        )

    @property
    def head_params(self) -> Dict[str, float]:
        if self._head is None:
            return {"emb_dim": -1, "mlp_width": self._head_width, "mlp_depth": self._head_depth, "dropout": self._head_dropout}
        return {
            "emb_dim": int(self._head.emb_dim),
            "mlp_width": int(self._head.width),
            "mlp_depth": int(self._head.depth),
            "dropout": float(self._head.dropout),
        }

    def forward(
        self,
        g1: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        g2: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._head is None:
            raise RuntimeError("GraphPhysicsModel head not built. Call build_head(emb_dim) first.")
        X1, A1, m1 = g1
        X2, A2, m2 = g2
        e1 = self.encoder(X1, A1, m1)
        e2 = self.encoder(X2, A2, m2)
        return self._head(e1, e2)


# ----------------------------
# Training config
# ----------------------------
@dataclass
class PhysicsTrainConfig:
    epochs: int = 1500
    lr: float = 1e-3
    weight_decay: float = 0.0
    clip_grad_norm: float = 5.0
    anchor_w_lambda: float = 1e-6
    device: str = "cuda"
    verbose_every: int = 100

    # checkpointing
    checkpoint_path: str = ""  # set by pipeline
    save_best: bool = True
    save_last: bool = True


# ----------------------------
# Checkpoint helpers
# ----------------------------
def infer_molgnn_params(encoder: nn.Module) -> Dict[str, int]:
    """Best-effort extraction of MolGNNEncoder architecture for reload."""
    params: Dict[str, int] = {}
    try:
        in_proj = getattr(encoder, "in_proj", None)
        if in_proj is not None and hasattr(in_proj, "in_features") and hasattr(in_proj, "out_features"):
            params["in_dim"] = int(in_proj.in_features)
            params["hidden_dim"] = int(in_proj.out_features)
        mlps = getattr(encoder, "mlps", None)
        if mlps is not None:
            params["layers"] = int(len(mlps))
        # out_proj is often a Sequential
        out_proj = getattr(encoder, "out_proj", None)
        if out_proj is not None:
            if isinstance(out_proj, nn.Sequential):
                # first Linear usually maps hidden->out
                for m in out_proj:
                    if isinstance(m, nn.Linear):
                        params["out_dim"] = int(m.out_features)
                        break
            elif isinstance(out_proj, nn.Linear):
                params["out_dim"] = int(out_proj.out_features)
    except Exception:
        return params
    return params


# ----------------------------
# Fixed-embedding training / prediction
# ----------------------------
def train_physics_siamese(
    X1: np.ndarray,
    X2: np.ndarray,
    T1: np.ndarray,
    T2: np.ndarray,
    x1: np.ndarray,
    Tm: np.ndarray,
    # Optional validation / test split for monitoring loss curves
    X1_val: Optional[np.ndarray] = None,
    X2_val: Optional[np.ndarray] = None,
    T1_val: Optional[np.ndarray] = None,
    T2_val: Optional[np.ndarray] = None,
    x1_val: Optional[np.ndarray] = None,
    Tm_val: Optional[np.ndarray] = None,
    *,
    emb_dim: int,
    mlp_width: int = 128,
    mlp_depth: int = 2,
    dropout: float = 0.2,
    cfg: PhysicsTrainConfig = PhysicsTrainConfig(),
    show_progress: bool = False,
) -> Tuple[ParamSiameseNet, Dict[str, List[float]]]:
    device = get_device(cfg.device)

    model = ParamSiameseNet(emb_dim=int(emb_dim), width=int(mlp_width), depth=int(mlp_depth), dropout=float(dropout)).to(device)

    X1t = torch.tensor(X1, dtype=torch.float32, device=device)
    X2t = torch.tensor(X2, dtype=torch.float32, device=device)
    T1t = torch.tensor(T1, dtype=torch.float32, device=device).view(-1)
    T2t = torch.tensor(T2, dtype=torch.float32, device=device).view(-1)
    x1t = torch.tensor(x1, dtype=torch.float32, device=device).view(-1)
    Tmt = torch.tensor(Tm, dtype=torch.float32, device=device).view(-1)

    # validation tensors (optional)
    has_val = (X1_val is not None) and (X2_val is not None) and (T1_val is not None) and (T2_val is not None) and (x1_val is not None) and (Tm_val is not None)
    if has_val:
        X1v = torch.tensor(X1_val, dtype=torch.float32, device=device)
        X2v = torch.tensor(X2_val, dtype=torch.float32, device=device)
        T1v = torch.tensor(T1_val, dtype=torch.float32, device=device).view(-1)
        T2v = torch.tensor(T2_val, dtype=torch.float32, device=device).view(-1)
        x1v = torch.tensor(x1_val, dtype=torch.float32, device=device).view(-1)
        Tmv = torch.tensor(Tm_val, dtype=torch.float32, device=device).view(-1)

    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    hist: Dict[str, List[float]] = {"train_loss": [], "test_loss": [], "loss": []}
    best_loss = float("inf")

    for ep in range(cfg.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)

        d1, d2, W = model(X1t, X2t)
        Tm_pred = physics_forward(d1, d2, W, T1t, T2t, x1t)
        loss = l1_loss(Tm_pred, Tmt) + cfg.anchor_w_lambda * torch.mean(W ** 2)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
        opt.step()

        cur_loss = float(loss.detach().cpu().item())
        hist["train_loss"].append(cur_loss)
        hist["loss"].append(cur_loss)

        # test/val loss for monitoring
        if has_val:
            model.eval()
            with torch.no_grad():
                d1v, d2v, Wv = model(X1v, X2v)
                Tm_v_pred = physics_forward(d1v, d2v, Wv, T1v, T2v, x1v)
                vloss = l1_loss(Tm_v_pred, Tmv) + cfg.anchor_w_lambda * torch.mean(Wv ** 2)
            hist["test_loss"].append(float(vloss.detach().cpu().item()))
        else:
            hist["test_loss"].append(float("nan"))

        if cfg.checkpoint_path:
            # save best
            if cfg.save_best and cur_loss < best_loss - 1e-12:
                best_loss = cur_loss
                _ensure_parent_dir(cfg.checkpoint_path)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "emb_dim": int(emb_dim),
                        "mlp_width": int(mlp_width),
                        "mlp_depth": int(mlp_depth),
                        "dropout": float(dropout),
                        "cfg": cfg.__dict__,
                        "epoch": ep,
                        "loss": cur_loss,
                    },
                    cfg.checkpoint_path,
                )

        if show_progress and (ep % cfg.verbose_every == 0):
            print(f"Epoch {ep:4d} | loss={cur_loss:.4f} | W(med)={float(W.median().detach().cpu()):.3f}")

    if cfg.checkpoint_path and cfg.save_last:
        last_path = cfg.checkpoint_path.replace(".pt", "_last.pt")
        _ensure_parent_dir(last_path)
        torch.save(
            {
                "model_state": model.state_dict(),
                "emb_dim": int(emb_dim),
                "mlp_width": int(mlp_width),
                "mlp_depth": int(mlp_depth),
                "dropout": float(dropout),
                "cfg": cfg.__dict__,
                "epoch": cfg.epochs - 1,
                "loss": hist["loss"][-1] if hist["loss"] else None,
            },
            last_path,
        )

    return model, hist


@torch.no_grad()
def predict_physics_siamese(
    model: ParamSiameseNet,
    X1: np.ndarray,
    X2: np.ndarray,
    T1: np.ndarray,
    T2: np.ndarray,
    x1: np.ndarray,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dev = get_device(device)
    model = model.to(dev)
    model.eval()

    X1t = torch.tensor(X1, dtype=torch.float32, device=dev)
    X2t = torch.tensor(X2, dtype=torch.float32, device=dev)
    T1t = torch.tensor(T1, dtype=torch.float32, device=dev).view(-1)
    T2t = torch.tensor(T2, dtype=torch.float32, device=dev).view(-1)
    x1t = torch.tensor(x1, dtype=torch.float32, device=dev).view(-1)

    d1, d2, W = model(X1t, X2t)
    Tm_pred = physics_forward(d1, d2, W, T1t, T2t, x1t)

    return (
        Tm_pred.detach().cpu().numpy(),
        d1.detach().cpu().numpy(),
        d2.detach().cpu().numpy(),
        W.detach().cpu().numpy(),
    )


# ----------------------------
# Graph-embedding training / prediction
# ----------------------------
def train_graph_physics(
    model: GraphPhysicsModel,
    g1: Tuple[np.ndarray, np.ndarray, np.ndarray],
    g2: Tuple[np.ndarray, np.ndarray, np.ndarray],
    T1: np.ndarray,
    T2: np.ndarray,
    x1: np.ndarray,
    Tm: np.ndarray,
    # Optional validation / test split for monitoring loss curves
    g1_val: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    g2_val: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    T1_val: Optional[np.ndarray] = None,
    T2_val: Optional[np.ndarray] = None,
    x1_val: Optional[np.ndarray] = None,
    Tm_val: Optional[np.ndarray] = None,
    cfg: PhysicsTrainConfig = PhysicsTrainConfig(),
    show_progress: bool = False,
) -> Tuple[GraphPhysicsModel, Dict[str, List[float]]]:
    device = get_device(cfg.device)
    model = model.to(device)

    X1b, A1b, m1b = [torch.tensor(t, dtype=torch.float32, device=device) for t in g1]
    X2b, A2b, m2b = [torch.tensor(t, dtype=torch.float32, device=device) for t in g2]
    T1t = torch.tensor(T1, dtype=torch.float32, device=device).view(-1)
    T2t = torch.tensor(T2, dtype=torch.float32, device=device).view(-1)
    x1t = torch.tensor(x1, dtype=torch.float32, device=device).view(-1)
    Tmt = torch.tensor(Tm, dtype=torch.float32, device=device).view(-1)

    # validation tensors (optional)
    has_val = (g1_val is not None) and (g2_val is not None) and (T1_val is not None) and (T2_val is not None) and (x1_val is not None) and (Tm_val is not None)
    if has_val:
        X1v, A1v, m1v = [torch.tensor(t, dtype=torch.float32, device=device) for t in g1_val]
        X2v, A2v, m2v = [torch.tensor(t, dtype=torch.float32, device=device) for t in g2_val]
        T1v = torch.tensor(T1_val, dtype=torch.float32, device=device).view(-1)
        T2v = torch.tensor(T2_val, dtype=torch.float32, device=device).view(-1)
        x1v = torch.tensor(x1_val, dtype=torch.float32, device=device).view(-1)
        Tmv = torch.tensor(Tm_val, dtype=torch.float32, device=device).view(-1)

    # lazily build head once we know embedding dim
    if model._head is None:
        with torch.no_grad():
            e_dummy = model.encoder(X1b[:1], A1b[:1], m1b[:1])
        model.build_head(int(e_dummy.shape[-1]))
        model = model.to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    hist: Dict[str, List[float]] = {"train_loss": [], "test_loss": [], "loss": []}
    best_loss = float("inf")

    for ep in range(cfg.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)

        d1, d2, W = model((X1b, A1b, m1b), (X2b, A2b, m2b))
        Tm_pred = physics_forward(d1, d2, W, T1t, T2t, x1t)
        loss = l1_loss(Tm_pred, Tmt) + cfg.anchor_w_lambda * torch.mean(W ** 2)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
        opt.step()

        cur_loss = float(loss.detach().cpu().item())
        hist["train_loss"].append(cur_loss)
        hist["loss"].append(cur_loss)

        # test/val loss for monitoring
        if has_val:
            model.eval()
            with torch.no_grad():
                d1v, d2v, Wv = model((X1v, A1v, m1v), (X2v, A2v, m2v))
                Tm_v_pred = physics_forward(d1v, d2v, Wv, T1v, T2v, x1v)
                vloss = l1_loss(Tm_v_pred, Tmv) + cfg.anchor_w_lambda * torch.mean(Wv ** 2)
            hist["test_loss"].append(float(vloss.detach().cpu().item()))
        else:
            hist["test_loss"].append(float("nan"))

        if cfg.checkpoint_path:
            if cfg.save_best and cur_loss < best_loss - 1e-12:
                best_loss = cur_loss
                _ensure_parent_dir(cfg.checkpoint_path)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "head_params": model.head_params,
                        "encoder_params": infer_molgnn_params(model.encoder),
                        "cfg": cfg.__dict__,
                        "epoch": ep,
                        "loss": cur_loss,
                    },
                    cfg.checkpoint_path,
                )

        if show_progress and (ep % cfg.verbose_every == 0):
            print(f"Epoch {ep:4d} | loss={cur_loss:.4f} | W(med)={float(W.median().detach().cpu()):.3f}")

    if cfg.checkpoint_path and cfg.save_last:
        last_path = cfg.checkpoint_path.replace(".pt", "_last.pt")
        _ensure_parent_dir(last_path)
        torch.save(
            {
                "model_state": model.state_dict(),
                "head_params": model.head_params,
                "encoder_params": infer_molgnn_params(model.encoder),
                "cfg": cfg.__dict__,
                "epoch": cfg.epochs - 1,
                "loss": hist["loss"][-1] if hist["loss"] else None,
            },
            last_path,
        )

    return model, hist


@torch.no_grad()
def predict_graph_physics(
    model: GraphPhysicsModel,
    g1: Tuple[np.ndarray, np.ndarray, np.ndarray],
    g2: Tuple[np.ndarray, np.ndarray, np.ndarray],
    T1: np.ndarray,
    T2: np.ndarray,
    x1: np.ndarray,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dev = get_device(device)
    model = model.to(dev)
    model.eval()

    X1b, A1b, m1b = [torch.tensor(t, dtype=torch.float32, device=dev) for t in g1]
    X2b, A2b, m2b = [torch.tensor(t, dtype=torch.float32, device=dev) for t in g2]
    T1t = torch.tensor(T1, dtype=torch.float32, device=dev).view(-1)
    T2t = torch.tensor(T2, dtype=torch.float32, device=dev).view(-1)
    x1t = torch.tensor(x1, dtype=torch.float32, device=dev).view(-1)

    d1, d2, W = model((X1b, A1b, m1b), (X2b, A2b, m2b))
    Tm_pred = physics_forward(d1, d2, W, T1t, T2t, x1t)

    return (
        Tm_pred.detach().cpu().numpy(),
        d1.detach().cpu().numpy(),
        d2.detach().cpu().numpy(),
        W.detach().cpu().numpy(),
    )


# ----------------------------
# Loading checkpoints (for reuse / inference)
# ----------------------------
def load_physics_siamese_checkpoint(path: str, device: str = "cpu") -> ParamSiameseNet:
    """Load a fixed-embedding physics model checkpoint saved by train_physics_siamese."""
    ckpt = torch.load(path, map_location=get_device(device))
    model = ParamSiameseNet(
        emb_dim=int(ckpt["emb_dim"]),
        width=int(ckpt.get("mlp_width", 128)),
        depth=int(ckpt.get("mlp_depth", 2)),
        dropout=float(ckpt.get("dropout", 0.2)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(get_device(device))
    model.eval()
    return model


def load_graph_physics_checkpoint(
    path: str,
    *,
    in_dim: int,
    gnn_hidden_dim: int,
    gnn_out_dim: int,
    gnn_layers: int,
    gnn_dropout: float,
    head_width: int,
    head_depth: int,
    head_dropout: float,
    device: str = "cpu",
) -> GraphPhysicsModel:
    """Load a GraphPhysicsModel checkpoint saved by train_graph_physics."""
    from embedding import MolGNNEncoder  # local import to avoid circular

    dev = get_device(device)
    encoder = MolGNNEncoder(
        in_dim=int(in_dim),
        hidden_dim=int(gnn_hidden_dim),
        out_dim=int(gnn_out_dim),
        layers=int(gnn_layers),
        dropout=float(gnn_dropout),
    )
    model = GraphPhysicsModel(
        gnn_encoder=encoder,
        head_width=int(head_width),
        head_depth=int(head_depth),
        dropout=float(head_dropout),
    )

    ckpt = torch.load(path, map_location=dev)
    emb_dim = int(ckpt.get("head_params", {}).get("emb_dim", gnn_out_dim))
    model.build_head(emb_dim)
    model.load_state_dict(ckpt["model_state"])
    model.to(dev).eval()
    return model
