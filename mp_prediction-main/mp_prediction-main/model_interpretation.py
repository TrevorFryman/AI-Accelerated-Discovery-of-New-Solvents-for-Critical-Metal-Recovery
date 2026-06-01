#!/usr/bin/env python3
"""
Model interpretation / monitoring utilities:
- training loss curves
- parity plots (true vs pred)
- residual histograms
- XGBoost feature importance plots
- optional SHAP (if installed)

This is intentionally lightweight and "necessary plots"-focused.
"""

from __future__ import annotations

from typing import Dict, Optional

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(
    history: Dict[str, list],
    out_png: Optional[str] = None,
    title: str = "Training loss",
    *,
    robust: bool = True,
    warmup: int = 0,
    yscale: str = "auto",  # "auto" | "linear" | "log"
) -> None:
    """Plot loss vs epoch (train + test/val).

    Supports histories produced by physics_learning.py:
      - train_loss: list[float]
      - test_loss : list[float] (optional; may contain NaNs)
    Backward compatible with:
      - loss: list[float]
    """
    train = history.get("train_loss", history.get("loss", []))
    test = history.get("test_loss", [])

    if not train:
        return

    y_tr = np.asarray(train, dtype=float)
    x = np.arange(1, len(y_tr) + 1)

    y_te = None
    if isinstance(test, list) and len(test) == len(y_tr):
        y_te = np.asarray(test, dtype=float)

    plt.figure()
    plt.plot(x, y_tr, label="train")
    if y_te is not None:
        plt.plot(x, y_te, label="test")
        plt.legend()

    plt.xlabel("Epoch")
    plt.ylabel("L1 loss")
    plt.title(title)

    def _auto_set_log(y: np.ndarray) -> None:
        yy = y[max(0, warmup):]
        yy = yy[np.isfinite(yy)]
        if len(yy) > 0 and np.all(yy > 0):
            r = float(np.nanmax(yy) / max(float(np.nanmin(yy)), 1e-12))
            if r > 50:
                plt.yscale("log")

    if yscale == "auto":
        _auto_set_log(y_tr)
        if y_te is not None:
            _auto_set_log(y_te)
    elif yscale in ("log", "linear"):
        if yscale == "log":
            plt.yscale("log")

    if robust:
        ys = [y_tr]
        if y_te is not None:
            ys.append(y_te)
        yy = np.concatenate([y[max(0, warmup):] for y in ys])
        yy = yy[np.isfinite(yy)]
        if len(yy) > 5:
            top = float(np.nanquantile(yy, 0.95))
            bottom = float(np.nanquantile(yy, 0.05))
            y0 = max(0.0, bottom * 0.9)
            y1 = max(top * 1.1, y0 + 1e-6)
            if plt.gca().get_yscale() == "linear":
                plt.ylim(y0, y1)

    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def parity_plot(y_true: np.ndarray, y_pred: np.ndarray, out_png: Optional[str] = None, title: str = "Parity plot") -> None:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    plt.figure()
    plt.scatter(y_true, y_pred, s=12, alpha=0.7)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("True Tm (K)")
    plt.ylabel("Pred Tm (K)")
    plt.title(title)
    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def residual_hist(y_true: np.ndarray, y_pred: np.ndarray, out_png: Optional[str] = None, title: str = "Residuals (pred-true)") -> None:
    res = np.asarray(y_pred).ravel() - np.asarray(y_true).ravel()
    plt.figure()
    plt.hist(res, bins=30)
    plt.xlabel("Residual (K)")
    plt.ylabel("Count")
    plt.title(title)
    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def xgb_feature_importance(xgb_model, out_png: Optional[str] = None, title: str = "XGBoost feature importance (gain)") -> None:
    booster = xgb_model.get_booster()
    score = booster.get_score(importance_type="gain")
    if not score:
        return
    keys = list(score.keys())
    vals = np.array([score[k] for k in keys], dtype=float)
    order = np.argsort(vals)[::-1][:30]  # top 30
    keys = [keys[i] for i in order]
    vals = vals[order]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(keys))[::-1], vals, tick_label=keys[::-1])
    plt.xlabel("Gain")
    plt.title(title)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def try_shap_summary(xgb_model, X: np.ndarray, out_png: Optional[str] = None, max_display: int = 20) -> bool:
    """Optional SHAP summary plot.

    SHAP + XGBoost sometimes breaks depending on library versions (e.g. parsing base_score).
    This helper must *never* crash the pipeline; it should quietly skip on any error.
    Returns True if plotted, else False.
    """
    try:
        import shap  # type: ignore
    except Exception:
        return False

    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X)
        plt.figure()
        shap.summary_plot(shap_values, X, show=False, max_display=max_display)
        if out_png:
            plt.savefig(out_png, dpi=200, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
        return True
    except Exception:
        try:
            plt.close()
        except Exception:
            pass
        return False


