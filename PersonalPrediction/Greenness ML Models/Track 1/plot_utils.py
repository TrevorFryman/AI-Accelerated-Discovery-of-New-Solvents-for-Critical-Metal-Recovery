"""
plot_utils.py
=============
Shared dark-theme plotting utilities for the GSK Solvent G-Score Prediction project.
Ensures premium visual aesthetics (harmony, micro-animations/transitions in style,
proper formatting, clear titles, labels, and legends) across all model pipelines.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import shap

# ---------------------------------------------------------------------------
# Visual Styling Config
# ---------------------------------------------------------------------------
def apply_dark_style():
    """
    Apply a consistent, high-fidelity dark style to matplotlib.
    """
    plt.style.use("dark_background")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Clean"]
    
    # Theme colors
    plt.rcParams["figure.facecolor"] = "#0D0D11"  # Very dark slate-gray
    plt.rcParams["axes.facecolor"] = "#16161D"    # Slightly lighter card color
    plt.rcParams["grid.color"] = "#2A2A35"        # Subtle grid lines
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.5
    
    # Border & axis styling
    plt.rcParams["axes.edgecolor"] = "#3E3E4F"
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.color"] = "#B0B0C3"
    plt.rcParams["ytick.color"] = "#B0B0C3"
    plt.rcParams["axes.labelcolor"] = "#FFFFFF"
    plt.rcParams["text.color"] = "#FFFFFF"
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.facecolor"] = "#16161D"
    plt.rcParams["legend.edgecolor"] = "#3E3E4F"

# Theme Color Palette
PALETTE = {
    "train": "#00E5FF",      # Vibrant Cyan
    "test": "#FF4081",       # Electric Rose / Neon Pink
    "accent": "#00E676",     # Neon Green
    "warning": "#FFEA00",    # Neon Yellow
    "muted": "#75758A",      # Cool Slate
    "gradient": ["#00E5FF", "#7C4DFF", "#FF4081"]  # Cyan -> Purple -> Pink
}

# ---------------------------------------------------------------------------
# 1. Parity Plot (Train vs Test Overlay)
# ---------------------------------------------------------------------------
def plot_parity(y_train, y_train_pred, y_test, y_test_pred, save_path):
    """
    Plot actual vs. predicted values for both train and test sets in an overlay.
    """
    apply_dark_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Scatter plots with transparency
    ax.scatter(y_train, y_train_pred, color=PALETTE["train"], alpha=0.6, edgecolors="#0D0D11", linewidths=0.5, label="Train Set", s=50)
    ax.scatter(y_test, y_test_pred, color=PALETTE["test"], alpha=0.8, edgecolors="#0D0D11", linewidths=0.5, label="Test Set", s=70)
    
    # Identity line
    min_val = min(y_train.min(), y_test.min()) - 5
    max_val = max(y_train.max(), y_test.max()) + 5
    ax.plot([min_val, max_val], [min_val, max_val], color=PALETTE["muted"], linestyle="--", alpha=0.8, label="Ideal (y = x)")
    
    # Calculate metrics for annotation box
    train_r2, train_rmse = r2_score(y_train, y_train_pred), root_mean_squared_error(y_train, y_train_pred)
    test_r2, test_rmse = r2_score(y_test, y_test_pred), root_mean_squared_error(y_test, y_test_pred)
    
    stats_text = (
        f"Train R²: {train_r2:.3f}\n"
        f"Train RMSE: {train_rmse:.2f}\n"
        f"Test R²: {test_r2:.3f}\n"
        f"Test RMSE: {test_rmse:.2f}"
    )
    ax.text(
        0.05, 0.95, stats_text, transform=ax.transAxes,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#16161D', edgecolor='#3E3E4F', alpha=0.85)
    )
    
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Actual G-Score", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted G-Score", fontsize=11, fontweight="bold")
    ax.set_title("Parity Plot: Actual vs. Predicted G-Scores", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True)
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

# ---------------------------------------------------------------------------
# 2. Residuals Plot (Scatter + Distribution)
# ---------------------------------------------------------------------------
def plot_residuals(y_train, y_train_pred, y_test, y_test_pred, save_path):
    """
    Plot residuals vs predicted values and their distributions.
    """
    apply_dark_style()
    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={'width_ratios': [2.5, 1]})
    
    # Calculate residuals
    res_train = y_train - y_train_pred
    res_test = y_test - y_test_pred
    
    # Left: Scatter of Residuals vs Predicted
    ax_scatter.scatter(y_train_pred, res_train, color=PALETTE["train"], alpha=0.5, edgecolors="#0D0D11", linewidths=0.5, label="Train Set", s=45)
    ax_scatter.scatter(y_test_pred, res_test, color=PALETTE["test"], alpha=0.75, edgecolors="#0D0D11", linewidths=0.5, label="Test Set", s=65)
    ax_scatter.axhline(0, color=PALETTE["muted"], linestyle="--", alpha=0.8)
    
    ax_scatter.set_xlabel("Predicted G-Score", fontsize=11, fontweight="bold")
    ax_scatter.set_ylabel("Residual (Actual - Predicted)", fontsize=11, fontweight="bold")
    ax_scatter.set_title("Residuals vs. Predicted Values", fontsize=12, fontweight="bold")
    ax_scatter.grid(True)
    ax_scatter.legend()
    
    # Right: Distribution of Residuals (KDE/Hist)
    sns.histplot(res_train, color=PALETTE["train"], kde=True, ax=ax_hist, label="Train", alpha=0.4, element="step")
    sns.histplot(res_test, color=PALETTE["test"], kde=True, ax=ax_hist, label="Test", alpha=0.6, element="step")
    ax_hist.axvline(0, color=PALETTE["muted"], linestyle="--", alpha=0.8)
    
    ax_hist.set_xlabel("Residual", fontsize=11, fontweight="bold")
    ax_hist.set_ylabel("Density / Count", fontsize=11, fontweight="bold")
    ax_hist.set_title("Residual Distribution", fontsize=12, fontweight="bold")
    ax_hist.grid(True)
    ax_hist.legend()
    
    plt.suptitle("Residuals Analysis", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

# ---------------------------------------------------------------------------
# 3. XGBoost Feature Importance (Gain-based)
# ---------------------------------------------------------------------------
def plot_feature_importance_xgb(model, feature_names, save_path, max_features=20):
    """
    Plot gain-based feature importance from XGBoost.
    """
    apply_dark_style()
    
    # Extract feature importances
    importances = model.feature_importances_
    
    # Handle case where feature_names might be shorter/longer than importances
    if len(feature_names) != len(importances):
        # Fallback names
        feature_names = [f"Feature {i}" for i in range(len(importances))]
        
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df = df.sort_values(by="Importance", ascending=False).head(max_features)
    
    fig, ax = plt.subplots(figsize=(8, min(max_features * 0.35 + 2.0, 8)))
    
    # Horizontal bar plot with custom color gradient
    colors = sns.color_palette("coolwarm", len(df))[::-1]  # Warm for high importance
    bars = ax.barh(df["Feature"], df["Importance"], color=colors, edgecolor="#0D0D11", height=0.6)
    
    # Add values on top of bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (df["Importance"].max() * 0.01),
            bar.get_y() + bar.get_height()/2,
            f"{width:.4f}",
            va='center', ha='left', fontsize=8, color="#B0B0C3"
        )
        
    ax.invert_yaxis()  # Top importance at the top
    ax.set_xlabel("Gain Importance", fontsize=11, fontweight="bold")
    ax.set_ylabel("Features", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {len(df)} XGBoost Feature Importances (Gain)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, axis="x")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

# ---------------------------------------------------------------------------
# 4. SHAP Summary Plot
# ---------------------------------------------------------------------------
def plot_shap_importance(shap_values, X, save_path, max_display=20):
    """
    Generate and save a SHAP summary plot.
    """
    apply_dark_style()
    # Create figure to draw into
    fig = plt.figure(figsize=(9, min(max_display * 0.35 + 2.0, 9)))
    
    # Customise the SHAP summary plot style to fit dark theme
    # Note: shap.summary_plot directly interacts with active matplotlib figure/axes
    shap.summary_plot(
        shap_values,
        X,
        max_display=max_display,
        show=False,
        plot_size=None,
        color_bar_label="Feature Value"
    )
    
    # Post-process SHAP plot labels and fonts for dark theme
    fig.patch.set_facecolor("#0D0D11")
    ax = plt.gca()
    ax.set_facecolor("#16161D")
    ax.xaxis.label.set_color("#FFFFFF")
    ax.yaxis.label.set_color("#FFFFFF")
    ax.tick_params(colors="#B0B0C3")
    
    # Change color bar tick colors if it exists
    for child in fig.get_children():
        if "colorbar" in str(type(child)).lower() or "Colorbar" in str(type(child)):
            pass # SHAP handles its own colorbar, but we adjust global text settings anyway
            
    plt.title(f"SHAP Feature Importance (Top {max_display})", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

# ---------------------------------------------------------------------------
# 5. Out-of-Fold (OOF) Prediction Plot
# ---------------------------------------------------------------------------
def plot_oof(y_train, oof_preds, save_path):
    """
    Plot cross-validated out-of-fold (OOF) predictions against actual values.
    """
    apply_dark_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    
    ax.scatter(y_train, oof_preds, color=PALETTE["train"], alpha=0.6, edgecolors="#0D0D11", linewidths=0.5, s=50, label="OOF Predictions")
    
    # Identity line
    min_val = min(y_train.min(), oof_preds.min()) - 5
    max_val = max(y_train.max(), oof_preds.max()) + 5
    ax.plot([min_val, max_val], [min_val, max_val], color=PALETTE["muted"], linestyle="--", alpha=0.8, label="Ideal")
    
    r2 = r2_score(y_train, oof_preds)
    rmse = root_mean_squared_error(y_train, oof_preds)
    mae = mean_absolute_error(y_train, oof_preds)
    
    stats_text = (
        f"OOF R²: {r2:.3f}\n"
        f"OOF RMSE: {rmse:.2f}\n"
        f"OOF MAE: {mae:.2f}"
    )
    ax.text(
        0.05, 0.95, stats_text, transform=ax.transAxes,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#16161D', edgecolor='#3E3E4F', alpha=0.85)
    )
    
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Actual G-Score", fontsize=11, fontweight="bold")
    ax.set_ylabel("OOF Predicted G-Score", fontsize=11, fontweight="bold")
    ax.set_title("Out-of-Fold (CV) Predictions vs. Actual", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True)
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

# ---------------------------------------------------------------------------
# 6. CV Fold Scores (Per-fold metrics)
# ---------------------------------------------------------------------------
def plot_cv_fold_scores(cv_metrics, metric_name, save_path):
    """
    Plot per-fold scores as a bar chart with mean horizontal line.
    
    Parameters
    ----------
    cv_metrics : list of float
        Metric values for each CV fold.
    metric_name : str
        Name of the metric (e.g. 'RMSE', 'R2').
    """
    apply_dark_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    folds = [f"Fold {i+1}" for i in range(len(cv_metrics))]
    mean_val = np.mean(cv_metrics)
    std_val = np.std(cv_metrics)
    
    # Beautiful teal bars
    bars = ax.bar(folds, cv_metrics, color=PALETTE["train"], alpha=0.75, edgecolor="#0D0D11", width=0.5)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + (max(cv_metrics) * 0.01),
            f"{height:.3f}",
            ha='center', va='bottom', fontsize=9, color="#B0B0C3"
        )
        
    # Mean line
    ax.axhline(mean_val, color=PALETTE["test"], linestyle="-.", alpha=0.8, linewidth=1.5,
               label=f"Mean: {mean_val:.3f} (±{std_val:.3f})")
    
    ax.set_ylabel(metric_name, fontsize=11, fontweight="bold")
    ax.set_title(f"5-Fold CV: {metric_name} Scores", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

# ---------------------------------------------------------------------------
# 7. Learning Curves
# ---------------------------------------------------------------------------
def plot_learning_curves(estimator, X, y, cv, save_path):
    """
    Plot learning curves (RMSE vs. Training Size).
    """
    apply_dark_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Use negative RMSE because learning_curve scoring uses utilities where higher is better
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv,
        scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1
    )
    
    # Convert negative RMSE back to positive RMSE
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = -np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    
    # Plot curves with shaded error bands
    ax.plot(train_sizes, train_scores_mean, 'o-', color=PALETTE["train"], linewidth=2, label="Training Score")
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.15, color=PALETTE["train"])
                     
    ax.plot(train_sizes, val_scores_mean, 'o-', color=PALETTE["test"], linewidth=2, label="Validation Score")
    ax.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.15, color=PALETTE["test"])
    
    ax.set_xlabel("Training Set Size (Molecules)", fontsize=11, fontweight="bold")
    ax.set_ylabel("RMSE (G-Score)", fontsize=11, fontweight="bold")
    ax.set_title("XGBoost Learning Curves (Group CV)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True)
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

# ---------------------------------------------------------------------------
# 8. Model Stability Analysis (Multiple Seeds)
# ---------------------------------------------------------------------------
def plot_stability(seed_scores, save_path):
    """
    Boxplot of model performance (e.g. RMSE, R2) across multiple seeds.
    
    Parameters
    ----------
    seed_scores : dict
        A dictionary mapping seed -> metric values (e.g. {'R2': [...], 'RMSE': [...]}).
    """
    apply_dark_style()
    metrics = list(seed_scores.keys())
    
    # We create a 1xN or 2x1 subplot depending on the number of metrics (typically R2 and RMSE)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5.5))
    if len(metrics) == 1:
        axes = [axes]
        
    for ax, metric in zip(axes, metrics):
        values = seed_scores[metric]
        
        # Plot individual data points with jitter overlaid on a boxplot
        sns.boxplot(y=values, ax=ax, color=PALETTE["train"], width=0.4,
                    boxprops=dict(alpha=0.6, edgecolor="#FFFFFF"),
                    medianprops=dict(color=PALETTE["test"], linewidth=2),
                    whiskerprops=dict(color="#FFFFFF"),
                    capprops=dict(color="#FFFFFF"))
                    
        # Stripplot for single points
        sns.stripplot(y=values, ax=ax, color=PALETTE["accent"], size=8, jitter=0.15, edgecolor="#0D0D11", linewidth=0.5)
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        ax.set_title(f"{metric} Stability\nMean: {mean_val:.3f} (±{std_val:.3f})", fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel(metric, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y")
        
    plt.suptitle("Model Stability across 5 Random Seeds", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
