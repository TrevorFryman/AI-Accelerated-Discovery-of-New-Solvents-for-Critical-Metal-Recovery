"""Compare Morgan, RDKit, and ChemBERTa XGBoost models before and after tuning.

Usage: run this script from project root. It will load descriptors from
`descriptors/*.npz`, perform CV/OoF evaluation of baseline models (if saved),
run randomized hyperparameter optimization using `hyperparameter_optimization.py`,
and save comparison tables/figures and a JSON of results.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from config import DIRS, COL_SMILES, COL_TARGET, XGB_PARAMS
from data_utils import load_dataset, make_split, assign_groups
import plot_utils
from hyperparameter_optimization import tune_xgb, default_param_space

from xgboost import XGBRegressor

STABILITY_SEEDS = [42, 123, 456, 789, 1337]


def load_features(pipeline_name: str):
    path = DIRS["descriptors"] / f"{pipeline_name}_fingerprints.npz" if pipeline_name == "morgan" else \
           DIRS["descriptors"] / f"{pipeline_name}_descriptors.npz" if pipeline_name == "rdkit" else \
           DIRS["descriptors"] / f"{pipeline_name}_embeddings.npz"
    data = np.load(path, allow_pickle=True)
    return data["features"], data.get("targets"), data.get("smiles")


def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def oof_cv_evaluate(estimator, X, y, splits):
    oof_preds = np.zeros(len(y))
    cv_scores = []
    for train_idx, val_idx in splits:
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        est = estimator
        est.fit(X_tr, y_tr)
        preds = est.predict(X_val)
        oof_preds[val_idx] = preds
        cv_scores.append(float(np.sqrt(mean_squared_error(y_val, preds))))
    return oof_preds, cv_scores


def evaluate_and_tune(pipeline_name: str, n_iter: int = 40):
    print(f"Processing pipeline: {pipeline_name}")
    X, targets, smiles = load_features(pipeline_name)
    y = targets

    df = load_dataset()
    train_idx, test_idx = make_split(df)
    groups = assign_groups(df)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    groups_train = groups[train_idx]
    from config import COL_CLASS
    strata_train = df.iloc[train_idx][COL_CLASS].values

    # Baseline: start from default XGB_PARAMS
    base = XGBRegressor(**XGB_PARAMS)
    # Build cv_splits using hyperparameter module helper (requires strata)
    from hyperparameter_optimization import build_cv_splits
    cv_splits = build_cv_splits(X_train, strata_train, groups_train)

    # OOF baseline
    base_oof, base_cv_rmse = oof_cv_evaluate(base, X_train, y_train, cv_splits)
    base_cv_metrics = {
        "mean_rmse": float(np.mean(base_cv_rmse)),
        "std_rmse": float(np.std(base_cv_rmse)),
    }

    # Fit final baseline on full train and evaluate on test
    base.fit(X_train, y_train)
    test_preds_base = base.predict(X_test)
    test_metrics_base = metrics(y_test, test_preds_base)

    # Hyperparameter tuning
    param_space = default_param_space()
    search, used_splits = tune_xgb(X_train, y_train, groups_train, param_space, n_iter=n_iter, strata=strata_train)
    best = search.best_estimator_

    # OOF tuned
    tuned_oof, tuned_cv_rmse = oof_cv_evaluate(best, X_train, y_train, used_splits)
    tuned_cv_metrics = {
        "mean_rmse": float(np.mean(tuned_cv_rmse)),
        "std_rmse": float(np.std(tuned_cv_rmse)),
    }

    # Fit final tuned on full train and evaluate test
    best.fit(X_train, y_train)
    test_preds_tuned = best.predict(X_test)
    test_metrics_tuned = metrics(y_test, test_preds_tuned)

    result = {
        "pipeline": pipeline_name,
        "baseline": {
            "cv_rmse_per_fold": base_cv_rmse,
            "cv_metrics": base_cv_metrics,
            "test_metrics": test_metrics_base,
            "oof_preds": base_oof.tolist(),
        },
        "tuned": {
            "best_params": search.best_params_,
            "cv_rmse_per_fold": tuned_cv_rmse,
            "cv_metrics": tuned_cv_metrics,
            "test_metrics": test_metrics_tuned,
            "oof_preds": tuned_oof.tolist(),
        }
    }

    # Save intermediate artifacts
    out_dir = DIRS["results"] / pipeline_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tuning_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    # Generate comparison figures: before vs after test RMSE
    fig_dir = DIRS["figures"] / pipeline_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    # Simple parity plots saved earlier; create before-vs-after bar
    df_comp = pd.DataFrame({
        "model": ["baseline", "tuned"],
        "test_rmse": [test_metrics_base["rmse"], test_metrics_tuned["rmse"]],
        "test_r2": [test_metrics_base["r2"], test_metrics_tuned["r2"]]
    })
    ax = df_comp.plot(x="model", y="test_rmse", kind="bar", legend=False, title=f"{pipeline_name} Test RMSE: Baseline vs Tuned")
    ax.set_ylabel("RMSE")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(fig_dir / "rmse_before_after.png", dpi=300)
    fig.clf()

    return result


def main():
    pipelines = ["morgan", "rdkit", "chemberta"]
    all_results = {}
    for p in pipelines:
        res = evaluate_and_tune(p, n_iter=30)
        all_results[p] = res

    # Save consolidated comparison
    with open(DIRS["results"] / "hyperparam_tuning_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Build a simple comparison table and save
    rows = []
    for p, r in all_results.items():
        rows.append({
            "pipeline": p,
            "baseline_test_rmse": r["baseline"]["test_metrics"]["rmse"],
            "tuned_test_rmse": r["tuned"]["test_metrics"]["rmse"],
            "baseline_test_r2": r["baseline"]["test_metrics"]["r2"],
            "tuned_test_r2": r["tuned"]["test_metrics"]["r2"],
        })
    df_table = pd.DataFrame(rows).sort_values("tuned_test_rmse")
    df_table.to_csv(DIRS["results"] / "model_tuning_comparison.csv", index=False)

    print("Finished comparison. Results saved to results/hyperparam_tuning_results.json and model_tuning_comparison.csv")


if __name__ == "__main__":
    main()
