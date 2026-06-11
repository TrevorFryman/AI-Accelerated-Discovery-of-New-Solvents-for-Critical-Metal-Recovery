"""
run_all_pipelines.py
====================
Master orchestrator for the GSK Solvent G-Score Prediction project.
Runs all feature generation and training scripts in sequence,
verifies outputs, loads final metrics, and prints/saves a beautiful
comparison table.
"""

import sys
import subprocess
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


def run_script(script_name: str):
    """
    Run a python script in a subprocess, logging its stdout/stderr.
    """
    script_path = PROJECT_ROOT / script_name
    log.info("=" * 70)
    log.info("Executing script: %s", script_name)
    log.info("=" * 70)
    
    # Run python script using the same executable that runs this script
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT)
    )
    
    # Print child process logs
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
        
    if result.returncode != 0:
        log.error("Script %s failed with exit code %d", script_name, result.returncode)
        sys.exit(result.returncode)
        
    log.info("Script %s finished successfully.\n", script_name)


def verify_outputs() -> bool:
    """
    Verify all expected outputs (npz, json, joblib, png, csv) exist.
    """
    expected_files = [
        # Features
        "descriptors/morgan_fingerprints.npz",
        "descriptors/rdkit_descriptors.npz",
        "descriptors/chemberta_embeddings.npz",
        
        # Models
        "models/morgan_xgb.json",
        "models/rdkit_xgb.json",
        "models/rdkit_preprocessor.joblib",
        "models/chemberta_xgb.json",
        "models/chemberta_scaler.joblib",
        
        # Results - Morgan
        "results/morgan/metrics.json",
        "results/morgan/cv_metrics.json",
        "results/morgan/stability_metrics.json",
        "results/morgan/oof_predictions.csv",
        "results/morgan/test_predictions.csv",
        
        # Results - RDKit
        "results/rdkit/metrics.json",
        "results/rdkit/cv_metrics.json",
        "results/rdkit/stability_metrics.json",
        "results/rdkit/oof_predictions.csv",
        "results/rdkit/test_predictions.csv",
        
        # Results - ChemBERTa
        "results/chemberta/metrics.json",
        "results/chemberta/cv_metrics.json",
        "results/chemberta/stability_metrics.json",
        "results/chemberta/oof_predictions.csv",
        "results/chemberta/test_predictions.csv",
    ]
    
    missing = []
    for rel_path in expected_files:
        p = PROJECT_ROOT / rel_path
        if not p.exists():
            missing.append(rel_path)
            
    if missing:
        log.warning("Verification failed. The following expected outputs are missing:")
        for m in missing:
            log.warning("  - %s", m)
        return False
        
    log.info("All expected files successfully generated and verified!")
    return True


def collect_metrics_and_compare():
    """
    Load metrics from all pipelines and print a beautiful summary table.
    """
    pipelines = ["morgan", "rdkit", "chemberta"]
    results = {}
    
    for pipe in pipelines:
        metrics_file = PROJECT_ROOT / f"results/{pipe}/metrics.json"
        cv_file = PROJECT_ROOT / f"results/{pipe}/cv_metrics.json"
        features_file = PROJECT_ROOT / f"descriptors/{pipe}_fingerprints.npz" if pipe == "morgan" else \
                        PROJECT_ROOT / f"descriptors/{pipe}_descriptors.npz" if pipe == "rdkit" else \
                        PROJECT_ROOT / f"descriptors/{pipe}_embeddings.npz"
        
        # Load main metrics
        with open(metrics_file) as f:
            metrics = json.load(f)
        # Load CV metrics
        with open(cv_file) as f:
            cv_metrics = json.load(f)
            
        # Get feature dimensions
        features_data = np.load(features_file, allow_pickle=True)
        feat_dim = features_data["features"].shape[1]
        
        # Store comparison details
        results[pipe] = {
            "Dimension": feat_dim,
            "Train RMSE": metrics["train_rmse"],
            "Train R²": metrics["train_r2"],
            "CV RMSE (Mean)": cv_metrics["mean_rmse"],
            "CV RMSE (Std)": cv_metrics["std_rmse"],
            "CV R² (Mean)": cv_metrics["mean_r2"],
            "CV R² (Std)": cv_metrics["std_r2"],
            "Test RMSE": metrics["test_rmse"],
            "Test R²": metrics["test_r2"],
        }
        
    # Build comparison DataFrame
    df_comp = pd.DataFrame(results).T
    df_comp.index.name = "Pipeline"
    
    # Save comparison to results
    comp_json_path = PROJECT_ROOT / "results/model_comparison.json"
    with open(comp_json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    # Format comparison table in Markdown
    log.info("\n" + "=" * 70)
    log.info("                      MODEL COMPARISON SUMMARY")
    log.info("=" * 70)
    
    markdown_table = (
        "| Pipeline | Features | Train RMSE | Train R² | CV RMSE (Mean ± Std) | CV R² (Mean ± Std) | Test RMSE | Test R² |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    )
    
    for pipe, res in results.items():
        name = "Pipeline A (Morgan)" if pipe == "morgan" else \
               "Pipeline B (RDKit)" if pipe == "rdkit" else \
               "Pipeline C (ChemBERTa)"
        markdown_table += (
            f"| {name} | {res['Dimension']} | {res['Train RMSE']:.3f} | {res['Train R²']:.3f} | "
            f"{res['CV RMSE (Mean)']:.3f} ± {res['CV RMSE (Std)']:.3f} | "
            f"{res['CV R² (Mean)']:.3f} ± {res['CV R² (Std)']:.3f} | "
            f"{res['Test RMSE']:.3f} | {res['Test R²']:.3f} |\n"
        )
        
    print(markdown_table)
    
    # Also save markdown comparison to results folder
    comp_md_path = PROJECT_ROOT / "results/model_comparison.md"
    with open(comp_md_path, "w") as f:
        f.write("# GSK Solvent G-Score — Model Comparison Report\n\n")
        f.write(markdown_table)
    log.info("Saved model comparison report to results/model_comparison.md\n")


def main():
    # 1. Feature generation scripts
    run_script("generate_morgan_features.py")
    run_script("generate_rdkit_features.py")
    run_script("generate_chemberta_features.py")
    
    # 2. Model training scripts
    run_script("train_morgan_xgb.py")
    run_script("train_rdkit_xgb.py")
    run_script("train_chemberta_xgb.py")
    
    # 3. Verification & Comparison
    if verify_outputs():
        collect_metrics_and_compare()
        log.info("All pipelines ran and compared successfully!")
    else:
        log.error("Pipeline run completed, but some output files are missing.")


if __name__ == "__main__":
    main()
