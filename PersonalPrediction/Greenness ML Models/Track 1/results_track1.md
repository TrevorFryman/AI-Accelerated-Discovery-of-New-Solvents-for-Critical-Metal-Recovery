# Track 1 — Results Summary

Comparison of the three feature-representation pipelines (**Morgan fingerprints**, **RDKit descriptors**, **ChemBERTa embeddings**) used to predict the greenness score (G-score). Data pulled from `results/*/metrics.json`, `results/*/cv_metrics.json`, `results/*/stability_metrics.json`, `results/*/tuning_summary.json`, and `results/model_comparison.json`. Figures pulled from `figures/*`.

> Image links below point to the `main` branch of this repository via `raw.githubusercontent.com`, so they render correctly on the GitHub wiki.

## 1. Headline Comparison (baseline models)

| Pipeline  | Feature Dim | Train RMSE | Train R² | CV RMSE (mean ± std) | CV R² (mean ± std) | Test RMSE | Test R² | Test MAE |
|-----------|------------:|-----------:|---------:|----------------------|---------------------|----------:|--------:|---------:|
| Morgan FP | 2048 | 0.4031 | 0.8920 | 0.9163 ± 0.1147 | 0.3990 ± 0.2023 | 0.9522 | 0.5664 | 0.7514 |
| RDKit     | 217  | 0.0568 | 0.9979 | 0.7838 ± 0.1331 | 0.5521 ± 0.1996 | **0.8001** | **0.6938** | **0.6670** |
| ChemBERTa | 768  | 0.0556 | 0.9979 | 1.1765 ± 0.0777 | 0.0408 ± 0.1571 | 1.4154 | 0.0419 | 1.1704 |

**Takeaways**
- **RDKit descriptors give the best generalization**, with the lowest test RMSE/MAE and highest test R² of the three pipelines.
- **Morgan fingerprints** are a solid middle ground — moderate train fit, second-best test performance.
- **ChemBERTa embeddings** overfit badly: near-perfect training fit (R² ≈ 0.998) but the worst test/CV performance (test R² ≈ 0.04), suggesting the embeddings + model combination is not generalizing for this dataset/size.

## 2. Hyperparameter Tuning (before vs. after)

| Pipeline  | Stage    | Best Params (subsample / n_estimators / max_depth / lr / colsample) | CV RMSE (mean) | Test RMSE | Test R² | Test MAE |
|-----------|----------|------------------------------------------------------------------------|---------------:|----------:|--------:|---------:|
| Morgan FP | Baseline | — | 0.9163 | 0.9522 | 0.5664 | 0.7514 |
| Morgan FP | Tuned    | 0.9 / 200 / 4 / 0.03 / 0.7 | 0.8670 | 0.9661 | 0.5537 | 0.7634 |
| RDKit     | Baseline | — | 0.7727 | 0.8458 | 0.6579 | 0.7241 |
| RDKit     | Tuned    | 0.6 / 200 / 4 / 0.2 / 1.0 | 0.8125 | 0.8568 | 0.6489 | 0.7018 |
| ChemBERTa | Baseline | — | 1.1765 | 1.4154 | 0.0419 | 1.1704 |
| ChemBERTa | Tuned    | 0.9 / 100 / 3 / 0.01 / 1.0 | 1.1412 | 1.3952 | 0.0691 | 1.1809 |

**Takeaways**
- Tuning improved CV RMSE for all three pipelines, but the effect on the held-out **test set was mixed**:
  - **Morgan FP**: tuning improved CV RMSE (0.9163 → 0.8670) but slightly *hurt* test RMSE (0.9522 → 0.9661) and MAE — sign of mild overfitting to CV folds.
  - **RDKit**: tuning slightly hurt both CV and test RMSE, but *improved* test MAE (0.7241 → 0.7018) — baseline RDKit remains the strongest overall on RMSE/R².
  - **ChemBERTa**: tuning improved both CV and test RMSE/R² somewhat, but the model is still far behind the other two pipelines.
- See the `rmse_before_after.png` figure per pipeline below for the visual before/after comparison.

## 3. Seed Stability (5 seeds: 42, 123, 456, 789, 1337)

| Pipeline  | CV RMSE range | CV R² range | Test RMSE range | Test R² range |
|-----------|----------------|--------------|-------------------|------------------|
| Morgan FP | 0.8682 – 0.9488 | 0.347 – 0.477 | 0.9519 – 0.9586 | 0.561 – 0.567 |
| RDKit     | 0.7500 – 0.8148 | 0.547 – 0.610 | 0.7991 – 0.8856 | 0.625 – 0.695 |
| ChemBERTa | 1.1116 – 1.1779 | 0.041 – 0.152 | 1.2955 – 1.4154 | 0.042 – 0.197 |

**Takeaways**
- **Morgan FP** is the most stable across seeds (test R² varies only ~0.56–0.57).
- **RDKit** is slightly more variable but consistently the strongest performer (test R² 0.62–0.69).
- **ChemBERTa** is both the weakest and the most variable (test R² 0.04–0.20), reinforcing that this representation isn't well-suited here.
- See the `stability.png` figure per pipeline below for the per-seed visualization.

## 4. Exploratory Data Analysis (EDA)

| Figure | Description |
|--------|-------------|
| ![G-score histogram](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/EDA/fig01_gscore_histogram.png) | Distribution of the target G-score |
| ![G-score boxplot](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/EDA/fig02_gscore_boxplot.png) | Boxplot of G-score |
| ![G-score violin by class](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/EDA/fig03_gscore_violin_by_class.png) | Violin plot of G-score by class |
| ![G-score strip by class](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/EDA/fig04_gscore_strip_by_class.png) | Strip plot of G-score by class |
| ![Correlation heatmap](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/EDA/fig05_correlation_heatmap.png) | Feature correlation heatmap |
| ![Missing values](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/EDA/fig06_missing_values.png) | Missing value summary |
| ![G-score box by class](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/EDA/fig07_gscore_box_by_class.png) | G-score boxplot by class (alt view) |

## 5. Per-Pipeline Figures

Each pipeline folder (`figures/morgan`, `figures/rdkit`, `figures/chemberta`) contains the same set of diagnostic plots.

### 5.1 Morgan Fingerprints

| Parity | OOF Parity |
|---|---|
| ![Parity](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/morgan/parity.png) | ![OOF Parity](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/morgan/oof_parity.png) |

| Residuals | Learning Curves |
|---|---|
| ![Residuals](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/morgan/residuals.png) | ![Learning Curves](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/morgan/learning_curves.png) |

| CV Fold Scores | RMSE Before/After Tuning |
|---|---|
| ![CV Fold Scores](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/morgan/cv_fold_scores.png) | ![RMSE Before/After](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/morgan/rmse_before_after.png) |

| Feature Importance | SHAP Importance |
|---|---|
| ![Feature Importance](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/morgan/feature_importance.png) | ![SHAP Importance](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/morgan/shap_importance.png) |

**Seed Stability**

![Stability](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/morgan/stability.png)

### 5.2 RDKit Descriptors

| Parity | OOF Parity |
|---|---|
| ![Parity](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/rdkit/parity.png) | ![OOF Parity](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/rdkit/oof_parity.png) |

| Residuals | Learning Curves |
|---|---|
| ![Residuals](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/rdkit/residuals.png) | ![Learning Curves](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/rdkit/learning_curves.png) |

| CV Fold Scores | RMSE Before/After Tuning |
|---|---|
| ![CV Fold Scores](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/rdkit/cv_fold_scores.png) | ![RMSE Before/After](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/rdkit/rmse_before_after.png) |

| Feature Importance | SHAP Importance |
|---|---|
| ![Feature Importance](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/rdkit/feature_importance.png) | ![SHAP Importance](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/rdkit/shap_importance.png) |

**Seed Stability**

![Stability](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/rdkit/stability.png)

### 5.3 ChemBERTa Embeddings

| Parity | OOF Parity |
|---|---|
| ![Parity](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/chemberta/parity.png) | ![OOF Parity](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/chemberta/oof_parity.png) |

| Residuals | Learning Curves |
|---|---|
| ![Residuals](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/chemberta/residuals.png) | ![Learning Curves](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/chemberta/learning_curves.png) |

| CV Fold Scores | RMSE Before/After Tuning |
|---|---|
| ![CV Fold Scores](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/chemberta/cv_fold_scores.png) | ![RMSE Before/After](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/chemberta/rmse_before_after.png) |

| Feature Importance | SHAP Importance |
|---|---|
| ![Feature Importance](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/chemberta/feature_importance.png) | ![SHAP Importance](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/chemberta/shap_importance.png) |

**Seed Stability**

![Stability](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%201/figures/chemberta/stability.png)

## 6. Overall Conclusion

**RDKit descriptors (217-dim) currently produce the best-generalizing model** (Test R² = 0.694, Test RMSE = 0.800), followed by Morgan fingerprints (Test R² = 0.566), with ChemBERTa embeddings trailing far behind (Test R² = 0.042) despite fitting the training data almost perfectly — a clear overfitting signal. For further gains, consider regularizing the ChemBERTa pipeline (stronger shrinkage/dropout, dimensionality reduction, or fewer estimators), and consider ensembling Morgan + RDKit features.
