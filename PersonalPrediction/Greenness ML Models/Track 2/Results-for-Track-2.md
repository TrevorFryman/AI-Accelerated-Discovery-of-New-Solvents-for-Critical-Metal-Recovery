# Track 2 — Results Summary

_This page is dedicated to all the results that were generated from the Track 2 models. This includes reports, figures, comparison tables, hyperparameters, and saved models._

This results summary was generated within VS Code by Claude Sonnet 4.6, based on the reports, metrics, and figures produced by the Track 2 pipeline scripts in `scripts/`, `reports/`, `results/`, and the final `Descriptor_Comparison_Table.csv` / `Final_Descriptor_Comparison_Report.md`.

Comparison of the three feature-representation pipelines (**Morgan fingerprints**, **RDKit descriptors**, **ChemBERTa embeddings**) used to predict the greenness score (G-score) for the GSK solvent dataset (154 molecules). All three pipelines use an identical methodology: 80/20 train-test split (`random_state=42`), 5-fold cross-validation (`KFold(n_splits=5, shuffle=True, random_state=42)`), and `RandomizedSearchCV(n_iter=100)` over the same XGBoost hyperparameter grid (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`).

> Image links below point to the `main` branch of this repository via `raw.githubusercontent.com`, so they render correctly on the GitHub wiki.

## 1. Headline Comparison (Optimized Models)

| Pipeline  | Feature Dim | Train R² (CV mean) | CV RMSE (mean ± std) | CV R² (mean ± std) | Test RMSE | Test R² | Test MAE |
|-----------|------------:|--------------------:|------------------------|----------------------|----------:|--------:|---------:|
| Morgan FP | 2048 | 0.9435 | 0.8645 ± 0.0941 | **0.5318 ± 0.0835** | **0.7432** | **0.6933** | **0.5697** |
| RDKit     | 7    | 0.8288 | 0.9278 ± 0.1036 | 0.4652 ± 0.0627 | 0.9475 | 0.5016 | 0.7329 |
| ChemBERTa | 768  | 0.9977 | 1.1710 ± 0.0709 | 0.1453 ± 0.0545 | 1.0996 | 0.3288 | 0.9075 |

**Takeaways**
- **Morgan fingerprints achieve the best raw predictive performance** (highest CV R² and test R², lowest test RMSE/MAE), but at the cost of the largest feature dimensionality (2048).
- **RDKit descriptors** are a close second on CV R² (0.4652 vs 0.5318) using only **7 features** — a 99.7% reduction in dimensionality versus Morgan — and show the smallest train/CV generalization gap of the three pipelines.
- **ChemBERTa embeddings** perform worst on every metric and show the most severe overfitting (train R² ≈ 0.998 vs CV R² = 0.145), suggesting the embedding + XGBoost combination does not generalize well for this small (154-sample) dataset.

## 2. Hyperparameter Tuning (Baseline vs Optimized)

| Pipeline  | Stage    | Best Params (subsample / n_estimators / max_depth / lr / colsample / min_child_weight) | OOF RMSE | OOF R² | Test RMSE | Test R² | Test MAE |
|-----------|----------|------------------------------------------------------------------------------------------|---------:|-------:|----------:|--------:|---------:|
| Morgan FP | Baseline | — (defaults) | 0.9830 | 0.4047 | 0.9207 | 0.5294 | 0.7488 |
| Morgan FP | Tuned    | 0.5 / 50 / 5 / 0.2 / 0.6 / 1 | 0.8691 | 0.5346 | 0.7432 | 0.6933 | 0.5697 |
| RDKit     | Baseline | — (defaults) | 0.9632 | 0.4284 | 0.9805 | 0.4663 | 0.7860 |
| RDKit     | Tuned    | 0.7 / 500 / 3 / 0.01 / 0.8 / 1 | 0.9331 | 0.4636 | 0.9475 | 0.5016 | 0.7329 |
| ChemBERTa | Baseline | — (defaults) | 1.2551 | 0.0295 | 1.2181 | 0.1763 | 1.0165 |
| ChemBERTa | Tuned    | 0.9 / 200 / 3 / 0.1 / 0.3 / 7 | 1.1734 | 0.1517 | 1.0996 | 0.3288 | 0.9075 |

**Takeaways**
- Hyperparameter tuning via `RandomizedSearchCV(n_iter=100)` improved both OOF and test metrics for **all three pipelines** — no pipeline got worse after tuning.
- **Morgan FP** saw the largest absolute improvement (test R² 0.529 → 0.693; OOF R² 0.405 → 0.535).
- **RDKit** improved more modestly (test R² 0.466 → 0.502; OOF R² 0.428 → 0.464), and the tuned model also reduced the train/CV gap from 0.579 (baseline) to 0.358 (tuned) — the most "regularized" of the three after tuning.
- **ChemBERTa** improved substantially in relative terms (test R² 0.176 → 0.329; OOF R² 0.030 → 0.152) but remains far behind the other two pipelines and still overfits severely (gap = 0.848).

## 3. Statistical Analysis (Paired Comparison of CV RMSE)

Because all three pipelines share identical 5-fold `KFold(random_state=42)` splits (fold assignment depends only on sample count, not feature values), per-fold CV RMSE values are paired across pipelines, enabling a paired t-test.

### 95% Confidence Intervals (5-Fold CV RMSE, t-distribution, df=4)

| Pipeline  | Mean CV RMSE | 95% CI |
|-----------|-------------:|--------|
| Morgan Fingerprints | 0.8645 | [0.7477, 0.9814] |
| RDKit Descriptors   | 0.9278 | [0.7991, 1.0565] |
| ChemBERTa Embeddings | 1.1710 | [1.0830, 1.2590] |

### Paired t-test (CV RMSE, n=5 folds)

| Comparison | Mean RMSE Difference | t-statistic | p-value | Interpretation |
|---|---:|---:|---:|---|
| Morgan vs RDKit     | -0.0633 | -2.294 | 0.0835 | Not significant (p ≥ 0.05) — difference may be due to random variation |
| Morgan vs ChemBERTa | -0.3065 | -11.604 | 0.0003 | Significant — Morgan has lower RMSE |
| RDKit vs ChemBERTa  | -0.2432 | -10.516 | 0.0005 | Significant — RDKit has lower RMSE |

**Takeaway:** Both Morgan and RDKit are statistically significantly better than ChemBERTa. The Morgan vs RDKit gap is **not** statistically significant at n=5 folds — consistent with RDKit being a viable, much cheaper alternative to Morgan.

## 4. Computational Cost Analysis

| Pipeline  | Dimensionality | Descriptor Gen. Time (s, 154 mols) | Training Time (s/fit) | Inference Time (ms) | Feature Memory (MB) | Feature File Size (MB) |
|-----------|---------------:|-----------------------------------:|-----------------------:|---------------------:|---------------------:|------------------------:|
| Morgan Fingerprints | 2048 | 0.0125 | 0.0571 | 0.7205 | 2.4063 | 1.2407 |
| RDKit Descriptors   | 7    | 0.0694 | 0.0597 | 0.5030 | 0.0082 | 0.0137 |
| ChemBERTa Embeddings | 768 | 3.6941 | 0.2978 | 0.4954 | 0.9023 | 2.2528 |

**Takeaway:** RDKit descriptors have the smallest memory footprint and storage size by far (7 named features), while Morgan fingerprints are the fastest to generate (no model loading required). ChemBERTa is by far the most expensive — generating embeddings for 154 molecules takes ~3.7s due to loading and running the pretrained transformer, roughly 53x slower than RDKit and ~300x slower than Morgan.

## 5. Model Robustness Analysis

| Pipeline  | CV R² mean | CV R² std (fold-to-fold) | Train R² (CV folds) | Train–CV R² Gap |
|-----------|-----------:|--------------------------:|----------------------:|-----------------:|
| RDKit Descriptors   | 0.4652 | 0.0627 | 0.8288 | **0.3635** |
| Morgan Fingerprints | 0.5318 | 0.0835 | 0.9435 | 0.4117 |
| ChemBERTa Embeddings | 0.1453 | 0.0545 | 0.9977 | 0.8524 |

### Robustness Ranking (Most → Least Robust)
1. **RDKit Descriptors** — smallest train/CV gap (0.3635); some overfitting but the most controlled of the three.
2. **Morgan Fingerprints** — moderate gap (0.4117); strong CV performance but more overfitting than RDKit.
3. **ChemBERTa Embeddings** — largest gap (0.8524); severe overfitting, least robust.

All three pipelines show evidence of overfitting (training R² substantially exceeds CV R²), expected given the small dataset (154 samples) relative to feature dimensionality — particularly for the high-dimensional Morgan and ChemBERTa representations. None of the pipelines show evidence of underfitting.

![Overfitting Gap](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/Descriptor_Comparison_Figures/overfitting_gap.png)

## 6. Interpretability Analysis

| Pipeline | Ease of Interpretation | Explainability | Top Drivers |
|---|---|---|---|
| **RDKit Descriptors** | High — 7 named physicochemical properties | Very high — direct SHAP/feature-importance on named features | TPSA > LogP > RingCount (SHAP); TPSA > NumHAcceptors > NumHDonors (gain) |
| **Morgan Fingerprints** | Low–moderate — bits map to substructures but require decoding | Moderate — SHAP per bit possible, but 2048 bits (466 active) make a global narrative hard | Substructure-level (not directly named) |
| **ChemBERTa Embeddings** | Low — 768 abstract learned dimensions | Low — no direct chemical meaning without probing studies | None (opaque) |

**RDKit feature importance (gain-based):**

| Rank | Descriptor | Importance |
|---|---|---:|
| 1 | TPSA | 0.2385 |
| 2 | NumHAcceptors | 0.1979 |
| 3 | NumHDonors | 0.1427 |
| 4 | NumRotatableBonds | 0.1267 |
| 5 | LogP | 0.1238 |
| 6 | MolWt | 0.0865 |
| 7 | RingCount | 0.0839 |

**RDKit SHAP analysis (mean |SHAP value|):**

| Rank | Descriptor | Mean \|SHAP\| |
|---|---|---:|
| 1 | TPSA | 0.6069 |
| 2 | LogP | 0.2991 |
| 3 | RingCount | 0.1477 |
| 4 | MolWt | 0.1184 |
| 5 | NumHAcceptors | 0.1168 |
| 6 | NumRotatableBonds | 0.1074 |
| 7 | NumHDonors | 0.0524 |

TPSA — a measure of polar surface area linked to hydrogen-bonding capacity — is consistently the strongest driver of G-score predictions, consistent with the EDA finding that TPSA had the strongest linear correlation with G-score (r = 0.5210) of the descriptors examined.

## 7. Exploratory Data Analysis (EDA)

- **Dataset:** 154 solvents, 10 columns, 11 classifications (Alcohols, Aromatics, Carbonates, Dipolar Aprotics, Esters, Ethers, Halogenated, Hydrocarbons, Ketones, Other, water and acids)
- **Data quality:** 0 missing values, 0 fully duplicated rows, 2 duplicate-SMILES pairs (Ethanol/IMS denatured ethanol; 2-Methylpentane/Petroleum spirit), 0 invalid SMILES, 0 G-score outliers (IQR method)
- **G-score distribution:** mean = 6.0388, median = 5.9819, std = 1.2782, min = 3.0151, max = 8.7589, skewness = -0.1127 (approximately symmetric)

| Descriptor | Min | Mean | Max | Correlation with G-score |
|---|---:|---:|---:|---:|
| MolWt | 18.02 | 110.76 | 416.05 | 0.0272 |
| LogP  | -1.67 | 1.16 | 6.20 | -0.1902 |
| TPSA  | 0.00 | 21.59 | 78.90 | 0.5210 |

| Figure | Description |
|--------|-------------|
| ![G-score Overview](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/EDA/publication_gscore_overview.png) | G-score histogram, density, and boxplot |
| ![Descriptor Overview](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/EDA/publication_descriptor_overview.png) | MolWt, LogP, TPSA histograms |
| ![Correlation Matrix](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/EDA/publication_correlation_matrix.png) | Correlation matrix of descriptors and G-score |
| ![G-score Histogram](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/EDA/g_score_histogram.png) | G-score histogram (individual figure) |
| ![G-score Density](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/EDA/g_score_density.png) | G-score density plot |
| ![G-score Boxplot](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/EDA/g_score_boxplot.png) | G-score boxplot |
| ![Descriptor Histograms](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/EDA/molecular_descriptor_histograms.png) | MolWt/LogP/TPSA histograms (individual figure) |
| ![Descriptor Pairplot](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/EDA/molecular_descriptor_pairplot.png) | Pairplot of MolWt, LogP, TPSA, G-score |
| ![Descriptor Correlation](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/EDA/molecular_descriptor_correlation.png) | Descriptor correlation matrix (individual figure) |

## 8. Per-Pipeline Figures

### 8.1 Morgan Fingerprints (2048-bit, radius=2)

| Learning Curve | Parity & Residual Plots |
|---|---|
| ![Learning Curve](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/Morgan/learning_curve.png) | ![Parity & Residuals](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/Morgan/parity_residual_plots.png) |

### 8.2 RDKit Descriptors (MolWt, LogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds, RingCount)

| Learning Curve | Parity & Residual Plots |
|---|---|
| ![Learning Curve](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/RDKit/learning_curve.png) | ![Parity & Residuals](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/RDKit/parity_residual_plots.png) |

| Feature Importance (Gain) | SHAP Importance (Bar) |
|---|---|
| ![Feature Importance](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/RDKit/feature_importance.png) | ![SHAP Importance](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/RDKit/shap_importance_bar.png) |

**SHAP Summary Plot**

![SHAP Summary](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/RDKit/shap_summary.png)

### 8.3 ChemBERTa Embeddings (768-dim, `seyonec/ChemBERTa-zinc-base-v1`)

| Learning Curve | Prediction & Residual Plots |
|---|---|
| ![Learning Curve](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/ChemBERTa/learning_curve.png) | ![Prediction & Residuals](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/ChemBERTa/prediction_residual_plots.png) |

## 9. Descriptor Comparison Figures

| Model Comparison Metrics | CV Distributions |
|---|---|
| ![Model Comparison Metrics](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/Descriptor_Comparison_Figures/model_comparison_metrics.png) | ![CV Distributions](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/Descriptor_Comparison_Figures/cv_distributions.png) |

| Overfitting Gap | Computational Cost |
|---|---|
| ![Overfitting Gap](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/Descriptor_Comparison_Figures/overfitting_gap.png) | ![Computational Cost](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/Descriptor_Comparison_Figures/computational_cost.png) |

## 10. Overall Conclusion

**Morgan fingerprints currently produce the best raw-performance model** (CV R² = 0.5318, Test R² = 0.6933), but this comes with the largest feature dimensionality (2048) and a larger train/CV overfitting gap (0.4117) than RDKit. **RDKit descriptors (7-dim)** are statistically indistinguishable from Morgan on CV RMSE (paired t-test p = 0.0835), show the smallest overfitting gap (0.3635), the lowest computational cost, and are fully interpretable via named physicochemical properties (TPSA, LogP, RingCount being the strongest drivers — directly relevant to hydrogen-bonding behavior). **ChemBERTa embeddings (768-dim)** are significantly worse than both other pipelines (p < 0.001) and severely overfit (gap = 0.8524), suggesting that for this small (154-sample), structurally narrow solvent dataset, the pretrained transformer embeddings did not provide a predictive advantage without dimensionality reduction or stronger regularization.

**Recommendation:** For future solvent-property prediction and Deep Eutectic Solvent (DES) melting-point prediction studies, **RDKit Descriptors** are recommended as the primary representation given their comparable performance, superior robustness, minimal computational overhead, and full interpretability — properties that are especially valuable for relating model behavior to the hydrogen-bonding and polarity mechanisms (TPSA, LogP, HBD/HBA) that govern DES melting-point depression. Morgan Fingerprints remain a strong secondary/ensemble option if marginal predictive gains are prioritized over interpretability and cost.
