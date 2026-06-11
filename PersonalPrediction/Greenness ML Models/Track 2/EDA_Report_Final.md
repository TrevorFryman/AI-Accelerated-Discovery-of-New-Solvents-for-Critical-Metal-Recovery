# Final EDA Report: GSK_dataset.csv

## 1. Dataset Overview

- **Rows:** 154
- **Columns:** 10
- **Columns:** `Unnamed: 0` (row index), `Classification`, `solvent_common_name`, `IPUAC name`, `solvent_SMILES`, `CAS Number`, `G-score`
- **Classifications represented:** 11 (Alcohols, Aromatics, Carbonates, Dipolar Aprotics, Esters, Ethers, Halogenated, Hydrocarbons, Ketones, Other, water and acids)

## 2. Data Quality Findings

- Missing values: 0
- Fully duplicated rows: 0
- Duplicate `solvent_SMILES` values: 2 (Ethanol/IMS denatured ethanol, and 2-Methylpentane/Petroleum spirit)
- Invalid SMILES (RDKit): 0
- G-score outliers (IQR method): 0

Full details: see `Data_Quality_Report.md`.

## 3. G-score Distribution

| Statistic | Value |
|---|---|
| Mean | 6.0388 |
| Median | 5.9819 |
| Standard Deviation | 1.2782 |
| Min | 3.0151 |
| Max | 8.7589 |
| Skewness | -0.1127 |

The G-score distribution is approximately symmetric (skewness = -0.1127), with mean (6.0388) and median (5.9819) close together and no outliers detected via the IQR method. Full details: see `target_variable_report.md`.

## 4. Molecular Diversity Analysis

| Descriptor | Min | Mean | Max | Correlation with G-score |
|---|---|---|---|---|
| MolWt | 18.02 | 110.76 | 416.05 | 0.0272 |
| LogP | -1.67 | 1.16 | 6.20 | -0.1902 |
| TPSA | 0.00 | 21.59 | 78.90 | 0.5210 |

TPSA shows the strongest correlation with G-score (positive), suggesting that more polar solvents tend to have higher (greener) G-scores. LogP is moderately negatively correlated, while MolWt shows essentially no correlation. Full details: see `molecular_diversity_report.md`.

## 5. Recommendations for Machine Learning

- Drop the unnamed index column (`Unnamed: 0`) before modeling.
- Review the duplicate-SMILES pairs (Ethanol/IMS, 2-Methylpentane/Petroleum spirit) — consider whether to merge, deduplicate, or retain as distinct samples, since they represent the same molecule under different commercial names.
- No missing values, invalid SMILES, or G-score outliers were found, so the dataset is largely ready for featurization without imputation or filtering.
- Molecular descriptors (MolWt, LogP, TPSA) provide a reasonable starting feature set; TPSA in particular shows a meaningful linear relationship with G-score and should be retained as a feature.
- Given the small dataset size (154 samples), favor simple, low-variance models (e.g., linear/regularized regression, small tree ensembles) and use cross-validation rather than a large held-out test set.
- Consider expanding the feature set with additional RDKit descriptors or fingerprints, since MolWt/LogP/TPSA alone show moderate correlations with G-score.

## 6. Figures

Publication-quality figures (saved to `results/EDA/`):

- `publication_gscore_overview.png` — G-score histogram, density, and boxplot
- `publication_descriptor_overview.png` — MolWt, LogP, TPSA histograms
- `publication_correlation_matrix.png` — Correlation matrix of descriptors and G-score

Additional individual EDA figures generated in earlier tasks are also available in `results/EDA/`.
