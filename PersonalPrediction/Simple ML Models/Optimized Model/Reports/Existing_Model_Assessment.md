# Existing Model Assessment

*Generated: 2026-06-17 10:32*

## 1. Model Architecture
- **Type**: `XGBRegressor` wrapped in a `sklearn.Pipeline`
- **Pipeline steps**: preprocessor → regressor

## 2. Preprocessing Pipeline
| Stage | Numeric Features | Categorical Features |
|---|---|---|
| Imputation | Median | Most-frequent |
| Scaling | StandardScaler | OneHotEncoder (drop=first) |

## 3. Input Features
**Numeric** (5): `Number of components`, `X#1 (molar fraction)`, `X#2 (molar fraction)`, `T#1`, `T#2`

**Categorical** (2): `Type of DES`, `Phase diagram (Yes/No)`

**Excluded** (ID / SMILES / target): `Component#1`, `Component#2`, `Reference (DOI)`, `Smiles#1`, `Smiles#2`

## 4. Target Variable
- **`Tmelt, K`** (melting temperature, Kelvin)
- Range in dataset: 182.5 K – 603.0 K

## 5. Stored Hyperparameters
| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 8 |
| `learning_rate` | 0.1 |
| `subsample` | None |
| `colsample_bytree` | None |
| `min_child_weight` | None |
| `gamma` | None |
| `reg_alpha` | None |
| `reg_lambda` | None |

## 6. Baseline Performance
| Split | RMSE (K) | MAE (K) | R² |
|---|---|---|---|
| Train | 3.141 | 2.053 | 0.9984 |
| Test  | 17.533 | 11.228 | 0.9510 |
| 5-Fold CV (mean) | 18.642 ± 1.359 | 11.537 | 0.9428 ± 0.0106 |

> **Overfitting gap**: Train R² − Test R² = **0.0474** — moderate overfitting detected.

## 7. Descriptor Utilization Check
| Descriptor Type | Used? |
|---|---|
| SMILES strings | No — excluded from features |
| Morgan fingerprints | No |
| RDKit descriptors | No (dataset available but not used) |
| ChemBERTa embeddings | No |

> **Note**: RDKit features file found at `data/RDKitDescriptorGeneration/DES_RDKit_Features.csv` with 14 additional molecular descriptors per component (MolWt, LogP, TPSA, HBD, HBA, RotBonds, RingCount for each component). These are not currently used.

## 8. Dataset Compatibility
- **Training dataset**: `Melting_temperature_appended_35il_03082026.csv` — 2006 samples ✓
- **Candidate dataset**: Same file used for screening (no separate candidate file provided) ✓
- **RDKit features dataset**: Same 2006 samples with additional descriptors ✓

## 9. Decision
**→ OPTIMIZE: Further optimize the existing XGBoost model.**

Overfitting gap (train R²=0.9984 vs test R²=0.9510) of 0.0474 indicates the existing model is overfit. The original GridSearchCV searched only 18 combinations; a wider RandomizedSearchCV with regularization parameters has not been attempted.

### Optimization Plan
1. Expand hyperparameter search to include regularization (`gamma`, `reg_alpha`, `reg_lambda`, `min_child_weight`)
2. Use `RandomizedSearchCV` with `n_iter=120` over a broad search space
3. Optimize for 5-fold CV RMSE (generalization, not training accuracy)
4. Retrain best model on 100% of data for deployment