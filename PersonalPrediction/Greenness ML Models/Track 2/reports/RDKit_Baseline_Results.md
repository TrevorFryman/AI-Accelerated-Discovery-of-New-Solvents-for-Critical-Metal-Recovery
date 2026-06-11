# RDKit Descriptor Baseline Model - Results

## Model

- Algorithm: XGBoost Regressor (default hyperparameters)
- random_state: 42
- Features: 7 RDKit descriptors (MolWt, LogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds, RingCount)
- Target: G-score

## 80/20 Train-Test Split

| Set | RMSE | MAE | R2 |
|---|---|---|---|
| Train | 0.1279 | 0.0247 | 0.9896 |
| Test | 0.9805 | 0.7860 | 0.4663 |

## 5-Fold Cross-Validation (Out-of-Fold)

| Metric | Value |
|---|---|
| RMSE | 0.9632 |
| MAE | 0.7529 |
| R2 | 0.4284 |

## Learning Curve / Fit Assessment

- Final training R2: 0.9917
- Final cross-validation R2: 0.4132
- Gap: 0.5785
- **Assessment: Overfitting**

The training R2 (0.992) is substantially higher than the cross-validation R2 (0.413), a gap of 0.579, indicating the model fits the training data much better than unseen data.

## Plots

- `results/RDKit/learning_curve.png`
- `results/RDKit/parity_residual_plots.png`
