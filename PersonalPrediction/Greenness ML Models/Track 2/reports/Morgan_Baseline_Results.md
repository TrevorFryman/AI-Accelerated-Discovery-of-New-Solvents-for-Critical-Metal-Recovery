# Morgan Fingerprint Baseline Model - Results

## Model

- Algorithm: XGBoost Regressor (default hyperparameters)
- random_state: 42
- Features: 2048 Morgan fingerprint bits (radius=2, nBits=2048)
- Target: G-score

## 80/20 Train-Test Split

| Set | RMSE | MAE | R2 |
|---|---|---|---|
| Train | 0.0426 | 0.0329 | 0.9989 |
| Test | 0.9207 | 0.7488 | 0.5294 |

## 5-Fold Cross-Validation (Out-of-Fold)

| Metric | Value |
|---|---|
| RMSE | 0.9830 |
| MAE | 0.7533 |
| R2 | 0.4047 |

## Learning Curve / Fit Assessment

- Final training R2: 0.9930
- Final cross-validation R2: 0.4041
- Gap: 0.5889
- **Assessment: Overfitting**

The training R2 (0.993) is substantially higher than the cross-validation R2 (0.404), a gap of 0.589, indicating the model fits the training data much better than unseen data.

## Plots

- `results/Morgan/learning_curve.png`
- `results/Morgan/parity_residual_plots.png`
