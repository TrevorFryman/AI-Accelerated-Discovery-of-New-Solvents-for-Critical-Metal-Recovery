# ChemBERTa Embedding Baseline Model - Results

## Model

- Algorithm: XGBoost Regressor (default hyperparameters)
- random_state: 42
- Features: 768 ChemBERTa embedding dimensions
- Target: G-score

## 80/20 Train-Test Split

| Set | RMSE | MAE | R2 |
|---|---|---|---|
| Train | 0.0004 | 0.0003 | 1.0000 |
| Test | 1.2181 | 1.0165 | 0.1763 |

## 5-Fold Cross-Validation (Out-of-Fold)

| Metric | Value |
|---|---|
| RMSE | 1.2551 |
| MAE | 1.0212 |
| R2 | 0.0295 |

## Learning Curve / Fit Assessment

- Final training R2: 0.9981
- Final cross-validation R2: 0.0028
- Gap: 0.9953
- **Assessment: Overfitting**

The training R2 (0.998) is substantially higher than the cross-validation R2 (0.003), a gap of 0.995, indicating the model fits the training data much better than unseen data.

## Plots

- `results/ChemBERTa/learning_curve.png`
- `results/ChemBERTa/prediction_residual_plots.png`
