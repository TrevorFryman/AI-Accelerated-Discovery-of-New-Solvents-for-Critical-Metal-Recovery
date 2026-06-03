# Walkthrough: Full Model Comparison

We have trained and evaluated four models on the DES dataset to predict melting temperature (`Tmelt, K`):
1. **Linear Regression** — Baseline pipeline
2. **Random Forest Regressor** — 5-Fold CV with OOF predictions
3. **Gradient Boosting Regressor** — 80/20 train/test split
4. **XGBoost Regressor** — 5-Fold GridSearchCV hyperparameter tuning

## Performance Comparison

| Metric | Linear Regression | Random Forest (5-Fold CV OOF) | Gradient Boosting | **XGBoost (Tuned)** |
|---|---|---|---|---|
| **MAE** | 39.17 K | 13.65 K | 15.97 K | **11.23 K** |
| **RMSE** | 48.77 K | 21.41 K | 23.25 K | **17.53 K** |
| **R²** | 0.6207 | 0.9260 | 0.9138 | **0.9510** |

> [!TIP]
> **XGBoost** achieves the best overall test performance with **R² = 0.951** and **MAE = 11.23 K**, after 5-fold hyperparameter tuning over 18 parameter combinations (90 total fits). This is a ~18% MAE improvement over Random Forest.

---

## 1. XGBoost Model Details

- **Hyperparameter Search Space**: `n_estimators` ∈ [100, 200], `max_depth` ∈ [4, 6, 8], `learning_rate` ∈ [0.05, 0.1, 0.2]
- **Best Parameters (5-Fold CV)**: `n_estimators=200`, `max_depth=8`, `learning_rate=0.1`
- **Best CV RMSE**: `19.40 K`
- **Automatic Pre-processing**: Dropped high-cardinality text/SMILES columns automatically. Numerical features were median-imputed and scaled; categorical features were one-hot encoded.
- **Model Serialization**: Best pipeline saved at [xgboost.pkl](file:///c:/dev/PersonalPrediction/models/xgboost.pkl).
- **Evaluation Metrics JSON**: [evaluation_metrics.json](file:///c:/dev/PersonalPrediction/results/XGBoost/evaluation_metrics.json).

---

## 2. Feature Importances (XGBoost)

The XGBoost model confirms the same key driver patterns seen in Random Forest:
- **T#1** and **T#2** (pure component melting points) are by far the most predictive features.
- **X#1 / X#2** (molar fractions) contribute moderately.
- `Type of DES` and `Phase diagram` add marginal gain.

---

## 3. Visualizations

Use the carousel below to see all evaluation plots across all four models:

````carousel
![XGBoost Feature Importance](C:/Users/trevo/.gemini/antigravity-ide/brain/a0a400d5-fbee-46a0-be5a-fe2d1b0ee064/xgb_feature_importance.png)
<!-- slide -->
![XGBoost Actual vs Predicted](C:/Users/trevo/.gemini/antigravity-ide/brain/a0a400d5-fbee-46a0-be5a-fe2d1b0ee064/xgb_actual_vs_predicted.png)
<!-- slide -->
![Gradient Boosting Actual vs Predicted](C:/Users/trevo/.gemini/antigravity-ide/brain/a0a400d5-fbee-46a0-be5a-fe2d1b0ee064/gb_actual_vs_predicted.png)
<!-- slide -->
![Gradient Boosting Residual Plot](C:/Users/trevo/.gemini/antigravity-ide/brain/a0a400d5-fbee-46a0-be5a-fe2d1b0ee064/gb_residual_plot.png)
<!-- slide -->
![Random Forest Feature Importance](C:/Users/trevo/.gemini/antigravity-ide/brain/a0a400d5-fbee-46a0-be5a-fe2d1b0ee064/rf_feature_importance.png)
<!-- slide -->
![Random Forest Actual vs Predicted](C:/Users/trevo/.gemini/antigravity-ide/brain/a0a400d5-fbee-46a0-be5a-fe2d1b0ee064/rf_actual_vs_predicted.png)
<!-- slide -->
![Linear Regression Actual vs Predicted](C:/Users/trevo/.gemini/antigravity-ide/brain/a0a400d5-fbee-46a0-be5a-fe2d1b0ee064/actual_vs_predicted.png)
````
