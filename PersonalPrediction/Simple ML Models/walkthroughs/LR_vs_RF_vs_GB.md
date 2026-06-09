# Walkthrough: Model Comparisons (Linear Regression vs. Random Forest vs. Gradient Boosting)

We have trained and evaluated three models on the DES dataset to predict melting temperature (`Tmelt, K`):
1. **Linear Regression** (Standard pipeline)
2. **Random Forest Regressor** (5-Fold Cross-Validation, OOF predictions)
3. **Gradient Boosting Regressor** (80/20 train/test split)

## Performance Comparison

Here is the comparative summary of results across all three models:

| Metric | Linear Regression (Test Set) | Random Forest Regressor (5-Fold CV OOF) | Gradient Boosting Regressor (Test Set) |
|---|---|---|---|
| **Mean Absolute Error (MAE)** | 39.17 K | **13.65 K** | 15.97 K |
| **Root Mean Squared Error (RMSE)** | 48.77 K | **21.41 K** | 23.25 K |
| **R² Coefficient of Determination** | 0.6207 | **0.9260** | 0.9138 |

> [!TIP]
> Both tree-based ensemble models (Random Forest and Gradient Boosting) achieve outstanding performance ($R^2 > 0.91$, MAE $< 16\text{ K}$), dramatically outperforming the simpler Linear Regression model ($R^2 \approx 0.62$). Random Forest achieves the top overall performance.

---

## 1. Gradient Boosting Model Details

- **Automatic Pre-processing**: Dropped high-cardinality text/identifying features (`Component#1`, `Component#2`, `Reference (DOI)`, `Smiles#1`, `Smiles#2`). Automatically handled categorical and numerical splits with imputation and standard scaling.
- **Residual Analysis**: Calculated residuals ($y_{test} - y_{pred}$) and generated a Residuals vs. Predicted Plot. The residuals are centered around zero, indicating that the model holds low systematic bias across the range of melting points.
- **Model Serialization**: The final trained pipeline is saved at [gradient_boosting.pkl](file:///c:/dev/PersonalPrediction/models/gradient_boosting.pkl).
- **Evaluation Metrics JSON**: [evaluation_metrics.json](file:///c:/dev/PersonalPrediction/results/GradientBoosting/evaluation_metrics.json).

---

## 2. Visualizations

Use the carousel below to view the evaluation plots for all three models:

````carousel
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
