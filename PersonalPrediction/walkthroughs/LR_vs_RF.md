# Walkthrough: Model Comparisons (Linear Regression vs. Random Forest)

We have trained and evaluated both a **Linear Regression** model and a **Random Forest Regressor** model to predict the melting temperature (`Tmelt, K`) of Deep Eutectic Solvents (DES).

## Performance Comparison

The table below summarizes the test performance of the Linear Regression model alongside the 5-fold cross-validated out-of-fold (OOF) performance of the Random Forest model:

| Metric | Linear Regression (Test Set) | Random Forest Regressor (5-Fold CV OOF) |
|---|---|---|
| **Mean Absolute Error (MAE)** | 39.17 K | **13.65 K** |
| **Root Mean Squared Error (RMSE)** | 48.77 K | **21.41 K** |
| **R² Coefficient of Determination** | 0.6207 | **0.9260** |

> [!TIP]
> The Random Forest Regressor shows a substantial improvement over Linear Regression, reducing MAE by **~65%** and explaining **92.6%** of the variance in melting temperature.

---

## 1. Random Forest Model details

- **Pre-processing**: High-cardinality text/identifying features (`Component#1`, `Component#2`, `Reference (DOI)`, `Smiles#1`, `Smiles#2`) were automatically dropped. Numerical columns were median-imputed and standardized. Categorical columns were most-frequent imputed and one-hot encoded.
- **Model Validation**: Evaluated using 5-fold cross-validation, reporting a standard deviation of only `~1.13 K` MAE across folds, demonstrating high model stability.
- **Model Serialization**: The final model trained on all data points is saved at [random_forest.pkl](file:///c:/dev/PersonalPrediction/models/random_forest.pkl).
- **Evaluation Metrics JSON**: [evaluation_metrics.json](file:///c:/dev/PersonalPrediction/results/RandomForest/evaluation_metrics.json).

### Feature Importances

According to the Random Forest model:
1. **T#2** (melting point of component 2) and **T#1** (melting point of component 1) are the most significant predictors of the DES melting temperature.
2. **X#2 (molar fraction)** and **X#1 (molar fraction)** also show moderate predictive importance.

---

## 2. Visualizations

Use the carousel below to view the Random Forest feature importances, the Random Forest actual vs. predicted plot, and the Linear Regression actual vs. predicted plot.

````carousel
![Random Forest Feature Importance](C:/Users/trevo/.gemini/antigravity-ide/brain/a0a400d5-fbee-46a0-be5a-fe2d1b0ee064/rf_feature_importance.png)
<!-- slide -->
![Random Forest Actual vs Predicted](C:/Users/trevo/.gemini/antigravity-ide/brain/a0a400d5-fbee-46a0-be5a-fe2d1b0ee064/rf_actual_vs_predicted.png)
<!-- slide -->
![Linear Regression Actual vs Predicted](C:/Users/trevo/.gemini/antigravity-ide/brain/a0a400d5-fbee-46a0-be5a-fe2d1b0ee064/actual_vs_predicted.png)
````
