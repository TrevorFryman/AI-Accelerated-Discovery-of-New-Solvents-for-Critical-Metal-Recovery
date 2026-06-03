# Walkthrough: Linear Regression Model Training and Evaluation

We have successfully built, trained, and evaluated the Linear Regression model pipeline for predicting the melting temperature (`Tmelt, K`).

## Summary of Changes

1. **Created model script**: [linear_regression_model.py](file:///c:/dev/PersonalPrediction/models/linear_regression_model.py)
   - Loads the melting temperature dataset from the `data` folder.
   - Drops rows with missing targets.
   - Built a robust preprocessing pipeline for both numeric and categorical columns.
   - Fits a scikit-learn Linear Regression model.
   - Evaluates performance metrics (MAE, RMSE, R²).
   - Plots actual vs predicted values.
   - Serializes the entire pipeline to `models/linear_regression.pkl`.
   - Saves results under `results/LinearRegression/`.

## Evaluation Results

The pipeline achieved the following performance on the test set (20% split):

| Metric | Value |
|---|---|
| **Mean Absolute Error (MAE)** | 39.17 K |
| **Root Mean Squared Error (RMSE)** | 48.77 K |
| **R² Coefficient of Determination** | 0.6207 |

The metrics are also saved as JSON in [evaluation_metrics.json](file:///c:/dev/PersonalPrediction/results/LinearRegression/evaluation_metrics.json).

## Visualizations

Below is the **Actual vs. Predicted Melting Temperature** plot saved in `results/LinearRegression/actual_vs_predicted.png`:

![Actual vs Predicted Plot](C:/Users/trevo/.gemini/antigravity-ide/brain/a0a400d5-fbee-46a0-be5a-fe2d1b0ee064/actual_vs_predicted.png)

## Serialized Model Artifact

The full preprocessing and estimator pipeline has been saved as a joblib-compatible pickle file:
- [linear_regression.pkl](file:///c:/dev/PersonalPrediction/models/linear_regression.pkl)
