# XGBoost Regressor Model Training and Evaluation

Create a Python script `xgboost_model.py` in the `models` folder to train an optimized XGBoost Regressor model on the DES dataset. The script will perform automatic feature preprocessing, 5-fold cross-validation grid search to tune key hyperparameters, evaluate performance on a holdout test set, and save the serialized model and plots.

## User Review Required

> [!IMPORTANT]
> The target variable is `Tmelt, K`.
> High-cardinality identifying columns (`Component#1`, `Component#2`, `Reference (DOI)`, `Smiles#1`, `Smiles#2`) will be dropped, and the remaining features will be classified into numeric and categorical types.
>
> We will train an `XGBRegressor` (XGBoost v3.2.0) with hyperparameter optimization via `GridSearchCV` with 5-fold cross-validation on the training set (80% split). The following parameters will be tuned:
> - `n_estimators`: [100, 200]
> - `max_depth`: [4, 6, 8]
> - `learning_rate`: [0.05, 0.1, 0.2]
>
> We will generate two plots in the results folder (`results/XGBoost`):
> 1. **Actual vs Predicted plot**: Showing predicted vs. actual melting temperatures on the test set using the best model.
> 2. **Feature Importance plot**: Visualizing the feature importances extracted from the best trained XGBoost estimator.
>
> The final optimized pipeline (preprocessing + XGBRegressor) will be serialized as `models/xgboost.pkl`.

## Open Questions

None.

## Proposed Changes

### Machine Learning Model Codebase

---

#### [NEW] [xgboost_model.py](file:///c:/dev/PersonalPrediction/models/xgboost_model.py)
This new script will perform the following actions:
1. **Load data**: Read the CSV `data/Melting_temperature_appended_35il_03082026.csv`.
2. **Handle target missing values**: Drop rows where the target `Tmelt, K` is missing.
3. **Feature Classification**:
   - Exclude ID/string columns: `Component#1`, `Component#2`, `Reference (DOI)`, `Smiles#1`, `Smiles#2`.
   - Categorical columns: `Type of DES`, `Phase diagram (Yes/No)`.
   - Numeric columns: `Number of components`, `X#1 (molar fraction)`, `X#2 (molar fraction)`, `T#1`, `T#2`.
4. **Preprocessing**:
   - Numeric: Median imputation and `StandardScaler`.
   - Categorical: Most frequent imputation and `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')`.
5. **Split data**: 80/20 train/test split (`random_state=42`).
6. **Hyperparameter Optimization (5-Fold GridSearchCV)**:
   - Run `GridSearchCV` with 5 folds and `scoring='neg_root_mean_squared_error'` over the parameter grid.
   - Output best hyperparameters and best cross-validation RMSE to console.
7. **Evaluation**:
   - Compute test set MAE, RMSE, and R² using the best estimator.
   - Print metrics to console and save alongside best hyperparameters to `results/XGBoost/evaluation_metrics.json`.
8. **Visualization**:
   - Create **Actual vs Predicted plot**, saving to `results/XGBoost/actual_vs_predicted.png`.
   - Create **Feature Importance plot** (sorted by importance), saving to `results/XGBoost/feature_importance.png`.
9. **Model Serialization**:
   - Save the best pipeline to `models/xgboost.pkl`.

## Verification Plan

### Automated Tests
- Run `python models/xgboost_model.py`.
- Verify generation of `models/xgboost.pkl`.
- Verify generation of results under `results/XGBoost/` (metrics JSON, prediction plot, feature importance plot).
- Inspect the console output showing best hyperparameters and final test metrics.

### Results Achieved
- **Best Parameters**: `n_estimators=200`, `max_depth=8`, `learning_rate=0.1`
- **Best CV RMSE**: `19.40 K`
- **Test MAE**: `11.23 K` | **Test RMSE**: `17.53 K` | **Test R²**: `0.9510`
