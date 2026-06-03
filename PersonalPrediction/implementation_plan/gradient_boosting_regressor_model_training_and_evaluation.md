# Gradient Boosting Regressor Model Training and Evaluation

Create a Python script `gradient_boosting_model.py` in the `models` folder to train a Gradient Boosting model on the DES dataset. The script will perform data cleaning, automatic feature preprocessing, an 80/20 train/test split, model training, performance evaluation, and save the serialized model and plots.

## User Review Required

> [!IMPORTANT]
> The target variable is `Tmelt, K`.
> High-cardinality identifying columns (`Component#1`, `Component#2`, `Reference (DOI)`, `Smiles#1`, `Smiles#2`) will be dropped, and the remaining features will be classified into numeric and categorical types.
>
> We will train a scikit-learn `GradientBoostingRegressor`. 
>
> We will generate two plots in the results folder (`results/GradientBoosting`):
> 1. **Actual vs Predicted plot**: Showing how closely the predicted melting temperatures match actual ones on the test set.
> 2. **Residual plot**: Showing predictions vs residuals ($y_{actual} - y_{predicted}$) on the test set, which helps diagnose non-linearity, heteroscedasticity, or outliers.
>
> The final trained pipeline (preprocessing + regressor) will be serialized as `models/gradient_boosting.pkl`.

## Open Questions

None.

## Proposed Changes

### Machine Learning Model Codebase

---

#### [NEW] [gradient_boosting_model.py](file:///c:/dev/PersonalPrediction/models/gradient_boosting_model.py)
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
5. **Split data**: 80/20 train/test split.
6. **Model training**: Fit a `GradientBoostingRegressor(n_estimators=100, random_state=42)` on the training set.
7. **Evaluation**:
   - Compute test set MAE, RMSE, and R².
   - Print metrics to console and save to `results/GradientBoosting/evaluation_metrics.json`.
8. **Visualization**:
   - Create **Actual vs Predicted plot**, saving to `results/GradientBoosting/actual_vs_predicted.png`.
   - Create **Residual plot** (Residuals vs Predicted), saving to `results/GradientBoosting/residual_plot.png`.
9. **Model Serialization**:
   - Save the pipeline to `models/gradient_boosting.pkl`.

## Verification Plan

### Automated Tests
- Run `python models/gradient_boosting_model.py`.
- Verify generation of `models/gradient_boosting.pkl`.
- Verify generation of results under `results/GradientBoosting/` (metrics JSON, prediction plot, residual plot).
- Inspect the evaluation metrics printout.
