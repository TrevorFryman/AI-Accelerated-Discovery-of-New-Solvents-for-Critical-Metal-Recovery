# Linear Regression Model Training and Evaluation

Create a Python script `linear_regression_model.py` in the `models` folder to train a Linear Regression model on the melting temperature dataset. The script will handle missing values, encode categorical variables, split the dataset, train the model, evaluate metrics, and save both the model and the results.

## User Review Required

> [!IMPORTANT]
> The target variable is `Tmelt, K`. The features are:
> - `Number of components` (numeric)
> - `Type of DES` (categorical: encoded via one-hot encoding)
> - `X#1 (molar fraction)` (numeric)
> - `X#2 (molar fraction)` (numeric)
> - `Phase diagram (Yes/No)` (categorical: encoded via one-hot/binary encoding)
> - `T#1` (numeric)
> - `T#2` (numeric)
>
> We will encapsulate the preprocessing (imputer, scaler, and encoder) and the estimator in a single `sklearn.pipeline.Pipeline` object. This ensures that any new data passed to the model for inference is processed consistently without manual step replication. The pipeline will be saved as `models/linear_regression.pkl`.

## Open Questions

None at this stage, the requirements are clear and complete.

## Proposed Changes

### Machine Learning Model Codebase

---

#### [NEW] [linear_regression_model.py](file:///c:/dev/PersonalPrediction/models/linear_regression_model.py)
This new script will perform the following actions:
1. **Load data**: Read the CSV `data/Melting_temperature_appended_35il_03082026.csv`.
2. **Handle target missing values**: Drop rows where the target `Tmelt, K` is missing (if any).
3. **Preprocessing Pipeline**:
   - Numeric features: Impute missing values with median, scale features using `StandardScaler`.
   - Categorical features: Impute missing values with most frequent value, encode categories using `OneHotEncoder(drop='first', handle_unknown='ignore')`.
4. **Split data**: Perform an 80/20 train/test split.
5. **Model training**: Train a `LinearRegression` model using the pipeline.
6. **Evaluation**:
   - Compute MAE, RMSE, and R² on the test set.
   - Print metrics to console.
   - Save metrics to `results/LinearRegression/evaluation_metrics.json`.
7. **Visualization**:
   - Create an Actual vs Predicted scatter plot using `matplotlib` and `seaborn`.
   - Save the plot to `results/LinearRegression/actual_vs_predicted.png`.
8. **Model Serialization**:
   - Save the complete pipeline as `models/linear_regression.pkl` using `pickle`.

## Verification Plan

### Automated Tests
- Run `python models/linear_regression_model.py` to execute the full workflow.
- Verify that the model `models/linear_regression.pkl` is saved successfully.
- Verify that `results/LinearRegression/evaluation_metrics.json` and `results/LinearRegression/actual_vs_predicted.png` are created and contain the correct content.
- Inspect the output metrics on the terminal.
