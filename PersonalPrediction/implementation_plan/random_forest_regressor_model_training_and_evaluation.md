# Random Forest Regressor Model Training and Evaluation

Create a Python script `random_forest_model.py` in the `models` folder to train a Random Forest model. The script will perform data cleaning, automatic preprocessing of numeric and categorical features, 5-fold cross validation, and generate evaluation plots and metrics.

## User Review Required

> [!IMPORTANT]
> The target variable is `Tmelt, K`.
> The feature selection will automatically drop high-cardinality text/identifying columns (`Component#1`, `Component#2`, `Reference (DOI)`, `Smiles#1`, `Smiles#2`), classifying the remaining features into numeric and categorical types.
>
> We will implement 5-Fold Cross-Validation:
> - For each fold, we fit the preprocessing and modeling pipeline on the training subset and validate on the holdout fold.
> - We will record the out-of-fold (OOF) predictions across the entire dataset to compute overall CV metrics (MAE, RMSE, R²).
> - The final model saved as `models/random_forest.pkl` will be trained on the *entire* dataset to leverage all available data points for future predictions.

## Open Questions

None at this stage.

## Proposed Changes

### Machine Learning Model Codebase

---

#### [NEW] [random_forest_model.py](file:///c:/dev/PersonalPrediction/models/random_forest_model.py)
This new script will perform the following:
1. **Load data**: Read the CSV `data/Melting_temperature_appended_35il_03082026.csv`.
2. **Handle target missing values**: Drop rows where the target `Tmelt, K` is missing.
3. **Feature Classification**:
   - Exclude ID/string columns: `Component#1`, `Component#2`, `Reference (DOI)`, `Smiles#1`, `Smiles#2`.
   - Categorical columns: `Type of DES`, `Phase diagram (Yes/No)`.
   - Numeric columns: `Number of components`, `X#1 (molar fraction)`, `X#2 (molar fraction)`, `T#1`, `T#2`.
4. **Preprocessing**:
   - Numeric: Median imputation and `StandardScaler`.
   - Categorical: Most frequent imputation and `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')`.
5. **5-Fold Cross Validation**:
   - Initialize `KFold(n_splits=5, shuffle=True, random_state=42)`.
   - Loop through folds, tracking fold-level MAE, RMSE, and R².
   - Accumulate out-of-fold predictions.
6. **Overall Metrics & Serialization**:
   - Print fold-level and mean CV metrics.
   - Save CV metrics to `results/RandomForest/evaluation_metrics.json`.
   - Train final model on the complete dataset.
   - Save model to `models/random_forest.pkl`.
7. **Visualization**:
   - Create **Actual vs Predicted plot** using OOF predictions, saving to `results/RandomForest/actual_vs_predicted.png`.
   - Extract feature importances from the final model, match them with the encoded feature names, and create a **Feature Importance plot**, saving to `results/RandomForest/feature_importance.png`.

## Verification Plan

### Automated Tests
- Run `python models/random_forest_model.py`.
- Verify successful generation of `models/random_forest.pkl`.
- Verify generation of results under `results/RandomForest/` (metrics JSON and both PNG plots).
- Check stdout for details of cross-validation folds.
