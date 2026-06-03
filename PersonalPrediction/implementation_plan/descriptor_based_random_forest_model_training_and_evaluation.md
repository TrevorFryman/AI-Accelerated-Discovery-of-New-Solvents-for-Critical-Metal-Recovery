# Descriptor-Based Random Forest Regressor Model Training and Evaluation

Create a Python script `descriptor_based_random_forest.py` in the `models` folder to train a Random Forest model using RDKit molecular descriptors plus additional structural features. The script will perform data cleaning, automatic preprocessing of numeric and categorical features, hyperparameter tuning, 5-fold cross validation, and generate evaluation plots and metrics.

## User Review Required

> [!IMPORTANT]
> The target variable is `Tmelt, K`.
>
> The model uses the following feature set:
> - **All RDKit Descriptors** (14 features):
>   - Component 1: `C1_MolWt`, `C1_LogP`, `C1_TPSA`, `C1_HBD`, `C1_HBA`, `C1_RotBonds`, `C1_RingCount`
>   - Component 2: `C2_MolWt`, `C2_LogP`, `C2_TPSA`, `C2_HBD`, `C2_HBA`, `C2_RotBonds`, `C2_RingCount`
> - **Additional Features** (5 features):
>   - `X#1 (molar fraction)` — Molar fraction of component 1
>   - `X#2 (molar fraction)` — Molar fraction of component 2
>   - `Type of DES` — Categorical: type of Deep Eutectic Solvent
>   - `Number of components` — Number of DES components
>   - `Phase diagram (Yes/No)` — Categorical: phase diagram availability
>
> **Total: 19 features**
>
> We will implement:
> - **Hyperparameter Tuning**: GridSearchCV with 5-fold CV over parameter combinations for `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.
> - **5-Fold Cross-Validation**: Using the best hyperparameters, we perform 5-fold CV and record out-of-fold (OOF) predictions across the entire dataset to compute overall CV metrics (MAE, RMSE, R²).
> - **Feature Importance Analysis**: Extract and visualize the most important features from the final model trained on the complete dataset.
> - The final model saved as `models/descriptor_based_random_forest.pkl` will be trained on the *entire* dataset to leverage all available data points for future predictions.

## Open Questions

None at this stage.

## Proposed Changes

### Machine Learning Model Codebase

---

#### [NEW] [descriptor_based_random_forest.py](file:///c:/dev/PersonalPrediction/models/descriptor_based_random_forest.py)
This new script will perform the following:
1. **Load data**: Read the CSV `data/RDKitDescriptorGeneration/DES_RDKit_Features.csv`.
2. **Handle target missing values**: Drop rows where the target `Tmelt, K` is missing.
3. **Feature Selection**:
   - Select all 14 RDKit descriptors (C1_* and C2_*).
   - Select additional structural features: `X#1 (molar fraction)`, `X#2 (molar fraction)`, `Type of DES`, `Number of components`, `Phase diagram (Yes/No)`.
   - Total of 19 features used in the model.
4. **Preprocessing**:
   - Numeric: Median imputation and `StandardScaler`.
   - Categorical: Most frequent imputation and `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')`.
5. **Hyperparameter Tuning**:
   - Initialize `GridSearchCV` with `RandomForestRegressor` and parameter grid: `n_estimators` ∈ [100, 200], `max_depth` ∈ [10, 15, 20], `min_samples_split` ∈ [2, 5], `min_samples_leaf` ∈ [1, 2].
   - Perform 5-fold CV to identify the best hyperparameter combination.
6. **5-Fold Cross Validation**:
   - Initialize `KFold(n_splits=5, shuffle=True, random_state=42)`.
   - Loop through folds using the best hyperparameters, tracking fold-level MAE, RMSE, and R².
   - Accumulate out-of-fold predictions.
7. **Overall Metrics & Serialization**:
   - Print fold-level and mean CV metrics.
   - Save CV metrics (including best hyperparameters and best CV R² score) to `results/DescriptorBasedRandomForest/evaluation_metrics.json`.
   - Train final model on the complete dataset using best hyperparameters.
   - Save model to `models/descriptor_based_random_forest.pkl`.
8. **Visualization**:
   - Create **Actual vs Predicted plot** using OOF predictions, saving to `results/DescriptorBasedRandomForest/actual_vs_predicted.png`.
   - Extract feature importances from the final model, match them with the encoded feature names, and create a **Feature Importance plot**, saving to `results/DescriptorBasedRandomForest/feature_importance.png`.
   - Save feature importances to CSV: `results/DescriptorBasedRandomForest/feature_importances.csv`.

## Verification Plan

### Automated Tests
- Run `python models/descriptor_based_random_forest.py`.
- Verify successful generation of `models/descriptor_based_random_forest.pkl`.
- Verify generation of results under `results/DescriptorBasedRandomForest/` (metrics JSON, PNG plots, and CSV feature importances).
- Check stdout for details of hyperparameter tuning results, cross-validation folds, and metrics summary.
