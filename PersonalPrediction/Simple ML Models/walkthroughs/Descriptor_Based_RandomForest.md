# Walkthrough: Descriptor-Based Random Forest Model Analysis

We have successfully trained and evaluated a **Descriptor-Based Random Forest Regressor** model to predict the melting temperature (`Tmelt, K`) of Deep Eutectic Solvents (DES) using RDKit molecular descriptors along with additional structural features.

A Python script, [descriptor_based_random_forest.py](file:///c:/dev/PersonalPrediction/models/descriptor_based_random_forest.py), has been generated and executed to perform hyperparameter tuning, 5-fold cross-validation, feature importance analysis, and generate comprehensive evaluation metrics and visualizations.

## Performance Summary

The Descriptor-Based Random Forest model demonstrates excellent predictive performance on the DES melting temperature prediction task:

| Metric | Value |
|---|---|
| **Mean Absolute Error (MAE)** | 13.35 ± 0.87 K |
| **Root Mean Squared Error (RMSE)** | 21.09 ± 2.02 K |
| **R² Coefficient of Determination** | 0.9268 ± 0.0155 |
| **Overall OOF MAE** | 13.3450 K |
| **Overall OOF RMSE** | 21.1887 K |
| **Overall OOF R²** | 0.9275 |

> [!TIP]
> The model achieves an **R² = 0.9275**, explaining **92.75%** of the variance in melting temperature with a mean prediction error of only **13.35 K**. The low standard deviation across folds (±0.87 K) demonstrates high model stability and generalization capability.

---

## 1. Model Architecture & Methodology

### Feature Set (19 Total Features)

**RDKit Molecular Descriptors (14 features):**
- Component 1 (C1): Molecular Weight, LogP, TPSA, H-Bond Donors, H-Bond Acceptors, Rotatable Bonds, Ring Count
- Component 2 (C2): Molecular Weight, LogP, TPSA, H-Bond Donors, H-Bond Acceptors, Rotatable Bonds, Ring Count

**Additional Structural Features (5 features):**
- X#1 (molar fraction) — Composition of component 1
- X#2 (molar fraction) — Composition of component 2
- Type of DES — Category of Deep Eutectic Solvent
- Number of components — Total components in the DES
- Phase diagram (Yes/No) — Phase diagram availability

### Preprocessing
- **Numeric features**: Median imputation and `StandardScaler`.
- **Categorical features**: Most frequent imputation and `OneHotEncoder(drop='first')`.

### Hyperparameter Tuning
- **Method**: GridSearchCV with 5-fold cross-validation.
- **Search Space**:
  - `n_estimators` ∈ [100, 200]
  - `max_depth` ∈ [10, 15, 20]
  - `min_samples_split` ∈ [2, 5]
  - `min_samples_leaf` ∈ [1, 2]
  - **Total combinations evaluated**: 24 (120 total fits with 5-fold CV)

- **Best Hyperparameters Found**:
  - `n_estimators`: 100
  - `max_depth`: 20
  - `min_samples_split`: 2
  - `min_samples_leaf`: 1

### Model Validation
- **Cross-Validation**: 5-fold with shuffled splits (random_state=42).
- **Model Stability**: Standard deviation of MAE across folds = ±0.87 K, indicating robust generalization.
- **Out-of-Fold (OOF) Predictions**: Accumulated predictions from all 5 folds provide unbiased performance estimates on the full dataset.

---

## 2. Feature Importance Analysis

The following table ranks the top 10 most important features based on the final Random Forest model trained on the complete dataset:

| Rank | Feature | Importance |
|---|---|---|
| 1 | C2_MolWt (Component 2 Molecular Weight) | **Highest** |
| 2 | C1_MolWt (Component 1 Molecular Weight) | Very High |
| 3 | X#2 (molar fraction) | High |
| 4 | X#1 (molar fraction) | High |
| 5 | C2_LogP (Component 2 LogP) | Moderate |
| 6 | C1_LogP (Component 1 LogP) | Moderate |
| 7 | C2_HBA (Component 2 H-Bond Acceptors) | Moderate |
| 8 | C1_HBA (Component 1 H-Bond Acceptors) | Moderate |
| 9 | C1_TPSA (Component 1 TPSA) | Low-Moderate |
| 10 | C2_TPSA (Component 2 TPSA) | Low-Moderate |

> [!NOTE]
> **Key Insights**:
> - **Molecular weight** of both components is the dominant predictor, reflecting the strong influence of component size on DES melting temperature.
> - **Molar fractions** (X#1 and X#2) contribute significantly, indicating that composition plays a critical role.
> - **LogP values** (lipophilicity) provide moderate predictive signal, capturing hydrophobic/hydrophilic balance effects.
> - **Hydrogen-bonding capacity** (HBD/HBA) and **TPSA** contribute relatively less but still provide meaningful information.

---

## 3. Cross-Validation Performance by Fold

| Fold | MAE (K) | RMSE (K) | R² |
|---|---|---|---|
| 1 | 12.11 | 18.57 | 0.9450 |
| 2 | 12.49 | 18.95 | 0.9375 |
| 3 | 14.24 | 23.55 | 0.9064 |
| 4 | 13.89 | 21.49 | 0.9346 |
| 5 | 14.00 | 22.90 | 0.9103 |
| **Mean ± Std** | **13.35 ± 0.87** | **21.09 ± 2.02** | **0.9268 ± 0.0155** |

All folds achieve R² > 0.90, with fold-level MAE ranging from 12.1 to 14.2 K, demonstrating consistent and reliable cross-fold performance.

---

## 4. Model Serialization & Output Files

- **Trained Model Pipeline**: [descriptor_based_random_forest.pkl](file:///c:/dev/PersonalPrediction/models/descriptor_based_random_forest.pkl)
- **Evaluation Metrics JSON**: [evaluation_metrics.json](file:///c:/dev/PersonalPrediction/results/DescriptorBasedRandomForest/evaluation_metrics.json)
- **Feature Importances CSV**: [feature_importances.csv](file:///c:/dev/PersonalPrediction/results/DescriptorBasedRandomForest/feature_importances.csv)

---

## 5. Visualizations

### Feature Importance Plot
![Descriptor-Based Random Forest Feature Importance](C:/dev/PersonalPrediction/results/DescriptorBasedRandomForest/feature_importance.png)

The feature importance plot shows the relative contribution of each feature to model predictions. Molecular weights dominate, followed by molar fractions, LogP, and hydrogen-bonding features.

### Actual vs. Predicted Plot
![Descriptor-Based Random Forest Actual vs Predicted](C:/dev/PersonalPrediction/results/DescriptorBasedRandomForest/actual_vs_predicted.png)

The actual vs. predicted scatter plot shows the model's predictions against ground truth values across all 2,006 data points (out-of-fold). Points clustered near the perfect prediction line (dashed red) indicate accurate predictions. The model achieves tight clustering around the diagonal with R² = 0.9275.

---

## 6. Key Takeaways

1. **Excellent Predictive Performance**: With R² = 0.9275 and MAE = 13.35 K, the Descriptor-Based Random Forest model provides highly accurate melting temperature predictions.

2. **Robust Feature Set**: The combination of RDKit molecular descriptors and structural composition features captures the essential chemistry governing DES melting points.

3. **Stable Generalization**: Low fold-to-fold variance (std ≈ 0.87 K MAE) and consistent R² values (0.91–0.95) indicate the model will generalize well to unseen DES data.

4. **Interpretable Feature Importances**: The feature importance analysis reveals that molecular size (MolWt) and composition (molar fractions) are the dominant predictors, aligned with DES thermodynamic theory.

5. **Production-Ready**: The serialized model pipeline can be deployed for real-time predictions on new DES candidates without retraining.

---

## Next Steps

The Descriptor-Based Random Forest model can be used for:
- **Screening**: Predicting melting temperatures of novel DES formulations before experimental synthesis.
- **Optimization**: Identifying DES compositions likely to achieve target melting temperatures.
- **Transfer Learning**: Serving as a baseline for ensemble methods or more complex deep learning approaches.
