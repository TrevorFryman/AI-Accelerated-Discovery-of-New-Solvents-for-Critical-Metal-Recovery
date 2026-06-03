# Exploratory Data Analysis Walkthrough

We have successfully performed a complete exploratory data analysis (EDA) on the dataset `Melting_temperature_appended_35il_03082026.csv`. 

A Python script, [exploratory_analysis.py](file:///c:/dev/PersonalPrediction/exploratory_analysis.py), has been generated and executed to process the dataset and generate high-quality visual plots.

## Key Findings

### 1. Dataset Shape & Columns
- **Shape**: 2,006 rows and 13 columns.
- **Missing Values**: 0 (no missing values found in the dataset).
- **Duplicate Rows**: 0 (no duplicate rows found in the dataset).

### 2. Column Types
- **Numeric columns (7)**: `Number of components`, `X#1 (molar fraction)`, `X#2 (molar fraction)`, `Tmelt, K` (Target), `T#1`, `T#2`, `Type of DES` (currently encoded numerically).
- **Categorical columns (6)**: `Component#1`, `Component#2`, `Phase diagram (Yes/No)`, `Reference (DOI)`, `Smiles#1`, `Smiles#2`.

### 3. Summary Statistics for Numeric Variables
| Column | Count | Mean | Std | Min | 50% (Median) | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Number of components** | 2006.0 | 2.00 | 0.00 | 2.00 | 2.00 | 2.00 |
| **X#1 (molar fraction)** | 2006.0 | 0.48 | 0.19 | 0.01 | 0.47 | 0.98 |
| **X#2 (molar fraction)** | 2006.0 | 0.52 | 0.19 | 0.02 | 0.53 | 0.98 |
| **Tmelt, K** (Target) | 2006.0 | 354.01 | 78.71 | 182.51 | 331.40 | 603.00 |
| **T#1** | 2006.0 | 477.23 | 99.40 | 253.15 | 478.16 | 633.15 |
| **T#2** | 2006.0 | 365.40 | 85.34 | 223.15 | 348.00 | 633.15 |

## Generated Visualizations
All generated visualizations have been saved in the folder `results/EDA`.

1. **Target Variable Distribution**
   - File: [target_distribution_analysis.png](file:///c:/dev/PersonalPrediction/results/EDA/target_distribution_analysis.png)
   - Shows the histogram, KDE, and boxplot of the melting temperature `Tmelt, K`. The melting temperatures show a right-skewed distribution centered around 300-350 K.
2. **Correlation Heatmap**
   - File: [correlation_matrix.png](file:///c:/dev/PersonalPrediction/results/EDA/correlation_matrix.png)
   - Displays the Pearson correlation matrix for the numeric variables. Demonstrates perfect negative correlation (-1.00) between `X#1` and `X#2` because they represent molar fractions of a binary system.
3. **Variable Histograms**
   - Histograms for all numeric columns have been generated and saved individually in the [results/EDA](file:///c:/dev/PersonalPrediction/results/EDA) directory.

## Preprocessing Recommendations for Machine Learning

1. **Feature Engineering on SMILES**: Convert the SMILES strings (`Smiles#1`, `Smiles#2`) into molecular descriptors (e.g., Mordred, RDKit) or molecular fingerprints (e.g., Morgan fingerprints) to capture chemical structure.
2. **Drop Constant Columns**: Drop `Number of components` as it is constant (exactly `2.0` for all rows) and has no variance.
3. **Drop Reference Identifiers**: Drop ID columns such as `Reference (DOI)` to prevent the model from overfitting to specific studies.
4. **Address Multicollinearity**: Drop one of the molar fraction columns (`X#1` or `X#2`) since they sum to `1.0` and are perfectly colinear.
5. **Scale Features**: Scale the descriptors using a standard scaler (e.g., `StandardScaler` or `MinMaxScaler`) before inputting into algorithms sensitive to scale (such as SVMs, Ridge/Lasso, and Neural Networks).
6. **Encode Categorical Columns**: Encode binary categories like `Phase diagram (Yes/No)` using binary (0/1) encoding.
