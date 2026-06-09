import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda():
    # 0. Setup directories
    output_dir = os.path.join('results', 'EDA')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    csv_path = os.path.join('data', 'Melting_temperature_appended_35il_03082026.csv')
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # 1. Dataset shape
    print("\n1. Dataset Shape:")
    print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # 2. Column names
    print("\n2. Column Names:")
    for col in df.columns:
        print(f"   - {col}")
        
    # 3. Missing values
    print("\n3. Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Values': missing, 'Percentage (%)': missing_pct})
    print(missing_df[missing_df['Missing Values'] > 0])
    if missing.sum() == 0:
        print("   No missing values detected in the dataset.")
        
    # 4. Duplicate rows
    print("\n4. Duplicate Rows:")
    duplicates = df.duplicated().sum()
    print(f"   Number of duplicate rows: {duplicates}")
    if duplicates > 0:
        print(f"   Percentage of duplicates: {(duplicates / len(df)) * 100:.2f}%")
        
    # 5. Summarize all numeric columns
    print("\n5. Summary Statistics for Numeric Columns:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(df[numeric_cols].describe().T)
    
    # 6. Identify categorical columns
    print("\n6. Categorical Columns:")
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in categorical_cols:
        unique_vals = df[col].nunique()
        print(f"   - {col}: {unique_vals} unique values (Example: {df[col].dropna().iloc[0] if unique_vals > 0 else 'N/A'})")
        
    # 7. Create histograms for numeric variables
    print("\n7. Generating Histograms...")
    # Plot style settings
    sns.set_theme(style="whitegrid")
    
    # We will plot individual histograms and save them
    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col].dropna(), kde=True, color='royalblue', bins=30)
        plt.title(f'Distribution of {col}', fontsize=14, pad=15)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        plot_name = col.replace(',', '').replace(' ', '_').replace('#', 'num').replace('(', '').replace(')', '').replace('/', '_')
        plt.savefig(os.path.join(output_dir, f'hist_{plot_name}.png'), dpi=300)
        plt.close()
        print(f"   Saved histogram for {col} to results/EDA/hist_{plot_name}.png")
        
    # 8. Generate a correlation matrix
    print("\n8. Generating Correlation Matrix...")
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        # Mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                    vmin=-1, vmax=1, square=True, linewidths=.5,
                    cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Numeric Variables', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300)
        plt.close()
        print("   Saved correlation matrix heatmap to results/EDA/correlation_matrix.png")
    else:
        print("   Not enough numeric columns for a correlation matrix.")
        
    # 9. Report the distribution of the target variable "Tmelt, K"
    target_col = "Tmelt, K"
    if target_col in df.columns:
        print(f"\n9. Target Variable '{target_col}' Distribution Details:")
        target_summary = df[target_col].describe()
        print(target_summary)
        
        # Specific plot for the target variable with QQ-plot and boxplot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histplot + KDE
        sns.histplot(df[target_col].dropna(), kde=True, color='teal', bins=30, ax=axes[0])
        axes[0].set_title(f'Histogram & KDE of {target_col}', fontsize=12)
        axes[0].set_xlabel(target_col)
        
        # Boxplot
        sns.boxplot(x=df[target_col].dropna(), color='lightseagreen', ax=axes[1])
        axes[1].set_title(f'Boxplot of {target_col}', fontsize=12)
        axes[1].set_xlabel(target_col)
        
        plt.suptitle(f'Distribution Analysis of Target: {target_col}', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'target_distribution_analysis.png'), dpi=300)
        plt.close()
        print(f"   Saved detailed target distribution plot to results/EDA/target_distribution_analysis.png")
    else:
        print(f"\n9. Warning: Target variable '{target_col}' not found in dataset columns.")
        
    # 10. Recommend preprocessing steps for machine learning
    print("\n10. Preprocessing Recommendations for ML:")
    print("   [A] Feature Engineering on SMILES & Component names:")
    print("       - The Smiles columns (Smiles#1, Smiles#2) can be used to generate molecular descriptors")
    print("         (e.g., Mordred, RDKit descriptors) or molecular fingerprints (e.g., Morgan / ECFP4)")
    print("         to capture structural information of the DES components.")
    print("       - Component name columns (Component#1, Component#2) are text labels and should be dropped")
    print("         or mapped if SMILES are used.")
    print("   [B] Handling Missing/Constant Columns:")
    print("       - Drop columns that are constant or have no predictive value (e.g., 'Number of components'")
    print("         which is likely constant, or reference columns like 'Reference (DOI)').")
    print("   [C] Scaling Numeric Features:")
    print("       - The numeric descriptors and molar fractions (X#1, X#2) have different scales.")
    print("       - Use StandardScaler or MinMaxScaler on the numeric features before training ML models")
    print("         (especially for models sensitive to scale, like Ridge, SVM, Neural Networks).")
    print("   [D] Handling Multicollinearity:")
    print("       - Check correlation matrix for highly correlated descriptors and remove redundant ones.")
    print("       - Molar fractions (X#1 and X#2) sum to 1.0 (or close to it) for binary systems, meaning one is")
    # Let's check if they sum to 1
    # We will write a check in python to see if X#1 + X#2 is indeed 1.
    print("         completely redundant. Drop one of the molar fraction columns to avoid perfect collinearity.")
    print("   [E] Categorical Encoding:")
    print("       - Encode 'Phase diagram (Yes/No)' to binary (1/0) or one-hot encode categorical features if used.")
    print("       - 'Type of DES' is categorical/ordinal (numeric but represents DES type category like Type III, etc.).")
    print("         Should be verified if it should be treated as categorical or numeric.")
    
    print("\n" + "=" * 60)
    print("EDA COMPLETE. All outputs and plots are saved.")
    print("=" * 60)

if __name__ == "__main__":
    run_eda()
