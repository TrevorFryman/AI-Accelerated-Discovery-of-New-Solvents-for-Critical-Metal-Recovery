import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_and_evaluate():
    # 1. Setup paths and directories
    dataset_path = os.path.join('data', 'Melting_temperature_appended_35il_03082026.csv')
    results_dir = os.path.join('results', 'LinearRegression')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 2. Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Target and Features
    target_col = 'Tmelt, K'
    
    # Map requested features to actual CSV column names
    feature_mapping = {
        'Number of components': 'Number of components',
        'Type of DES': 'Type of DES',
        'X#1 (molar fraction)': 'X#1 (molar fraction)',
        'X#2 (molar fraction)': 'X#2 (molar fraction)',
        'Phase diagram': 'Phase diagram (Yes/No)',
        'T#1': 'T#1',
        'T#2': 'T#2'
    }
    
    # Verify columns exist in dataframe
    for feat_name, col_name in feature_mapping.items():
        if col_name not in df.columns:
            raise KeyError(f"Required feature column '{col_name}' (for {feat_name}) not found in the dataset.")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in the dataset.")
        
    # 3. Handle missing values
    # Drop rows where target is missing, as we cannot train or validate on them
    initial_len = len(df)
    df = df.dropna(subset=[target_col])
    dropped_target_rows = initial_len - len(df)
    if dropped_target_rows > 0:
        print(f"Dropped {dropped_target_rows} rows with missing target values.")
        
    X = df[list(feature_mapping.values())]
    y = df[target_col]
    
    # 4. Split data 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Data split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Define numeric and categorical columns
    numeric_features = [
        feature_mapping['Number of components'],
        feature_mapping['X#1 (molar fraction)'],
        feature_mapping['X#2 (molar fraction)'],
        feature_mapping['T#1'],
        feature_mapping['T#2']
    ]
    categorical_features = [
        feature_mapping['Type of DES'],
        feature_mapping['Phase diagram']
    ]
    
    # 5. Build Preprocessing & Modeling Pipeline
    # Impute and scale numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Impute and encode categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Full pipeline: Preprocessor + LinearRegression
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # 6. Train model
    print("Training Linear Regression model...")
    pipeline.fit(X_train, y_train)
    print("Model training completed successfully.")
    
    # 7. Evaluate
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "=" * 40)
    print("EVALUATION METRICS (Test Set):")
    print(f"  Mean Absolute Error (MAE):     {mae:.4f} K")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f} K")
    print(f"  R² Score:                       {r2:.4f}")
    print("=" * 40 + "\n")
    
    # Save metrics to results directory
    metrics = {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2)
    }
    metrics_path = os.path.join(results_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved evaluation metrics to: {metrics_path}")
    
    # 8. Create Actual vs Predicted plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 7))
    
    # Scatter plot
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='royalblue', edgecolor='w', s=60)
    
    # Perfect prediction reference line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='firebrick', linestyle='--', linewidth=2, label='Perfect Prediction')
    
    # Customise plot design
    plt.title('Actual vs. Predicted Melting Temperature (Linear Regression)', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Actual Tmelt (K)', fontsize=12)
    plt.ylabel('Predicted Tmelt (K)', fontsize=12)
    
    # Add textbox with metrics
    metric_text = f"MAE: {mae:.2f} K\nRMSE: {rmse:.2f} K\nR²: {r2:.4f}"
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
    plt.text(0.05, 0.95, metric_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=bbox_props)
    
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, 'actual_vs_predicted.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Actual vs Predicted plot to: {plot_path}")
    
    # 9. Save model pipeline
    model_path = os.path.join('models', 'linear_regression.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Saved serialized model pipeline to: {model_path}")
    
    print("\nWorkflow completed successfully.")

if __name__ == '__main__':
    train_and_evaluate()
