import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_and_evaluate_gb():
    # 1. Setup paths and directories
    dataset_path = os.path.join('data', 'Melting_temperature_appended_35il_03082026.csv')
    results_dir = os.path.join('results', 'GradientBoosting')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 2. Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Target definition
    target_col = 'Tmelt, K'
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in the dataset.")
        
    # Handle missing targets (drop rows where Tmelt, K is null)
    initial_len = len(df)
    df = df.dropna(subset=[target_col])
    dropped_target_rows = initial_len - len(df)
    if dropped_target_rows > 0:
        print(f"Dropped {dropped_target_rows} rows with missing target values.")
        
    # 3. Automatic Feature Identification
    # Exclude ID, reference and structural text/SMILES columns from features
    exclude_cols = ['Component#1', 'Component#2', 'Reference (DOI)', 'Smiles#1', 'Smiles#2', target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Automatically separate numeric and categorical columns
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()
    
    print("\nAutomatically identified features:")
    print(f"  Numeric features:     {numeric_features}")
    print(f"  Categorical features: {categorical_features}")
    print(f"  Total features count: {len(numeric_features) + len(categorical_features)}")
    print(f"  Dataset shape:        {X.shape}\n")
    
    # 4. Setup Preprocessing Pipelines
    # Preprocessing for numerical data: median imputation and scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical data: most frequent imputation and one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Define complete pipeline with Gradient Boosting Regressor
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
    
    # 5. Perform Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Data split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # 6. Train model
    print("Training Gradient Boosting Regressor...")
    pipeline.fit(X_train, y_train)
    print("Model training completed successfully.")
    
    # 7. Evaluate on Test Set
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "=" * 45)
    print("EVALUATION METRICS (Test Set):")
    print(f"  Mean Absolute Error (MAE):     {mae:.4f} K")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f} K")
    print(f"  R² Score:                       {r2:.4f}")
    print("=" * 45 + "\n")
    
    # Save metrics to results directory
    metrics_summary = {
        'test_metrics': {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2)
        }
    }
    metrics_path = os.path.join(results_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Saved evaluation metrics to: {metrics_path}")
    
    # 8. Create Actual vs Predicted Plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 7))
    
    # Scatter plot
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='darkviolet', edgecolor='w', s=60)
    
    # Perfect prediction reference line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='firebrick', linestyle='--', linewidth=2, label='Perfect Prediction')
    
    # Customise plot design
    plt.title('Actual vs. Predicted Melting Temperature (Gradient Boosting)', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Actual Tmelt (K)', fontsize=12)
    plt.ylabel('Predicted Tmelt (K)', fontsize=12)
    
    # Add textbox with metrics
    metric_text = f"MAE: {mae:.2f} K\nRMSE: {rmse:.2f} K\nR²: {r2:.4f}"
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
    plt.text(0.05, 0.95, metric_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=bbox_props)
    
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    
    pred_plot_path = os.path.join(results_dir, 'actual_vs_predicted.png')
    plt.savefig(pred_plot_path, dpi=300)
    plt.close()
    print(f"Saved Actual vs Predicted plot to: {pred_plot_path}")
    
    # 9. Create Residual Plot (Residuals vs Predicted)
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    
    # Scatter plot of residuals
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color='mediumorchid', edgecolor='w', s=60)
    
    # Zero line
    plt.axhline(y=0, color='firebrick', linestyle='--', linewidth=2, label='Zero Residual')
    
    # Customise plot design
    plt.title('Residuals vs. Predicted Melting Temperature (Gradient Boosting)', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Predicted Tmelt (K)', fontsize=12)
    plt.ylabel('Residuals (K)', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    
    residual_plot_path = os.path.join(results_dir, 'residual_plot.png')
    plt.savefig(residual_plot_path, dpi=300)
    plt.close()
    print(f"Saved Residual plot to: {residual_plot_path}")
    
    # 10. Save Serialized Model Pipeline
    model_path = os.path.join('models', 'gradient_boosting.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Saved serialized model pipeline to: {model_path}")
    
    print("\nWorkflow completed successfully.")

if __name__ == '__main__':
    train_and_evaluate_gb()
