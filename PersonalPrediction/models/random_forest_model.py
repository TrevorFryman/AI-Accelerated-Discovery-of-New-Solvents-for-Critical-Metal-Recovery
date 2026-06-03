import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

def train_and_evaluate_rf():
    # 1. Setup paths and directories
    dataset_path = os.path.join('data', 'Melting_temperature_appended_35il_03082026.csv')
    results_dir = os.path.join('results', 'RandomForest')
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
    
    # Define complete pipeline with Random Forest Regressor
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # 5. 5-Fold Cross-Validation
    print("Executing 5-Fold Cross-Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(df))
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Clone the pipeline to ensure independence of folds
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)
        
        # Generate predictions on validation fold
        val_preds = fold_pipeline.predict(X_val)
        oof_predictions[val_idx] = val_preds
        
        # Calculate metrics for the current fold
        fold_mae = mean_absolute_error(y_val, val_preds)
        fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        fold_r2 = r2_score(y_val, val_preds)
        
        fold_metrics.append({
            'fold': fold + 1,
            'MAE': float(fold_mae),
            'RMSE': float(fold_rmse),
            'R2': float(fold_r2)
        })
        print(f"  Fold {fold + 1} - MAE: {fold_mae:.4f} K, RMSE: {fold_rmse:.4f} K, R²: {fold_r2:.4f}")
        
    # Calculate overall metrics based on out-of-fold predictions
    cv_mae = mean_absolute_error(y, oof_predictions)
    cv_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    cv_r2 = r2_score(y, oof_predictions)
    
    # Calculate mean and standard deviation across folds
    mean_mae = np.mean([f['MAE'] for f in fold_metrics])
    std_mae = np.std([f['MAE'] for f in fold_metrics])
    mean_rmse = np.mean([f['RMSE'] for f in fold_metrics])
    std_rmse = np.std([f['RMSE'] for f in fold_metrics])
    mean_r2 = np.mean([f['R2'] for f in fold_metrics])
    std_r2 = np.std([f['R2'] for f in fold_metrics])
    
    print("\n" + "=" * 55)
    print("5-FOLD CROSS-VALIDATION SUMMARY:")
    print(f"  MAE (Mean ± Std):   {mean_mae:.4f} ± {std_mae:.4f} K")
    print(f"  RMSE (Mean ± Std):  {mean_rmse:.4f} ± {std_rmse:.4f} K")
    print(f"  R² (Mean ± Std):    {mean_r2:.4f} ± {std_r2:.4f}")
    print("-" * 55)
    print("OVERALL OUT-OF-FOLD (OOF) PERFORMANCE:")
    print(f"  Overall OOF MAE:    {cv_mae:.4f} K")
    print(f"  Overall OOF RMSE:   {cv_rmse:.4f} K")
    print(f"  Overall OOF R²:     {cv_r2:.4f}")
    print("=" * 55 + "\n")
    
    # Save metrics to results directory
    metrics_summary = {
        'fold_metrics': fold_metrics,
        'cv_averages': {
            'mean_MAE': mean_mae,
            'std_MAE': std_mae,
            'mean_RMSE': mean_rmse,
            'std_RMSE': std_rmse,
            'mean_R2': mean_r2,
            'std_R2': std_r2
        },
        'overall_oof': {
            'MAE': cv_mae,
            'RMSE': cv_rmse,
            'R2': cv_r2
        }
    }
    metrics_path = os.path.join(results_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Saved evaluation metrics to: {metrics_path}")
    
    # 6. Train Final Model on Complete Dataset
    print("Training final Random Forest model on complete dataset...")
    pipeline.fit(X, y)
    print("Final model training completed successfully.")
    
    # 7. Generate Feature Importance Plot
    # Retrieve feature names out of the preprocessing ColumnTransformer
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    # Clean feature names (remove ColumnTransformer class prefixes: num__ and cat__)
    clean_feature_names = []
    for name in feature_names:
        if name.startswith('num__'):
            clean_feature_names.append(name[5:])
        elif name.startswith('cat__'):
            clean_feature_names.append(name[5:])
        else:
            clean_feature_names.append(name)
            
    # Retrieve feature importances from regressor
    importances = pipeline.named_steps['regressor'].feature_importances_
    
    # Create DataFrame and sort features by importance
    importance_df = pd.DataFrame({
        'Feature': clean_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Generate feature importance bar plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', hue='Feature', legend=False)
    plt.title('Feature Importances (Random Forest Regressor)', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Importance Value', fontsize=12)
    plt.ylabel('Feature Name', fontsize=12)
    plt.tight_layout()
    
    importance_plot_path = os.path.join(results_dir, 'feature_importance.png')
    plt.savefig(importance_plot_path, dpi=300)
    plt.close()
    print(f"Saved Feature Importance plot to: {importance_plot_path}")
    
    # 8. Create Actual vs Predicted Plot
    plt.figure(figsize=(8, 7))
    
    # Scatter plot of actual vs OOF predicted
    sns.scatterplot(x=y, y=oof_predictions, alpha=0.6, color='teal', edgecolor='w', s=60)
    
    # Perfect prediction reference line
    min_val = min(y.min(), oof_predictions.min())
    max_val = max(y.max(), oof_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='firebrick', linestyle='--', linewidth=2, label='Perfect Prediction')
    
    # Customise plot design
    plt.title('Actual vs. Predicted Melting Temperature (5-Fold CV OOF)', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Actual Tmelt (K)', fontsize=12)
    plt.ylabel('Predicted Tmelt (K)', fontsize=12)
    
    # Add textbox with metrics
    metric_text = f"OOF MAE: {cv_mae:.2f} K\nOOF RMSE: {cv_rmse:.2f} K\nOOF R²: {cv_r2:.4f}"
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
    plt.text(0.05, 0.95, metric_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=bbox_props)
    
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    
    pred_plot_path = os.path.join(results_dir, 'actual_vs_predicted.png')
    plt.savefig(pred_plot_path, dpi=300)
    plt.close()
    print(f"Saved Actual vs Predicted plot to: {pred_plot_path}")
    
    # 9. Save Serialized Model Pipeline
    model_path = os.path.join('models', 'random_forest.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Saved serialized model pipeline to: {model_path}")
    
    print("\nWorkflow completed successfully.")

if __name__ == '__main__':
    train_and_evaluate_rf()
