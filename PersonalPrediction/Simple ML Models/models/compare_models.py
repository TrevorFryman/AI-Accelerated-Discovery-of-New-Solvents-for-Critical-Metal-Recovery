import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_and_compare_models():
    # 1. Setup paths and directories
    dataset_path = os.path.join('data', 'RDKitDescriptorGeneration', 'DES_RDKit_Features.csv')
    results_dir = os.path.join('results', 'ModelComparison')
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
        
    # Handle missing targets
    initial_len = len(df)
    df = df.dropna(subset=[target_col])
    dropped_target_rows = initial_len - len(df)
    if dropped_target_rows > 0:
        print(f"Dropped {dropped_target_rows} rows with missing target values.")
    
    # 3. Feature Selection
    exclude_cols = ['Component#1', 'Component#2', 'Reference (DOI)', 'Smiles#1', 'Smiles#2', target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Automatically separate numeric and categorical columns
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()
    
    print(f"\nDataset Summary:")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    print(f"  Dataset shape: {X.shape}\n")
    
    # 4. Train/Test Split (80/20) - Identical for all models
    print("Creating train/test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"  Train set size: {len(X_train)} samples")
    print(f"  Test set size: {len(X_test)} samples\n")
    
    # 5. Setup Preprocessing Pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # 6. Define Models
    models = {
        'Linear Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ]),
        'Gradient Boosting': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]),
        'XGBoost': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ])
    }
    
    # 7. Train and Evaluate Models
    print("=" * 70)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 70)
    
    results = {}
    
    for model_name, pipeline in models.items():
        print(f"\nTraining {model_name}...")
        pipeline.fit(X_train, y_train)
        
        # Predictions on both train and test sets
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        results[model_name] = {
            'train': {
                'MAE': float(train_mae),
                'RMSE': float(train_rmse),
                'R2': float(train_r2)
            },
            'test': {
                'MAE': float(test_mae),
                'RMSE': float(test_rmse),
                'R2': float(test_r2)
            }
        }
        
        print(f"  Train - MAE: {train_mae:.4f} K, RMSE: {train_rmse:.4f} K, R²: {train_r2:.4f}")
        print(f"  Test  - MAE: {test_mae:.4f} K, RMSE: {test_rmse:.4f} K, R²: {test_r2:.4f}")
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70 + "\n")
    
    # 8. Create Comparison Table
    comparison_data = []
    for model_name, metrics in results.items():
        test_metrics = metrics['test']
        comparison_data.append({
            'Model': model_name,
            'Test MAE (K)': test_metrics['MAE'],
            'Test RMSE (K)': test_metrics['RMSE'],
            'Test R²': test_metrics['R2']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test MAE (K)')
    
    print("Test Set Performance Comparison:")
    print(comparison_df.to_string(index=False))
    print()
    
    # 9. Ranking
    print("Model Ranking by MAE (lower is better):")
    for idx, (_, row) in enumerate(comparison_df.iterrows(), 1):
        print(f"  {idx}. {row['Model']}: MAE = {row['Test MAE (K)']:.4f} K, R² = {row['Test R²']:.4f}")
    
    print("\nModel Ranking by R² (higher is better):")
    r2_ranking = comparison_df.sort_values('Test R²', ascending=False)
    for idx, (_, row) in enumerate(r2_ranking.iterrows(), 1):
        print(f"  {idx}. {row['Model']}: R² = {row['Test R²']:.4f}, MAE = {row['Test MAE (K)']:.4f} K")
    
    # 10. Recommendation
    best_mae_model = comparison_df.iloc[0]['Model']
    best_mae_value = comparison_df.iloc[0]['Test MAE (K)']
    
    best_r2_model = r2_ranking.iloc[0]['Model']
    best_r2_value = r2_ranking.iloc[0]['Test R²']
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(f"\n✓ BEST by MAE: {best_mae_model}")
    print(f"  Mean Absolute Error: {best_mae_value:.4f} K")
    
    print(f"\n✓ BEST by R²: {best_r2_model}")
    print(f"  R² Score: {best_r2_value:.4f}")
    
    # Combined score: lower MAE + higher R²
    if best_mae_model == best_r2_model:
        print(f"\n✓ OVERALL BEST: {best_mae_model}")
        print(f"  This model excels in both MAE and R² metrics.")
    else:
        # Calculate normalized scores for overall ranking
        mae_scores = comparison_df['Test MAE (K)'].values
        r2_scores = comparison_df['Test R²'].values
        
        mae_normalized = (mae_scores - mae_scores.min()) / (mae_scores.max() - mae_scores.min())
        r2_normalized = (r2_scores.max() - r2_scores) / (r2_scores.max() - r2_scores.min())
        
        combined_scores = mae_normalized + r2_normalized
        best_overall_idx = np.argmin(combined_scores)
        best_overall_model = comparison_df.iloc[best_overall_idx]['Model']
        
        print(f"\n✓ OVERALL BEST (Combined MAE + R²): {best_overall_model}")
        print(f"  Balanced performance across both metrics.")
    
    print("\n" + "=" * 70 + "\n")
    
    # 11. Save Results to JSON
    results['comparison_summary'] = {
        'best_by_mae': {
            'model': best_mae_model,
            'mae': float(best_mae_value)
        },
        'best_by_r2': {
            'model': best_r2_model,
            'r2': float(best_r2_value)
        },
        'dataset_info': {
            'total_samples': int(len(df)),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'total_features': int(len(feature_cols)),
            'train_test_split': '80/20'
        }
    }
    
    metrics_path = os.path.join(results_dir, 'comparison_results.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved comparison results to: {metrics_path}")
    
    # 12. Save Comparison Table to CSV
    csv_path = os.path.join(results_dir, 'model_comparison.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved comparison table to: {csv_path}")
    
    # 13. Generate Bar Chart Comparing Metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MAE Comparison
    axes[0].bar(comparison_df['Model'], comparison_df['Test MAE (K)'], color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_title('Mean Absolute Error (MAE)\nLower is Better', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('MAE (K)', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(comparison_df['Test MAE (K)']):
        axes[0].text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    
    # RMSE Comparison
    axes[1].bar(comparison_df['Model'], comparison_df['Test RMSE (K)'], color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_title('Root Mean Squared Error (RMSE)\nLower is Better', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('RMSE (K)', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(comparison_df['Test RMSE (K)']):
        axes[1].text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    
    # R² Comparison
    axes[2].bar(comparison_df['Model'], comparison_df['Test R²'], color='mediumseagreen', alpha=0.7, edgecolor='black')
    axes[2].set_title('R² Score\nHigher is Better', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('R²', fontsize=11)
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].set_ylim([0.8, 1.0])
    for i, v in enumerate(comparison_df['Test R²']):
        axes[2].text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    chart_path = os.path.join(results_dir, 'model_comparison_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison chart to: {chart_path}")
    
    # 14. Generate Ranking Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create ranking based on MAE (primary) and R² (secondary)
    ranking_data = comparison_df.copy()
    ranking_data['Rank'] = range(1, len(ranking_data) + 1)
    
    colors = ['gold', 'silver', '#CD7F32', 'lightblue']
    bars = ax.barh(ranking_data['Model'], ranking_data['Test MAE (K)'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Test MAE (K)', fontsize=12, fontweight='bold')
    ax.set_title('Model Ranking by MAE (Lower is Better)', fontsize=14, fontweight='bold', pad=20)
    
    # Add rank numbers and MAE values
    for i, (model, mae, r2) in enumerate(zip(ranking_data['Model'], ranking_data['Test MAE (K)'], ranking_data['Test R²'])):
        rank = i + 1
        ax.text(mae + 0.2, i, f"#{rank}: MAE={mae:.2f}K, R²={r2:.4f}", 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    ranking_path = os.path.join(results_dir, 'model_ranking.png')
    plt.savefig(ranking_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved model ranking chart to: {ranking_path}")
    
    # 15. Create Detailed Comparison Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for heatmap (normalized)
    heatmap_data = comparison_df[['Model', 'Test MAE (K)', 'Test RMSE (K)', 'Test R²']].set_index('Model')
    
    # Normalize each column to 0-1 scale for better visualization
    heatmap_normalized = heatmap_data.copy()
    heatmap_normalized['Test MAE (K)'] = 1 - (heatmap_data['Test MAE (K)'] - heatmap_data['Test MAE (K)'].min()) / (heatmap_data['Test MAE (K)'].max() - heatmap_data['Test MAE (K)'].min())
    heatmap_normalized['Test RMSE (K)'] = 1 - (heatmap_data['Test RMSE (K)'] - heatmap_data['Test RMSE (K)'].min()) / (heatmap_data['Test RMSE (K)'].max() - heatmap_data['Test RMSE (K)'].min())
    heatmap_normalized['Test R²'] = (heatmap_data['Test R²'] - heatmap_data['Test R²'].min()) / (heatmap_data['Test R²'].max() - heatmap_data['Test R²'].min())
    
    sns.heatmap(heatmap_normalized, annot=heatmap_data, fmt='.4f', cmap='RdYlGn', 
                cbar_kws={'label': 'Normalized Score (Higher is Better)'}, ax=ax, linewidths=1, linecolor='black')
    
    ax.set_title('Model Performance Heatmap\n(Normalized Scores: Higher is Better)', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    heatmap_path = os.path.join(results_dir, 'model_comparison_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison heatmap to: {heatmap_path}")
    
    print("\nWorkflow completed successfully.")

if __name__ == '__main__':
    train_and_compare_models()
