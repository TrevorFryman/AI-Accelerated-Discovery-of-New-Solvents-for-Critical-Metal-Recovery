import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

def train_and_evaluate_xgb():
    # 1. Setup paths and directories
    dataset_path = os.path.join('data', 'Melting_temperature_appended_35il_03082026.csv')
    results_dir = os.path.join('results', 'XGBoost')
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
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with missing target values.")

    # 3. Automatic Feature Identification
    # Exclude ID, reference and structural text/SMILES columns from features
    exclude_cols = [
        'Component#1', 'Component#2', 'Reference (DOI)',
        'Smiles#1', 'Smiles#2', target_col
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df[target_col]

    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()

    print("\nAutomatically identified features:")
    print(f"  Numeric features:     {numeric_features}")
    print(f"  Categorical features: {categorical_features}")
    print(f"  Total features count: {len(numeric_features) + len(categorical_features)}")
    print(f"  Dataset shape:        {X.shape}\n")

    # 4. Setup Preprocessing Pipelines
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

    # Full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ))
    ])

    # 5. Perform 80/20 Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Data split completed. Train: {X_train.shape}, Test: {X_test.shape}")

    # 6. Hyperparameter Optimization via 5-Fold GridSearchCV
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [4, 6, 8],
        'regressor__learning_rate': [0.05, 0.1, 0.2]
    }

    print(f"\nStarting 5-Fold GridSearchCV over {2 * 3 * 3} parameter combinations...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_cv_rmse = -grid_search.best_score_

    print("\n" + "=" * 50)
    print("BEST HYPERPARAMETERS (5-Fold CV):")
    for param, value in best_params.items():
        clean_param = param.replace('regressor__', '')
        print(f"  {clean_param}: {value}")
    print(f"  Best CV RMSE: {best_cv_rmse:.4f} K")
    print("=" * 50 + "\n")

    # 7. Evaluate Best Model on Test Set
    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("EVALUATION METRICS (Test Set):")
    print(f"  Mean Absolute Error (MAE):      {mae:.4f} K")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f} K")
    print(f"  R² Score:                       {r2:.4f}")
    print("=" * 50 + "\n")

    # Save metrics and best parameters to JSON
    metrics_summary = {
        'best_hyperparameters': {
            k.replace('regressor__', ''): v for k, v in best_params.items()
        },
        'cv_best_rmse': float(best_cv_rmse),
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

    # 8. Feature Importance Plot
    # Retrieve feature names from the best pipeline's preprocessor
    feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()
    clean_feature_names = []
    for name in feature_names:
        if name.startswith('num__'):
            clean_feature_names.append(name[5:])
        elif name.startswith('cat__'):
            clean_feature_names.append(name[5:])
        else:
            clean_feature_names.append(name)

    importances = best_pipeline.named_steps['regressor'].feature_importances_

    importance_df = pd.DataFrame({
        'Feature': clean_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Importance', y='Feature',
        data=importance_df,
        palette='plasma',
        hue='Feature',
        legend=False
    )
    plt.title('Feature Importances (XGBoost Regressor)', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Importance Value (gain)', fontsize=12)
    plt.ylabel('Feature Name', fontsize=12)
    plt.tight_layout()

    importance_plot_path = os.path.join(results_dir, 'feature_importance.png')
    plt.savefig(importance_plot_path, dpi=300)
    plt.close()
    print(f"Saved Feature Importance plot to: {importance_plot_path}")

    # 9. Actual vs Predicted Plot
    plt.figure(figsize=(8, 7))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='darkorange', edgecolor='w', s=60)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot(
        [min_val, max_val], [min_val, max_val],
        color='firebrick', linestyle='--', linewidth=2, label='Perfect Prediction'
    )

    plt.title('Actual vs. Predicted Melting Temperature (XGBoost)', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Actual Tmelt (K)', fontsize=12)
    plt.ylabel('Predicted Tmelt (K)', fontsize=12)

    metric_text = f"MAE: {mae:.2f} K\nRMSE: {rmse:.2f} K\nR²: {r2:.4f}"
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
    plt.text(
        0.05, 0.95, metric_text,
        transform=plt.gca().transAxes, fontsize=11,
        verticalalignment='top', bbox=bbox_props
    )

    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()

    pred_plot_path = os.path.join(results_dir, 'actual_vs_predicted.png')
    plt.savefig(pred_plot_path, dpi=300)
    plt.close()
    print(f"Saved Actual vs Predicted plot to: {pred_plot_path}")

    # 10. Save Best Model Pipeline
    model_path = os.path.join('models', 'xgboost.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
    print(f"Saved serialized model pipeline to: {model_path}")

    print("\nWorkflow completed successfully.")

if __name__ == '__main__':
    train_and_evaluate_xgb()
