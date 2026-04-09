"""
Training Module V3 - Enhanced Decision Tree (Allowed improvements only)

Allowed improvements:
1. Location features (country/region encoding)
2. Feature engineering (interactions: exp×size, region×exp, etc.)
3. Outlier removal (IQR method)
4. Target transformation (PowerTransformer)
5. Hyperparameter tuning (GridSearchCV on Decision Tree)
6. Development index (country GDP proxy)

NOT allowed (per assignment):
- Random Forest
- XGBoost
- Ensemble methods

Author: Salary Prediction App
Version: 3.0.2 (FIXED - No duplicate functions, uses preprocess.py)
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from salary_src.data_loader import load_salaries_dataset
from salary_src.preprocess import (
    prepare_features, 
    get_target, 
    save_encoding_maps,
    load_encoding_maps,
    validate_input_data,
    validate_encoded_features,
    TargetTransformer,
    engineer_features,  # Import from preprocess (single source of truth)
    FULL_FEATURE_ORDER  # Import the feature order
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class ModelConfigV3:
    """Decision Tree configuration with allowed enhancements"""
    
    # Decision Tree hyperparameter search space (expanded for tuning)
    DT_PARAMS = {
        'max_depth': [8, 10, 12, 15, 18, 20, 25],
        'min_samples_split': [5, 10, 15, 20, 30],
        'min_samples_leaf': [2, 4, 6, 8, 10],
        'max_features': ['sqrt', 'log2', None],
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
    }
    
    # Training parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # Allowed enhancements
    USE_OUTLIER_REMOVAL = True
    OUTLIER_IQR_MULTIPLIER = 2.5
    
    USE_TARGET_TRANSFORM = True  # PowerTransformer
    
    USE_INTERACTIONS = True
    USE_DEVELOPMENT_INDEX = True
    
    # File paths
    MODEL_PATH = Path("models/decision_tree.pkl")
    TRANSFORMER_PATH = Path("models/transformer.pkl")
    METRICS_PATH = Path("models/metrics.json")
    ENCODING_MAPS_PATH = Path("models/encoding_maps.json")


# ============================================================================
# Outlier Removal (Allowed - data cleaning)
# ============================================================================

def remove_outliers(
    X: pd.DataFrame, 
    y: pd.Series,
    multiplier: float = 2.5
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Remove outliers using IQR method.
    
    Args:
        X: Features DataFrame
        y: Target Series
        multiplier: IQR multiplier (2.5 is less aggressive than 3.0)
    
    Returns:
        Tuple of (X_cleaned, y_cleaned)
    """
    if not ModelConfigV3.USE_OUTLIER_REMOVAL:
        return X, y
    
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - multiplier * IQR)  # Cannot be negative
    upper_bound = Q3 + multiplier * IQR
    
    mask = (y >= lower_bound) & (y <= upper_bound)
    removed_count = (~mask).sum()
    
    if removed_count > 0:
        logger.info(f"✅ Removed {removed_count} outliers ({removed_count/len(y)*100:.1f}%)")
        logger.info(f"   Salary range kept: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
    
    return X[mask], y[mask]


# ============================================================================
# Decision Tree Training with Hyperparameter Tuning
# ============================================================================

def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: np.ndarray
) -> Tuple[DecisionTreeRegressor, Dict]:
    """
    Train Decision Tree with GridSearchCV hyperparameter tuning.
    This is the ONLY model allowed per assignment.
    
    Args:
        X_train: Training features
        y_train: Training target (already transformed)
    
    Returns:
        Tuple of (trained model, best parameters)
    """
    logger.info("=" * 60)
    logger.info("Training Decision Tree with Hyperparameter Tuning")
    logger.info("=" * 60)
    
    base_model = DecisionTreeRegressor(random_state=ModelConfigV3.RANDOM_STATE)
    
    # Calculate total combinations
    total_combinations = 1
    for v in ModelConfigV3.DT_PARAMS.values():
        total_combinations *= len(v)
    
    logger.info(f"Searching {total_combinations} parameter combinations...")
    logger.info(f"This may take 1-2 minutes...")
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=ModelConfigV3.DT_PARAMS,
        cv=ModelConfigV3.CV_FOLDS,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"\n✅ Best R² score (CV): {grid_search.best_score_:.4f}")
    logger.info(f"✅ Best parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_, grid_search.best_params_


# ============================================================================
# Model Evaluation
# ============================================================================

def evaluate_model(
    model: DecisionTreeRegressor,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    target_transformer: Optional[TargetTransformer] = None
) -> Dict[str, float]:
    """
    Evaluate Decision Tree model performance.
    """
    y_pred = model.predict(X_test)
    
    # Inverse transform if needed
    if target_transformer and ModelConfigV3.USE_TARGET_TRANSFORM:
        y_pred_original = target_transformer.inverse_transform(y_pred)
        y_test_original = target_transformer.inverse_transform(y_test.reshape(-1, 1)).flatten()
    else:
        y_pred_original = y_pred
        y_test_original = y_test if isinstance(y_test, np.ndarray) else y_test.values
    
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred_original)
    
    mean_salary = np.mean(y_test_original)
    mae_percentage = (mae / mean_salary) * 100
    
    metrics = {
        'mae': float(mae),
        'mae_percentage': float(mae_percentage),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'mean_salary': float(mean_salary),
        'median_salary': float(np.median(y_test_original)),
        'test_samples': len(y_test)
    }
    
    print("\n" + "=" * 60)
    print("DECISION TREE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Set Size:     {metrics['test_samples']} samples")
    print(f"Mean Salary:       ${metrics['mean_salary']:,.2f}")
    print(f"Median Salary:     ${metrics['median_salary']:,.2f}")
    print(f"\n📊 Regression Metrics:")
    print(f"  MAE:  ${metrics['mae']:,.2f} ({metrics['mae_percentage']:.2f}%)")
    print(f"  RMSE: ${metrics['rmse']:,.2f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print("=" * 60)
    
    baseline_r2 = 0.223
    improvement = metrics['r2'] - baseline_r2
    
    if metrics['r2'] >= 0.5:
        print("✅ Excellent Decision Tree! Well above baseline")
    elif metrics['r2'] >= 0.4:
        print(f"👍 Great improvement! +{improvement*100:.1f}% from V1 baseline")
    elif metrics['r2'] >= 0.3:
        print(f"👌 Decent improvement! +{improvement*100:.1f}% from V1 baseline")
    else:
        print("⚠️ Try different hyperparameters")
    
    return metrics


# ============================================================================
# Save/Load Functions
# ============================================================================

def save_model(model, transformer, metrics, best_params, feature_order):
    """Save all model artifacts."""
    ModelConfigV3.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, ModelConfigV3.MODEL_PATH)
    logger.info(f"✅ Model saved to: {ModelConfigV3.MODEL_PATH}")
    
    if transformer:
        joblib.dump(transformer, ModelConfigV3.TRANSFORMER_PATH)
        logger.info(f"✅ Transformer saved to: {ModelConfigV3.TRANSFORMER_PATH}")
    
    metrics_with_metadata = {
        **metrics,
        'model_type': 'DecisionTreeRegressor',
        'model_version': 'v3',
        'training_date': datetime.now().isoformat(),
        'best_params': best_params,
        'feature_order': feature_order,
        'features_used': {
            'outlier_removal': ModelConfigV3.USE_OUTLIER_REMOVAL,
            'target_transform': ModelConfigV3.USE_TARGET_TRANSFORM,
            'interactions': ModelConfigV3.USE_INTERACTIONS,
            'development_index': ModelConfigV3.USE_DEVELOPMENT_INDEX
        }
    }
    
    with open(ModelConfigV3.METRICS_PATH, 'w') as f:
        json.dump(metrics_with_metadata, f, indent=2)
    
    logger.info(f"✅ Metrics saved to: {ModelConfigV3.METRICS_PATH}")
    
    txt_path = ModelConfigV3.METRICS_PATH.with_suffix('.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DECISION TREE - EVALUATION METRICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training Date: {datetime.now().isoformat()}\n")
        f.write(f"Model Type: DecisionTreeRegressor\n\n")
        f.write(f"Test R²: {metrics['r2']:.4f}\n")
        f.write(f"Test MAE: ${metrics['mae']:,.2f} ({metrics['mae_percentage']:.2f}%)\n")
        f.write(f"Test RMSE: ${metrics['rmse']:,.2f}\n\n")
        f.write("Best Hyperparameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
    
    logger.info(f"✅ Human-readable metrics saved to: {txt_path}")


# ============================================================================
# Feature Importance Analysis
# ============================================================================

def analyze_feature_importance(model: DecisionTreeRegressor, X_train: pd.DataFrame):
    """Display and save feature importance from Decision Tree."""
    importances = model.feature_importances_
    feature_names = X_train.columns
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\n" + "=" * 60)
    print("DECISION TREE FEATURE IMPORTANCE")
    print("=" * 60)
    
    for idx, row in importance_df.head(15).iterrows():
        bar_length = int(row['importance'] * 40)
        bar = "█" * bar_length
        print(f"  {row['feature']:25s} {bar} {row['importance']:.3f}")
    
    print("-" * 60)
    
    importance_path = ModelConfigV3.MODEL_PATH.parent / "feature_importance.json"
    importance_df.to_json(importance_path, orient='records', indent=2)
    logger.info(f"✅ Feature importance saved to: {importance_path}")
    
    return importance_df


# ============================================================================
# Sanity Check
# ============================================================================

def run_sanity_check(
    model: DecisionTreeRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_transformer: Optional[TargetTransformer] = None
):
    """Run sanity check with sample predictions."""
    print("\n" + "=" * 60)
    print("SANITY CHECK - Sample Predictions")
    print("=" * 60)
    
    sample_idx = np.random.choice(len(X_train), 5, replace=False)
    X_samples = X_train.iloc[sample_idx]
    y_actual = y_train.iloc[sample_idx]
    
    y_pred_transformed = model.predict(X_samples)
    
    if target_transformer and ModelConfigV3.USE_TARGET_TRANSFORM:
        y_pred = target_transformer.inverse_transform(y_pred_transformed)
        y_actual_original = y_actual.values
    else:
        y_pred = y_pred_transformed
        y_actual_original = y_actual.values
    
    print("\n📊 Sample Predictions vs Actual:")
    print("-" * 65)
    print(f"{'Sample':<8} {'Actual':<15} {'Predicted':<15} {'Error %':<10}")
    print("-" * 65)
    
    errors = []
    for i, (actual, pred) in enumerate(zip(y_actual_original, y_pred)):
        error_pct = abs(actual - pred) / actual * 100
        errors.append(error_pct)
        indicator = "✅" if error_pct < 20 else "⚠️" if error_pct < 40 else "❌"
        print(f"Sample {i+1}:  ${actual:>12,.2f}  ${pred:>12,.2f}  {error_pct:>6.1f}%     {indicator}")
    
    print("-" * 65)
    print(f"Average Error: {np.mean(errors):.1f}%")
    
    # Custom test case
    print("\n🔮 Custom Example Prediction:")
    custom_example = {
        'experience_level': 'SE',
        'employment_type': 'FT',
        'job_title': 'Data Scientist',
        'company_size': 'L',
        'remote_ratio': 100,
        'work_year': 2024,
        'employee_residence': 'US',
        'company_location': 'US'
    }
    
    from salary_src.preprocess import preprocess_single_row
    encoding_maps = load_encoding_maps(ModelConfigV3.ENCODING_MAPS_PATH)
    X_custom = preprocess_single_row(custom_example, encoding_maps)
    
    # Ensure same columns as training
    missing_cols = set(X_train.columns) - set(X_custom.columns)
    for col in missing_cols:
        X_custom[col] = 0
    
    X_custom = X_custom[X_train.columns]
    
    pred_transformed = model.predict(X_custom)[0]
    if target_transformer and ModelConfigV3.USE_TARGET_TRANSFORM:
        pred = target_transformer.inverse_transform(np.array([pred_transformed]))[0]
    else:
        pred = pred_transformed
    
    print(f"  Job: Senior Data Scientist at Large Company (US)")
    print(f"  Remote: 100% | Year: 2024")
    print(f"  Predicted Salary: ${pred:,.2f}")
    
    if 80000 < pred < 250000:
        print("\n✅ Sanity check PASSED")
    else:
        print("\n⚠️ Sanity check WARNING - Prediction outside expected range")


# ============================================================================
# Main Training Pipeline
# ============================================================================

def run_training_pipeline() -> Tuple[DecisionTreeRegressor, Dict]:
    """Execute complete Decision Tree training pipeline."""
    
    logger.info("=" * 60)
    logger.info("DECISION TREE TRAINING PIPELINE")
    logger.info("Allowed enhancements: Location, Interactions, Outlier Removal, Target Transform")
    logger.info("=" * 60)
    
    # Step 1: Load data
    logger.info("\n📂 Step 1: Loading data...")
    df = load_salaries_dataset()
    logger.info(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Step 2: Validate
    logger.info("\n🔍 Step 2: Validating data...")
    is_valid, issues = validate_input_data(df)
    if not is_valid:
        raise ValueError(f"Validation failed: {issues}")
    logger.info("   ✅ Validation passed")
    
    # Step 3: Prepare base features
    logger.info("\n⚙️ Step 3: Preparing base features (with location)...")
    X_base, job_title_freq_map = prepare_features(df, fit_job_title=True)
    y = get_target(df)
    logger.info(f"   Base features: {X_base.shape[1]} columns")
    logger.info(f"   Target range: ${y.min():,.2f} - ${y.max():,.2f}")
    
    # Step 4: Feature engineering (using preprocess.engineer_features)
    if ModelConfigV3.USE_INTERACTIONS:
        logger.info("\n🔧 Step 4: Engineering interaction features...")
        X = engineer_features(X_base, df)
        logger.info(f"   Features after engineering: {X.shape[1]} columns")
        logger.info(f"   Feature order: {list(X.columns)}")
    
    # Step 5: Remove outliers
    logger.info("\n📊 Step 5: Removing outliers...")
    X, y = remove_outliers(X, y, multiplier=ModelConfigV3.OUTLIER_IQR_MULTIPLIER)
    logger.info(f"   Data after outlier removal: {len(X)} rows")
    
    # Step 6: Train/test split
    logger.info("\n📊 Step 6: Creating train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ModelConfigV3.TEST_SIZE, random_state=ModelConfigV3.RANDOM_STATE
    )
    logger.info(f"   Train: {len(X_train)} rows")
    logger.info(f"   Test: {len(X_test)} rows")
    
    # Step 7: Target transformation
    target_transformer = None
    if ModelConfigV3.USE_TARGET_TRANSFORM:
        logger.info("\n🔄 Step 7: Transforming target variable...")
        target_transformer = TargetTransformer()
        y_train_transformed = target_transformer.fit_transform(y_train)
        y_test_transformed = target_transformer.transform(y_test)
        logger.info("   ✅ Target transformed")
    else:
        y_train_transformed = y_train.values
        y_test_transformed = y_test.values
    
    # Step 8: Train Decision Tree
    logger.info("\n🌲 Step 8: Training Decision Tree with GridSearchCV...")
    model, best_params = train_decision_tree(X_train, y_train_transformed)
    
    # Step 9: Evaluate
    logger.info("\n📈 Step 9: Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test_transformed, target_transformer)
    
    # Step 10: Feature importance
    logger.info("\n🔬 Step 10: Analyzing feature importance...")
    analyze_feature_importance(model, X_train)
    
    # Step 11: Save artifacts
    logger.info("\n💾 Step 11: Saving artifacts...")
    save_encoding_maps(job_title_freq_map, ModelConfigV3.ENCODING_MAPS_PATH)
    save_model(model, target_transformer, metrics, best_params, list(X.columns))
    
    # Step 12: Sanity check
    logger.info("\n🔧 Step 12: Running sanity check...")
    run_sanity_check(model, X_train, y_train, target_transformer)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ DECISION TREE TRAINING COMPLETE!")
    logger.info("=" * 60)
    
    return model, metrics


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 DECISION TREE TRAINER")
    print("=" * 60)
    print("\nThis script trains an enhanced Decision Tree with:")
    print("  ✅ Location features (region encoding)")
    print("  ✅ Feature interactions (exp×size, region×exp, etc.)")
    print("  ✅ Outlier removal (IQR method)")
    print("  ✅ Target transformation (PowerTransformer)")
    print("  ✅ Hyperparameter tuning (GridSearchCV)")
    print("  ✅ Development index (country GDP proxy)")
    print("\n" + "=" * 60)
    
    try:
        model, metrics = run_training_pipeline()
        
        print("\n" + "=" * 60)
        print("🎉 TRAINING SUCCESSFUL!")
        print("=" * 60)
        print(f"\n📁 Generated files:")
        print(f"   - {ModelConfigV3.MODEL_PATH}")
        print(f"   - {ModelConfigV3.METRICS_PATH}")
        print(f"   - {ModelConfigV3.ENCODING_MAPS_PATH}")
        if ModelConfigV3.USE_TARGET_TRANSFORM:
            print(f"   - {ModelConfigV3.TRANSFORMER_PATH}")
        print(f"   - {Path('models/feature_importance.json')}")
        
        print(f"\n📊 Final Results:")
        print(f"   R² Score:  {metrics['r2']:.4f}")
        print(f"   MAE:       ${metrics['mae']:,.2f} ({metrics['mae_percentage']:.1f}%)")
        print(f"   RMSE:      ${metrics['rmse']:,.2f}")
        
        baseline_r2 = 0.223
        improvement = (metrics['r2'] - baseline_r2) / baseline_r2 * 100
        print(f"\n📈 Improvement from baseline (0.223): +{improvement:.0f}%")
        
        if metrics['r2'] >= 0.45:
            print("\n🎯 Excellent! Decision Tree is performing well within assignment constraints.")
        elif metrics['r2'] >= 0.35:
            print("\n👍 Good improvement! Model is ready for deployment.")
        else:
            print("\n⚠️ Consider adjusting hyperparameters or adding more features.")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise