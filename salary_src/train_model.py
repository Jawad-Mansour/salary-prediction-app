"""
Training Module V4 - Decision Tree with Balanced Sampling

Author: Salary Prediction App
Version: 4.0.0
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
    engineer_features,
    FULL_FEATURE_ORDER
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

class ModelConfigV4:
    """Decision Tree configuration with balancing"""
    
    DT_PARAMS = {
        'max_depth': [8, 10, 12, 15, 18, 20, 25],
        'min_samples_split': [5, 10, 15, 20, 30],
        'min_samples_leaf': [2, 4, 6, 8, 10],
        'max_features': ['sqrt', 'log2', None],
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
    }
    
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    USE_OUTLIER_REMOVAL = True
    OUTLIER_IQR_MULTIPLIER = 2.5
    
    USE_TARGET_TRANSFORM = True
    USE_INTERACTIONS = True
    USE_DEVELOPMENT_INDEX = True
    
    # ============ NEW: BALANCING CONFIGURATION ============
    USE_BALANCED_SAMPLING = True
    BALANCE_METHOD = 'weighted'  # Options: 'weighted', 'smote', 'undersample'
    
    # Sample weights to balance experience levels
    EXPERIENCE_WEIGHTS = {
        'EN': 3.5,   # Entry level (underrepresented at 8.5%)
        'MI': 1.2,   # Mid level (21%)
        'SE': 0.6,   # Senior level (overrepresented at 67%)
        'EX': 5.0    # Executive level (rarest at 3%)
    }
    # ======================================================
    
    MODEL_PATH = Path("models/decision_tree_v4.pkl")
    TRANSFORMER_PATH = Path("models/transformer_v4.pkl")
    METRICS_PATH = Path("models/metrics_v4.json")
    ENCODING_MAPS_PATH = Path("models/encoding_maps.json")


# ============================================================================
# Outlier Removal
# ============================================================================

def remove_outliers(
    X: pd.DataFrame, 
    y: pd.Series,
    multiplier: float = 2.5
) -> Tuple[pd.DataFrame, pd.Series]:
    if not ModelConfigV4.USE_OUTLIER_REMOVAL:
        return X, y
    
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - multiplier * IQR)
    upper_bound = Q3 + multiplier * IQR
    
    mask = (y >= lower_bound) & (y <= upper_bound)
    removed_count = (~mask).sum()
    
    if removed_count > 0:
        logger.info(f"✅ Removed {removed_count} outliers ({removed_count/len(y)*100:.1f}%)")
        logger.info(f"   Salary range kept: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
    
    return X[mask], y[mask]


# ============================================================================
# NEW: Balancing Function
# ============================================================================

def apply_balancing(X_train: pd.DataFrame, y_train: pd.Series, 
                    df_train_original: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
    """
    Apply balancing to handle imbalanced experience levels.
    
    Returns:
        X_train: Features (possibly modified)
        y_train: Target values (as numpy array)
        sample_weights: Sample weights for model training (or None)
    """
    if not ModelConfigV4.USE_BALANCED_SAMPLING:
        return X_train, y_train.values, None
    
    logger.info("=" * 60)
    logger.info("Applying Data Balancing")
    logger.info("=" * 60)
    
    exp_levels = df_train_original['experience_level']
    
    # Print class distribution before balancing
    logger.info("Experience level distribution BEFORE balancing:")
    for level, count in exp_levels.value_counts().items():
        pct = count / len(exp_levels) * 100
        logger.info(f"  {level}: {count} ({pct:.1f}%)")
    
    if ModelConfigV4.BALANCE_METHOD == 'weighted':
        # Method 1: Sample weights (preferred - doesn't modify data)
        sample_weights = exp_levels.map(ModelConfigV4.EXPERIENCE_WEIGHTS).values
        logger.info(f"\n✅ Applied sample weights: {ModelConfigV4.EXPERIENCE_WEIGHTS}")
        return X_train, y_train.values, sample_weights
    
    elif ModelConfigV4.BALANCE_METHOD == 'smote':
        # Method 2: SMOTE oversampling
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=ModelConfigV4.RANDOM_STATE)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            logger.info(f"\n✅ SMOTE applied: {len(X_train)} → {len(X_resampled)} samples")
            return X_resampled, y_resampled, None
        except ImportError:
            logger.warning("⚠️ imbalanced-learn not installed. Falling back to weighted method.")
            sample_weights = exp_levels.map(ModelConfigV4.EXPERIENCE_WEIGHTS).values
            return X_train, y_train.values, sample_weights
    
    else:  # 'undersample'
        # Method 3: Undersample majority class
        from sklearn.utils import resample
        
        train_indices = X_train.index
        df_train_balanced = df_train_original.loc[train_indices].copy()
        
        # Undersample Senior level to 1000 samples
        se_indices = df_train_balanced[df_train_balanced['experience_level'] == 'SE'].index
        if len(se_indices) > 1000:
            se_undersampled = resample(se_indices, replace=False, n_samples=1000, 
                                       random_state=ModelConfigV4.RANDOM_STATE)
            keep_indices = df_train_balanced[df_train_balanced['experience_level'] != 'SE'].index.tolist() + \
                          se_undersampled.tolist()
            X_train = X_train.loc[keep_indices]
            y_train = y_train.loc[keep_indices]
            logger.info(f"\n✅ Undersampled SE: {len(se_indices)} → 1000 samples")
        
        return X_train, y_train.values, None


# ============================================================================
# Decision Tree Training
# ============================================================================

def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weights: Optional[np.ndarray] = None
) -> Tuple[DecisionTreeRegressor, Dict]:
    
    logger.info("=" * 60)
    logger.info("Training Decision Tree with Hyperparameter Tuning")
    logger.info("=" * 60)
    
    base_model = DecisionTreeRegressor(random_state=ModelConfigV4.RANDOM_STATE)
    
    total_combinations = 1
    for v in ModelConfigV4.DT_PARAMS.values():
        total_combinations *= len(v)
    
    logger.info(f"Searching {total_combinations} parameter combinations...")
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=ModelConfigV4.DT_PARAMS,
        cv=ModelConfigV4.CV_FOLDS,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    # Pass sample_weights if provided
    if sample_weights is not None:
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
        logger.info("✅ Training with sample weights")
    else:
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
    
    y_pred = model.predict(X_test)
    
    if target_transformer and ModelConfigV4.USE_TARGET_TRANSFORM:
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
    
    return metrics


# ============================================================================
# Save/Load Functions
# ============================================================================

def save_model(model, transformer, metrics, best_params, feature_order):
    ModelConfigV4.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, ModelConfigV4.MODEL_PATH)
    logger.info(f"✅ Model saved to: {ModelConfigV4.MODEL_PATH}")
    
    if transformer:
        joblib.dump(transformer, ModelConfigV4.TRANSFORMER_PATH)
        logger.info(f"✅ Transformer saved to: {ModelConfigV4.TRANSFORMER_PATH}")
    
    metrics_with_metadata = {
        **metrics,
        'model_type': 'DecisionTreeRegressor',
        'model_version': 'v4',
        'training_date': datetime.now().isoformat(),
        'best_params': best_params,
        'feature_order': feature_order,
        'features_used': {
            'outlier_removal': ModelConfigV4.USE_OUTLIER_REMOVAL,
            'target_transform': ModelConfigV4.USE_TARGET_TRANSFORM,
            'interactions': ModelConfigV4.USE_INTERACTIONS,
            'development_index': ModelConfigV4.USE_DEVELOPMENT_INDEX,
            'balanced_sampling': ModelConfigV4.USE_BALANCED_SAMPLING,
            'balance_method': ModelConfigV4.BALANCE_METHOD
        }
    }
    
    with open(ModelConfigV4.METRICS_PATH, 'w') as f:
        json.dump(metrics_with_metadata, f, indent=2)
    
    logger.info(f"✅ Metrics saved to: {ModelConfigV4.METRICS_PATH}")


# ============================================================================
# Feature Importance
# ============================================================================

def analyze_feature_importance(model: DecisionTreeRegressor, X_train: pd.DataFrame):
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
    
    importance_path = ModelConfigV4.MODEL_PATH.parent / "feature_importance_v4.json"
    importance_df.to_json(importance_path, orient='records', indent=2)
    logger.info(f"✅ Feature importance saved to: {importance_path}")
    
    return importance_df


# ============================================================================
# Main Training Pipeline
# ============================================================================

def run_training_pipeline() -> Tuple[DecisionTreeRegressor, Dict]:
    
    logger.info("=" * 60)
    logger.info("DECISION TREE TRAINING PIPELINE V4 (WITH BALANCING)")
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
    logger.info("\n⚙️ Step 3: Preparing base features...")
    X_base, job_title_freq_map = prepare_features(df, fit_job_title=True)
    y = get_target(df)
    logger.info(f"   Base features: {X_base.shape[1]} columns")
    logger.info(f"   Target range: ${y.min():,.2f} - ${y.max():,.2f}")
    
    # Step 4: Feature engineering
    if ModelConfigV4.USE_INTERACTIONS:
        logger.info("\n🔧 Step 4: Engineering interaction features...")
        X = engineer_features(X_base, df)
        logger.info(f"   Features after engineering: {X.shape[1]} columns")
    
    # Step 5: Remove outliers
    logger.info("\n📊 Step 5: Removing outliers...")
    X, y = remove_outliers(X, y, multiplier=ModelConfigV4.OUTLIER_IQR_MULTIPLIER)
    # Also filter df to match
    df_filtered = df.loc[X.index]
    logger.info(f"   Data after outlier removal: {len(X)} rows")
    
    # Step 6: Train/test split
    logger.info("\n📊 Step 6: Creating train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ModelConfigV4.TEST_SIZE, random_state=ModelConfigV4.RANDOM_STATE
    )
    df_train = df_filtered.loc[X_train.index]
    logger.info(f"   Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    
    # ============ NEW: Step 6.5 - Apply Balancing ============
    logger.info("\n⚖️ Step 6.5: Applying data balancing...")
    X_train, y_train_array, sample_weights = apply_balancing(X_train, y_train, df_train)
    # =========================================================
    
    # Step 7: Target transformation
    target_transformer = None
    if ModelConfigV4.USE_TARGET_TRANSFORM:
        logger.info("\n🔄 Step 7: Transforming target variable...")
        target_transformer = TargetTransformer()
        y_train_transformed = target_transformer.fit_transform(pd.Series(y_train_array))
        y_test_transformed = target_transformer.transform(y_test)
        logger.info("   ✅ Target transformed")
    else:
        y_train_transformed = y_train_array
        y_test_transformed = y_test.values
    
    # Step 8: Train Decision Tree
    logger.info("\n🌲 Step 8: Training Decision Tree with GridSearchCV...")
    model, best_params = train_decision_tree(X_train, y_train_transformed, sample_weights)
    
    # Step 9: Evaluate
    logger.info("\n📈 Step 9: Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test_transformed, target_transformer)
    
    # Step 10: Feature importance
    logger.info("\n🔬 Step 10: Analyzing feature importance...")
    analyze_feature_importance(model, X_train)
    
    # Step 11: Save artifacts
    logger.info("\n💾 Step 11: Saving artifacts...")
    save_encoding_maps(job_title_freq_map, ModelConfigV4.ENCODING_MAPS_PATH)
    save_model(model, target_transformer, metrics, best_params, list(X.columns))
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ DECISION TREE TRAINING COMPLETE!")
    logger.info(f"   Model saved as: {ModelConfigV4.MODEL_PATH}")
    logger.info(f"   R² Score: {metrics['r2']:.4f}")
    logger.info("=" * 60)
    
    return model, metrics


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 DECISION TREE TRAINER V4 (WITH BALANCING)")
    print("=" * 60)
    print("\nThis script trains an enhanced Decision Tree with:")
    print("  ✅ Location features (region encoding)")
    print("  ✅ Feature interactions (exp×size, region×exp, etc.)")
    print("  ✅ Outlier removal (IQR method)")
    print("  ✅ Target transformation (PowerTransformer)")
    print("  ✅ Hyperparameter tuning (GridSearchCV)")
    print("  ✅ Development index (country GDP proxy)")
    print("  ✅ Balanced sampling (experience level weighting)")
    print("\n" + "=" * 60)
    
    try:
        model, metrics = run_training_pipeline()
        print(f"\n🎯 Final R² Score: {metrics['r2']:.4f}")
        print(f"   MAE: ${metrics['mae']:,.2f} ({metrics['mae_percentage']:.1f}%)")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise