"""
Preprocessing Module - Production Grade

This module handles ALL data preprocessing for the Salary Prediction Application.
It is used by:
1. train_model.py (training pipeline)
2. FastAPI (inference API)
3. Local pipeline (batch predictions)

The encoding must be EXACTLY IDENTICAL between training and inference.
That's why we save encoding maps to JSON and reuse them.

Author: Salary Prediction App
Version: 1.0.1 (Fixed column name consistency)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
from sklearn.model_selection import train_test_split
import logging

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PART 1: ENCODING MAPS (Must be IDENTICAL for training and inference)
# ============================================================================

# Ordinal encoding maps - Order MATTERS!
# These are based on our EDA findings
ORDINAL_MAPS = {
    "experience_level": {
        "EN": 0,  # Entry-level (lowest)
        "MI": 1,  # Mid-level
        "SE": 2,  # Senior
        "EX": 3   # Executive (highest)
    },
    "employment_type": {
        "FT": 0,  # Full-time (most stable/benefits)
        "PT": 1,  # Part-time
        "CT": 2,  # Contract
        "FL": 3   # Freelance (least stable)
    },
    "company_size": {
        "S": 0,   # Small (<50 employees)
        "M": 1,   # Medium (50-250 employees)
        "L": 2    # Large (>250 employees)
    }
}

# Reverse maps for debugging/interpretability
REVERSE_MAPS = {
    category: {v: k for k, v in mapping.items()}
    for category, mapping in ORDINAL_MAPS.items()
}

# Features that are already numerical (no encoding needed)
NUMERICAL_FEATURES = ["remote_ratio", "work_year"]

# All features in correct order (important for consistent prediction)
# IMPORTANT: These column names must match what prepare_features() outputs
FEATURE_ORDER = [
    "experience_level_encoded",
    "employment_type_encoded", 
    "company_size_encoded",
    "job_title_encoded",      # ← Fixed: was 'job_title_freq'
    "remote_ratio",
    "work_year"
]


# ============================================================================
# PART 2: CORE PREPROCESSING FUNCTIONS
# ============================================================================

def encode_ordinal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns using predefined ordinal mappings.
    
    Args:
        df: DataFrame with raw categorical columns
        
    Returns:
        DataFrame with encoded columns (original columns replaced)
        
    Raises:
        ValueError: If unmapped values are found in any column
        
    Example:
        >>> df = pd.DataFrame({'experience_level': ['SE', 'EN', 'EX']})
        >>> encode_ordinal_columns(df)
           experience_level
        0                 2
        1                 0
        2                 3
    """
    df_encoded = df.copy()
    
    for col_name, mapping in ORDINAL_MAPS.items():
        if col_name not in df_encoded.columns:
            logger.warning(f"Column '{col_name}' not found in DataFrame. Skipping.")
            continue
            
        # Apply mapping
        encoded_col = df_encoded[col_name].map(mapping)
        
        # Check for unmapped values (data quality issue)
        if encoded_col.isnull().any():
            unmapped_values = df_encoded[col_name][encoded_col.isnull()].unique()
            raise ValueError(
                f"Unmapped values found in column '{col_name}': {unmapped_values}\n"
                f"Expected one of: {list(mapping.keys())}\n"
                f"Please check your data or update the mapping."
            )
        
        # Replace with encoded values
        df_encoded[col_name] = encoded_col
        
    return df_encoded


def frequency_encode_job_title(
    df: pd.DataFrame, 
    freq_map: Optional[Dict[str, float]] = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Frequency encode job_title column.
    
    Frequency encoding replaces each job title with its frequency in the dataset.
    This works well for high-cardinality categorical variables with Decision Trees.
    
    Args:
        df: DataFrame with 'job_title' column
        freq_map: Pre-computed frequency map (for inference/prediction)
        fit: If True, compute new frequency map from data.
             If False, use provided freq_map.
    
    Returns:
        Tuple of (encoded DataFrame, frequency map dictionary)
        
    Example:
        >>> df = pd.DataFrame({'job_title': ['DS', 'DS', 'ML', 'DE']})
        >>> encoded, freq_map = frequency_encode_job_title(df, fit=True)
        >>> encoded
           job_title_encoded
        0              0.50  (2/4)
        1              0.50
        2              0.25  (1/4)
        3              0.25  (1/4)
    """
    df_encoded = df.copy()
    
    if fit:
        # Compute frequency of each job title
        value_counts = df['job_title'].value_counts()
        total_count = len(df)
        
        # Frequency = count / total
        freq_map = {title: count / total_count for title, count in value_counts.items()}
        
        logger.info(f"Created frequency map with {len(freq_map)} unique job titles")
        logger.info(f"Top 5 job titles by frequency: {list(freq_map.items())[:5]}")
        
        # Optional: Log rare titles (for data quality monitoring)
        rare_threshold = 0.01  # 1% frequency
        rare_titles = [title for title, freq in freq_map.items() if freq < rare_threshold]
        if rare_titles:
            logger.info(f"Found {len(rare_titles)} rare job titles (<{rare_threshold:.1%} frequency)")
    
    # Apply frequency encoding
    df_encoded['job_title_encoded'] = df['job_title'].map(freq_map)
    
    # Handle missing values (titles not in freq_map during inference)
    if not fit and df_encoded['job_title_encoded'].isnull().any():
        missing_titles = df['job_title'][df_encoded['job_title_encoded'].isnull()].unique()
        logger.warning(f"Found {len(missing_titles)} job titles not in frequency map during inference")
        logger.warning(f"Setting them to 0.0 (minimum frequency)")
        df_encoded['job_title_encoded'] = df_encoded['job_title_encoded'].fillna(0.0)
    
    # Drop original job_title column (keep encoded version)
    df_encoded = df_encoded.drop('job_title', axis=1)
    
    return df_encoded, freq_map


def prepare_features(
    df: pd.DataFrame,
    job_title_freq_map: Optional[Dict[str, float]] = None,
    fit_job_title: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Complete feature preparation pipeline.
    
    This function applies ALL preprocessing steps in the correct order:
    1. Select required features
    2. Encode ordinal categoricals (experience_level, employment_type, company_size)
    3. Frequency encode job_title
    4. Keep numerical features as-is
    
    Args:
        df: Raw DataFrame with all columns
        job_title_freq_map: Pre-computed frequency map (for inference)
        fit_job_title: If True, compute new frequency map
    
    Returns:
        Tuple of (X_features DataFrame, job_title_freq_map)
        
    Example:
        >>> df = load_raw_data()
        >>> X, freq_map = prepare_features(df, fit_job_title=True)
        >>> print(X.head())
           experience_level_encoded  employment_type_encoded  ...  work_year
        0                         2                        0  ...       2023
        1                         0                        1  ...       2023
    """
    # Select only the columns we need for features
    feature_columns = ['experience_level', 'employment_type', 'job_title', 
                       'company_size', 'remote_ratio', 'work_year']
    
    # Check if all required columns exist
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Create feature DataFrame
    X = df[feature_columns].copy()
    
    # Step 1: Encode ordinal categorical columns
    X = encode_ordinal_columns(X)
    
    # Step 2: Rename encoded columns for clarity
    X = X.rename(columns={
        'experience_level': 'experience_level_encoded',
        'employment_type': 'employment_type_encoded',
        'company_size': 'company_size_encoded'
    })
    
    # Step 3: Frequency encode job_title (creates 'job_title_encoded' column)
    X, job_title_freq_map = frequency_encode_job_title(
        X, 
        freq_map=job_title_freq_map,
        fit=fit_job_title
    )
    
    # Step 4: Ensure all features are in correct order
    # (Important for consistent prediction across different runs)
    for col in FEATURE_ORDER:
        if col not in X.columns:
            raise ValueError(f"Missing feature column after preprocessing: {col}")
    
    X = X[FEATURE_ORDER]
    
    logger.info(f"Prepared features: {X.shape[0]} rows, {X.shape[1]} columns")
    logger.info(f"Feature columns: {list(X.columns)}")
    
    return X, job_title_freq_map


def get_target(df: pd.DataFrame) -> pd.Series:
    """
    Extract target variable (salary_in_usd) from DataFrame.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Series with target values
        
    Raises:
        ValueError: If target column is missing
    """
    target_column = 'salary_in_usd'
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    y = df[target_column].copy()
    
    # Check for missing values in target
    if y.isnull().any():
        missing_count = y.isnull().sum()
        logger.warning(f"Found {missing_count} missing values in target. Dropping them.")
        logger.warning("Consider investigating why target values are missing!")
        y = y.dropna()
    
    # Check for unrealistic values
    if (y <= 0).any():
        logger.error(f"Found { (y <= 0).sum() } target values <= 0. This is unrealistic for salary.")
        raise ValueError("Target contains zero or negative salary values")
    
    logger.info(f"Target extracted: {len(y)} rows, range: ${y.min():,.2f} - ${y.max():,.2f}")
    
    return y


def train_test_split_reproducible(
    X: pd.DataFrame, 
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create train/test split with reproducibility.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion for test set (default: 0.2 = 80/20 split)
        random_state: Seed for reproducibility
        stratify: Optional column to stratify on (e.g., for imbalanced classification)
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # For regression problems, we don't typically stratify
    # But we keep the parameter for flexibility
    stratify_param = stratify if stratify is not None else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    logger.info(f"Train/Test split: {len(X_train)} train, {len(X_test)} test ({test_size*100:.0f}% test)")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# PART 3: SAVE AND LOAD ENCODING MAPS (Critical for consistency)
# ============================================================================

def save_encoding_maps(
    job_title_freq_map: Dict[str, float],
    filepath: Union[str, Path] = "models/encoding_maps.json"
) -> None:
    """
    Save all encoding maps to JSON file.
    
    This is CRITICAL because FastAPI must use the EXACT same mappings
    that were used during training. Without this, predictions will be wrong.
    
    Args:
        job_title_freq_map: Frequency map for job titles
        filepath: Where to save the JSON file
    
    Example:
        >>> save_encoding_maps(freq_map, "models/encoding_maps.json")
        ✅ Saved encoding maps to models/encoding_maps.json
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    maps_to_save = {
        "ordinal_maps": ORDINAL_MAPS,
        "reverse_maps": REVERSE_MAPS,
        "job_title_freq_map": job_title_freq_map,
        "feature_order": FEATURE_ORDER,
        "numerical_features": NUMERICAL_FEATURES,
        "version": "1.0.1",
        "created_at": pd.Timestamp.now().isoformat()
    }
    
    with open(filepath, 'w') as f:
        json.dump(maps_to_save, f, indent=2, default=str)
    
    logger.info(f"✅ Saved encoding maps to {filepath}")
    logger.info(f"   - {len(ORDINAL_MAPS)} ordinal maps")
    logger.info(f"   - {len(job_title_freq_map)} job title frequencies")


def load_encoding_maps(
    filepath: Union[str, Path] = "models/encoding_maps.json"
) -> Dict:
    """
    Load encoding maps from JSON file.
    
    Used by FastAPI and any inference pipeline to ensure consistent encoding.
    
    Args:
        filepath: Path to the JSON file
    
    Returns:
        Dictionary containing all encoding maps
    
    Example:
        >>> maps = load_encoding_maps("models/encoding_maps.json")
        >>> print(maps.keys())
        dict_keys(['ordinal_maps', 'reverse_maps', 'job_title_freq_map', ...])
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Encoding maps not found at {filepath}\n"
            f"Please run training first to generate the maps."
        )
    
    with open(filepath, 'r') as f:
        maps = json.load(f)
    
    logger.info(f"✅ Loaded encoding maps from {filepath}")
    logger.info(f"   Version: {maps.get('version', 'unknown')}")
    
    return maps


# ============================================================================
# PART 4: VALIDATION AND QUALITY CHECKS
# ============================================================================

def validate_input_data(
    df: pd.DataFrame,
    required_columns: Optional[list] = None
) -> Tuple[bool, list]:
    """
    Validate that input data has all required columns and no critical issues.
    
    Args:
        df: Input DataFrame
        required_columns: List of columns that must exist
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    if required_columns is None:
        required_columns = ['experience_level', 'employment_type', 'job_title', 
                           'company_size', 'remote_ratio', 'work_year', 'salary_in_usd']
    
    issues = []
    
    # Check for missing columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check for empty DataFrame
    if len(df) == 0:
        issues.append("DataFrame is empty")
    
    # Check for too many missing values (more than 50% in any column)
    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct > 50:
            issues.append(f"Column '{col}' has {missing_pct:.1f}% missing values (>50% threshold)")
    
    is_valid = len(issues) == 0
    
    if not is_valid:
        logger.error(f"Data validation failed with {len(issues)} issues")
        for issue in issues:
            logger.error(f"  - {issue}")
    
    return is_valid, issues


def validate_encoded_features(X: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validate that encoded features are within expected ranges.
    
    Args:
        X: Encoded feature DataFrame
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check ordinal columns are within expected ranges
    for col, mapping in ORDINAL_MAPS.items():
        encoded_col = f"{col}_encoded"
        if encoded_col in X.columns:
            min_val = min(mapping.values())
            max_val = max(mapping.values())
            
            if X[encoded_col].min() < min_val or X[encoded_col].max() > max_val:
                issues.append(
                    f"Column '{encoded_col}' has values outside expected range "
                    f"[{min_val}, {max_val}]"
                )
    
    # Check job_title_encoded is between 0 and 1
    if 'job_title_encoded' in X.columns:
        if X['job_title_encoded'].min() < 0 or X['job_title_encoded'].max() > 1:
            issues.append(
                f"Column 'job_title_encoded' has values outside [0,1] range: "
                f"[{X['job_title_encoded'].min():.3f}, {X['job_title_encoded'].max():.3f}]"
            )
    
    # Check numerical features are reasonable
    if 'remote_ratio' in X.columns:
        valid_remote = X['remote_ratio'].isin([0, 50, 100])
        if not valid_remote.all():
            invalid_count = (~valid_remote).sum()
            issues.append(f"Column 'remote_ratio' has {invalid_count} invalid values (must be 0, 50, or 100)")
    
    if 'work_year' in X.columns:
        current_year = pd.Timestamp.now().year
        if (X['work_year'] < 2020).any() or (X['work_year'] > current_year + 1).any():
            issues.append(f"Column 'work_year' has values outside reasonable range (2020-{current_year+1})")
    
    is_valid = len(issues) == 0
    
    return is_valid, issues


# ============================================================================
# PART 5: CONVENIENCE WRAPPER FUNCTIONS
# ============================================================================

def preprocess_single_row(
    row_data: Dict,
    encoding_maps: Dict,
    fit_job_title: bool = False
) -> pd.DataFrame:
    """
    Preprocess a single row of data (for API calls).
    
    Args:
        row_data: Dictionary with raw feature values
        encoding_maps: Pre-loaded encoding maps from load_encoding_maps()
        fit_job_title: Must be False for inference
    
    Returns:
        DataFrame with encoded features ready for prediction
    
    Example:
        >>> row = {
        ...     'experience_level': 'SE',
        ...     'employment_type': 'FT',
        ...     'job_title': 'Data Scientist',
        ...     'company_size': 'M',
        ...     'remote_ratio': 100,
        ...     'work_year': 2024
        ... }
        >>> X = preprocess_single_row(row, encoding_maps)
        >>> prediction = model.predict(X)
    """
    # Convert single dict to DataFrame
    df = pd.DataFrame([row_data])
    
    # Get frequency map from loaded maps
    job_title_freq_map = encoding_maps.get('job_title_freq_map')
    
    if job_title_freq_map is None:
        raise ValueError("Encoding maps missing 'job_title_freq_map'")
    
    # Prepare features (using existing frequency map, not fitting)
    X, _ = prepare_features(
        df,
        job_title_freq_map=job_title_freq_map,
        fit_job_title=False  # CRITICAL: Don't fit during inference!
    )
    
    return X


def preprocess_batch(
    df: pd.DataFrame,
    encoding_maps: Dict,
    fit_job_title: bool = False
) -> pd.DataFrame:
    """
    Preprocess a batch of data (for pipeline).
    
    Args:
        df: DataFrame with raw feature values
        encoding_maps: Pre-loaded encoding maps
        fit_job_title: Must be False for inference
    
    Returns:
        DataFrame with encoded features
    """
    job_title_freq_map = encoding_maps.get('job_title_freq_map')
    
    if job_title_freq_map is None:
        raise ValueError("Encoding maps missing 'job_title_freq_map'")
    
    X, _ = prepare_features(
        df,
        job_title_freq_map=job_title_freq_map,
        fit_job_title=fit_job_title
    )
    
    return X


# ============================================================================
# PART 6: MAIN BLOCK (For testing and validation)
# ============================================================================

if __name__ == "__main__":
    """
    Run this script directly to test the preprocessing pipeline.
    
    Usage:
        python -m salary_src.preprocess
    """
    print("=" * 60)
    print("Testing Preprocessing Module")
    print("=" * 60)
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'experience_level': ['SE', 'EN', 'EX', 'MI'],
        'employment_type': ['FT', 'PT', 'CT', 'FL'],
        'job_title': ['Data Scientist', 'ML Engineer', 'Data Scientist', 'Data Engineer'],
        'company_size': ['L', 'M', 'S', 'L'],
        'remote_ratio': [100, 50, 0, 100],
        'work_year': [2023, 2023, 2023, 2023],
        'salary_in_usd': [150000, 80000, 200000, 120000]
    })
    
    print("\n1. Testing prepare_features()...")
    X, freq_map = prepare_features(sample_data, fit_job_title=True)
    print(f"   ✅ Features shape: {X.shape}")
    print(f"   Feature columns: {list(X.columns)}")
    print(f"   First row:\n{X.iloc[0]}")
    
    print("\n2. Testing get_target()...")
    y = get_target(sample_data)
    print(f"   ✅ Target: {y.values}")
    
    print("\n3. Testing train_test_split...")
    X_train, X_test, y_train, y_test = train_test_split_reproducible(X, y)
    print(f"   ✅ Train: {len(X_train)} rows")
    print(f"   ✅ Test: {len(X_test)} rows")
    
    print("\n4. Testing save/load encoding maps...")
    save_encoding_maps(freq_map, "models/test_encoding_maps.json")
    loaded_maps = load_encoding_maps("models/test_encoding_maps.json")
    print(f"   ✅ Loaded keys: {list(loaded_maps.keys())}")
    
    print("\n5. Testing single row preprocessing...")
    single_row = {
        'experience_level': 'SE',
        'employment_type': 'FT',
        'job_title': 'Data Scientist',
        'company_size': 'M',
        'remote_ratio': 100,
        'work_year': 2024
    }
    X_single = preprocess_single_row(single_row, loaded_maps)
    print(f"   ✅ Single row features:\n{X_single.iloc[0]}")
    
    print("\n6. Testing validation...")
    is_valid, issues = validate_input_data(sample_data)
    print(f"   ✅ Input valid: {is_valid}")
    if issues:
        print(f"   Issues: {issues}")
    
    is_valid, issues = validate_encoded_features(X)
    print(f"   ✅ Encoded features valid: {is_valid}")
    if issues:
        print(f"   Issues: {issues}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Preprocessing module is ready.")
    print("=" * 60)
    
    # Clean up test file
    test_file = Path("models/test_encoding_maps.json")
    if test_file.exists():
        test_file.unlink()
        print("   Cleaned up test file.")