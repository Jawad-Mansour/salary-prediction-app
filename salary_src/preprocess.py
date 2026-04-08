"""
Preprocessing Module - Production Grade

This module handles ALL data preprocessing for the Salary Prediction Application.
It is used by:
1. Training pipeline (train_model.py)
2. FastAPI service (for real-time predictions)
3. Local pipeline (for batch predictions)

IMPORTANT: Any change here MUST be tested with both training and inference.
The encoding maps are saved and loaded to ensure consistency.

Author: Salary Prediction Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Tuple, Dict, Optional, Any
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1: ENCODING MAPS (Hardcoded from EDA)
# ============================================================================

# Ordinal encoding maps (order matters for Decision Tree)
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
REVERSE_ORDINAL_MAPS = {
    "experience_level": {v: k for k, v in ORDINAL_MAPS["experience_level"].items()},
    "employment_type": {v: k for k, v in ORDINAL_MAPS["employment_type"].items()},
    "company_size": {v: k for k, v in ORDINAL_MAPS["company_size"].items()}
}

# Features that are already numeric (no encoding needed)
NUMERIC_FEATURES = ["remote_ratio", "work_year"]

# All feature columns (order matters for consistency)
FEATURE_COLUMNS = [
    "experience_level",
    "employment_type", 
    "job_title",
    "company_size",
    "remote_ratio",
    "work_year"
]

# Target column
TARGET_COLUMN = "salary_in_usd"


# ============================================================================
# SECTION 2: VALIDATION FUNCTIONS
# ============================================================================

def validate_input_data(df: pd.DataFrame, stage: str = "raw") -> None:
    """
    Validate that input data has expected structure.
    
    Args:
        df: DataFrame to validate
        stage: "raw" (before preprocessing) or "processed" (after)
    
    Raises:
        ValueError: If validation fails
    """
    if stage == "raw":
        # Check required columns exist
        missing_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check no missing values in required columns
        for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
            if df[col].isnull().any():
                raise ValueError(f"Missing values found in column: {col}")
        
        # Validate categorical values are known
        for col, mapping in ORDINAL_MAPS.items():
            unique_vals = df[col].unique()
            unknown_vals = [v for v in unique_vals if v not in mapping]
            if unknown_vals:
                raise ValueError(f"Unknown values in {col}: {unknown_vals}")
    
    elif stage == "processed":
        # Check no missing values after preprocessing
        if df.isnull().any().any():
            raise ValueError("Missing values found in processed data")
        
        # Check all columns are numeric
        non_numeric = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            raise ValueError(f"Non-numeric columns after preprocessing: {non_numeric}")
    
    logger.info(f"✅ Validation passed for stage: {stage}")


def validate_prediction_input(job_title: str, experience_level: str, 
                              employment_type: str, company_size: str,
                              remote_ratio: int, work_year: int) -> None:
    """
    Validate prediction inputs for FastAPI endpoint.
    
    Args:
        job_title: Job title string
        experience_level: EN, MI, SE, or EX
        employment_type: FT, PT, CT, or FL
        company_size: S, M, or L
        remote_ratio: 0, 50, or 100
        work_year: Year (e.g., 2023)
    
    Raises:
        ValueError: If any input is invalid
    """
    # Validate experience_level
    if experience_level not in ORDINAL_MAPS["experience_level"]:
        raise ValueError(f"Invalid experience_level: {experience_level}. Must be one of {list(ORDINAL_MAPS['experience_level'].keys())}")
    
    # Validate employment_type
    if employment_type not in ORDINAL_MAPS["employment_type"]:
        raise ValueError(f"Invalid employment_type: {employment_type}. Must be one of {list(ORDINAL_MAPS['employment_type'].keys())}")
    
    # Validate company_size
    if company_size not in ORDINAL_MAPS["company_size"]:
        raise ValueError(f"Invalid company_size: {company_size}. Must be one of {list(ORDINAL_MAPS['company_size'].keys())}")
    
    # Validate remote_ratio
    if remote_ratio not in [0, 50, 100]:
        raise ValueError(f"Invalid remote_ratio: {remote_ratio}. Must be 0, 50, or 100")
    
    # Validate work_year
    if not isinstance(work_year, int) or work_year < 2020 or work_year > 2030:
        raise ValueError(f"Invalid work_year: {work_year}. Must be between 2020 and 2030")
    
    logger.info(f"✅ Prediction input validation passed for: {job_title}")


# ============================================================================
# SECTION 3: CORE PREPROCESSING FUNCTIONS
# ============================================================================

def encode_ordinal_columns(df: pd.DataFrame, maps: Dict = None) -> pd.DataFrame:
    """
    Encode ordinal categorical columns using predefined mappings.
    
    Args:
        df: DataFrame with raw categorical columns
        maps: Optional custom maps (uses ORDINAL_MAPS by default)
    
    Returns:
        DataFrame with encoded numeric columns
    """
    if maps is None:
        maps = ORDINAL_MAPS
    
    df_encoded = df.copy()
    
    for col, mapping in maps.items():
        if col in df_encoded.columns:
            # Apply encoding
            df_encoded[col] = df_encoded[col].map(mapping)
            
            # Check for unmapped values (should have been caught in validation)
            if df_encoded[col].isnull().any():
                unmapped = df[col][df_encoded[col].isnull()].unique()
                raise ValueError(f"Unmapped values found in {col}: {unmapped}")
    
    logger.info(f"✅ Encoded ordinal columns: {list(maps.keys())}")
    return df_encoded


def frequency_encode_job_title(df: pd.DataFrame, 
                                fit: bool = True,
                                freq_map: Optional[Dict] = None,
                                min_frequency: float = 0.001) -> Tuple[pd.DataFrame, Dict]:
    """
    Frequency encode job_title based on occurrence in dataset.
    
    Frequency encoding replaces each category with its frequency (count/total).
    This works well for Decision Trees because:
    1. Handles high cardinality (50+ unique values)
    2. Preserves importance (common titles get higher values)
    3. No order assumption (unlike ordinal encoding)
    
    Args:
        df: DataFrame with raw job_title column
        fit: If True, compute frequency map from this data
        freq_map: Existing frequency map (used when fit=False)
        min_frequency: Minimum frequency threshold (rare titles get this value)
    
    Returns:
        Tuple of (encoded DataFrame, frequency map dictionary)
    """
    df_encoded = df.copy()
    
    if fit:
        # Compute frequency of each job title
        freq = df['job_title'].value_counts()
        total = len(df)
        
        # Create frequency map (count / total)
        freq_map = {title: count / total for title, count in freq.items()}
        
        # Optional: Cap rare titles at min_frequency
        # This prevents extremely rare titles from having too much influence
        for title in freq_map:
            if freq_map[title] < min_frequency:
                freq_map[title] = min_frequency
        
        logger.info(f"✅ Created frequency map for {len(freq_map)} unique job titles")
        logger.info(f"   Most common: {max(freq_map, key=freq_map.get)} = {max(freq_map.values()):.4f}")
        logger.info(f"   Least common: {min(freq_map, key=freq_map.get)} = {min(freq_map.values()):.4f}")
    
    # Apply frequency encoding
    df_encoded['job_title_encoded'] = df['job_title'].map(freq_map)
    
    # Handle rare titles not in map (shouldn't happen with fit=True)
    if not fit:
        df_encoded['job_title_encoded'] = df_encoded['job_title_encoded'].fillna(min_frequency)
    
    # Drop original job_title column (we only need the encoded version)
    df_encoded = df_encoded.drop('job_title', axis=1)
    
    return df_encoded, freq_map


def prepare_features(df: pd.DataFrame,
                     job_title_freq_map: Optional[Dict] = None,
                     fit_job_title: bool = True,
                     validate: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete feature preparation pipeline.
    
    This function orchestrates all preprocessing steps:
    1. Validate input data
    2. Select feature columns
    3. Encode ordinal categoricals
    4. Frequency encode job_title
    5. Validate output
    
    Args:
        df: Raw DataFrame with all columns
        job_title_freq_map: Existing frequency map (if fit_job_title=False)
        fit_job_title: If True, compute new frequency map
        validate: If True, run validation checks
    
    Returns:
        Tuple of (X_features DataFrame, job_title_freq_map)
    """
    if validate:
        validate_input_data(df, stage="raw")
    
    # Select only feature columns
    X = df[FEATURE_COLUMNS].copy()
    
    # Step 1: Encode ordinal categoricals
    X = encode_ordinal_columns(X)
    
    # Step 2: Frequency encode job_title
    X, job_title_freq_map = frequency_encode_job_title(
        X, 
        fit=fit_job_title, 
        freq_map=job_title_freq_map
    )
    
    # Step 3: Ensure numeric columns are correct type
    for col in NUMERIC_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype(int)
    
    if validate:
        validate_input_data(X, stage="processed")
    
    logger.info(f"✅ Feature preparation complete. Shape: {X.shape}")
    logger.info(f"   Features: {X.columns.tolist()}")
    
    return X, job_title_freq_map


def get_target(df: pd.DataFrame) -> pd.Series:
    """
    Extract target variable from DataFrame.
    
    Args:
        df: Raw DataFrame
    
    Returns:
        Series containing target values
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame")
    
    y = df[TARGET_COLUMN].copy()
    
    # Basic sanity checks
    if y.min() < 0:
        logger.warning(f"⚠️ Negative target values found: min={y.min()}")
    
    if y.max() > 1_000_000:
        logger.warning(f"⚠️ Very large target values found: max={y.max()}")
    
    return y


def train_test_split_reproducible(X: pd.DataFrame, 
                                   y: pd.Series,
                                   test_size: float = 0.2,
                                   random_state: int = 42,
                                   stratify: Optional[pd.Series] = None) -> Tuple:
    """
    Create reproducible train/test split.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion for testing (default 0.2 = 80/20)
        random_state: Fixed seed for reproducibility
        stratify: Optional stratification column
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if stratify is not None:
        # For classification, stratify by target bins
        # For regression, stratify by quantiles of target
        if y.nunique() > 10:  # Regression case
            stratify_col = pd.qcut(y, q=10, labels=False, duplicates='drop')
        else:
            stratify_col = y
    else:
        stratify_col = None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    logger.info(f"✅ Train/test split complete")
    logger.info(f"   Train: {X_train.shape[0]} rows")
    logger.info(f"   Test: {X_test.shape[0]} rows")
    logger.info(f"   Random state: {random_state}")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# SECTION 4: SAVE/LOAD ENCODING MAPS (For FastAPI Compatibility)
# ============================================================================

def save_encoding_maps(job_title_freq_map: Dict,
                       path: str = "models/encoding_maps.json",
                       include_ordinal: bool = True) -> None:
    """
    Save encoding maps to JSON file for reuse in FastAPI.
    
    This ensures that the FastAPI service uses EXACTLY the same
    encoding as the training pipeline.
    
    Args:
        job_title_freq_map: Frequency map for job titles
        path: Where to save the JSON file
        include_ordinal: If True, include ordinal maps as well
    """
    # Create directory if it doesn't exist
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    maps_to_save = {
        "version": "1.0.0",
        "description": "Encoding maps for Salary Prediction Application",
        "created_at": pd.Timestamp.now().isoformat(),
    }
    
    if include_ordinal:
        maps_to_save["ordinal_maps"] = ORDINAL_MAPS
        maps_to_save["reverse_ordinal_maps"] = REVERSE_ORDINAL_MAPS
    
    maps_to_save["job_title_freq_map"] = job_title_freq_map
    maps_to_save["numeric_features"] = NUMERIC_FEATURES
    maps_to_save["feature_columns"] = FEATURE_COLUMNS
    
    # Convert numpy values to Python native types for JSON serialization
    maps_to_save["job_title_freq_map"] = {
        k: float(v) for k, v in job_title_freq_map.items()
    }
    
    with open(path, 'w') as f:
        json.dump(maps_to_save, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Saved encoding maps to {path}")
    logger.info(f"   Job titles in map: {len(job_title_freq_map)}")


def load_encoding_maps(path: str = "models/encoding_maps.json") -> Dict:
    """
    Load encoding maps from JSON file (for FastAPI).
    
    Args:
        path: Path to the JSON file
    
    Returns:
        Dictionary containing all encoding maps
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Encoding maps not found at {path}. Run training first.")
    
    with open(path, 'r') as f:
        maps = json.load(f)
    
    logger.info(f"✅ Loaded encoding maps from {path}")
    logger.info(f"   Version: {maps.get('version', 'unknown')}")
    
    return maps


def encode_prediction_input(job_title: str,
                            experience_level: str,
                            employment_type: str,
                            company_size: str,
                            remote_ratio: int,
                            work_year: int,
                            encoding_maps: Dict) -> np.ndarray:
    """
    Encode a single prediction input (for FastAPI).
    
    This function mirrors prepare_features() but for a single prediction.
    
    Args:
        job_title: Raw job title string
        experience_level: Raw experience level (EN/MI/SE/EX)
        employment_type: Raw employment type (FT/PT/CT/FL)
        company_size: Raw company size (S/M/L)
        remote_ratio: Integer (0/50/100)
        work_year: Integer year
        encoding_maps: Loaded encoding maps from load_encoding_maps()
    
    Returns:
        Numpy array of encoded features ready for model prediction
    """
    # Validate input first
    validate_prediction_input(job_title, experience_level, employment_type,
                              company_size, remote_ratio, work_year)
    
    # Get maps
    ordinal_maps = encoding_maps["ordinal_maps"]
    job_title_freq_map = encoding_maps["job_title_freq_map"]
    
    # Encode ordinal
    exp_encoded = ordinal_maps["experience_level"][experience_level]
    emp_encoded = ordinal_maps["employment_type"][employment_type]
    size_encoded = ordinal_maps["company_size"][company_size]
    
    # Frequency encode job title (default to 0.001 if not found)
    job_title_encoded = job_title_freq_map.get(job_title, 0.001)
    
    # Create feature array in the correct order
    # Order must match what the model was trained on
    features = np.array([
        exp_encoded,      # experience_level
        emp_encoded,      # employment_type
        job_title_encoded, # job_title_encoded
        size_encoded,     # company_size
        remote_ratio,     # remote_ratio
        work_year         # work_year
    ], dtype=np.float32)
    
    logger.info(f"✅ Encoded prediction input: {features}")
    
    return features


# ============================================================================
# SECTION 5: MAIN FUNCTION (For Testing)
# ============================================================================

def test_preprocessing():
    """
    Test the entire preprocessing pipeline.
    Run this module directly to verify everything works.
    """
    print("\n" + "=" * 60)
    print("TESTING PREPROCESSING MODULE")
    print("=" * 60)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'experience_level': ['SE', 'EN', 'EX', 'MI'],
        'employment_type': ['FT', 'PT', 'CT', 'FL'],
        'job_title': ['Data Scientist', 'ML Engineer', 'Data Scientist', 'Data Engineer'],
        'company_size': ['L', 'M', 'S', 'L'],
        'remote_ratio': [100, 50, 0, 100],
        'work_year': [2023, 2023, 2023, 2023],
        'salary_in_usd': [150000, 80000, 200000, 120000]
    })
    
    print("\n1. Sample data:")
    print(sample_data)
    
    # Test feature preparation
    print("\n2. Preparing features...")
    X, freq_map = prepare_features(sample_data, fit_job_title=True)
    print(f"   Result shape: {X.shape}")
    print(f"   Features: {X.columns.tolist()}")
    print(f"   First row:\n{X.iloc[0]}")
    
    # Test target extraction
    print("\n3. Extracting target...")
    y = get_target(sample_data)
    print(f"   Target: {y.values}")
    
    # Test train/test split
    print("\n4. Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split_reproducible(X, y)
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Test saving maps
    print("\n5. Saving encoding maps...")
    save_encoding_maps(freq_map, path="models/test_encoding_maps.json")
    
    # Test loading maps
    print("\n6. Loading encoding maps...")
    loaded_maps = load_encoding_maps(path="models/test_encoding_maps.json")
    print(f"   Loaded keys: {list(loaded_maps.keys())}")
    
    # Test single prediction encoding
    print("\n7. Encoding single prediction input...")
    encoded = encode_prediction_input(
        job_title="Data Scientist",
        experience_level="SE",
        employment_type="FT",
        company_size="L",
        remote_ratio=100,
        work_year=2023,
        encoding_maps=loaded_maps
    )
    print(f"   Encoded features: {encoded}")
    
    # Clean up test file
    import os
    if os.path.exists("models/test_encoding_maps.json"):
        os.remove("models/test_encoding_maps.json")
        print("\n✅ Test file cleaned up")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_preprocessing()