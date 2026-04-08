"""
Preprocessing Module - Production Grade (V2 with Location Features)

This module handles ALL data preprocessing for the Salary Prediction Application.
V2 adds location-based features for improved predictions.

Author: Salary Prediction App
Version: 2.0.0
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

# Country to region mapping (for better generalization)
# Group countries by economic similarity
COUNTRY_REGION_MAP = {
    # North America (High salaries)
    "US": "NA", "CA": "NA",
    
    # Western Europe (High salaries)
    "GB": "WE", "DE": "WE", "FR": "WE", "NL": "WE", "CH": "WE", 
    "ES": "WE", "IT": "WE", "BE": "WE", "AT": "WE", "IE": "WE",
    "PT": "WE", "SE": "WE", "DK": "WE", "FI": "WE", "NO": "WE",
    
    # Eastern Europe (Medium salaries)
    "PL": "EE", "CZ": "EE", "HU": "EE", "RO": "EE", "BG": "EE",
    "SK": "EE", "HR": "EE", "RS": "EE", "LT": "EE", "LV": "EE",
    "EE": "EE", "GR": "EE",
    
    # Asia Pacific (Varied)
    "AU": "AP", "NZ": "AP", "SG": "AP", "JP": "AP", "KR": "AP",
    "CN": "AP", "IN": "AP", "MY": "AP", "TH": "AP", "VN": "AP",
    "PH": "AP", "ID": "AP", "PK": "AP",
    
    # Middle East
    "AE": "ME", "SA": "ME", "IL": "ME", "QA": "ME", "KW": "ME",
    
    # Latin America
    "BR": "LA", "MX": "LA", "AR": "LA", "CL": "LA", "CO": "LA",
    "PE": "LA", "UY": "LA",
    
    # Africa
    "ZA": "AF", "NG": "AF", "KE": "AF", "EG": "AF", "MA": "AF",
    
    # Remote/Other
    "remote": "REMOTE"
}

# Region encoding (ordinal by average salary)
REGION_ENCODING = {
    "NA": 4,      # North America (highest)
    "WE": 3,      # Western Europe
    "AP": 2,      # Asia Pacific
    "ME": 2,      # Middle East (tied with AP)
    "LA": 1,      # Latin America
    "EE": 1,      # Eastern Europe (tied with LA)
    "AF": 0,      # Africa (lowest)
    "REMOTE": 2   # Remote (medium)
}

# Reverse maps for debugging
REVERSE_MAPS = {
    category: {v: k for k, v in mapping.items()}
    for category, mapping in ORDINAL_MAPS.items()
}

# Features that are already numerical
NUMERICAL_FEATURES = ["remote_ratio", "work_year"]

# All features in correct order (V2 with location features)
FEATURE_ORDER = [
    "experience_level_encoded",
    "employment_type_encoded", 
    "company_size_encoded",
    "job_title_encoded",
    "region_encoded",           # NEW: Location feature
    "remote_ratio",
    "work_year"
]


# ============================================================================
# PART 2: CORE PREPROCESSING FUNCTIONS
# ============================================================================

def encode_ordinal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns using predefined ordinal mappings."""
    df_encoded = df.copy()
    
    for col_name, mapping in ORDINAL_MAPS.items():
        if col_name not in df_encoded.columns:
            logger.warning(f"Column '{col_name}' not found. Skipping.")
            continue
            
        encoded_col = df_encoded[col_name].map(mapping)
        
        if encoded_col.isnull().any():
            unmapped_values = df_encoded[col_name][encoded_col.isnull()].unique()
            raise ValueError(
                f"Unmapped values found in '{col_name}': {unmapped_values}\n"
                f"Expected: {list(mapping.keys())}"
            )
        
        df_encoded[col_name] = encoded_col
        
    return df_encoded


def encode_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode location features from employee_residence and company_location.
    
    Creates:
    - region_encoded: Numerical encoding of geographic region
    - is_remote_match: Boolean if employee lives in same country as company
    """
    df_encoded = df.copy()
    
    # Map employee residence to region
    df_encoded['employee_region'] = df_encoded['employee_residence'].map(COUNTRY_REGION_MAP)
    df_encoded['employee_region'] = df_encoded['employee_region'].fillna("REMOTE")
    
    # Map company location to region
    df_encoded['company_region'] = df_encoded['company_location'].map(COUNTRY_REGION_MAP)
    df_encoded['company_region'] = df_encoded['company_region'].fillna("REMOTE")
    
    # Use employee region as primary (employee location matters more)
    df_encoded['region_encoded'] = df_encoded['employee_region'].map(REGION_ENCODING)
    df_encoded['region_encoded'] = df_encoded['region_encoded'].fillna(2)  # Default to medium
    
    # Bonus feature: Does employee live in same country as company?
    df_encoded['is_colocated'] = (
        df_encoded['employee_residence'] == df_encoded['company_location']
    ).astype(int)
    
    # Clean up intermediate columns
    df_encoded = df_encoded.drop(['employee_region', 'company_region'], axis=1)
    
    logger.info(f"Location features encoded - Regions found: {df_encoded['region_encoded'].unique()}")
    logger.info(f"Colocated employees: {df_encoded['is_colocated'].sum()}/{len(df_encoded)}")
    
    return df_encoded


def frequency_encode_job_title(
    df: pd.DataFrame, 
    freq_map: Optional[Dict[str, float]] = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Frequency encode job_title column."""
    df_encoded = df.copy()
    
    if fit:
        value_counts = df['job_title'].value_counts()
        total_count = len(df)
        freq_map = {title: count / total_count for title, count in value_counts.items()}
        
        logger.info(f"Created frequency map with {len(freq_map)} unique job titles")
        logger.info(f"Top 5: {list(freq_map.items())[:5]}")
    
    df_encoded['job_title_encoded'] = df['job_title'].map(freq_map)
    
    if not fit and df_encoded['job_title_encoded'].isnull().any():
        df_encoded['job_title_encoded'] = df_encoded['job_title_encoded'].fillna(0.0)
    
    df_encoded = df_encoded.drop('job_title', axis=1)
    
    return df_encoded, freq_map


def prepare_features(
    df: pd.DataFrame,
    job_title_freq_map: Optional[Dict[str, float]] = None,
    fit_job_title: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Complete feature preparation pipeline (V2 with location).
    """
    # Select all available features (including location columns)
    feature_columns = ['experience_level', 'employment_type', 'job_title', 
                       'company_size', 'remote_ratio', 'work_year',
                       'employee_residence', 'company_location']  # NEW: location
    
    available_columns = [col for col in feature_columns if col in df.columns]
    missing_columns = [col for col in feature_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"Missing columns (will skip): {missing_columns}")
    
    X = df[available_columns].copy()
    
    # Step 1: Encode ordinal categoricals
    X = encode_ordinal_columns(X)
    
    # Step 2: Encode location features (NEW)
    X = encode_location_features(X)
    
    # Step 3: Rename encoded columns
    rename_map = {
        'experience_level': 'experience_level_encoded',
        'employment_type': 'employment_type_encoded',
        'company_size': 'company_size_encoded'
    }
    X = X.rename(columns={k: v for k, v in rename_map.items() if k in X.columns})
    
    # Step 4: Frequency encode job_title
    X, job_title_freq_map = frequency_encode_job_title(
        X, freq_map=job_title_freq_map, fit=fit_job_title
    )
    
    # Step 5: Ensure all features are in correct order
    for col in FEATURE_ORDER:
        if col not in X.columns:
            logger.warning(f"Missing feature column: {col}")
    
    # Keep only columns that exist
    existing_features = [col for col in FEATURE_ORDER if col in X.columns]
    X = X[existing_features]
    
    logger.info(f"Prepared features: {X.shape[0]} rows, {X.shape[1]} columns")
    logger.info(f"Feature columns: {list(X.columns)}")
    
    return X, job_title_freq_map


def get_target(df: pd.DataFrame) -> pd.Series:
    """Extract target variable (salary_in_usd)."""
    target_column = 'salary_in_usd'
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    y = df[target_column].copy()
    
    if y.isnull().any():
        missing_count = y.isnull().sum()
        logger.warning(f"Found {missing_count} missing target values. Dropping.")
        y = y.dropna()
    
    if (y <= 0).any():
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
    """Create train/test split with reproducibility."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    logger.info(f"Train/Test split: {len(X_train)} train, {len(X_test)} test")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# PART 3: SAVE AND LOAD ENCODING MAPS
# ============================================================================

def save_encoding_maps(
    job_title_freq_map: Dict[str, float],
    filepath: Union[str, Path] = "models/encoding_maps_v2.json"
) -> None:
    """Save all encoding maps to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    maps_to_save = {
        "ordinal_maps": ORDINAL_MAPS,
        "reverse_maps": REVERSE_MAPS,
        "country_region_map": COUNTRY_REGION_MAP,
        "region_encoding": REGION_ENCODING,
        "job_title_freq_map": job_title_freq_map,
        "feature_order": FEATURE_ORDER,
        "numerical_features": NUMERICAL_FEATURES,
        "version": "2.0.0",
        "created_at": pd.Timestamp.now().isoformat()
    }
    
    with open(filepath, 'w') as f:
        json.dump(maps_to_save, f, indent=2, default=str)
    
    logger.info(f"✅ Saved encoding maps to {filepath}")


def load_encoding_maps(
    filepath: Union[str, Path] = "models/encoding_maps_v2.json"
) -> Dict:
    """Load encoding maps from JSON file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Encoding maps not found at {filepath}")
    
    with open(filepath, 'r') as f:
        maps = json.load(f)
    
    logger.info(f"✅ Loaded encoding maps from {filepath}")
    
    return maps


# ============================================================================
# PART 4: VALIDATION
# ============================================================================

def validate_input_data(df: pd.DataFrame) -> Tuple[bool, list]:
    """Validate input data."""
    required_columns = ['experience_level', 'employment_type', 'job_title', 
                       'company_size', 'remote_ratio', 'work_year', 'salary_in_usd']
    
    issues = []
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    if len(df) == 0:
        issues.append("DataFrame is empty")
    
    return len(issues) == 0, issues


def validate_encoded_features(X: pd.DataFrame) -> Tuple[bool, list]:
    """Validate encoded features."""
    issues = []
    
    # Check numerical features are reasonable
    if 'remote_ratio' in X.columns:
        valid_remote = X['remote_ratio'].isin([0, 50, 100])
        if not valid_remote.all():
            issues.append("Invalid remote_ratio values")
    
    return len(issues) == 0, issues


def preprocess_single_row(row_data: Dict, encoding_maps: Dict) -> pd.DataFrame:
    """Preprocess a single row for API calls."""
    df = pd.DataFrame([row_data])
    job_title_freq_map = encoding_maps.get('job_title_freq_map')
    
    X, _ = prepare_features(df, job_title_freq_map=job_title_freq_map, fit_job_title=False)
    
    return X


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Preprocessing Module V2 (with Location Features)")
    print("=" * 60)
    
    from salary_src.data_loader import load_salaries_dataset
    
    df = load_salaries_dataset()
    print(f"Loaded {len(df)} rows")
    
    X, freq_map = prepare_features(df, fit_job_title=True)
    print(f"\n✅ Features shape: {X.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    save_encoding_maps(freq_map, "models/encoding_maps_v2.json")
    print("\n✅ Encoding maps saved")