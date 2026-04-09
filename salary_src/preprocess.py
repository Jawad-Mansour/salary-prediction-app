"""
Preprocessing Module - Production Grade (V2 with Location Features)

This module handles ALL data preprocessing for the Salary Prediction Application.
V2 adds location-based features for improved predictions.

Author: Salary Prediction App
Version: 2.0.4 (FIXED - Single source of truth for feature order)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, Optional, Union, List
from sklearn.model_selection import train_test_split
import logging

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Target Transformation Class (for consistency between training and inference)
# ============================================================================

class TargetTransformer:
    """
    Apply PowerTransformer (Yeo-Johnson) to target variable.
    Handles both transformation and inverse transformation.
    This class is defined here so it can be loaded by both training and API.
    """
    
    def __init__(self):
        from sklearn.preprocessing import PowerTransformer
        self.transformer = PowerTransformer(method='yeo-johnson')
        self.fitted = False
    
    def fit_transform(self, y):
        """Fit transformer and transform target."""
        import pandas as pd
        import numpy as np
        y_array = y.values.reshape(-1, 1) if hasattr(y, 'values') else y.reshape(-1, 1)
        y_transformed = self.transformer.fit_transform(y_array)
        self.fitted = True
        logger.info(f"✅ Target transformed with PowerTransformer")
        if hasattr(y, 'skew'):
            logger.info(f"   Skewness before: {y.skew():.3f}")
            logger.info(f"   Skewness after: {pd.Series(y_transformed.flatten()).skew():.3f}")
        return y_transformed.flatten()
    
    def transform(self, y):
        """Transform target using fitted transformer."""
        import numpy as np
        if not self.fitted:
            raise ValueError("Transformer not fitted yet")
        y_array = y.values.reshape(-1, 1) if hasattr(y, 'values') else y.reshape(-1, 1)
        return self.transformer.transform(y_array).flatten()
    
    def inverse_transform(self, y_transformed):
        """Convert back to original scale."""
        import numpy as np
        y_array = y_transformed.reshape(-1, 1) if isinstance(y_transformed, np.ndarray) else np.array(y_transformed).reshape(-1, 1)
        return self.transformer.inverse_transform(y_array).flatten()


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

# Base features order (without engineered features)
BASE_FEATURE_ORDER = [
    "experience_level_encoded",
    "employment_type_encoded", 
    "company_size_encoded",
    "job_title_encoded",
    "region_encoded",
    "remote_ratio",
    "work_year"
]

# THIS IS CRITICAL - The EXACT order your trained model expects
# This order comes from training and MUST match exactly
FULL_FEATURE_ORDER = [
    'region_encoded', 'region_x_exp', 'job_title_encoded', 'title_x_region',
    'dev_index', 'exp_x_size', 'work_year', 'remote_x_exp', 'work_year_squared',
    'remote_ratio', 'company_size_encoded', 'same_country', 
    'experience_level_encoded', 'employment_type_encoded'
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
    - is_colocated: Boolean if employee lives in same country as company
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
    """
    Frequency encode job_title column.
    
    Args:
        df: DataFrame with 'job_title' column
        freq_map: Pre-computed frequency map (REQUIRED when fit=False)
        fit: If True, compute new frequency map from data.
             If False, use provided freq_map (must not be None)
    
    Returns:
        Tuple of (encoded DataFrame, frequency map dictionary)
    """
    df_encoded = df.copy()
    
    if fit:
        # Compute frequency of each job title
        value_counts = df['job_title'].value_counts()
        total_count = len(df)
        freq_map = {title: count / total_count for title, count in value_counts.items()}
        
        logger.info(f"Created frequency map with {len(freq_map)} unique job titles")
        logger.info(f"Top 5: {list(freq_map.items())[:5]}")
    else:
        # When not fitting, freq_map MUST be provided
        if freq_map is None:
            raise ValueError(
                "When fit=False, freq_map must be provided. "
                "Load encoding maps first using load_encoding_maps()"
            )
    
    # Apply frequency encoding
    df_encoded['job_title_encoded'] = df['job_title'].map(freq_map)
    
    # Handle missing values (titles not in freq_map during inference)
    if not fit and df_encoded['job_title_encoded'].isnull().any():
        missing_titles = df['job_title'][df_encoded['job_title_encoded'].isnull()].unique()
        logger.warning(f"Found {len(missing_titles)} job titles not in frequency map")
        logger.warning(f"Setting them to 0.0 (minimum frequency)")
        df_encoded['job_title_encoded'] = df_encoded['job_title_encoded'].fillna(0.0)
    
    # Drop original job_title column (keep encoded version)
    df_encoded = df_encoded.drop('job_title', axis=1)
    
    return df_encoded, freq_map


def engineer_features(X: pd.DataFrame, df_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Create interaction features for Decision Tree.
    This function MUST be used for both training and inference.
    Returns features in FULL_FEATURE_ORDER.
    """
    X_eng = X.copy()
    
    # 1. Experience × Company Size interaction
    if 'experience_level_encoded' in X.columns and 'company_size_encoded' in X.columns:
        X_eng['exp_x_size'] = X['experience_level_encoded'] * X['company_size_encoded']
    else:
        X_eng['exp_x_size'] = 0
    
    # 2. Remote Ratio × Experience interaction
    if 'remote_ratio' in X.columns and 'experience_level_encoded' in X.columns:
        X_eng['remote_x_exp'] = (X['remote_ratio'] / 100) * X['experience_level_encoded']
    else:
        X_eng['remote_x_exp'] = 0
    
    # 3. Work year squared (capture non-linear trends)
    if 'work_year' in X.columns:
        X_eng['work_year_squared'] = X['work_year'] ** 2
    else:
        X_eng['work_year_squared'] = 0
    
    # 4. Region × Experience interaction
    if 'region_encoded' in X.columns and 'experience_level_encoded' in X.columns:
        X_eng['region_x_exp'] = X['region_encoded'] * X['experience_level_encoded']
    else:
        X_eng['region_x_exp'] = 0
    
    # 5. Job Title × Region interaction
    if 'job_title_encoded' in X.columns and 'region_encoded' in X.columns:
        X_eng['title_x_region'] = X['job_title_encoded'] * X['region_encoded']
    else:
        X_eng['title_x_region'] = 0
    
    # 6. Development Index (Country GDP proxy)
    development_index = {
        'US': 100, 'CA': 95, 'GB': 90, 'DE': 90, 'FR': 85,
        'ES': 85, 'IT': 85, 'NL': 95, 'SE': 95, 'NO': 95,
        'DK': 95, 'FI': 95, 'CH': 100, 'AU': 95, 'NZ': 90,
        'SG': 100, 'JP': 85, 'KR': 85, 'CN': 70, 'IN': 45,
        'BR': 60, 'MX': 60, 'ZA': 55, 'AE': 85, 'IL': 90
    }
    
    if df_raw is not None and 'employee_residence' in df_raw.columns:
        X_eng['dev_index'] = df_raw['employee_residence'].map(development_index).fillna(50)
    else:
        X_eng['dev_index'] = 50
    
    # 7. Same country colocation (employee lives where company is)
    if df_raw is not None and 'employee_residence' in df_raw.columns and 'company_location' in df_raw.columns:
        X_eng['same_country'] = (df_raw['employee_residence'] == df_raw['company_location']).astype(int)
    else:
        X_eng['same_country'] = 0
    
    # Return in the EXACT order the model expects
    # Ensure all columns exist
    for col in FULL_FEATURE_ORDER:
        if col not in X_eng.columns:
            X_eng[col] = 0
    
    return X_eng[FULL_FEATURE_ORDER]


def prepare_features(
    df: pd.DataFrame,
    job_title_freq_map: Optional[Dict[str, float]] = None,
    fit_job_title: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Complete feature preparation pipeline (base features only).
    
    Args:
        df: Raw DataFrame with all columns
        job_title_freq_map: Pre-computed frequency map (for inference)
        fit_job_title: If True, compute new frequency map
    
    Returns:
        Tuple of (X_features DataFrame, job_title_freq_map)
    """
    # Select all available features (including location columns)
    feature_columns = ['experience_level', 'employment_type', 'job_title', 
                       'company_size', 'remote_ratio', 'work_year',
                       'employee_residence', 'company_location']
    
    available_columns = [col for col in feature_columns if col in df.columns]
    missing_columns = [col for col in feature_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"Missing columns (will skip): {missing_columns}")
    
    X = df[available_columns].copy()
    
    # Step 1: Encode ordinal categoricals
    X = encode_ordinal_columns(X)
    
    # Step 2: Encode location features
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
        X, 
        freq_map=job_title_freq_map, 
        fit=fit_job_title
    )
    
    # Step 5: Ensure all base features are present
    for col in BASE_FEATURE_ORDER:
        if col not in X.columns:
            logger.warning(f"Missing base feature column: {col}")
            X[col] = 0
    
    # Keep only base features
    X = X[BASE_FEATURE_ORDER]
    
    logger.info(f"Prepared base features: {X.shape[0]} rows, {X.shape[1]} columns")
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
    filepath: Union[str, Path] = "models/encoding_maps.json"
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
        "base_feature_order": BASE_FEATURE_ORDER,
        "full_feature_order": FULL_FEATURE_ORDER,
        "numerical_features": NUMERICAL_FEATURES,
        "version": "2.0.4",
        "created_at": pd.Timestamp.now().isoformat()
    }
    
    with open(filepath, 'w') as f:
        json.dump(maps_to_save, f, indent=2, default=str)
    
    logger.info(f"✅ Saved encoding maps to {filepath}")


def load_encoding_maps(
    filepath: Union[str, Path] = "models/encoding_maps.json"
) -> Dict:
    """Load encoding maps from JSON file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Encoding maps not found at {filepath}")
    
    with open(filepath, 'r') as f:
        maps = json.load(f)
    
    logger.info(f"✅ Loaded encoding maps from {filepath}")
    logger.info(f"   Version: {maps.get('version', 'unknown')}")
    
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
            invalid_count = (~valid_remote).sum()
            issues.append(f"Invalid remote_ratio values: {invalid_count} rows")
    
    # Check region_encoded is within range
    if 'region_encoded' in X.columns:
        if X['region_encoded'].min() < 0 or X['region_encoded'].max() > 4:
            issues.append(f"region_encoded outside expected range [0-4]")
    
    return len(issues) == 0, issues


def preprocess_single_row(row_data: Dict, encoding_maps: Dict) -> pd.DataFrame:
    """
    Preprocess a single row for API calls.
    Includes ALL engineered features to match training.
    Returns features in the EXACT order the model expects.
    
    Args:
        row_data: Dictionary with raw feature values
        encoding_maps: Pre-loaded encoding maps from load_encoding_maps()
    
    Returns:
        DataFrame with FULL encoded features in correct order
    """
    df = pd.DataFrame([row_data])
    job_title_freq_map = encoding_maps.get('job_title_freq_map')
    
    if job_title_freq_map is None:
        raise ValueError("Encoding maps missing 'job_title_freq_map'")
    
    # Step 1: Get base features
    X_base, _ = prepare_features(
        df, 
        job_title_freq_map=job_title_freq_map, 
        fit_job_title=False
    )
    
    # Step 2: Add engineered features (returns in FULL_FEATURE_ORDER)
    X_full = engineer_features(X_base, df)
    
    logger.info(f"Preprocessed single row: {X_full.shape[1]} features")
    
    return X_full


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
        DataFrame with FULL encoded features (including engineered)
    """
    job_title_freq_map = encoding_maps.get('job_title_freq_map')
    
    if job_title_freq_map is None:
        raise ValueError("Encoding maps missing 'job_title_freq_map'")
    
    # Step 1: Get base features
    X_base, _ = prepare_features(
        df,
        job_title_freq_map=job_title_freq_map,
        fit_job_title=fit_job_title
    )
    
    # Step 2: Add engineered features
    X_full = engineer_features(X_base, df)
    
    return X_full


def get_full_feature_order() -> List[str]:
    """Return the full feature order the model expects."""
    return FULL_FEATURE_ORDER


# ============================================================================
# PART 5: MAIN TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Preprocessing Module V2 (with Location Features)")
    print("=" * 60)
    
    from salary_src.data_loader import load_salaries_dataset
    
    df = load_salaries_dataset()
    print(f"Loaded {len(df)} rows")
    
    # Test training mode (fit_job_title=True)
    X_base, freq_map = prepare_features(df, fit_job_title=True)
    print(f"\n✅ Base features shape: {X_base.shape}")
    print(f"Base feature columns: {list(X_base.columns)}")
    
    # Test full features with engineering
    X_full = engineer_features(X_base, df)
    print(f"\n✅ Full features shape: {X_full.shape}")
    print(f"Full feature columns: {list(X_full.columns)}")
    
    # Verify order
    if list(X_full.columns) == FULL_FEATURE_ORDER:
        print(f"   ✅ Feature order matches FULL_FEATURE_ORDER")
    else:
        print(f"   ⚠️ Feature order mismatch!")
    
    # Save encoding maps
    save_encoding_maps(freq_map, "models/encoding_maps.json")
    print("\n✅ Encoding maps saved to models/encoding_maps.json")
    
    # Test loading
    loaded_maps = load_encoding_maps("models/encoding_maps.json")
    print(f"✅ Encoding maps loaded: {len(loaded_maps)} sections")
    
    # Test single row preprocessing (for API)
    test_row = {
        'experience_level': 'SE',
        'employment_type': 'FT',
        'job_title': 'Data Scientist',
        'company_size': 'L',
        'remote_ratio': 100,
        'work_year': 2024,
        'employee_residence': 'US',
        'company_location': 'US'
    }
    
    X_single = preprocess_single_row(test_row, loaded_maps)
    print(f"\n✅ Single row preprocessing successful")
    print(f"   Features: {X_single.shape[1]} columns")
    print(f"   Feature names: {list(X_single.columns)}")
    
    # Verify all expected features are present
    expected_features = FULL_FEATURE_ORDER
    missing_features = [f for f in expected_features if f not in X_single.columns]
    if not missing_features:
        print(f"   ✅ All {len(expected_features)} expected features present")
    else:
        print(f"   ⚠️ Missing features: {missing_features}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Preprocessing module is ready.")
    print("=" * 60)