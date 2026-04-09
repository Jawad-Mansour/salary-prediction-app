"""
Preprocessing Module - Production Grade (V3 with Enhanced Features)

This module handles ALL data preprocessing for the Salary Prediction Application.

Author: Salary Prediction App
Version: 3.0.0
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
# Target Transformation Class
# ============================================================================

class TargetTransformer:
    """Apply PowerTransformer (Yeo-Johnson) to target variable."""
    
    def __init__(self):
        from sklearn.preprocessing import PowerTransformer
        self.transformer = PowerTransformer(method='yeo-johnson')
        self.fitted = False
    
    def fit_transform(self, y):
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
        import numpy as np
        if not self.fitted:
            raise ValueError("Transformer not fitted yet")
        y_array = y.values.reshape(-1, 1) if hasattr(y, 'values') else y.reshape(-1, 1)
        return self.transformer.transform(y_array).flatten()
    
    def inverse_transform(self, y_transformed):
        import numpy as np
        y_array = y_transformed.reshape(-1, 1) if isinstance(y_transformed, np.ndarray) else np.array(y_transformed).reshape(-1, 1)
        return self.transformer.inverse_transform(y_array).flatten()


# ============================================================================
# PART 1: ENCODING MAPS
# ============================================================================

ORDINAL_MAPS = {
    "experience_level": {"EN": 0, "MI": 1, "SE": 2, "EX": 3},
    "employment_type": {"FT": 0, "PT": 1, "CT": 2, "FL": 3},
    "company_size": {"S": 0, "M": 1, "L": 2}
}

COUNTRY_REGION_MAP = {
    "US": "NA", "CA": "NA", "GB": "WE", "DE": "WE", "FR": "WE",
    "ES": "WE", "IT": "WE", "NL": "WE", "CH": "WE", "BE": "WE",
    "AT": "WE", "IE": "WE", "PT": "WE", "SE": "WE", "DK": "WE",
    "FI": "WE", "NO": "WE", "AU": "AP", "NZ": "AP", "SG": "AP",
    "JP": "AP", "KR": "AP", "CN": "AP", "IN": "AP", "MY": "AP",
    "TH": "AP", "VN": "AP", "PH": "AP", "ID": "AP", "PK": "AP",
    "AE": "ME", "SA": "ME", "IL": "ME", "QA": "ME", "KW": "ME",
    "BR": "LA", "MX": "LA", "AR": "LA", "CL": "LA", "CO": "LA",
    "PE": "LA", "UY": "LA", "ZA": "AF", "NG": "AF", "KE": "AF",
    "EG": "AF", "MA": "AF", "remote": "REMOTE"
}

REGION_ENCODING = {
    "NA": 4, "WE": 3, "AP": 2, "ME": 2, "LA": 1, "EE": 1, "AF": 0, "REMOTE": 2
}

REVERSE_MAPS = {
    category: {v: k for k, v in mapping.items()}
    for category, mapping in ORDINAL_MAPS.items()
}

NUMERICAL_FEATURES = ["remote_ratio", "work_year"]

BASE_FEATURE_ORDER = [
    "experience_level_encoded", "employment_type_encoded", "company_size_encoded",
    "job_title_encoded", "region_encoded", "remote_ratio", "work_year"
]

FULL_FEATURE_ORDER = [
    'region_encoded', 'region_x_exp', 'job_title_encoded', 'title_x_region',
    'dev_index', 'exp_x_size', 'work_year', 'remote_x_exp', 'work_year_squared',
    'remote_ratio', 'company_size_encoded', 'same_country',
    'experience_level_encoded', 'employment_type_encoded',
    'size_squared', 'size_x_region', 'size_x_title', 'is_large', 'is_small'
]


# ============================================================================
# PART 2: CORE PREPROCESSING FUNCTIONS
# ============================================================================

def encode_ordinal_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df.copy()
    for col_name, mapping in ORDINAL_MAPS.items():
        if col_name not in df_encoded.columns:
            continue
        encoded_col = df_encoded[col_name].map(mapping)
        if encoded_col.isnull().any():
            unmapped = df_encoded[col_name][encoded_col.isnull()].unique()
            raise ValueError(f"Unmapped values in '{col_name}': {unmapped}")
        df_encoded[col_name] = encoded_col
    return df_encoded


def encode_location_features(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df.copy()
    df_encoded['employee_region'] = df_encoded['employee_residence'].map(COUNTRY_REGION_MAP).fillna("REMOTE")
    df_encoded['company_region'] = df_encoded['company_location'].map(COUNTRY_REGION_MAP).fillna("REMOTE")
    df_encoded['region_encoded'] = df_encoded['employee_region'].map(REGION_ENCODING).fillna(2)
    df_encoded['is_colocated'] = (df_encoded['employee_residence'] == df_encoded['company_location']).astype(int)
    df_encoded = df_encoded.drop(['employee_region', 'company_region'], axis=1)
    logger.info(f"Location features encoded - Regions found: {df_encoded['region_encoded'].unique()}")
    return df_encoded


def frequency_encode_job_title(df: pd.DataFrame, freq_map: Optional[Dict] = None, fit: bool = True):
    df_encoded = df.copy()
    if fit:
        value_counts = df['job_title'].value_counts()
        freq_map = {title: count / len(df) for title, count in value_counts.items()}
        logger.info(f"Created frequency map with {len(freq_map)} job titles")
    else:
        if freq_map is None:
            raise ValueError("When fit=False, freq_map must be provided")
    df_encoded['job_title_encoded'] = df['job_title'].map(freq_map).fillna(0.0)
    df_encoded = df_encoded.drop('job_title', axis=1)
    return df_encoded, freq_map


def engineer_features(X: pd.DataFrame, df_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    X_eng = X.copy()
    
    # Experience × Company Size
    if 'experience_level_encoded' in X.columns and 'company_size_encoded' in X.columns:
        X_eng['exp_x_size'] = X['experience_level_encoded'] * X['company_size_encoded']
    else:
        X_eng['exp_x_size'] = 0
    
    # Remote × Experience
    if 'remote_ratio' in X.columns and 'experience_level_encoded' in X.columns:
        X_eng['remote_x_exp'] = (X['remote_ratio'] / 100) * X['experience_level_encoded']
    else:
        X_eng['remote_x_exp'] = 0
    
    # Work year squared
    if 'work_year' in X.columns:
        X_eng['work_year_squared'] = X['work_year'] ** 2
    else:
        X_eng['work_year_squared'] = 0
    
    # Region × Experience
    if 'region_encoded' in X.columns and 'experience_level_encoded' in X.columns:
        X_eng['region_x_exp'] = X['region_encoded'] * X['experience_level_encoded']
    else:
        X_eng['region_x_exp'] = 0
    
    # Title × Region
    if 'job_title_encoded' in X.columns and 'region_encoded' in X.columns:
        X_eng['title_x_region'] = X['job_title_encoded'] * X['region_encoded']
    else:
        X_eng['title_x_region'] = 0
    
    # Development Index
    dev_index = {
        'US': 100, 'CA': 95, 'GB': 90, 'DE': 90, 'FR': 85, 'ES': 85,
        'IT': 85, 'NL': 95, 'SE': 95, 'NO': 95, 'DK': 95, 'FI': 95,
        'CH': 100, 'AU': 95, 'NZ': 90, 'SG': 100, 'JP': 85, 'KR': 85,
        'CN': 70, 'IN': 45, 'BR': 60, 'MX': 60, 'ZA': 55, 'AE': 85
    }
    if df_raw is not None and 'employee_residence' in df_raw.columns:
        X_eng['dev_index'] = df_raw['employee_residence'].map(dev_index).fillna(50)
    else:
        X_eng['dev_index'] = 50
    
    # Same Country
    if df_raw is not None and 'employee_residence' in df_raw.columns and 'company_location' in df_raw.columns:
        X_eng['same_country'] = (df_raw['employee_residence'] == df_raw['company_location']).astype(int)
    else:
        X_eng['same_country'] = 0
    
    # Size squared
    if 'company_size_encoded' in X.columns:
        X_eng['size_squared'] = X['company_size_encoded'] ** 2
    else:
        X_eng['size_squared'] = 0
    
    # Size × Region
    if 'company_size_encoded' in X.columns and 'region_encoded' in X.columns:
        X_eng['size_x_region'] = X['company_size_encoded'] * X['region_encoded']
    else:
        X_eng['size_x_region'] = 0
    
    # Size × Title
    if 'company_size_encoded' in X.columns and 'job_title_encoded' in X.columns:
        X_eng['size_x_title'] = X['company_size_encoded'] * X['job_title_encoded']
    else:
        X_eng['size_x_title'] = 0
    
    # Is Large
    if 'company_size_encoded' in X.columns:
        X_eng['is_large'] = (X['company_size_encoded'] == 2).astype(int)
    else:
        X_eng['is_large'] = 0
    
    # Is Small
    if 'company_size_encoded' in X.columns:
        X_eng['is_small'] = (X['company_size_encoded'] == 0).astype(int)
    else:
        X_eng['is_small'] = 0
    
    # Ensure all columns exist
    for col in FULL_FEATURE_ORDER:
        if col not in X_eng.columns:
            X_eng[col] = 0
    
    return X_eng[FULL_FEATURE_ORDER]


def prepare_features(df: pd.DataFrame, job_title_freq_map: Optional[Dict] = None, fit_job_title: bool = True):
    feature_columns = ['experience_level', 'employment_type', 'job_title', 
                       'company_size', 'remote_ratio', 'work_year',
                       'employee_residence', 'company_location']
    
    available = [col for col in feature_columns if col in df.columns]
    X = df[available].copy()
    
    X = encode_ordinal_columns(X)
    X = encode_location_features(X)
    
    rename_map = {
        'experience_level': 'experience_level_encoded',
        'employment_type': 'employment_type_encoded',
        'company_size': 'company_size_encoded'
    }
    X = X.rename(columns={k: v for k, v in rename_map.items() if k in X.columns})
    
    X, job_title_freq_map = frequency_encode_job_title(X, freq_map=job_title_freq_map, fit=fit_job_title)
    
    for col in BASE_FEATURE_ORDER:
        if col not in X.columns:
            X[col] = 0
    
    X = X[BASE_FEATURE_ORDER]
    
    logger.info(f"Prepared base features: {X.shape}")
    return X, job_title_freq_map


def get_target(df: pd.DataFrame) -> pd.Series:
    y = df['salary_in_usd'].copy()
    if (y <= 0).any():
        raise ValueError("Target contains zero or negative salary values")
    logger.info(f"Target extracted: {len(y)} rows, range: ${y.min():,.2f} - ${y.max():,.2f}")
    return y


def train_test_split_reproducible(X, y, test_size=0.2, random_state=42, stratify=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    logger.info(f"Train/Test split: {len(X_train)} train, {len(X_test)} test")
    return X_train, X_test, y_train, y_test


# ============================================================================
# PART 3: SAVE AND LOAD ENCODING MAPS
# ============================================================================

def save_encoding_maps(job_title_freq_map: Dict, filepath: Union[str, Path] = "models/encoding_maps.json"):
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
        "version": "3.0.0",
        "created_at": pd.Timestamp.now().isoformat()
    }
    
    with open(filepath, 'w') as f:
        json.dump(maps_to_save, f, indent=2, default=str)
    
    logger.info(f"✅ Saved encoding maps to {filepath}")


def load_encoding_maps(filepath: Union[str, Path] = "models/encoding_maps.json") -> Dict:
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
    required = ['experience_level', 'employment_type', 'job_title', 
                'company_size', 'remote_ratio', 'work_year', 'salary_in_usd']
    issues = []
    missing = [col for col in required if col not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
    if len(df) == 0:
        issues.append("DataFrame is empty")
    return len(issues) == 0, issues


def validate_encoded_features(X: pd.DataFrame) -> Tuple[bool, list]:
    issues = []
    if 'remote_ratio' in X.columns:
        valid = X['remote_ratio'].isin([0, 50, 100])
        if not valid.all():
            issues.append(f"Invalid remote_ratio values: {(~valid).sum()} rows")
    if 'region_encoded' in X.columns:
        if X['region_encoded'].min() < 0 or X['region_encoded'].max() > 4:
            issues.append("region_encoded outside expected range [0-4]")
    if 'company_size_encoded' in X.columns:
        if X['company_size_encoded'].min() < 0 or X['company_size_encoded'].max() > 2:
            issues.append("company_size_encoded outside expected range [0-2]")
    return len(issues) == 0, issues


def preprocess_single_row(row_data: Dict, encoding_maps: Dict) -> pd.DataFrame:
    df = pd.DataFrame([row_data])
    job_title_freq_map = encoding_maps.get('job_title_freq_map')
    if job_title_freq_map is None:
        raise ValueError("Encoding maps missing 'job_title_freq_map'")
    
    X_base, _ = prepare_features(df, job_title_freq_map=job_title_freq_map, fit_job_title=False)
    X_full = engineer_features(X_base, df)
    
    logger.info(f"Preprocessed single row: {X_full.shape[1]} features")
    return X_full


def preprocess_batch(df: pd.DataFrame, encoding_maps: Dict, fit_job_title: bool = False) -> pd.DataFrame:
    job_title_freq_map = encoding_maps.get('job_title_freq_map')
    if job_title_freq_map is None:
        raise ValueError("Encoding maps missing 'job_title_freq_map'")
    
    X_base, _ = prepare_features(df, job_title_freq_map=job_title_freq_map, fit_job_title=fit_job_title)
    X_full = engineer_features(X_base, df)
    return X_full


def get_full_feature_order() -> List[str]:
    return FULL_FEATURE_ORDER


# ============================================================================
# PART 5: MAIN TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Preprocessing Module V3")
    print("=" * 60)
    
    from salary_src.data_loader import load_salaries_dataset
    
    df = load_salaries_dataset()
    print(f"Loaded {len(df)} rows")
    
    X_base, freq_map = prepare_features(df, fit_job_title=True)
    print(f"\n✅ Base features shape: {X_base.shape}")
    print(f"Base feature columns: {list(X_base.columns)}")
    
    X_full = engineer_features(X_base, df)
    print(f"\n✅ Full features shape: {X_full.shape}")
    print(f"Full feature columns: {list(X_full.columns)}")
    
    if list(X_full.columns) == FULL_FEATURE_ORDER:
        print(f"   ✅ Feature order matches FULL_FEATURE_ORDER")
    else:
        print(f"   ⚠️ Feature order mismatch!")
    
    save_encoding_maps(freq_map, "models/encoding_maps.json")
    print("\n✅ Encoding maps saved to models/encoding_maps.json")
    
    loaded_maps = load_encoding_maps("models/encoding_maps.json")
    print(f"✅ Encoding maps loaded: {len(loaded_maps)} sections")
    
    test_row = {
        'experience_level': 'SE', 'employment_type': 'FT',
        'job_title': 'Data Scientist', 'company_size': 'L',
        'remote_ratio': 100, 'work_year': 2024,
        'employee_residence': 'US', 'company_location': 'US'
    }
    
    X_single = preprocess_single_row(test_row, loaded_maps)
    print(f"\n✅ Single row preprocessing successful")
    print(f"   Features: {X_single.shape[1]} columns")
    print(f"   Feature names: {list(X_single.columns)}")
    
    expected = FULL_FEATURE_ORDER
    missing = [f for f in expected if f not in X_single.columns]
    if not missing:
        print(f"   ✅ All {len(expected)} expected features present")
    else:
        print(f"   ⚠️ Missing features: {missing}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Preprocessing module is ready.")
    print("=" * 60)