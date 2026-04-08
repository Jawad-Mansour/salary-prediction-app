# ============================================================
# config.py — Single source of truth for all encoding maps
# Used by preprocess.py, train_model.py, and fastapi_app/utils.py
# ============================================================

EXPERIENCE_LEVEL_MAP = {
    "EN": 0,  # Entry level
    "MI": 1,  # Mid level
    "SE": 2,  # Senior
    "EX": 3   # Executive
}

EMPLOYMENT_TYPE_MAP = {
    "FT": 0,  # Full time
    "PT": 1,  # Part time
    "CT": 2,  # Contract
    "FL": 3   # Freelance
}

COMPANY_SIZE_MAP = {
    "S": 0,  # Small
    "M": 1,  # Medium
    "L": 2   # Large
}

REMOTE_RATIO_VALUES = [0, 50, 100]

WORK_YEAR_MIN = 2020
WORK_YEAR_MAX = 2023

TARGET_COLUMN = "salary_in_usd"

FEATURES = [
    "experience_level",
    "employment_type",
    "company_size",
    "remote_ratio",
    "work_year",
    "job_title_encoded"
]

MODEL_PATH = "models/decision_tree_v1.pkl"
FREQ_ENCODING_PATH = "models/job_title_freq_map.pkl"