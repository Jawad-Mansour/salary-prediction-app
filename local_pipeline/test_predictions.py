"""Test salary predictions for different scenarios"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import joblib
import numpy as np
from salary_src.preprocess import load_encoding_maps, preprocess_single_row

# Load model and transformer
print("Loading model and transformer...")
model = joblib.load(project_root / 'models/decision_tree.pkl')
transformer = joblib.load(project_root / 'models/transformer.pkl')
maps = load_encoding_maps(project_root / 'models/encoding_maps.json')
print("✅ Model loaded successfully!\n")

# =========================================================
# TEST 1: SALARY BY LOCATION
# =========================================================
print("=" * 70)
print("🌍 TEST 1: SALARY BY LOCATION")
print("=" * 70)
print("Job: Data Scientist, Senior Level, Large Company, 100% Remote")
print("-" * 70)

locations = [
    ('NG', 'Nigeria'),
    ('IN', 'India'),
    ('BR', 'Brazil'),
    ('PL', 'Poland'),
    ('GB', 'United Kingdom'),
    ('US', 'United States'),
]

print(f"\n{'Location':<20} {'Salary (USD)':<15}")
print("-" * 38)

for code, name in locations:
    test = {
        'job_title': 'Data Scientist',
        'experience_level': 'SE',
        'employment_type': 'FT',
        'company_size': 'L',
        'remote_ratio': 100,
        'work_year': 2024,
        'employee_residence': code,
        'company_location': code
    }
    
    X = preprocess_single_row(test, maps)
    pred_transformed = model.predict(X)[0]
    pred = transformer.inverse_transform(np.array([[pred_transformed]]))[0]
    
    print(f"{name:<20} ${pred:>12,.2f}")

# =========================================================
# TEST 2: CAREER PROGRESSION
# =========================================================
print("\n" + "=" * 70)
print("📈 TEST 2: CAREER PROGRESSION")
print("=" * 70)
print("Job: Data Scientist, USA, Large Company")
print("-" * 70)

levels = [
    ('EN', 'Entry Level'),
    ('MI', 'Mid Level'),
    ('SE', 'Senior'),
    ('EX', 'Executive'),
]

print(f"\n{'Level':<15} {'Salary (USD)':<15}")
print("-" * 32)

for code, name in levels:
    test = {
        'job_title': 'Data Scientist',
        'experience_level': code,
        'employment_type': 'FT',
        'company_size': 'L',
        'remote_ratio': 100,
        'work_year': 2024,
        'employee_residence': 'US',
        'company_location': 'US'
    }
    
    X = preprocess_single_row(test, maps)
    pred_transformed = model.predict(X)[0]
    pred = transformer.inverse_transform(np.array([[pred_transformed]]))[0]
    
    print(f"{name:<15} ${pred:>12,.2f}")

# =========================================================
# TEST 3: JOB TITLE COMPARISON
# =========================================================
print("\n" + "=" * 70)
print("💼 TEST 3: JOB TITLE COMPARISON")
print("=" * 70)
print("Location: USA, Senior Level, Large Company")
print("-" * 70)

titles = [
    'Data Analyst',
    'Data Engineer',
    'Data Scientist',
    'Machine Learning Engineer',
]

print(f"\n{'Job Title':<25} {'Salary (USD)':<15}")
print("-" * 42)

for title in titles:
    test = {
        'job_title': title,
        'experience_level': 'SE',
        'employment_type': 'FT',
        'company_size': 'L',
        'remote_ratio': 100,
        'work_year': 2024,
        'employee_residence': 'US',
        'company_location': 'US'
    }
    
    X = preprocess_single_row(test, maps)
    pred_transformed = model.predict(X)[0]
    pred = transformer.inverse_transform(np.array([[pred_transformed]]))[0]
    
    print(f"{title:<25} ${pred:>12,.2f}")

# =========================================================
# TEST 4: COMPANY SIZE IMPACT
# =========================================================
print("\n" + "=" * 70)
print("🏢 TEST 4: COMPANY SIZE IMPACT")
print("=" * 70)
print("Job: Data Scientist, USA, Senior Level")
print("-" * 70)

sizes = [
    ('S', 'Small'),
    ('M', 'Medium'),
    ('L', 'Large'),
]

print(f"\n{'Company Size':<15} {'Salary (USD)':<15}")
print("-" * 32)

for code, name in sizes:
    test = {
        'job_title': 'Data Scientist',
        'experience_level': 'SE',
        'employment_type': 'FT',
        'company_size': code,
        'remote_ratio': 100,
        'work_year': 2024,
        'employee_residence': 'US',
        'company_location': 'US'
    }
    
    X = preprocess_single_row(test, maps)
    pred_transformed = model.predict(X)[0]
    pred = transformer.inverse_transform(np.array([[pred_transformed]]))[0]
    
    print(f"{name:<15} ${pred:>12,.2f}")

print("\n" + "=" * 70)
print("✅ ALL TESTS COMPLETED!")
print("=" * 70)
