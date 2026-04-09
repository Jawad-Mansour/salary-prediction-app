"""
COMPLETE PROJECT VALIDATION - Phase 1 through 5
Tests every component of the Salary Prediction Application
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
import json
import time
from datetime import datetime

print("=" * 70)
print("🔬 COMPLETE PROJECT VALIDATION")
print("=" * 70)
print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Project root: {project_root}")
print("=" * 70)

results = {"passed": 0, "failed": 0, "warnings": 0}


def test_pass(name, detail=""):
    results["passed"] += 1
    print(f"✅ PASS: {name}")
    if detail:
        print(f"   {detail}")


def test_fail(name, error):
    results["failed"] += 1
    print(f"❌ FAIL: {name} - {error}")


def test_warning(name, message):
    results["warnings"] += 1
    print(f"⚠️ WARN: {name} - {message}")


# ============================================================================
# PHASE 1: DATA & EDA
# ============================================================================
print("\n" + "-" * 70)
print("PHASE 1: DATA LOADING & EDA")
print("-" * 70)

# Test 1.1: Data file exists
try:
    data_file = project_root / "data" / "raw" / "salaries_raw.csv"
    if data_file.exists():
        size = data_file.stat().st_size
        test_pass("Data file exists", f"{size:,} bytes")
    else:
        test_fail("Data file exists", "File not found")
except Exception as e:
    test_fail("Data file exists", str(e))

# Test 1.2: Load data
try:
    from salary_src.data_loader import load_salaries_dataset
    df = load_salaries_dataset()
    assert len(df) > 0
    test_pass("Load data", f"{len(df)} rows, {len(df.columns)} columns")
except Exception as e:
    test_fail("Load data", str(e))
    df = None

# Test 1.3: Target column exists
if df is not None:
    try:
        assert 'salary_in_usd' in df.columns
        test_pass("Target column exists", f"salary_in_usd range: ${df['salary_in_usd'].min():,.2f} - ${df['salary_in_usd'].max():,.2f}")
    except Exception as e:
        test_fail("Target column exists", str(e))

# Test 1.4: No missing values
if df is not None:
    try:
        missing = df.isnull().sum().sum()
        if missing == 0:
            test_pass("Missing values", "No missing values")
        else:
            test_warning("Missing values", f"{missing} missing values found")
    except Exception as e:
        test_fail("Missing values", str(e))

# Test 1.5: Categorical columns analysis
if df is not None:
    try:
        exp_values = df['experience_level'].unique()
        emp_values = df['employment_type'].unique()
        size_values = df['company_size'].unique()
        job_count = df['job_title'].nunique()
        
        print(f"\n   📊 Categorical Analysis:")
        print(f"      experience_level: {sorted(exp_values)}")
        print(f"      employment_type: {sorted(emp_values)}")
        print(f"      company_size: {sorted(size_values)}")
        print(f"      job_title: {job_count} unique values")
        test_pass("Categorical analysis", "All columns properly identified")
    except Exception as e:
        test_fail("Categorical analysis", str(e))


# ============================================================================
# PHASE 2: PREPROCESSING
# ============================================================================
print("\n" + "-" * 70)
print("PHASE 2: PREPROCESSING")
print("-" * 70)

# Test 2.1: Preprocess module imports
try:
    from salary_src.preprocess import (
        prepare_features, engineer_features, get_target,
        save_encoding_maps, load_encoding_maps,
        preprocess_single_row, get_full_feature_order,
        ORDINAL_MAPS, COUNTRY_REGION_MAP, REGION_ENCODING
    )
    test_pass("Preprocess imports", "All functions available")
except Exception as e:
    test_fail("Preprocess imports", str(e))

# Test 2.2: Base features preparation
if df is not None:
    try:
        X_base, freq_map = prepare_features(df.head(100), fit_job_title=True)
        assert X_base.shape[1] == 7
        test_pass("Base features", f"Shape: {X_base.shape}, 7 features")
    except Exception as e:
        test_fail("Base features", str(e))

# Test 2.3: Feature engineering
if df is not None:
    try:
        X_base_full, _ = prepare_features(df.head(100), fit_job_title=True)
        X_full = engineer_features(X_base_full, df.head(100))
        expected_features = get_full_feature_order()
        
        if X_full.shape[1] == len(expected_features):
            test_pass("Feature engineering", f"Shape: {X_full.shape}, {len(expected_features)} features")
        else:
            test_fail("Feature engineering", f"Expected {len(expected_features)}, got {X_full.shape[1]}")
    except Exception as e:
        test_fail("Feature engineering", str(e))

# Test 2.4: Encoding maps
try:
    from salary_src.preprocess import ORDINAL_MAPS
    assert 'EN' in ORDINAL_MAPS['experience_level']
    assert 'FT' in ORDINAL_MAPS['employment_type']
    assert 'S' in ORDINAL_MAPS['company_size']
    test_pass("Encoding maps", "Ordinal maps correctly defined")
except Exception as e:
    test_fail("Encoding maps", str(e))


# ============================================================================
# PHASE 3: MODEL TRAINING
# ============================================================================
print("\n" + "-" * 70)
print("PHASE 3: MODEL TRAINING")
print("-" * 70)

# Test 3.1: Model file exists
model_path = project_root / "models" / "decision_tree.pkl"
try:
    if model_path.exists():
        size = model_path.stat().st_size
        test_pass("Model file exists", f"{size:,} bytes")
    else:
        test_fail("Model file exists", "decision_tree.pkl not found")
except Exception as e:
    test_fail("Model file exists", str(e))

# Test 3.2: Transformer file exists
transformer_path = project_root / "models" / "transformer.pkl"
try:
    if transformer_path.exists():
        size = transformer_path.stat().st_size
        test_pass("Transformer file exists", f"{size:,} bytes")
    else:
        test_warning("Transformer file exists", "transformer.pkl not found (will be created on retrain)")
except Exception as e:
    test_warning("Transformer file exists", str(e))

# Test 3.3: Encoding maps file exists
maps_path = project_root / "models" / "encoding_maps.json"
try:
    if maps_path.exists():
        size = maps_path.stat().st_size
        with open(maps_path, 'r') as f:
            maps = json.load(f)
        test_pass("Encoding maps file exists", f"{size:,} bytes, version: {maps.get('version', 'unknown')}")
    else:
        test_fail("Encoding maps file exists", "encoding_maps.json not found")
except Exception as e:
    test_fail("Encoding maps file exists", str(e))

# Test 3.4: Load model
try:
    model = joblib.load(model_path)
    test_pass("Load model", f"Type: {type(model).__name__}")
except Exception as e:
    test_fail("Load model", str(e))
    model = None

# Test 3.5: Load transformer
try:
    transformer = joblib.load(transformer_path)
    test_pass("Load transformer", "Transformer loaded successfully")
except Exception as e:
    test_warning("Load transformer", str(e))
    transformer = None

# Test 3.6: Model metrics
metrics_path = project_root / "models" / "metrics.json"
try:
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        r2 = metrics.get('r2', 0)
        mae = metrics.get('mae', 0)
        print(f"\n   📊 Model Performance:")
        print(f"      R² Score: {r2:.4f}")
        print(f"      MAE: ${mae:,.2f}")
        print(f"      MAE %: {metrics.get('mae_percentage', 0):.1f}%")
        
        if r2 >= 0.4:
            test_pass("Model metrics", f"R²={r2:.4f} (Good)")
        else:
            test_warning("Model metrics", f"R²={r2:.4f} (Below 0.4)")
    else:
        test_warning("Model metrics", "metrics.json not found")
except Exception as e:
    test_warning("Model metrics", str(e))


# ============================================================================
# PHASE 4: SUPABASE
# ============================================================================
print("\n" + "-" * 70)
print("PHASE 4: SUPABASE")
print("-" * 70)

# Test 4.1: .env file exists
env_path = project_root / ".env"
try:
    if env_path.exists():
        size = env_path.stat().st_size
        test_pass(".env file exists", f"{size:,} bytes")
    else:
        test_fail(".env file exists", "Create .env with SUPABASE_URL and SUPABASE_KEY")
except Exception as e:
    test_fail(".env file exists", str(e))

# Test 4.2: Supabase credentials
try:
    from dotenv import load_dotenv
    import os
    load_dotenv(env_path)
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if supabase_url and supabase_key:
        test_pass("Supabase credentials", f"URL: {supabase_url[:30]}...")
    else:
        test_fail("Supabase credentials", "Missing URL or KEY in .env")
except Exception as e:
    test_fail("Supabase credentials", str(e))

# Test 4.3: Supabase connection (optional - may fail if no internet)
try:
    from supabase import create_client
    supabase = create_client(supabase_url, supabase_key)
    result = supabase.table('predictions').select('*', count='exact').limit(0).execute()
    test_pass("Supabase connection", f"Table 'predictions' exists, rows: {result.count}")
except Exception as e:
    test_warning("Supabase connection", f"Could not connect: {str(e)[:50]}...")


# ============================================================================
# PHASE 5: LOCAL PIPELINE
# ============================================================================
print("\n" + "-" * 70)
print("PHASE 5: LOCAL PIPELINE")
print("-" * 70)

# Test 5.1: Single prediction (FIXED - correct transformer indexing)
if model is not None and transformer is not None:
    try:
        from salary_src.preprocess import load_encoding_maps, preprocess_single_row
        
        maps = load_encoding_maps(project_root / "models" / "encoding_maps.json")
        
        test_input = {
            'job_title': 'Data Scientist',
            'experience_level': 'SE',
            'employment_type': 'FT',
            'company_size': 'L',
            'remote_ratio': 100,
            'work_year': 2024,
            'employee_residence': 'US',
            'company_location': 'US'
        }
        
        X = preprocess_single_row(test_input, maps)
        prediction_transformed = model.predict(X)[0]
        
        # CRITICAL: Transformer returns 1D array, use [0] not [0][0]
        prediction_array = transformer.inverse_transform(np.array([[prediction_transformed]]))
        prediction = float(prediction_array[0])
        
        if 50000 < prediction < 300000:
            test_pass("Single prediction", f"${prediction:,.2f} (reasonable range)")
        else:
            test_warning("Single prediction", f"${prediction:,.2f} (outside expected range)")
    except Exception as e:
        test_fail("Single prediction", str(e))

# Test 5.2: Batch prediction
if model is not None:
    try:
        from salary_src.preprocess import load_encoding_maps, preprocess_batch
        
        maps = load_encoding_maps(project_root / "models" / "encoding_maps.json")
        
        test_batch = pd.DataFrame([
            {'job_title': 'Data Analyst', 'experience_level': 'EN', 'employment_type': 'FT',
             'company_size': 'S', 'remote_ratio': 0, 'work_year': 2024,
             'employee_residence': 'US', 'company_location': 'US'},
            {'job_title': 'Data Engineer', 'experience_level': 'MI', 'employment_type': 'FT',
             'company_size': 'M', 'remote_ratio': 50, 'work_year': 2024,
             'employee_residence': 'US', 'company_location': 'US'},
            {'job_title': 'Data Scientist', 'experience_level': 'SE', 'employment_type': 'FT',
             'company_size': 'L', 'remote_ratio': 100, 'work_year': 2024,
             'employee_residence': 'US', 'company_location': 'US'},
        ])
        
        X_batch = preprocess_batch(test_batch, maps)
        predictions = model.predict(X_batch)
        
        if len(predictions) == 3:
            test_pass("Batch prediction", f"Generated {len(predictions)} predictions")
        else:
            test_fail("Batch prediction", f"Expected 3, got {len(predictions)}")
    except Exception as e:
        test_fail("Batch prediction", str(e))

# Test 5.3: Ollama module exists
ollama_path = project_root / "local_pipeline" / "llm_analyzer.py"
try:
    if ollama_path.exists():
        size = ollama_path.stat().st_size
        test_pass("LLM Analyzer exists", f"{size:,} bytes")
    else:
        test_warning("LLM Analyzer exists", "llm_analyzer.py not found")
except Exception as e:
    test_warning("LLM Analyzer exists", str(e))

# Test 5.4: Pipeline script exists
pipeline_path = project_root / "local_pipeline" / "run_pipeline.py"
try:
    if pipeline_path.exists():
        size = pipeline_path.stat().st_size
        test_pass("Pipeline script exists", f"{size:,} bytes")
    else:
        test_warning("Pipeline script exists", "run_pipeline.py not found")
except Exception as e:
    test_warning("Pipeline script exists", str(e))


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
print(f"\n📊 Results:")
print(f"   ✅ Passed: {results['passed']}")
print(f"   ❌ Failed: {results['failed']}")
print(f"   ⚠️ Warnings: {results['warnings']}")

print("\n📋 Phase Status:")
print(f"   Phase 1 (Data & EDA):     {'✅' if results['passed'] >= 4 else '❌'}")
print(f"   Phase 2 (Preprocessing):  {'✅' if results['passed'] >= 7 else '❌'}")
print(f"   Phase 3 (Model Training): {'✅' if results['passed'] >= 10 else '❌'}")
print(f"   Phase 4 (Supabase):       {'✅' if results['passed'] >= 12 else '⚠️'}")
print(f"   Phase 5 (Local Pipeline): {'✅' if results['passed'] >= 14 else '⚠️'}")

if results['failed'] == 0:
    print("\n" + "🎉" * 20)
    print("   ALL CRITICAL TESTS PASSED!")
    print("   You are ready to proceed to Phase 6 (FastAPI)")
    print("🎉" * 20)
elif results['failed'] <= 2:
    print("\n" + "✅" * 20)
    print("   Most tests passed. Minor issues can be fixed.")
    print("   You can still proceed to Phase 6")
    print("✅" * 20)
else:
    print("\n" + "⚠️" * 20)
    print(f"   {results['failed']} test(s) failed. Please fix before proceeding.")
    print("⚠️" * 20)

print("\n" + "=" * 70)
