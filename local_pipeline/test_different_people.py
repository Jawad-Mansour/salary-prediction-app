"""
Test different people from different backgrounds
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import joblib
import numpy as np
from salary_src.preprocess import load_encoding_maps, preprocess_single_row
from dotenv import load_dotenv
from supabase import create_client
import os
import time

load_dotenv()

print("=" * 70)
print("👥 TESTING DIFFERENT PEOPLE FROM AROUND THE WORLD")
print("=" * 70)

# Load model, transformer, and maps
model = joblib.load('models/decision_tree.pkl')
transformer = joblib.load('models/transformer.pkl')
maps = load_encoding_maps('models/encoding_maps.json')

# Supabase
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

# =========================================================
# TEST DIFFERENT PEOPLE
# =========================================================

people = [
    # (Name, Job Title, Experience, Employment, Company Size, Remote, Location)
    
    # USA - Different roles
    ("John (USA)", "Data Analyst", "EN", "FT", "M", 0, "US"),
    ("Sarah (USA)", "Data Engineer", "MI", "FT", "L", 50, "US"),
    ("Mike (USA)", "Data Scientist", "SE", "FT", "L", 100, "US"),
    ("Lisa (USA)", "ML Engineer", "SE", "FT", "L", 100, "US"),
    ("David (USA)", "Data Science Manager", "EX", "FT", "L", 0, "US"),
    
    # Europe - Different countries
    ("Emma (UK)", "Data Scientist", "SE", "FT", "L", 100, "GB"),
    ("Hans (Germany)", "Data Engineer", "MI", "FT", "L", 50, "DE"),
    ("Pierre (France)", "Data Analyst", "EN", "FT", "M", 0, "FR"),
    ("Maria (Spain)", "ML Engineer", "MI", "FT", "M", 100, "ES"),
    ("Luca (Italy)", "Data Scientist", "EN", "FT", "S", 50, "IT"),
    
    # Asia - Different countries
    ("Priya (India)", "Data Scientist", "SE", "FT", "L", 100, "IN"),
    ("Kenji (Japan)", "Data Engineer", "MI", "FT", "L", 50, "JP"),
    ("Wei (China)", "Data Analyst", "EN", "FT", "L", 0, "CN"),
    ("Ahmad (Singapore)", "ML Engineer", "SE", "FT", "M", 100, "SG"),
    ("Min (Korea)", "Data Scientist", "MI", "FT", "L", 50, "KR"),
    
    # Remote workers (living in low-cost countries, working for US companies)
    ("Remote1 (India for US)", "Data Scientist", "SE", "FT", "L", 100, "IN"),
    ("Remote2 (Brazil for US)", "Data Engineer", "MI", "FT", "L", 100, "BR"),
    ("Remote3 (Philippines for US)", "Data Analyst", "MI", "FT", "M", 100, "PH"),
    
    # Different company sizes
    ("Small Co (USA)", "Data Scientist", "MI", "FT", "S", 50, "US"),
    ("Medium Co (USA)", "Data Scientist", "MI", "FT", "M", 50, "US"),
    ("Large Co (USA)", "Data Scientist", "MI", "FT", "L", 50, "US"),
    
    # Different employment types
    ("Freelance (USA)", "Data Scientist", "SE", "FL", "M", 100, "US"),
    ("Contract (USA)", "Data Engineer", "MI", "CT", "L", 100, "US"),
    ("Part-time (UK)", "Data Analyst", "EN", "PT", "M", 50, "GB"),
]

print("\n📊 Generating predictions for different people...")
print("-" * 70)

results = []

for name, title, exp, emp, size, remote, loc in people:
    test_input = {
        'job_title': title,
        'experience_level': exp,
        'employment_type': emp,
        'company_size': size,
        'remote_ratio': remote,
        'work_year': 2024,
        'employee_residence': loc,
        'company_location': loc
    }
    
    X = preprocess_single_row(test_input, maps)
    pred_transformed = model.predict(X)[0]
    pred = transformer.inverse_transform(np.array([[pred_transformed]]))[0]
    
    results.append((name, title, exp, emp, size, remote, loc, pred))
    
    # Insert into Supabase
    record = {
        'job_title': title,
        'experience_level': exp,
        'employment_type': emp,
        'company_size': size,
        'remote_ratio': remote,
        'work_year': 2024,
        'employee_residence': loc,
        'company_location': loc,
        'predicted_salary_usd': pred,
        'prediction_version': 'test_people'
    }
    
    supabase.table('predictions').insert(record).execute()
    time.sleep(0.1)  # Small delay

print("\n" + "=" * 70)
print("📊 RESULTS - SALARY BY PERSON")
print("=" * 70)

# Group by location type
us_results = [r for r in results if r[6] == "US"]
uk_results = [r for r in results if r[6] == "GB"]
de_results = [r for r in results if r[6] == "DE"]
in_results = [r for r in results if r[6] == "IN"]
remote_results = [r for r in results if "Remote" in r[0]]

print("\n🇺🇸 USA Professionals:")
print("-" * 50)
for name, title, exp, emp, size, remote, loc, salary in us_results:
    print(f"  {name:<25} | {title:<20} | {exp} | ${salary:>10,.2f}")

print("\n🇬🇧 UK / 🇩🇪 Europe:")
print("-" * 50)
for name, title, exp, emp, size, remote, loc, salary in uk_results + de_results:
    print(f"  {name:<25} | {title:<20} | {exp} | ${salary:>10,.2f}")

print("\n🇮🇳 India / Asia:")
print("-" * 50)
for name, title, exp, emp, size, remote, loc, salary in in_results:
    print(f"  {name:<25} | {title:<20} | {exp} | ${salary:>10,.2f}")

print("\n🏠 Remote Workers (Living in low-cost countries):")
print("-" * 50)
for name, title, exp, emp, size, remote, loc, salary in remote_results:
    print(f"  {name:<25} | {title:<20} | {exp} | ${salary:>10,.2f}")

print("\n🏢 Company Size Comparison (Same role, same level, USA):")
print("-" * 50)
size_comparison = [r for r in results if "Co" in r[0]]
for name, title, exp, emp, size, remote, loc, salary in size_comparison:
    print(f"  {name:<25} | {size:<6} | ${salary:>10,.2f}")

print("\n💼 Employment Type Comparison (Same role, same level, USA):")
print("-" * 50)
emp_comparison = [r for r in results if r[3] in ["FL", "CT", "PT"]]
for name, title, exp, emp, size, remote, loc, salary in emp_comparison:
    print(f"  {name:<25} | {emp:<8} | ${salary:>10,.2f}")

print("\n" + "=" * 70)
print("✅ All predictions inserted into Supabase!")
print("=" * 70)

# Summary statistics
all_salaries = [r[7] for r in results]
print(f"\n📊 Summary Statistics:")
print(f"   Total predictions: {len(results)}")
print(f"   Min salary: ${min(all_salaries):,.2f}")
print(f"   Max salary: ${max(all_salaries):,.2f}")
print(f"   Average salary: ${sum(all_salaries)/len(all_salaries):,.2f}")

