"""Manual Supabase Test"""

from dotenv import load_dotenv
from supabase import create_client
import os
from pathlib import Path

# Load environment variables
load_dotenv()

# Create Supabase client
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

print("=" * 60)
print("MANUAL SUPABASE TEST")
print("=" * 60)

# ============================================
# TEST 1: INSERT A ROW
# ============================================
print("\n📝 TEST 1: Inserting a row...")

data = {
    'job_title': 'Manual Test',
    'experience_level': 'SE',
    'employment_type': 'FT',
    'company_size': 'L',
    'remote_ratio': 100,
    'work_year': 2024,
    'employee_residence': 'US',
    'company_location': 'US',
    'predicted_salary_usd': 175000,
    'llm_narrative': 'This is a manual test from the terminal'
}

result = supabase.table('predictions').insert(data).execute()
inserted_id = result.data[0]['id']

print(f"✅ Inserted!")
print(f"   ID: {inserted_id}")
print(f"   Job: {result.data[0]['job_title']}")
print(f"   Salary: ${result.data[0]['predicted_salary_usd']:,.2f}")

# ============================================
# TEST 2: READ ALL ROWS
# ============================================
print("\n📖 TEST 2: Reading all rows...")

result = supabase.table('predictions').select('*').order('created_at', desc=True).limit(5).execute()

print(f"✅ Found {len(result.data)} rows:")
for row in result.data:
    print(f"   - {row['job_title']}: ${row['predicted_salary_usd']:,.2f} ({row['created_at'][:10]})")

# ============================================
# TEST 3: DELETE THE TEST ROW
# ============================================
print("\n🗑️ TEST 3: Deleting the test row...")

supabase.table('predictions').delete().eq('id', inserted_id).execute()
print(f"✅ Deleted test row: {inserted_id}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)