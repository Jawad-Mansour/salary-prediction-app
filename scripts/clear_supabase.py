"""Clear all predictions from Supabase"""

from dotenv import load_dotenv
from supabase import create_client
import os

load_dotenv()

supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

print("=" * 50)
print("🗑️ CLEARING SUPABASE DATABASE")
print("=" * 50)

# Get current count
result = supabase.table('predictions').select('*', count='exact').execute()
current_count = result.count
print(f"Current rows: {current_count}")

if current_count > 0:
    # Delete all rows
    result = supabase.table('predictions').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
    print(f"✅ Deleted {current_count} rows")
    
    # Verify
    result = supabase.table('predictions').select('*', count='exact').execute()
    print(f"Remaining rows: {result.count}")
else:
    print("✅ Database already empty")

print("=" * 50)
print("✅ Database cleared! Ready for fresh tests.")
