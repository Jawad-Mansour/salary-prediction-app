"""View predictions from Supabase"""

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

# Get predictions
result = supabase.table('predictions').select('*').order('created_at', desc=True).limit(10).execute()

print('=' * 80)
print('📊 PREDICTIONS IN SUPABASE')
print('=' * 80)
print(f'Total rows shown: {len(result.data)}')
print()

for row in result.data:
    has_narrative = '✅' if row.get('llm_narrative') else '❌'
    has_chart = '✅' if row.get('chart_base64') else '❌'
    
    print(f'Job: {row["job_title"]:<25} | Exp: {row["experience_level"]:<2} | '
          f'Salary: ${row["predicted_salary_usd"]:>12,.2f} | '
          f'Narrative: {has_narrative} | Chart: {has_chart}')
    
    if has_narrative == '✅':
        narrative_preview = row.get('llm_narrative', '')[:100]
        print(f'   📝 Preview: {narrative_preview}...')
    print()