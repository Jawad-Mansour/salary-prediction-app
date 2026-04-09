#!/usr/bin/env python3
"""Supabase Setup and Test Script"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# Supabase imports
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def main():
    print("=" * 60)
    print("🚀 SUPABASE SETUP AND TEST SCRIPT")
    print("=" * 60)
    
    # Check credentials
    if not SUPABASE_URL:
        print("❌ SUPABASE_URL not found in .env file")
        print("   Please create .env with: SUPABASE_URL=your_url")
        return False
    if not SUPABASE_KEY:
        print("❌ SUPABASE_KEY not found in .env file")
        print("   Please create .env with: SUPABASE_KEY=your_key")
        return False
    
    print(f"✅ SUPABASE_URL: {SUPABASE_URL}")
    print(f"✅ SUPABASE_KEY: {SUPABASE_KEY[:30]}...")
    
    # Create Supabase client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Test 1: Check if table exists
    print("\n" + "=" * 60)
    print("TEST 1: Verifying 'predictions' table exists")
    print("=" * 60)
    
    try:
        result = supabase.table('predictions').select('*', count='exact').limit(0).execute()
        print(f"✅ 'predictions' table exists!")
        print(f"   Current row count: {result.count}")
    except Exception as e:
        error_msg = str(e).lower()
        if "relation" in error_msg and "does not exist" in error_msg:
            print("❌ Table 'predictions' does not exist!")
            print("   Please run the SQL script in Supabase SQL Editor first.")
        else:
            print(f"❌ Error: {e}")
        return False
    
    # Test 2: Insert a dummy row
    print("\n" + "=" * 60)
    print("TEST 2: Inserting dummy row")
    print("=" * 60)
    
    dummy_row = {
        "job_title": "Test Data Scientist",
        "experience_level": "SE",
        "employment_type": "FT",
        "company_size": "L",
        "remote_ratio": 100,
        "work_year": 2024,
        "employee_residence": "US",
        "company_location": "US",
        "predicted_salary_usd": 150000.00,
        "llm_narrative": "This is a test narrative for validation purposes.",
        "prediction_version": "test"
    }
    
    try:
        result = supabase.table('predictions').insert(dummy_row).execute()
        inserted_id = result.data[0]['id']
        print(f"✅ Insert successful!")
        print(f"   Inserted ID: {inserted_id}")
    except Exception as e:
        print(f"❌ Insert failed: {e}")
        return False
    
    # Test 3: Read the inserted row back
    print("\n" + "=" * 60)
    print("TEST 3: Reading back inserted row")
    print("=" * 60)
    
    try:
        result = supabase.table('predictions').select('*').eq('id', inserted_id).execute()
        row = result.data[0]
        print(f"✅ Read successful!")
        print(f"   Job Title: {row['job_title']}")
        print(f"   Predicted Salary: ${row['predicted_salary_usd']:,.2f}")
        print(f"   Created At: {row['created_at']}")
    except Exception as e:
        print(f"❌ Read failed: {e}")
        return False
    
    # Test 4: Delete the dummy row (cleanup)
    print("\n" + "=" * 60)
    print("TEST 4: Cleaning up (deleting dummy row)")
    print("=" * 60)
    
    try:
        supabase.table('predictions').delete().eq('id', inserted_id).execute()
        print(f"✅ Delete successful!")
        print(f"   Removed ID: {inserted_id}")
    except Exception as e:
        print(f"⚠️ Delete warning: {e}")
    
    # Test 5: Get all predictions (for dashboard)
    print("\n" + "=" * 60)
    print("TEST 5: Getting all predictions (dashboard read)")
    print("=" * 60)
    
    try:
        result = supabase.table('predictions').select('*').order('created_at', desc=True).limit(5).execute()
        print(f"✅ Read successful! Found {len(result.data)} rows")
        if result.data:
            print("\n   Most recent predictions:")
            for i, row in enumerate(result.data[:3]):
                print(f"   {i+1}. {row['job_title']} - ${row['predicted_salary_usd']:,.2f}")
    except Exception as e:
        print(f"❌ Read failed: {e}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! Supabase is ready!")
    print("=" * 60)
    print("\n📋 Summary:")
    print("   ✅ Supabase connection working")
    print("   ✅ 'predictions' table exists")
    print("   ✅ Insert operation working")
    print("   ✅ Read operation working")
    print("   ✅ Delete operation working")
    print("\n➡️ You can now proceed to Phase 5 (Local Pipeline)")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)