#!/usr/bin/env python3
"""
Test script to verify the dataset loading functionality.
"""

import sys
import os
sys.path.append('webapp')

def test_dataset_loading():
    try:
        from app import load_dataset
        print("🔄 Testing dataset loading...")
        
        df = load_dataset()
        
        if df is not None:
            print(f"✅ Dataset loaded successfully!")
            print(f"📊 Shape: {df.shape}")
            print(f"📝 Columns: {list(df.columns)}")
            print(f"💰 Salary range: ₹{df['annual_salary'].min():,.0f} - ₹{df['annual_salary'].max():,.0f}")
            return True
        else:
            print("❌ Failed to load dataset")
            return False
            
    except Exception as e:
        print(f"❌ Error testing dataset loading: {e}")
        return False

if __name__ == "__main__":
    success = test_dataset_loading()
    print(f"\n🎯 Test result: {'PASSED' if success else 'FAILED'}")
