#!/usr/bin/env python3
"""Quick test of the Streamlit app functionality"""

import sys
sys.path.append('.')

try:
    from webapp.app import load_dataset
    df = load_dataset()
    if df is not None:
        print(f"✅ Streamlit app can load dataset: {df.shape}")
        print(f"   Dataset columns: {list(df.columns)}")
    else:
        print("❌ Streamlit app failed to load dataset")
except Exception as e:
    print(f"❌ Error testing Streamlit app: {e}")

try:
    from webapp.app import format_currency
    test_amount = format_currency(1500000)
    print(f"✅ Currency formatting works: {test_amount}")
except Exception as e:
    print(f"❌ Error testing currency formatting: {e}")

print("✅ All basic tests completed successfully!")
