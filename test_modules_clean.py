#!/usr/bin/env python3
"""
Simple test script to validate all project modules and dataset.
"""

import sys
import os
sys.path.append('.')

def test_dataset():
    """Test if dataset exists and is readable."""
    try:
        import pandas as pd
        
        # Test different paths
        paths_to_try = [
            'data/raw/employee_salary_dataset.csv',
            './data/raw/employee_salary_dataset.csv'
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"Dataset found at {path}: {df.shape}")
                print(f"   Columns: {list(df.columns)}")
                return True
        
        print("Dataset not found at any expected location")
        return False
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

def test_utils():
    """Test utils module."""
    try:
        from src.utils import logger
        print("Utils module imported successfully")
        return True
    except Exception as e:
        print(f"Error importing utils: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing module."""
    try:
        from src.data_preprocessing import DataPreprocessor
        print("Data preprocessing module imported successfully")
        return True
    except Exception as e:
        print(f"Error importing data preprocessing: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering module."""
    try:
        from src.feature_engineering import FeatureEngineer
        print("Feature engineering module imported successfully")
        return True
    except Exception as e:
        print(f"Error importing feature engineering: {e}")
        return False

def test_model_training():
    """Test model training module."""
    try:
        from src.model_training import ModelTrainer
        print("Model training module imported successfully")
        return True
    except Exception as e:
        print(f"Error importing model training: {e}")
        return False

def test_streamlit_app():
    """Test Streamlit app imports."""
    try:
        from webapp.app import load_dataset, format_currency
        print("Streamlit app modules imported successfully")
        
        # Test dataset loading function
        df = load_dataset()
        if df is not None:
            print(f"   Dataset loaded via app function: {df.shape}")
        else:
            print("   Dataset loading returned None (expected if no dataset)")
        
        # Test utility functions
        test_amount = format_currency(1500000)
        print(f"   Currency formatting test: {test_amount}")
        
        return True
    except Exception as e:
        print(f"Error testing Streamlit app: {e}")
        return False

def main():
    """Run all tests."""
    print("Running Project Module Tests")
    print("=" * 50)
    
    tests = [
        ("Dataset", test_dataset),
        ("Utils Module", test_utils),
        ("Data Preprocessing", test_data_preprocessing),
        ("Feature Engineering", test_feature_engineering),
        ("Model Training", test_model_training),
        ("Streamlit App", test_streamlit_app)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"   {status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("ALL TESTS PASSED! Project is ready to run.")
        print("\nNext steps:")
        print("   1. Run: streamlit run webapp/app.py")
        print("   2. Or use: python start.py")
    else:
        print("Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main()
