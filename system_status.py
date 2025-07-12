#!/usr/bin/env python3
"""
Employee Salary Prediction System - Final Status Report
========================================================

This script provides a comprehensive status check of the entire system.
"""

import sys
import os
import pandas as pd
sys.path.append('.')

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_dataset():
    """Check dataset availability and quality."""
    print_header("DATASET STATUS")
    
    try:
        df = pd.read_csv('data/raw/employee_salary_dataset.csv')
        print(f"✅ Dataset loaded successfully")
        print(f"   📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"   💰 Salary range: ₹{df['annual_salary'].min():,.0f} - ₹{df['annual_salary'].max():,.0f}")
        print(f"   🏢 Company sizes: {df['company_size'].nunique()} unique sizes")
        print(f"   🎯 Departments: {df['department'].nunique()} unique departments")
        print(f"   🌍 Cities: {df['city'].nunique()} unique cities")
        return True
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        return False

def check_modules():
    """Check all core modules."""
    print_header("MODULE STATUS")
    
    modules = [
        ("Utils", "src.utils", "logger"),
        ("Data Preprocessing", "src.data_preprocessing", "DataPreprocessor"),
        ("Feature Engineering", "src.feature_engineering", "FeatureEngineer"),
        ("Model Training", "src.model_training", "ModelTrainer"),
    ]
    
    all_good = True
    for name, module_path, class_name in modules:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {name} module ready")
        except Exception as e:
            print(f"❌ {name} module error: {e}")
            all_good = False
    
    return all_good

def check_webapp():
    """Check Streamlit web application."""
    print_header("WEB APPLICATION STATUS")
    
    try:
        from webapp.app import load_dataset, format_currency
        
        # Test dataset loading
        df = load_dataset()
        if df is not None:
            print(f"✅ Web app can load dataset: {df.shape}")
        else:
            print("❌ Web app cannot load dataset")
            return False
        
        # Test utility functions
        test_amount = format_currency(1500000)
        print(f"✅ Currency formatting: {test_amount}")
        
        print("✅ Streamlit app is ready to run")
        return True
        
    except Exception as e:
        print(f"❌ Web app error: {e}")
        return False

def check_project_structure():
    """Check project directory structure."""
    print_header("PROJECT STRUCTURE")
    
    required_dirs = [
        'data/raw',
        'data/processed', 
        'models/model_artifacts',
        'notebooks',
        'src',
        'webapp',
        'logs'
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ Missing: {dir_path}/")
            all_good = False
    
    # Check key files
    key_files = [
        'requirements.txt',
        'README.md',
        'start.py',
        'config.json'
    ]
    
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_good = False
    
    return all_good

def main():
    """Run complete system check."""
    print_header("EMPLOYEE SALARY PREDICTION SYSTEM - STATUS CHECK")
    print("🔍 Performing comprehensive system validation...")
    
    checks = [
        ("Project Structure", check_project_structure),
        ("Dataset", check_dataset),
        ("Core Modules", check_modules),
        ("Web Application", check_webapp)
    ]
    
    results = []
    for check_name, check_func in checks:
        result = check_func()
        results.append((check_name, result))
    
    # Final summary
    print_header("FINAL SYSTEM STATUS")
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ READY" if passed else "❌ ISSUES"
        print(f"   {status}: {check_name}")
        if not passed:
            all_passed = False
    
    print(f"\n{'='*60}")
    
    if all_passed:
        print("🎉 SYSTEM FULLY OPERATIONAL!")
        print("\n🚀 Ready to use:")
        print("   • Streamlit Web App: streamlit run webapp/app.py")
        print("   • Interactive Menu: python start.py")
        print("   • Jupyter Notebooks: jupyter notebook")
        print("\n💡 Features available:")
        print("   • Salary prediction with 10,000 employee dataset")
        print("   • Advanced data preprocessing and feature engineering")
        print("   • Multiple ML algorithms (Random Forest, XGBoost, LightGBM)")
        print("   • Interactive web interface with visualizations")
        print("   • Model interpretability with SHAP values")
        print("   • Production-ready code with proper logging")
    else:
        print("⚠️  SYSTEM HAS ISSUES - Check errors above")
    
    return all_passed

if __name__ == "__main__":
    main()
