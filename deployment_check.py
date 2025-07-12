#!/usr/bin/env python3
"""
Comprehensive deployment readiness check for Employee Salary Prediction System.
"""

import os
import sys
import warnings
import importlib.util
from pathlib import Path

warnings.filterwarnings('ignore')

def check_file_structure():
    """Check if all required files and directories exist."""
    print("📁 Checking file structure...")
    
    required_files = [
        'webapp/app.py',
        'src/data_preprocessing.py',
        'src/feature_engineering.py', 
        'src/model_training.py',
        'src/utils.py',
        'requirements.txt',
        'README.md',
        'ensure_dataset.py',
        '.streamlit/config.toml',
        'Procfile',
        'setup.sh'
    ]
    
    required_dirs = [
        'webapp',
        'src', 
        'data',
        'data/raw',
        'models',
        'notebooks',
        '.streamlit'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_files or missing_dirs:
        print("❌ File structure check failed:")
        for file in missing_files:
            print(f"   Missing file: {file}")
        for dir in missing_dirs:
            print(f"   Missing directory: {dir}")
        return False
    else:
        print("✅ All required files and directories present")
        return True

def check_dependencies():
    """Check if all required dependencies can be imported."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'sklearn',
        'joblib',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Dependency check failed:")
        for package in missing_packages:
            print(f"   Missing package: {package}")
        return False
    else:
        print("✅ All required dependencies available")
        return True

def check_dataset():
    """Check dataset availability and quality."""
    print("\n📊 Checking dataset...")
    
    try:
        # Try to load dataset
        sys.path.append('webapp')
        from app import load_dataset
        
        df = load_dataset()
        
        if df is None:
            print("❌ Dataset loading failed")
            return False
        
        # Check dataset structure
        expected_columns = ['employee_id', 'age', 'gender', 'education_level', 
                           'years_experience', 'city', 'city_tier', 'department', 
                           'company_size', 'performance_rating', 'annual_salary']
        
        missing_cols = [col for col in expected_columns if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Dataset missing required columns: {missing_cols}")
            return False
        
        # Check data quality
        if df.empty:
            print("❌ Dataset is empty")
            return False
        
        if df['annual_salary'].isnull().all():
            print("❌ All salary values are missing")
            return False
        
        print(f"✅ Dataset loaded successfully: {df.shape}")
        print(f"   • Columns: {len(df.columns)}")
        print(f"   • Records: {len(df):,}")
        print(f"   • Salary range: ₹{df['annual_salary'].min():,.0f} - ₹{df['annual_salary'].max():,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset check failed: {e}")
        return False

def check_streamlit_config():
    """Check Streamlit configuration."""
    print("\n⚙️ Checking Streamlit configuration...")
    
    config_path = '.streamlit/config.toml'
    
    if not os.path.exists(config_path):
        print("❌ Streamlit config file missing")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Check for deprecated settings
        if 'client.caching' in config_content:
            print("❌ Deprecated 'client.caching' found in config")
            return False
        
        # Check for required sections
        if '[server]' not in config_content:
            print("⚠️ [server] section missing in config")
        
        print("✅ Streamlit configuration valid")
        return True
        
    except Exception as e:
        print(f"❌ Config check failed: {e}")
        return False

def check_app_pages():
    """Check if all app pages can be imported and basic functions work."""
    print("\n🌐 Checking app functionality...")
    
    try:
        sys.path.append('webapp')
        from app import (load_dataset, generate_dataset_if_missing, 
                        predict_salary_demo, format_currency)
        
        # Test dataset functions
        df = load_dataset()
        if df is None:
            print("❌ load_dataset() returned None")
            return False
        
        # Test prediction function
        test_features = {
            'years_experience': 5,
            'education_level': 'Bachelor',
            'department': 'Technology',
            'city_tier': 'Tier 1',
            'performance_rating': 4
        }
        
        prediction = predict_salary_demo(test_features)
        if not isinstance(prediction, (int, float)) or prediction <= 0:
            print("❌ predict_salary_demo() returned invalid result")
            return False
        
        # Test currency formatting
        formatted = format_currency(500000)
        if not formatted or '₹' not in formatted:
            print("❌ format_currency() not working properly")
            return False
        
        print("✅ All core app functions working")
        return True
        
    except Exception as e:
        print(f"❌ App functionality check failed: {e}")
        return False

def check_deployment_files():
    """Check deployment-specific files."""
    print("\n🚀 Checking deployment files...")
    
    checks = []
    
    # Check requirements.txt
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        required_packages = ['streamlit', 'pandas', 'numpy', 'plotly', 'scikit-learn']
        missing_req = [pkg for pkg in required_packages if pkg not in requirements]
        
        if missing_req:
            print(f"⚠️ requirements.txt missing: {missing_req}")
            checks.append(False)
        else:
            print("✅ requirements.txt looks good")
            checks.append(True)
    else:
        print("❌ requirements.txt missing")
        checks.append(False)
    
    # Check Procfile
    if os.path.exists('Procfile'):
        with open('Procfile', 'r') as f:
            procfile = f.read()
        if 'streamlit run' in procfile:
            print("✅ Procfile configured for Streamlit")
            checks.append(True)
        else:
            print("⚠️ Procfile doesn't contain streamlit run command")
            checks.append(False)
    else:
        print("⚠️ Procfile missing (optional for Streamlit Cloud)")
        checks.append(True)  # Not critical for Streamlit Cloud
    
    # Check setup.sh
    if os.path.exists('setup.sh'):
        print("✅ setup.sh present")
        checks.append(True)
    else:
        print("⚠️ setup.sh missing (optional)")
        checks.append(True)  # Not critical
    
    return all(checks)

def run_comprehensive_check():
    """Run all checks and provide deployment readiness report."""
    print("🔍 Employee Salary Prediction System - Deployment Readiness Check")
    print("=" * 70)
    
    checks = [
        check_file_structure(),
        check_dependencies(), 
        check_dataset(),
        check_streamlit_config(),
        check_app_pages(),
        check_deployment_files()
    ]
    
    print("\n" + "=" * 70)
    print("📋 DEPLOYMENT READINESS REPORT")
    print("=" * 70)
    
    if all(checks):
        print("🎉 ✅ ALL CHECKS PASSED!")
        print("🚀 Your app is ready for deployment!")
        print("\n📌 Deployment Instructions:")
        print("   1. Commit and push all changes to GitHub")
        print("   2. Connect repository to Streamlit Cloud")
        print("   3. Deploy using webapp/app.py as main file")
        print("   4. The app will auto-generate dataset on first run")
        return True
    else:
        print("❌ SOME CHECKS FAILED!")
        print("🔧 Please fix the issues above before deploying")
        failed_count = sum(1 for check in checks if not check)
        print(f"📊 Status: {len(checks) - failed_count}/{len(checks)} checks passed")
        return False

if __name__ == "__main__":
    run_comprehensive_check()
