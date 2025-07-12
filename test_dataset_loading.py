#!/usr/bin/env python3
"""
Test script to verify the dataset loading functionality.
"""

import sys
import os
import warnings

# Suppress all warnings including Streamlit warnings
warnings.filterwarnings('ignore')
os.environ['STREAMLIT_CONFIG_LOGGING_LEVEL'] = 'error'

sys.path.append('webapp')

def validate_environment():
    """Validate that the environment is set up correctly."""
    validation_issues = []
    
    # Check if webapp directory exists
    if not os.path.exists('webapp'):
        validation_issues.append("webapp directory not found")
    
    # Check if app.py exists
    if not os.path.exists('webapp/app.py'):
        validation_issues.append("webapp/app.py file not found")
    
    # Check if data directory exists
    if not os.path.exists('data'):
        validation_issues.append("data directory not found")
    
    # Check if raw data directory exists
    if not os.path.exists('data/raw'):
        validation_issues.append("data/raw directory not found")
    
    return validation_issues

def test_dataset_loading():
    """Test dataset loading functionality with proper error handling."""
    try:
        # Suppress Streamlit output during import
        import io
        from contextlib import redirect_stderr
        
        f = io.StringIO()
        with redirect_stderr(f):
            from app import load_dataset  # type: ignore
            
        print("🔄 Testing dataset loading...")
        
        df = load_dataset()
        
        if df is not None:
            print("✅ Dataset loaded successfully!")
            print(f"📊 Shape: {df.shape}")
            print(f"📝 Columns: {list(df.columns)}")
            print(f"💰 Salary range: ₹{df['annual_salary'].min():,.0f} - ₹{df['annual_salary'].max():,.0f}")
            
            # Additional validation
            required_columns = ['employee_id', 'age', 'gender', 'education_level', 
                              'years_experience', 'city', 'city_tier', 'department', 
                              'company_size', 'performance_rating', 'annual_salary']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"⚠️  Missing expected columns: {missing_columns}")
            else:
                print("✅ All expected columns present")
                
            # Check data quality
            print("📊 Data Quality:")
            print(f"   • Missing values: {df.isnull().sum().sum()}")
            print(f"   • Duplicate rows: {df.duplicated().sum()}")
            print(f"   • Unique employees: {df['employee_id'].nunique()}")
            
            return True
        else:
            print("❌ Failed to load dataset")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure the webapp directory is accessible and contains app.py")
        return False
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("💡 Dataset file might be missing. Run ensure_dataset.py first.")
        return False
    except (ValueError, KeyError) as e:
        print(f"❌ Data validation error: {e}")
        print("💡 Dataset structure might be incorrect.")
        return False
    except (RuntimeError, OSError) as e:
        print(f"❌ System error: {e}")
        print("💡 Check system permissions and available resources.")
        return False

if __name__ == "__main__":
    print("🔍 Employee Salary Prediction System - Dataset Loading Test")
    print("=" * 60)
    
    # First validate environment
    print("1️⃣ Validating environment...")
    env_issues = validate_environment()
    
    if env_issues:
        print("❌ Environment validation failed:")
        for issue in env_issues:
            print(f"   • {issue}")
        print("\n💡 Please fix these issues before running the test.")
        print("🎯 Test result: FAILED (Environment)")
    else:
        print("✅ Environment validation passed")
        
        # Run dataset loading test
        print("\n2️⃣ Testing dataset loading...")
        success = test_dataset_loading()
        
        print(f"\n🎯 Test result: {'PASSED' if success else 'FAILED'}")
        
        if success:
            print("🎉 All tests completed successfully!")
            print("🚀 Your Employee Salary Prediction System is ready to use!")
        else:
            print("❌ Some tests failed. Please check the errors above.")
