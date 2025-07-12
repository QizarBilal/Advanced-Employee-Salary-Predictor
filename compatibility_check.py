#!/usr/bin/env python3
"""
Quick deployment compatibility test for Streamlit Cloud
"""

import sys
print(f"Python version: {sys.version}")

try:
    import streamlit as st
    print(f"✅ Streamlit: {st.__version__}")
except ImportError as e:
    print(f"❌ Streamlit: {e}")

try:
    import pandas as pd
    print(f"✅ Pandas: {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas: {e}")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy: {e}")

try:
    import sklearn
    print(f"✅ Scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ Scikit-learn: {e}")

try:
    import plotly
    print(f"✅ Plotly: {plotly.__version__}")
except ImportError as e:
    print(f"❌ Plotly: {e}")

print("\n🎉 Basic compatibility check completed!")
print("📋 Requirements updated for Streamlit Cloud Python 3.13 compatibility")
