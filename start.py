"""
Quick start script for the Employee Salary Prediction System.

This script provides easy commands to run different parts of the system.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def get_python_executable():
    """Get the correct Python executable path."""
    return "C:/Users/We/AppData/Local/Programs/Python/Python310/python.exe"

def run_streamlit_app():
    """Launch the Streamlit web application."""
    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "streamlit", "run", "webapp/app.py"]
    
    print("🚀 Starting Streamlit Web Application...")
    print("📱 The app will open in your default browser")
    print("🌐 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        print("\n✅ Streamlit app stopped successfully!")

def run_jupyter_notebook():
    """Launch Jupyter Notebook."""
    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "jupyter", "notebook"]
    
    print("📔 Starting Jupyter Notebook...")
    print("📱 Jupyter will open in your default browser")
    print("📂 Navigate to notebooks/ folder to see analysis")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        print("\n✅ Jupyter Notebook stopped successfully!")

def generate_dataset():
    """Generate the employee salary dataset."""
    python_exe = get_python_executable()
    notebook_path = "notebooks/01_data_generation.ipynb"
    
    print("📊 Generating Employee Salary Dataset...")
    print(f"📝 Executing: {notebook_path}")
    print("-" * 50)
    
    cmd = [python_exe, "-m", "jupyter", "nbconvert", "--execute", "--to", "notebook", "--inplace", notebook_path]
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Dataset generated successfully!")
            print("📁 Check data/raw/ folder for the generated dataset")
        else:
            print("❌ Error generating dataset:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ Error: {e}")

def install_dependencies():
    """Install required Python packages."""
    python_exe = get_python_executable()
    cmd = [python_exe, "-m", "pip", "install", "-r", "requirements.txt"]
    
    print("📦 Installing Python Dependencies...")
    print("⏳ This might take a few minutes...")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        if result.returncode == 0:
            print("✅ Dependencies installed successfully!")
        else:
            print("❌ Error installing dependencies")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_modules():
    """Test if all modules are working correctly."""
    python_exe = get_python_executable()
    
    print("🧪 Testing System Modules...")
    print("-" * 50)
    
    tests = [
        ("Data Preprocessing", "from src.data_preprocessing import DataPreprocessor; print('✅ Data preprocessing ready')"),
        ("Feature Engineering", "from src.feature_engineering import FeatureEngineer; print('✅ Feature engineering ready')"),
        ("Utilities", "from src.utils import PATHS, performance_monitor; print('✅ Utilities ready')"),
        ("Core Libraries", "import pandas, numpy, sklearn, matplotlib, seaborn, plotly, streamlit; print('✅ All libraries imported')")
    ]
    
    for test_name, test_code in tests:
        try:
            result = subprocess.run([python_exe, "-c", test_code], 
                                  cwd=Path(__file__).parent, 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {test_name}: {result.stdout.strip()}")
            else:
                print(f"❌ {test_name}: Error - {result.stderr.strip()}")
        except Exception as e:
            print(f"❌ {test_name}: Exception - {e}")

def show_project_info():
    """Display project information and structure."""
    print("💼 Employee Salary Prediction System")
    print("=" * 50)
    print()
    print("📋 Project Structure:")
    print("├── 📁 data/                 # Dataset files")
    print("│   ├── 📁 raw/              # Original datasets")
    print("│   ├── 📁 processed/        # Cleaned data")
    print("│   └── 📁 external/         # External data sources")
    print("├── 📁 notebooks/            # Jupyter analysis notebooks")
    print("├── 📁 src/                  # Source code modules")
    print("├── 📁 models/               # Trained ML models")
    print("├── 📁 webapp/               # Streamlit web application")
    print("├── 📄 requirements.txt     # Python dependencies")
    print("├── 📄 config.json          # System configuration")
    print("└── 📄 README.md             # Project documentation")
    print()
    print("🚀 Available Commands:")
    print("1. 🌐 Launch Web App       - streamlit")
    print("2. 📔 Open Jupyter         - jupyter")
    print("3. 📊 Generate Dataset     - generate")
    print("4. 📦 Install Dependencies - install")
    print("5. 🧪 Test Modules         - test")
    print("6. ℹ️  Show Info            - info")
    print("7. ❌ Exit                  - exit")

def main():
    """Main interactive menu."""
    while True:
        print("\n" + "=" * 60)
        print("💼 Employee Salary Prediction System - Quick Start")
        print("=" * 60)
        print()
        print("Choose an option:")
        print("1. 🌐 Launch Streamlit Web Application")
        print("2. 📔 Start Jupyter Notebook")
        print("3. 📊 Generate Dataset")
        print("4. 📦 Install Dependencies")
        print("5. 🧪 Test System Modules")
        print("6. ℹ️  Show Project Information")
        print("7. ❌ Exit")
        print()
        
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            run_streamlit_app()
        elif choice == '2':
            run_jupyter_notebook()
        elif choice == '3':
            generate_dataset()
        elif choice == '4':
            install_dependencies()
        elif choice == '5':
            test_modules()
        elif choice == '6':
            show_project_info()
        elif choice == '7':
            print("\n👋 Thank you for using the Employee Salary Prediction System!")
            print("🚀 Happy Data Science!")
            break
        else:
            print("\n❌ Invalid choice. Please enter a number between 1-7.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Check if this is the first run
    if not os.path.exists("data"):
        print("🎉 Welcome to the Employee Salary Prediction System!")
        print("📁 Setting up project directories...")
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("data/external", exist_ok=True)
        os.makedirs("models/model_artifacts", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        print("✅ Project setup complete!")
    
    main()
