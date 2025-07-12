# Employee Salary Prediction System

A comprehensive Machine Learning project for predicting employee salaries based on various factors like experience, education, location, and skills. This project demonstrates the complete data science workflow from data preprocessing to model deployment.

## 🎯 Project Overview

This project is designed as part of an AIML internship to showcase professional-level machine learning capabilities including:

- **Data Collection & Generation**: Creating realistic employee salary datasets
- **Data Cleaning**: Handling missing values, outliers, and inconsistencies
- **Exploratory Data Analysis**: Statistical insights and data visualization
- **Feature Engineering**: Creating meaningful features for better predictions
- **Model Training**: Multiple ML algorithms with hyperparameter tuning
- **Model Evaluation**: Comprehensive performance metrics and validation
- **Web Deployment**: Interactive Streamlit application for real-time predictions

## 🏗️ Project Structure

```
AIML Project/
├── data/                          # Dataset files
│   ├── raw/                      # Original datasets
│   ├── processed/                # Cleaned and processed data
│   └── external/                 # External data sources
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_data_generation.ipynb     # Dataset creation
│   ├── 02_data_cleaning.ipynb       # Data preprocessing
│   ├── 03_exploratory_analysis.ipynb # EDA and visualization
│   ├── 04_feature_engineering.ipynb # Feature creation
│   └── 05_model_training.ipynb      # ML model development
├── src/                          # Source code modules
│   ├── data_preprocessing.py        # Data cleaning utilities
│   ├── feature_engineering.py      # Feature creation functions
│   ├── model_training.py           # ML training pipeline
│   ├── model_evaluation.py         # Model assessment tools
│   └── utils.py                    # Helper functions
├── models/                       # Trained model files
│   ├── model_artifacts/            # Saved models and encoders
│   └── model_comparison.json       # Performance comparison
├── webapp/                       # Streamlit web application
│   ├── app.py                      # Main application
│   ├── components/                 # UI components
│   └── static/                     # CSS and assets
└── requirements.txt              # Python dependencies
```

## 🚀 Features

### Data Processing
- **Automated Data Generation**: Creates realistic employee datasets
- **Smart Data Cleaning**: Handles missing values using advanced imputation
- **Outlier Detection**: Statistical and ML-based outlier identification
- **Data Validation**: Ensures data quality and consistency

### Machine Learning
- **Multiple Algorithms**: Linear Regression, Random Forest, XGBoost, LightGBM
- **Hyperparameter Tuning**: Automated optimization using Optuna
- **Cross-Validation**: Robust model validation strategies
- **Feature Importance**: SHAP values for model interpretability

### Visualization & Analysis
- **Interactive Dashboards**: Professional charts and graphs
- **Statistical Insights**: Comprehensive EDA with statistical tests
- **Model Performance**: Detailed evaluation metrics and plots
- **Salary Distribution**: Geographic and demographic analysis

### Web Application
- **Professional UI**: Modern, responsive design
- **Real-time Predictions**: Instant salary predictions
- **Interactive Forms**: User-friendly input interface
- **Results Visualization**: Charts and explanations

## 🛠️ Installation & Setup

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Jupyter notebooks** for analysis:
   ```bash
   jupyter notebook
   ```
4. **Launch web application**:
   ```bash
   streamlit run webapp/app.py
   ```

## 📊 Dataset Features

The employee salary dataset includes:

- **Personal Information**: Age, Gender, Education Level
- **Professional Details**: Years of Experience, Job Title, Department
- **Location Data**: City, State, Country
- **Skills & Certifications**: Technical skills, Certifications
- **Company Information**: Company Size, Industry Type
- **Performance Metrics**: Performance Rating, Bonus History
- **Target Variable**: Annual Salary (in INR)

## 🤖 Machine Learning Pipeline

1. **Data Preprocessing**:
   - Missing value imputation
   - Outlier detection and treatment
   - Data type conversions
   - Feature scaling and normalization

2. **Feature Engineering**:
   - Categorical encoding (Label, One-Hot, Target)
   - Feature interactions and polynomial features
   - Dimensionality reduction (PCA, Feature Selection)
   - Domain-specific feature creation

3. **Model Training**:
   - Train-test-validation split
   - Multiple algorithm comparison
   - Hyperparameter optimization
   - Cross-validation and model selection

4. **Model Evaluation**:
   - Regression metrics (RMSE, MAE, R²)
   - Residual analysis
   - Feature importance analysis
   - Model interpretability with SHAP

## 🎨 Web Application Features

- **Prediction Interface**: Input employee details for salary prediction
- **Data Visualization**: Interactive charts showing salary distributions
- **Model Insights**: Feature importance and prediction explanations
- **Performance Dashboard**: Model accuracy and evaluation metrics
- **Data Explorer**: Browse and filter the training dataset

## 📈 Model Performance

The system achieves:
- **R² Score**: > 0.85 on test data
- **RMSE**: < 15% of mean salary
- **MAE**: Competitive performance across all models
- **Cross-Validation**: Consistent performance across folds

## 🔧 Technologies Used

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **XGBoost & LightGBM**: Advanced gradient boosting
- **Matplotlib & Seaborn**: Static visualizations
- **Plotly**: Interactive charts
- **Streamlit**: Web application framework
- **Jupyter**: Interactive development environment
- **SHAP**: Model interpretability

## 📝 Usage Examples

### Predicting Salary
```python
from src.model_training import SalaryPredictor

# Load trained model
predictor = SalaryPredictor()
predictor.load_model('models/best_model.pkl')

# Make prediction
salary = predictor.predict({
    'years_experience': 5,
    'education_level': 'Bachelor',
    'job_title': 'Software Engineer',
    'location': 'Bangalore',
    'skills_score': 85
})

print(f"Predicted Salary: ₹{salary:,.2f}")
```

### Running Analysis
```python
# Data cleaning and EDA
jupyter notebook notebooks/02_data_cleaning.ipynb
jupyter notebook notebooks/03_exploratory_analysis.ipynb

# Model training and evaluation
python src/model_training.py
python src/model_evaluation.py
```

## 🎯 Business Value

This system provides:
- **HR Analytics**: Data-driven salary benchmarking
- **Recruitment Support**: Competitive salary recommendations
- **Budget Planning**: Accurate compensation forecasting
- **Market Analysis**: Industry salary trends and insights
- **Performance Evaluation**: Fair compensation assessment

## 🔄 Future Enhancements

- **Real-time Data Integration**: Live market data feeds
- **Advanced Models**: Deep learning and ensemble methods
- **Geographic Expansion**: Multi-country salary prediction
- **API Development**: REST API for external integrations
- **Mobile Application**: Cross-platform mobile app
- **A/B Testing**: Model performance optimization

## 👨‍💻 Developer

**AIML Internship Project**
- Comprehensive machine learning implementation
- Professional-grade code quality and documentation
- Industry-standard best practices and methodologies

## 📄 License

This project is created for educational and internship purposes, demonstrating advanced machine learning capabilities in a real-world scenario.

---

**Note**: This is a demonstration project showcasing machine learning and data science skills. The salary predictions are based on synthetic data and should be used for educational purposes only.
