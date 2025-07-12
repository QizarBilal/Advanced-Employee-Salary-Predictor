"""
Utility functions for the Employee Salary Prediction system.

This module contains helper functions used across the project for
common operations like data validation, file operations, and configuration.
"""

import json
import pickle
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/salary_prediction.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ProjectPaths:
    """Centralized path management for the project."""
    
    def __init__(self, root_dir: Optional[str] = None):
        """Initialize project paths."""
        if root_dir is None:
            self.root = Path(__file__).parent.parent
        else:
            self.root = Path(root_dir)
        
        self.data = self.root / "data"
        self.data_raw = self.data / "raw"
        self.data_processed = self.data / "processed"
        self.data_external = self.data / "external"
        
        self.models = self.root / "models"
        self.model_artifacts = self.models / "model_artifacts"
        
        self.notebooks = self.root / "notebooks"
        self.src = self.root / "src"
        self.webapp = self.root / "webapp"
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create project directories if they don't exist."""
        directories = [
            self.data, self.data_raw, self.data_processed, self.data_external,
            self.models, self.model_artifacts, self.notebooks, self.src, self.webapp
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Global paths instance
PATHS = ProjectPaths()

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = PATHS.root / config_path
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Default configuration
        default_config = {
            "random_seed": 42,
            "test_size": 0.2,
            "val_size": 0.15,
            "cv_folds": 5,
            "target_currency": "INR",
            "model_params": {
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42
                },
                "xgboost": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": 42
                }
            }
        }
        save_config(default_config, config_path)
        return default_config

def save_config(config: Dict[str, Any], config_path: str = "config.json") -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_file = PATHS.root / config_path
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def save_model(model: Any, filepath: Union[str, Path], metadata: Optional[Dict] = None) -> None:
    """
    Save trained model with metadata.
    
    Args:
        model: Trained model object
        filepath: Path to save the model
        metadata: Additional metadata about the model
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata if provided
    if metadata:
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
    
    logger.info(f"Model saved to {filepath}")

def load_model(filepath: Union[str, Path]) -> Any:
    """
    Load trained model from file.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model object
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Model loaded from {filepath}")
    return model

def load_model_metadata(filepath: Union[str, Path]) -> Optional[Dict]:
    """
    Load model metadata if available.
    
    Args:
        filepath: Path to the model file
        
    Returns:
        Model metadata dictionary or None
    """
    metadata_path = Path(filepath).with_suffix('.json')
    
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate DataFrame structure and required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if valid, False otherwise
    """
    if df.empty:
        logger.error("DataFrame is empty")
        return False
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    return True

def get_numerical_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of numerical columns from DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of numerical column names
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of categorical columns from DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of categorical column names
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def calculate_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate memory usage statistics for DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with memory usage statistics
    """
    memory_usage = df.memory_usage(deep=True)
    
    return {
        'total_mb': memory_usage.sum() / 1024**2,
        'per_column_mb': (memory_usage / 1024**2).to_dict(),
        'shape': df.shape,
        'dtypes': df.dtypes.value_counts().to_dict()
    }

def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    logger.info(f"Random seed set to {seed}")

def format_currency(amount: float, currency: str = "INR") -> str:
    """
    Format currency amount with proper formatting.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency == "INR":
        if amount >= 10000000:  # 1 crore
            return f"₹{amount/10000000:.2f} Cr"
        elif amount >= 100000:  # 1 lakh
            return f"₹{amount/100000:.2f} L"
        else:
            return f"₹{amount:,.0f}"
    else:
        return f"{currency} {amount:,.2f}"

def create_log_directory() -> None:
    """Create logs directory if it doesn't exist."""
    log_dir = PATHS.root / "logs"
    log_dir.mkdir(exist_ok=True)

# Initialize logging directory
create_log_directory()

class DataValidator:
    """Data validation utilities for the salary prediction system."""
    
    @staticmethod
    def validate_salary_range(salary: float, min_salary: float = 100000, max_salary: float = 50000000) -> bool:
        """
        Validate if salary is within reasonable range.
        
        Args:
            salary: Salary amount to validate
            min_salary: Minimum acceptable salary
            max_salary: Maximum acceptable salary
            
        Returns:
            True if valid, False otherwise
        """
        return min_salary <= salary <= max_salary
    
    @staticmethod
    def validate_experience_years(years: float) -> bool:
        """
        Validate years of experience.
        
        Args:
            years: Years of experience
            
        Returns:
            True if valid, False otherwise
        """
        return 0 <= years <= 50
    
    @staticmethod
    def validate_age(age: int) -> bool:
        """
        Validate age range.
        
        Args:
            age: Age to validate
            
        Returns:
            True if valid, False otherwise
        """
        return 18 <= age <= 70

class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step/iteration number
        """
        if name not in self.metrics:
            self.metrics[name] = []
        
        metric_entry = {'value': value}
        if step is not None:
            metric_entry['step'] = step
        
        self.metrics[name].append(metric_entry)
        logger.info(f"Metric {name}: {value}")
    
    def get_metrics(self) -> Dict[str, List[Dict]]:
        """Get all logged metrics."""
        return self.metrics
    
    def save_metrics(self, filepath: Union[str, Path]) -> None:
        """Save metrics to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=4)
        
        logger.info(f"Metrics saved to {filepath}")

# Global performance monitor
performance_monitor = PerformanceMonitor()
