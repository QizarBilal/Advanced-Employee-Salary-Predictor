"""
Data preprocessing module for Employee Salary Prediction System.

This module handles data cleaning, missing value imputation, outlier detection,
and data transformation for the salary prediction pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats

from src.utils import DataValidator, performance_monitor, logger

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for employee salary data.
    
    This class handles all aspects of data preprocessing including:
    - Missing value imputation
    - Outlier detection and treatment
    - Data type conversions
    - Feature scaling and encoding
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary for preprocessing parameters
        """
        self.config = config or self._get_default_config()
        self.validator = DataValidator()
        self.fitted_transformers = {}
        self.preprocessing_stats = {}
        self.preprocessor = None
        
    def _get_default_config(self) -> Dict:
        """Get default preprocessing configuration."""
        return {
            'missing_value_strategy': 'knn',  # 'simple', 'knn', 'iterative'
            'outlier_method': 'iqr',  # 'iqr', 'zscore', 'isolation_forest'
            'outlier_threshold': 3.0,
            'scaling_method': 'standard',  # 'standard', 'minmax', 'robust'
            'encoding_method': 'onehot',  # 'label', 'onehot', 'target'
            'handle_outliers': True,
            'drop_outliers': False,  # If True, drops outliers; if False, caps them
            'random_state': 42
        }
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning process")
        df_cleaned = df.copy()
        
        # Remove completely empty rows and columns
        df_cleaned = df_cleaned.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df_cleaned.columns = self._clean_column_names(df_cleaned.columns)
        
        # Handle data type conversions
        df_cleaned = self._convert_data_types(df_cleaned)
        
        # Remove duplicate rows
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        duplicates_removed = initial_rows - len(df_cleaned)
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        # Validate data integrity
        df_cleaned = self._validate_data_integrity(df_cleaned)
        
        logger.info(f"Data cleaning completed. Shape: {df_cleaned.shape}")
        return df_cleaned
    
    def _clean_column_names(self, columns: pd.Index) -> List[str]:
        """Clean and standardize column names."""
        cleaned_names = []
        for col in columns:
            # Convert to lowercase and replace spaces with underscores
            cleaned = str(col).lower().replace(' ', '_').replace('-', '_')
            # Remove special characters except underscores
            cleaned = ''.join(c for c in cleaned if c.isalnum() or c == '_')
            cleaned_names.append(cleaned)
        return cleaned_names
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types."""
        df_converted = df.copy()
        
        # Define expected data types based on common patterns
        type_mappings = {
            'salary': 'float64',
            'annual_salary': 'float64',
            'years_experience': 'float64',
            'age': 'int64',
            'performance_rating': 'float64',
            'bonus': 'float64'
        }
        
        for col, dtype in type_mappings.items():
            if col in df_converted.columns:
                try:
                    if dtype == 'float64':
                        df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                    elif dtype == 'int64':
                        df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce').astype('Int64')
                except (ValueError, TypeError) as e:
                    logger.warning("Could not convert %s to %s: %s", col, dtype, e)
        
        return df_converted
    
    def _validate_data_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data based on business rules."""
        df_validated = df.copy()
        
        # Validate salary ranges
        if 'salary' in df_validated.columns:
            mask = df_validated['salary'].apply(
                lambda x: self.validator.validate_salary_range(x) if pd.notna(x) else True
            )
            invalid_salaries = (~mask).sum()
            if invalid_salaries > 0:
                logger.warning(f"Found {invalid_salaries} records with invalid salary ranges")
                # Set invalid salaries to NaN for imputation
                df_validated.loc[~mask, 'salary'] = np.nan
        
        # Validate experience years
        if 'years_experience' in df_validated.columns:
            mask = df_validated['years_experience'].apply(
                lambda x: self.validator.validate_experience_years(x) if pd.notna(x) else True
            )
            invalid_exp = (~mask).sum()
            if invalid_exp > 0:
                logger.warning(f"Found {invalid_exp} records with invalid experience years")
                df_validated.loc[~mask, 'years_experience'] = np.nan
        
        # Validate age
        if 'age' in df_validated.columns:
            mask = df_validated['age'].apply(
                lambda x: self.validator.validate_age(x) if pd.notna(x) else True
            )
            invalid_age = (~mask).sum()
            if invalid_age > 0:
                logger.warning(f"Found {invalid_age} records with invalid ages")
                df_validated.loc[~mask, 'age'] = np.nan
        
        return df_validated
    
    def handle_missing_values(self, df: pd.DataFrame, target_column: str = 'salary') -> pd.DataFrame:
        """
        Handle missing values using various imputation strategies.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            DataFrame with imputed missing values
        """
        logger.info("Handling missing values")
        df_imputed = df.copy()
        
        # Calculate missing value statistics
        missing_stats = self._calculate_missing_stats(df_imputed)
        self.preprocessing_stats['missing_values'] = missing_stats
        
        # Separate features and target
        features = df_imputed.drop(columns=[target_column] if target_column in df_imputed.columns else [])
        target = df_imputed[target_column] if target_column in df_imputed.columns else None
        
        # Get numerical and categorical columns
        numerical_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = features.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle numerical missing values
        if numerical_cols:
            features[numerical_cols] = self._impute_numerical(features[numerical_cols])
        
        # Handle categorical missing values
        if categorical_cols:
            features[categorical_cols] = self._impute_categorical(features[categorical_cols])
        
        # Combine features and target
        if target is not None:
            df_imputed = pd.concat([features, target], axis=1)
        else:
            df_imputed = features
        
        logger.info("Missing value imputation completed")
        return df_imputed
    
    def _calculate_missing_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate missing value statistics."""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        return {
            'total_missing': missing_counts.sum(),
            'missing_by_column': missing_counts.to_dict(),
            'missing_percentage_by_column': missing_percentages.to_dict(),
            'columns_with_missing': missing_counts[missing_counts > 0].index.tolist()
        }
    
    def _impute_numerical(self, df_num: pd.DataFrame) -> pd.DataFrame:
        """Impute numerical missing values."""
        strategy = self.config['missing_value_strategy']
        
        if strategy == 'simple':
            imputer = SimpleImputer(strategy='median')
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:  # Default to median
            imputer = SimpleImputer(strategy='median')
        
        imputed_values = imputer.fit_transform(df_num)
        self.fitted_transformers['numerical_imputer'] = imputer
        
        return pd.DataFrame(imputed_values, columns=df_num.columns, index=df_num.index)
    
    def _impute_categorical(self, df_cat: pd.DataFrame) -> pd.DataFrame:
        """Impute categorical missing values."""
        df_imputed = df_cat.copy()
        
        for col in df_cat.columns:
            # Use mode for categorical imputation
            mode_value = df_cat[col].mode()
            if len(mode_value) > 0:
                df_imputed[col] = df_imputed[col].fillna(mode_value[0])
            else:
                df_imputed[col] = df_imputed[col].fillna('Unknown')
        
        return df_imputed
    
    def detect_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Detect outliers using specified method.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers (if None, uses all numerical columns)
            
        Returns:
            Dictionary mapping column names to boolean arrays indicating outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_indices = {}
        method = self.config['outlier_method']
        
        for col in columns:
            if col in df.columns:
                if method == 'iqr':
                    outliers = self._detect_outliers_iqr(df[col])
                elif method == 'zscore':
                    outliers = self._detect_outliers_zscore(df[col])
                elif method == 'isolation_forest':
                    outliers = self._detect_outliers_isolation_forest(df[[col]])
                else:
                    outliers = np.zeros(len(df), dtype=bool)
                
                outlier_indices[col] = outliers
                logger.info(f"Detected {outliers.sum()} outliers in column '{col}'")
        
        return outlier_indices
    
    def _detect_outliers_iqr(self, series: pd.Series) -> np.ndarray:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(self, series: pd.Series) -> np.ndarray:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(series.dropna()))
        threshold = self.config['outlier_threshold']
        
        outliers = np.zeros(len(series), dtype=bool)
        outliers[series.notna()] = z_scores > threshold
        
        return outliers
    
    def _detect_outliers_isolation_forest(self, df: pd.DataFrame) -> np.ndarray:
        """Detect outliers using Isolation Forest."""
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(
            contamination=0.1,
            random_state=self.config['random_state']
        )
        outliers = iso_forest.fit_predict(df.dropna()) == -1
        
        result = np.zeros(len(df), dtype=bool)
        result[df.notna().all(axis=1)] = outliers
        
        return result
    
    def handle_outliers(self, df: pd.DataFrame, outlier_indices: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Handle detected outliers by either removing or capping them.
        
        Args:
            df: Input DataFrame
            outlier_indices: Dictionary of outlier indices per column
            
        Returns:
            DataFrame with outliers handled
        """
        if not self.config['handle_outliers']:
            return df
        
        df_handled = df.copy()
        
        if self.config['drop_outliers']:
            # Drop rows with outliers
            all_outliers = np.zeros(len(df), dtype=bool)
            for col, outliers in outlier_indices.items():
                all_outliers |= outliers
            
            df_handled = df_handled[~all_outliers]
            logger.info(f"Dropped {all_outliers.sum()} outlier rows")
        else:
            # Cap outliers at reasonable bounds
            for col, outliers in outlier_indices.items():
                if outliers.any():
                    Q1 = df[col].quantile(0.05)
                    Q99 = df[col].quantile(0.95)
                    
                    df_handled.loc[df_handled[col] < Q1, col] = Q1
                    df_handled.loc[df_handled[col] > Q99, col] = Q99
                    
                    logger.info(f"Capped {outliers.sum()} outliers in column '{col}'")
        
        return df_handled
    
    def create_preprocessing_pipeline(self, df: pd.DataFrame, target_column: str) -> ColumnTransformer:
        """
        Create a comprehensive preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Fitted ColumnTransformer pipeline
        """
        features = df.drop(columns=[target_column])
        
        # Identify column types
        numerical_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = features.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create transformers
        transformers = []
        
        # Numerical transformer
        if numerical_cols:
            if self.config['scaling_method'] == 'standard':
                scaler = StandardScaler()
            elif self.config['scaling_method'] == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            
            numerical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', scaler)
            ])
            
            transformers.append(('num', numerical_transformer, numerical_cols))
        
        # Categorical transformer
        if categorical_cols:
            if self.config['encoding_method'] == 'onehot':
                encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
            else:
                encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
            
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('encoder', encoder)
            ])
            
            transformers.append(('cat', categorical_transformer, categorical_cols))
        
        # Create and fit the preprocessor
        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
        
        return preprocessor
    
    def fit_transform(self, df: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, pd.Series]:
        """
        Fit the preprocessing pipeline and transform the data.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (transformed features, target)
        """
        logger.info("Starting preprocessing pipeline")
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Handle missing values
        df_imputed = self.handle_missing_values(df_clean, target_column)
        
        # Detect and handle outliers
        if self.config['handle_outliers']:
            outlier_indices = self.detect_outliers(df_imputed)
            df_processed = self.handle_outliers(df_imputed, outlier_indices)
        else:
            df_processed = df_imputed
        
        # Create and fit preprocessing pipeline
        self.preprocessor = self.create_preprocessing_pipeline(df_processed, target_column)
        
        # Separate features and target
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        
        # Transform features
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Store preprocessing statistics
        self.preprocessing_stats.update({
            'original_shape': df.shape,
            'final_shape': (X_transformed.shape[0], X_transformed.shape[1]),
            'records_removed': df.shape[0] - X_transformed.shape[0],
            'features_created': X_transformed.shape[1]
        })
        
        logger.info(f"Preprocessing completed. Final shape: {X_transformed.shape}")
        performance_monitor.log_metric('preprocessing_features', X_transformed.shape[1])
        performance_monitor.log_metric('preprocessing_samples', X_transformed.shape[0])
        
        return X_transformed, y
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using the fitted preprocessor.
        
        Args:
            df: New DataFrame to transform
            
        Returns:
            Transformed features
        """
        if not hasattr(self, 'preprocessor'):
            raise ValueError("Preprocessor has not been fitted. Call fit_transform first.")
        
        # Apply basic cleaning (without target column handling)
        df_clean = df.copy()
        df_clean.columns = self._clean_column_names(df_clean.columns)
        df_clean = self._convert_data_types(df_clean)
        
        # Transform using fitted preprocessor
        X_transformed = self.preprocessor.transform(df_clean)
        
        return X_transformed
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names after preprocessing.
        
        Returns:
            List of feature names
        """
        if not hasattr(self, 'preprocessor'):
            raise ValueError("Preprocessor has not been fitted. Call fit_transform first.")
        
        try:
            return self.preprocessor.get_feature_names_out().tolist()
        except AttributeError:
            # Fallback for older sklearn versions
            return [f'feature_{i}' for i in range(self.preprocessor.transform([[0]*len(self.preprocessor.transformers)]).shape[1])]
    
    def get_preprocessing_report(self) -> Dict:
        """
        Generate a comprehensive preprocessing report.
        
        Returns:
            Dictionary containing preprocessing statistics and information
        """
        return {
            'config': self.config,
            'statistics': self.preprocessing_stats,
            'transformers_fitted': list(self.fitted_transformers.keys()),
            'preprocessing_steps': [
                'Data cleaning',
                'Missing value imputation',
                'Outlier detection and handling',
                'Feature scaling and encoding'
            ]
        }
