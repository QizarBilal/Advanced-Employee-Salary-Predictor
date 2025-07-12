"""
Model Training module for Employee Salary Prediction System.

This module provides comprehensive machine learning model training capabilities including:
- Multiple algorithm support (Random Forest, XGBoost, Linear Regression, etc.)
- Hyperparameter tuning with Optuna
- Cross-validation and model evaluation
- Model persistence and artifact management
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import optuna
import json
from datetime import datetime
import warnings

from src.utils import save_model, performance_monitor, logger, PATHS
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer

warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Comprehensive model training pipeline for salary prediction.
    
    This class handles:
    - Multiple ML algorithm training and comparison
    - Hyperparameter optimization
    - Model evaluation and validation
    - Model persistence and metadata management
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the model trainer.
        
        Args:
            config: Configuration dictionary for training parameters
        """
        self.config = config or self._get_default_config()
        self.models = {}
        self.trained_models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = None
        self.feature_engineer = None
        
    def _get_default_config(self) -> Dict:
        """Get default training configuration."""
        return {
            'test_size': 0.2,
            'val_size': 0.15,
            'cv_folds': 5,
            'random_state': 42,
            'scoring_metric': 'r2',
            'algorithms': ['random_forest', 'xgboost', 'linear_regression'],
            'hyperparameter_tuning': True,
            'n_trials': 50,
            'timeout': 1800,  # 30 minutes
            'n_jobs': -1,
            'save_models': True,
            'model_params': {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                },
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1
                },
                'lightgbm': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'num_leaves': 31,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1
                },
                'linear_regression': {
                    'fit_intercept': True
                },
                'ridge_regression': {
                    'alpha': 1.0,
                    'random_state': 42
                },
                'lasso_regression': {
                    'alpha': 1.0,
                    'random_state': 42
                }
            }
        }
    
    def initialize_models(self) -> None:
        """Initialize ML models with default parameters."""
        self.models = {}
        
        if 'random_forest' in self.config['algorithms']:
            self.models['random_forest'] = RandomForestRegressor(
                **self.config['model_params']['random_forest']
            )
        
        if 'xgboost' in self.config['algorithms']:
            self.models['xgboost'] = xgb.XGBRegressor(
                **self.config['model_params']['xgboost']
            )
        
        if 'lightgbm' in self.config['algorithms']:
            self.models['lightgbm'] = lgb.LGBMRegressor(
                **self.config['model_params']['lightgbm']
            )
        
        if 'linear_regression' in self.config['algorithms']:
            self.models['linear_regression'] = LinearRegression(
                **self.config['model_params']['linear_regression']
            )
        
        if 'ridge_regression' in self.config['algorithms']:
            self.models['ridge_regression'] = Ridge(
                **self.config['model_params']['ridge_regression']
            )
        
        if 'lasso_regression' in self.config['algorithms']:
            self.models['lasso_regression'] = Lasso(
                **self.config['model_params']['lasso_regression']
            )
        
        logger.info("Initialized %d models: %s", len(self.models), list(self.models.keys()))
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=None  # Can't stratify continuous target
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = self.config['val_size'] / (1 - self.config['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config['random_state']
        )
        
        logger.info("Data split - Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance using multiple metrics.
        
        Args:
            model: Trained model
            X: Features
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            y_pred = model.predict(X)
            
            metrics = {
                'r2_score': r2_score(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred),
                'mape': np.mean(np.abs((y - y_pred) / y)) * 100
            }
            
            return metrics
        except (ValueError, RuntimeError, AttributeError) as e:
            logger.error("Error evaluating model: %s", e)
            return {'r2_score': 0, 'rmse': float('inf'), 'mae': float('inf'), 'mape': float('inf')}
    
    def cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Perform cross-validation on model.
        
        Args:
            model: Model to validate
            X: Features
            y: Target variable
            
        Returns:
            Dictionary of cross-validation metrics
        """
        try:
            # R² scores
            r2_scores = cross_val_score(
                model, X, y,
                cv=self.config['cv_folds'],
                scoring='r2',
                n_jobs=self.config['n_jobs']
            )
            
            # Negative MSE scores (convert to positive RMSE)
            mse_scores = -cross_val_score(
                model, X, y,
                cv=self.config['cv_folds'],
                scoring='neg_mean_squared_error',
                n_jobs=self.config['n_jobs']
            )
            rmse_scores = np.sqrt(mse_scores)
            
            # MAE scores
            mae_scores = -cross_val_score(
                model, X, y,
                cv=self.config['cv_folds'],
                scoring='neg_mean_absolute_error',
                n_jobs=self.config['n_jobs']
            )
            
            cv_metrics = {
                'cv_r2_mean': r2_scores.mean(),
                'cv_r2_std': r2_scores.std(),
                'cv_rmse_mean': rmse_scores.mean(),
                'cv_rmse_std': rmse_scores.std(),
                'cv_mae_mean': mae_scores.mean(),
                'cv_mae_std': mae_scores.std()
            }
            
            return cv_metrics
        except (ValueError, RuntimeError) as e:
            logger.error("Error in cross-validation: %s", e)
            return {
                'cv_r2_mean': 0, 'cv_r2_std': 0,
                'cv_rmse_mean': float('inf'), 'cv_rmse_std': 0,
                'cv_mae_mean': float('inf'), 'cv_mae_std': 0
            }
    
    def optimize_hyperparameters_optuna(self, model_name: str, X_train: pd.DataFrame, 
                                       y_train: pd.Series, X_val: pd.DataFrame, 
                                       y_val: pd.Series) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            model_name: Name of the model to optimize
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary with best parameters and score
        """
        def objective(trial):
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 5, 25),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'random_state': self.config['random_state'],
                    'n_jobs': self.config['n_jobs']
                }
                model = RandomForestRegressor(**params)
                
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': self.config['random_state'],
                    'n_jobs': self.config['n_jobs']
                }
                model = xgb.XGBRegressor(**params)
                
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': self.config['random_state'],
                    'n_jobs': self.config['n_jobs']
                }
                model = lgb.LGBMRegressor(**params)
                
            elif model_name == 'ridge_regression':
                params = {
                    'alpha': trial.suggest_float('alpha', 0.1, 100.0, log=True),
                    'random_state': self.config['random_state']
                }
                model = Ridge(**params)
                
            elif model_name == 'lasso_regression':
                params = {
                    'alpha': trial.suggest_float('alpha', 0.1, 100.0, log=True),
                    'random_state': self.config['random_state']
                }
                model = Lasso(**params)
                
            else:
                return 0  # Skip optimization for linear regression
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                return score
            except (ValueError, RuntimeError):
                return 0
        
        if model_name == 'linear_regression':
            return {'best_params': {}, 'best_score': 0, 'optimized': False}
        
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(
                objective,
                n_trials=self.config['n_trials'],
                timeout=self.config['timeout'],
                show_progress_bar=True
            )
            
            return {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'optimized': True
            }
        except (ValueError, RuntimeError) as e:
            logger.error("Hyperparameter optimization failed for %s: %s", model_name, e)
            return {'best_params': {}, 'best_score': 0, 'optimized': False}
    
    def train_single_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame, 
                          y_test: pd.Series) -> Dict[str, Any]:
        """
        Train a single model with optional hyperparameter optimization.
        
        Args:
            model_name: Name of the model to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            
        Returns:
            Dictionary with trained model and evaluation results
        """
        logger.info("Training %s...", model_name)
        
        # Get base model
        model = self.models[model_name]
        
        # Hyperparameter optimization
        if self.config['hyperparameter_tuning'] and model_name != 'linear_regression':
            logger.info("Optimizing hyperparameters for %s...", model_name)
            optimization_result = self.optimize_hyperparameters_optuna(
                model_name, X_train, y_train, X_val, y_val
            )
            
            if optimization_result['optimized']:
                # Create model with optimized parameters
                if model_name == 'random_forest':
                    model = RandomForestRegressor(**optimization_result['best_params'])
                elif model_name == 'xgboost':
                    model = xgb.XGBRegressor(**optimization_result['best_params'])
                elif model_name == 'lightgbm':
                    model = lgb.LGBMRegressor(**optimization_result['best_params'])
                elif model_name == 'ridge_regression':
                    model = Ridge(**optimization_result['best_params'])
                elif model_name == 'lasso_regression':
                    model = Lasso(**optimization_result['best_params'])
                
                logger.info("Best parameters for %s: %s", model_name, optimization_result['best_params'])
        
        # Train the model
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate on all sets
        train_metrics = self.evaluate_model(model, X_train, y_train)
        val_metrics = self.evaluate_model(model, X_val, y_val)
        test_metrics = self.evaluate_model(model, X_test, y_test)
        
        # Cross-validation
        cv_metrics = self.cross_validate_model(model, X_train, y_train)
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        
        results = {
            'model': model,
            'model_name': model_name,
            'training_time': training_time,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'cv_metrics': cv_metrics,
            'feature_importance': feature_importance,
            'hyperparameters': model.get_params()
        }
        
        logger.info("%s training completed. Test R²: %.4f", model_name, test_metrics['r2_score'])
        performance_monitor.log_metric(f'{model_name}_test_r2', test_metrics['r2_score'])
        
        return results
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train all configured models and compare performance.
        
        Args:
            Training, validation, and test data splits
            
        Returns:
            Dictionary with all model results
        """
        logger.info("Starting model training pipeline...")
        self.initialize_models()
        
        all_results = {}
        
        for model_name in self.models.keys():
            try:
                results = self.train_single_model(
                    model_name, X_train, y_train, X_val, y_val, X_test, y_test
                )
                all_results[model_name] = results
                self.trained_models[model_name] = results['model']
                self.model_scores[model_name] = results['test_metrics']['r2_score']
                
            except (ValueError, RuntimeError, AttributeError) as e:
                logger.error("Failed to train %s: %s", model_name, e)
                continue
        
        # Find best model
        if self.model_scores:
            self.best_model_name = max(self.model_scores, key=self.model_scores.get)
            self.best_model = self.trained_models[self.best_model_name]
            
            logger.info("Best model: %s with R² = %.4f", self.best_model_name, self.model_scores[self.best_model_name])
            performance_monitor.log_metric('best_model_r2', self.model_scores[self.best_model_name])
        
        return all_results
    
    def fit(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Complete training pipeline from raw data to trained models.
        
        Args:
            df: Raw dataset
            target_column: Name of target column
            
        Returns:
            Dictionary with all training results
        """
        logger.info("Starting complete training pipeline...")
        
        # Data preprocessing
        logger.info("Step 1: Data preprocessing...")
        self.preprocessor = DataPreprocessor()
        X_processed, y = self.preprocessor.fit_transform(df, target_column)
        
        # Feature engineering
        logger.info("Step 2: Feature engineering...")
        self.feature_engineer = FeatureEngineer()
        
        # Convert processed array back to DataFrame for feature engineering
        if isinstance(X_processed, np.ndarray):
            feature_names = self.preprocessor.get_feature_names()
            X_processed = pd.DataFrame(X_processed, columns=feature_names, index=df.index[:len(X_processed)])
        
        X_final, y = self.feature_engineer.fit_transform(
            pd.concat([X_processed, y], axis=1), target_column
        )
        
        # Data splitting
        logger.info("Step 3: Data splitting...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_final, y)
        
        # Model training
        logger.info("Step 4: Model training and evaluation...")
        training_results = self.train_all_models(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Save models if configured
        if self.config['save_models']:
            self.save_trained_models()
        
        # Generate final report
        final_results = {
            'training_results': training_results,
            'best_model_name': self.best_model_name,
            'best_model_score': self.model_scores.get(self.best_model_name, 0),
            'model_comparison': self.model_scores,
            'data_info': {
                'original_shape': df.shape,
                'final_shape': X_final.shape,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
            },
            'preprocessing_info': self.preprocessor.get_preprocessing_report(),
            'feature_engineering_info': self.feature_engineer.get_feature_importance_report()
        }
        
        logger.info("Training pipeline completed successfully!")
        return final_results
    
    def save_trained_models(self) -> None:
        """Save all trained models and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.trained_models.items():
            # Save model
            model_path = PATHS.model_artifacts / f"{model_name}_{timestamp}.pkl"
            save_model(model, model_path)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'timestamp': timestamp,
                'test_score': self.model_scores.get(model_name, 0),
                'hyperparameters': model.get_params(),
                'is_best_model': model_name == self.best_model_name
            }
            
            metadata_path = PATHS.model_artifacts / f"{model_name}_{timestamp}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)
        
        # Save best model separately
        if self.best_model is not None:
            best_model_path = PATHS.models / "best_salary_prediction_model.pkl"
            save_model(self.best_model, best_model_path)
            
            # Save model comparison
            comparison_path = PATHS.models / "model_comparison.json"
            with open(comparison_path, 'w', encoding='utf-8') as f:
                json.dump(self.model_scores, f, indent=4)
            
            logger.info("Best model saved: %s", best_model_path)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best trained model."""
        if self.best_model is None:
            raise ValueError("No trained model available. Run fit() first.")
        
        # Apply same preprocessing and feature engineering
        X_processed = self.preprocessor.transform(X)
        
        if isinstance(X_processed, np.ndarray):
            feature_names = self.preprocessor.get_feature_names()
            X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        
        X_final = self.feature_engineer.transform(X_processed)
        
        return self.best_model.predict(X_final)
    
    def get_model_comparison_report(self) -> pd.DataFrame:
        """Generate a comparison report of all trained models."""
        if not self.model_scores:
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, score in self.model_scores.items():
            comparison_data.append({
                'Model': model_name,
                'R² Score': score,
                'Is Best': model_name == self.best_model_name
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('R² Score', ascending=False)
        
        return df_comparison
