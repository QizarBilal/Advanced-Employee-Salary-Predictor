"""
Feature Engineering module for Employee Salary Prediction System.

This module provides comprehensive feature engineering capabilities including:
- Feature creation and transformation
- Feature selection techniques
- Feature encoding and scaling
- Domain-specific feature engineering for salary prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

from src.utils import performance_monitor, logger

class FeatureEngineer:
    """
    Comprehensive feature engineering pipeline for salary prediction.
    
    This class handles advanced feature engineering including:
    - Domain-specific feature creation
    - Interaction features
    - Polynomial features
    - Feature selection and dimensionality reduction
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration dictionary for feature engineering parameters
        """
        self.config = config or self._get_default_config()
        self.fitted_transformers = {}
        self.feature_importance = {}
        self.selected_features = []
        
    def _get_default_config(self) -> Dict:
        """Get default feature engineering configuration."""
        return {
            'create_interaction_features': True,
            'create_polynomial_features': False,
            'polynomial_degree': 2,
            'feature_selection_method': 'random_forest',  # 'univariate', 'rfe', 'random_forest', 'lasso'
            'n_features_to_select': 50,
            'apply_pca': False,
            'pca_variance_threshold': 0.95,
            'create_domain_features': True,
            'random_state': 42
        }
    
    def create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific features for salary prediction.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional domain features
        """
        df_features = df.copy()
        
        # Experience categories
        df_features['experience_category'] = pd.cut(
            df_features['years_experience'], 
            bins=[-1, 2, 5, 10, 20, float('inf')],
            labels=['Entry', 'Junior', 'Mid', 'Senior', 'Expert']
        )
        
        # Age categories
        df_features['age_category'] = pd.cut(
            df_features['age'],
            bins=[0, 25, 30, 35, 45, float('inf')],
            labels=['Young', 'Early_Career', 'Mid_Career', 'Experienced', 'Senior']
        )
        
        # Skill level categories
        if 'technical_skills_score' in df_features.columns:
            df_features['skill_level'] = pd.cut(
                df_features['technical_skills_score'],
                bins=[0, 40, 60, 80, 100],
                labels=['Beginner', 'Intermediate', 'Advanced', 'Expert']
            )
        
        # Experience efficiency (skills per year of experience)
        if 'technical_skills_score' in df_features.columns:
            df_features['skill_efficiency'] = (
                df_features['technical_skills_score'] / 
                (df_features['years_experience'] + 1)  # +1 to avoid division by zero
            )
        
        # Education-Experience mismatch
        education_weights = {'High School': 1, 'Diploma': 2, 'Bachelor': 3, 'Master': 4, 'PhD': 5}
        if 'education_level' in df_features.columns:
            df_features['education_weight'] = df_features['education_level'].map(education_weights)
            df_features['education_experience_ratio'] = (
                df_features['education_weight'] / (df_features['years_experience'] + 1)
            )
        
        # Performance categories
        if 'performance_rating' in df_features.columns:
            df_features['high_performer'] = (df_features['performance_rating'] >= 4).astype(int)
        
        # Certification value (certifications per year of experience)
        if 'certifications_count' in df_features.columns:
            df_features['certification_rate'] = (
                df_features['certifications_count'] / (df_features['years_experience'] + 1)
            )
        
        # Tech department indicator
        df_features['is_tech_department'] = (df_features['department'] == 'Technology').astype(int)
        
        # Metro city indicator
        df_features['is_metro_city'] = (df_features['city_tier'] == 'Tier 1').astype(int)
        
        # Large company indicator
        large_companies = ['Large (1000-5000)', 'Enterprise (5000+)']
        df_features['is_large_company'] = df_features['company_size'].isin(large_companies).astype(int)
        
        # MNC indicator
        df_features['is_mnc'] = (df_features['company_type'] == 'MNC').astype(int)
        
        # Senior role indicator (contains senior, lead, manager, director)
        senior_keywords = ['Senior', 'Lead', 'Manager', 'Director', 'Head', 'VP', 'Chief']
        df_features['is_senior_role'] = df_features['job_title'].str.contains(
            '|'.join(senior_keywords), case=False, na=False
        ).astype(int)
        
        logger.info("Created %d domain-specific features", len(df_features.columns) - len(df.columns))
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        df_interactions = df.copy()
        
        # Experience × Education interaction
        if 'years_experience' in df.columns and 'education_weight' in df.columns:
            df_interactions['experience_education_interaction'] = (
                df_interactions['years_experience'] * df_interactions['education_weight']
            )
        
        # Skills × Experience interaction
        if 'technical_skills_score' in df.columns and 'years_experience' in df.columns:
            df_interactions['skills_experience_interaction'] = (
                df_interactions['technical_skills_score'] * df_interactions['years_experience']
            )
        
        # Performance × Skills interaction
        if 'performance_rating' in df.columns and 'technical_skills_score' in df.columns:
            df_interactions['performance_skills_interaction'] = (
                df_interactions['performance_rating'] * df_interactions['technical_skills_score']
            )
        
        # Age × Experience interaction (career progression rate)
        if 'age' in df.columns and 'years_experience' in df.columns:
            df_interactions['career_progression_rate'] = (
                df_interactions['years_experience'] / (df_interactions['age'] - 20)
            )
        
        # Tech department × Skills interaction
        if 'is_tech_department' in df.columns and 'technical_skills_score' in df.columns:
            df_interactions['tech_skills_boost'] = (
                df_interactions['is_tech_department'] * df_interactions['technical_skills_score']
            )
        
        # Metro × Experience interaction (location premium)
        if 'is_metro_city' in df.columns and 'years_experience' in df.columns:
            df_interactions['metro_experience_premium'] = (
                df_interactions['is_metro_city'] * df_interactions['years_experience']
            )
        
        logger.info("Created %d interaction features", len(df_interactions.columns) - len(df.columns))
        return df_interactions
    
    def create_polynomial_features(self, df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
        """
        Create polynomial features for numerical columns.
        
        Args:
            df: Input DataFrame
            numerical_cols: List of numerical columns for polynomial features
            
        Returns:
            DataFrame with polynomial features
        """
        if not self.config['create_polynomial_features']:
            return df
        
        df_poly = df.copy()
        
        # Select important numerical features for polynomial transformation
        important_numerical = []
        for col in numerical_cols:
            if col in ['years_experience', 'age', 'technical_skills_score', 'performance_rating']:
                important_numerical.append(col)
        
        if len(important_numerical) > 0:
            # Create polynomial features
            poly = PolynomialFeatures(
                degree=self.config['polynomial_degree'],
                include_bias=False,
                interaction_only=False
            )
            
            poly_features = poly.fit_transform(df[important_numerical])
            poly_feature_names = poly.get_feature_names_out(important_numerical)
            
            # Add polynomial features to dataframe
            poly_df = pd.DataFrame(
                poly_features[:, len(important_numerical):],  # Exclude original features
                columns=poly_feature_names[len(important_numerical):],
                index=df.index
            )
            
            df_poly = pd.concat([df_poly, poly_df], axis=1)
            self.fitted_transformers['polynomial'] = poly
            
            logger.info("Created %d polynomial features", len(poly_df.columns))
        
        return df_poly
    
    def select_features_univariate(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """
        Select features using univariate statistical tests.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        # Only use numerical features for univariate selection
        numerical_X = X.select_dtypes(include=[np.number])
        
        selector = SelectKBest(score_func=f_regression, k=min(n_features, len(numerical_X.columns)))
        selector.fit(numerical_X, y)
        
        selected_features = numerical_X.columns[selector.get_support()].tolist()
        
        # Add important categorical features
        categorical_features = []
        for col in X.columns:
            if col not in numerical_X.columns and col in ['department', 'city_tier', 'company_size']:
                categorical_features.append(col)
        
        selected_features.extend(categorical_features)
        
        self.fitted_transformers['univariate_selector'] = selector
        return selected_features[:n_features]
    
    def select_features_rfe(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        # Prepare data for RFE (only numerical features)
        numerical_X = X.select_dtypes(include=[np.number])
        
        estimator = RandomForestRegressor(
            n_estimators=50,
            random_state=self.config['random_state'],
            n_jobs=-1
        )
        
        selector = RFE(
            estimator=estimator,
            n_features_to_select=min(n_features, len(numerical_X.columns)),
            step=1
        )
        
        selector.fit(numerical_X, y)
        selected_features = numerical_X.columns[selector.get_support()].tolist()
        
        self.fitted_transformers['rfe_selector'] = selector
        return selected_features
    
    def select_features_random_forest(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """
        Select features using Random Forest feature importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        # Prepare data for Random Forest
        numerical_X = X.select_dtypes(include=[np.number])
        
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=self.config['random_state'],
            n_jobs=-1
        )
        
        rf.fit(numerical_X, y)
        
        # Get feature importance
        importance_scores = pd.Series(rf.feature_importances_, index=numerical_X.columns)
        importance_scores = importance_scores.sort_values(ascending=False)
        
        # Store feature importance for analysis
        self.feature_importance['random_forest'] = importance_scores.to_dict()
        
        selected_features = importance_scores.head(n_features).index.tolist()
        
        self.fitted_transformers['random_forest_selector'] = rf
        return selected_features
    
    def select_features_lasso(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """
        Select features using Lasso regularization.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        from sklearn.linear_model import LassoCV
        
        # Prepare data for Lasso
        numerical_X = X.select_dtypes(include=[np.number])
        
        # Scale features for Lasso
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numerical_X)
        
        # Fit Lasso with cross-validation
        lasso = LassoCV(cv=5, random_state=self.config['random_state'], n_jobs=-1)
        lasso.fit(X_scaled, y)
        
        # Select features with non-zero coefficients
        selected_mask = lasso.coef_ != 0
        selected_features = numerical_X.columns[selected_mask].tolist()
        
        # If too many features selected, take top ones by coefficient magnitude
        if len(selected_features) > n_features:
            coef_importance = pd.Series(
                np.abs(lasso.coef_[selected_mask]),
                index=numerical_X.columns[selected_mask]
            ).sort_values(ascending=False)
            selected_features = coef_importance.head(n_features).index.tolist()
        
        self.fitted_transformers['lasso_selector'] = lasso
        self.fitted_transformers['lasso_scaler'] = scaler
        
        return selected_features
    
    def apply_pca(self, X: pd.DataFrame, variance_threshold: float = 0.95) -> Tuple[np.ndarray, PCA]:
        """
        Apply Principal Component Analysis for dimensionality reduction.
        
        Args:
            X: Feature matrix
            variance_threshold: Cumulative variance threshold to retain
            
        Returns:
            Tuple of (transformed features, fitted PCA object)
        """
        # Only apply PCA to numerical features
        numerical_X = X.select_dtypes(include=[np.number])
        
        # Scale features before PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numerical_X)
        
        # Apply PCA
        pca = PCA(random_state=self.config['random_state'])
        pca.fit(X_scaled)
        
        # Find number of components for desired variance
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
        
        # Refit with optimal number of components
        pca_final = PCA(n_components=n_components, random_state=self.config['random_state'])
        X_pca_final = pca_final.fit_transform(X_scaled)
        
        self.fitted_transformers['pca'] = pca_final
        self.fitted_transformers['pca_scaler'] = scaler
        
        logger.info("PCA reduced %d features to %d components (%.1f%% variance retained)",
                   len(numerical_X.columns), n_components, variance_threshold*100)
        
        return X_pca_final, pca_final
    
    def fit_transform(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (engineered features DataFrame, target Series)
        """
        logger.info("Starting feature engineering pipeline")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Step 1: Create domain-specific features
        if self.config['create_domain_features']:
            X = self.create_domain_features(X)
        
        # Step 2: Create interaction features
        if self.config['create_interaction_features']:
            X = self.create_interaction_features(X)
        
        # Step 3: Create polynomial features
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if self.config['create_polynomial_features']:
            X = self.create_polynomial_features(X, numerical_cols)
        
        # Step 4: Feature selection
        method = self.config['feature_selection_method']
        n_features = self.config['n_features_to_select']
        
        if method == 'univariate':
            selected_features = self.select_features_univariate(X, y, n_features)
        elif method == 'rfe':
            selected_features = self.select_features_rfe(X, y, n_features)
        elif method == 'random_forest':
            selected_features = self.select_features_random_forest(X, y, n_features)
        elif method == 'lasso':
            selected_features = self.select_features_lasso(X, y, n_features)
        else:
            # Use all numerical features if no valid method specified
            selected_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        self.selected_features = selected_features
        
        # Keep only selected features
        X_selected = X[selected_features]
        
        # Step 5: Apply PCA if requested
        if self.config['apply_pca']:
            X_pca_final, _ = self.apply_pca(X_selected, self.config['pca_variance_threshold'])
            # Convert back to DataFrame with PCA component names
            pca_columns = [f'PC_{i+1}' for i in range(X_pca_final.shape[1])]
            X_final = pd.DataFrame(X_pca_final, columns=pca_columns, index=X.index)
        else:
            X_final = X_selected
        
        logger.info("Feature engineering completed. Final shape: %s", X_final.shape)
        performance_monitor.log_metric('feature_engineering_features', X_final.shape[1])
        
        return X_final, y
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the fitted feature engineering pipeline.
        
        Args:
            df: New DataFrame to transform
            
        Returns:
            Transformed features DataFrame
        """
        X = df.copy()
        
        # Apply same transformations as fit_transform
        if self.config['create_domain_features']:
            X = self.create_domain_features(X)
        
        if self.config['create_interaction_features']:
            X = self.create_interaction_features(X)
        
        if self.config['create_polynomial_features'] and 'polynomial' in self.fitted_transformers:
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            important_numerical = []
            for col in numerical_cols:
                if col in ['years_experience', 'age', 'technical_skills_score', 'performance_rating']:
                    important_numerical.append(col)
            
            if len(important_numerical) > 0:
                poly = self.fitted_transformers['polynomial']
                poly_features = poly.transform(X[important_numerical])
                poly_feature_names = poly.get_feature_names_out(important_numerical)
                
                poly_df = pd.DataFrame(
                    poly_features[:, len(important_numerical):],
                    columns=poly_feature_names[len(important_numerical):],
                    index=X.index
                )
                
                X = pd.concat([X, poly_df], axis=1)
        
        # Select same features as training
        X_selected = X[self.selected_features]
        
        # Apply PCA if used in training
        if self.config['apply_pca'] and 'pca' in self.fitted_transformers:
            numerical_X = X_selected.select_dtypes(include=[np.number])
            scaler = self.fitted_transformers['pca_scaler']
            pca = self.fitted_transformers['pca']
            
            X_scaled = scaler.transform(numerical_X)
            X_pca = pca.transform(X_scaled)
            
            pca_columns = [f'PC_{i+1}' for i in range(X_pca.shape[1])]
            X_final = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
        else:
            X_final = X_selected
        
        return X_final
    
    def get_feature_importance_report(self) -> Dict:
        """
        Generate a comprehensive feature importance report.
        
        Returns:
            Dictionary containing feature importance analysis
        """
        report = {
            'selected_features': self.selected_features,
            'feature_selection_method': self.config['feature_selection_method'],
            'total_features_created': len(self.selected_features),
            'feature_importance': self.feature_importance
        }
        
        if 'random_forest' in self.feature_importance:
            # Get top 10 most important features
            rf_importance = self.feature_importance['random_forest']
            top_features = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            report['top_10_features'] = dict(top_features)
        
        return report
