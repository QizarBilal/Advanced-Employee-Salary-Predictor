#!/usr/bin/env python3
"""
Simple script to generate the employee salary dataset.
This extracts and runs the key data generation code from the notebook.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_employee_dataset(num_records=10000):
    """Generate comprehensive employee salary dataset."""
    
    print(f"🚀 Generating {num_records:,} employee records...")
    
    # Define data categories
    cities = [
        'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad',
        'Surat', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Thane', 'Bhopal',
        'Visakhapatnam', 'Pimpri-Chinchwad', 'Patna', 'Vadodara', 'Ghaziabad', 'Ludhiana'
    ]
    
    city_tiers = {
        'Tier 1': ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad'],
        'Tier 2': ['Surat', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Thane', 'Bhopal'],
        'Tier 3': ['Visakhapatnam', 'Pimpri-Chinchwad', 'Patna', 'Vadodara', 'Ghaziabad', 'Ludhiana']
    }
    
    departments = ['Technology', 'Finance', 'Marketing', 'Sales', 'HR', 'Operations']
    education_levels = ['High School', 'Diploma', 'Bachelor', 'Master', 'PhD']
    company_sizes = ['Startup (<50)', 'Small (50-200)', 'Medium (200-1000)', 'Large (1000-5000)', 'Enterprise (5000+)']
    company_types = ['Private', 'Public', 'Startup', 'MNC', 'Government']
    industries = ['IT Services', 'Banking', 'Healthcare', 'Manufacturing', 'E-commerce', 
                 'Consulting', 'Education', 'Retail', 'Telecommunications', 'Automotive']
    
    # Generate data
    data = []
    
    for i in range(num_records):
        # Basic demographics
        age = np.random.normal(32, 8)
        age = max(22, min(65, int(age)))
        
        gender = np.random.choice(['Male', 'Female'], p=[0.65, 0.35])
        education = np.random.choice(education_levels, p=[0.05, 0.10, 0.45, 0.35, 0.05])
        
        # Professional details
        years_experience = max(0, min(age - 22, np.random.exponential(5)))
        department = np.random.choice(departments)
        
        # Location
        city = np.random.choice(cities)
        city_tier = next(tier for tier, cities_list in city_tiers.items() if city in cities_list)
        
        # Company details
        company_size = np.random.choice(company_sizes)
        company_type = np.random.choice(company_types)
        industry = np.random.choice(industries)
        
        # Skills and performance
        technical_skills_score = max(20, min(100, np.random.normal(70, 15)))
        performance_rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.30, 0.40, 0.15])
        certifications_count = np.random.poisson(2)
        
        # Salary calculation with realistic business logic
        base_salary = 300000  # Base salary in INR
        
        # Experience factor
        exp_multiplier = 1 + (years_experience * 0.08)
        
        # Education factor
        edu_factors = {'High School': 0.7, 'Diploma': 0.85, 'Bachelor': 1.0, 'Master': 1.3, 'PhD': 1.6}
        edu_multiplier = edu_factors[education]
        
        # Department factor
        dept_factors = {'Technology': 1.4, 'Finance': 1.2, 'Marketing': 1.0, 'Sales': 1.1, 'HR': 0.9, 'Operations': 1.0}
        dept_multiplier = dept_factors[department]
        
        # City tier factor
        city_factors = {'Tier 1': 1.3, 'Tier 2': 1.1, 'Tier 3': 1.0}
        city_multiplier = city_factors[city_tier]
        
        # Company size factor
        size_factors = {'Startup (<50)': 0.9, 'Small (50-200)': 1.0, 'Medium (200-1000)': 1.1, 
                       'Large (1000-5000)': 1.2, 'Enterprise (5000+)': 1.3}
        size_multiplier = size_factors[company_size]
        
        # Performance factor
        perf_multiplier = performance_rating / 3.0
        
        # Skills factor
        skills_multiplier = 1 + (technical_skills_score / 100) * 0.3
        
        # Calculate final salary
        calculated_salary = (base_salary * exp_multiplier * edu_multiplier * dept_multiplier * 
                           city_multiplier * size_multiplier * perf_multiplier * skills_multiplier)
        
        # Add some randomness
        salary_noise = np.random.normal(1, 0.1)
        annual_salary = max(200000, min(5000000, calculated_salary * salary_noise))
        
        # Create record
        record = {
            'employee_id': f'EMP_{i+1:06d}',
            'age': age,
            'gender': gender,
            'education_level': education,
            'years_experience': round(years_experience, 1),
            'department': department,
            'city': city,
            'city_tier': city_tier,
            'company_size': company_size,
            'company_type': company_type,
            'industry': industry,
            'technical_skills_score': round(technical_skills_score, 1),
            'performance_rating': performance_rating,
            'certifications_count': certifications_count,
            'annual_salary': round(annual_salary, 0)
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values to simulate real data
    missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[missing_indices, 'technical_skills_score'] = np.nan
    
    missing_indices = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
    df.loc[missing_indices, 'certifications_count'] = np.nan
    
    print(f"✅ Dataset generated successfully!")
    print(f"📊 Shape: {df.shape}")
    print(f"💰 Salary range: ₹{df['annual_salary'].min():,.0f} - ₹{df['annual_salary'].max():,.0f}")
    print(f"💰 Average salary: ₹{df['annual_salary'].mean():,.0f}")
    
    return df

def main():
    """Main function to generate and save dataset."""
    
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Generate dataset
    df = generate_employee_dataset(10000)
    
    # Save dataset
    output_path = 'data/raw/employee_salary_dataset.csv'
    df.to_csv(output_path, index=False)
    
    print(f"💾 Dataset saved to: {output_path}")
    print(f"📈 Ready for analysis and modeling!")
    
    # Display basic info
    print("\n📋 Dataset Info:")
    print(df.info())
    
    print("\n📊 Sample Records:")
    print(df.head())

if __name__ == "__main__":
    main()
