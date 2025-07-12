#!/usr/bin/env python3
"""
Ensure the dataset exists - generate it if missing.
This script is designed to run before the Streamlit app starts.
"""

import os
import pandas as pd
import numpy as np

def ensure_dataset_exists():
    """Ensure dataset exists, generate if missing."""
    
    # Check if dataset exists
    dataset_paths = [
        'data/raw/employee_salary_dataset.csv',
        './data/raw/employee_salary_dataset.csv'
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"✅ Dataset found at: {path}")
            return True
    
    print("📊 Dataset not found. Generating new dataset...")
    
    try:
        # Create directories
        os.makedirs('data/raw', exist_ok=True)
        
        # Generate dataset
        df = generate_simple_dataset()
        
        # Save dataset
        dataset_path = 'data/raw/employee_salary_dataset.csv'
        df.to_csv(dataset_path, index=False)
        
        print(f"✅ Dataset generated successfully: {dataset_path}")
        print(f"📊 Dataset shape: {df.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Error generating dataset: {e}")
        return False

def generate_simple_dataset(num_records=5000):
    """Generate a simple employee dataset."""
    np.random.seed(42)
    
    # Define categories
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad']
    city_tiers = {'Mumbai': 'Tier 1', 'Delhi': 'Tier 1', 'Bangalore': 'Tier 1', 'Hyderabad': 'Tier 1',
                  'Chennai': 'Tier 1', 'Kolkata': 'Tier 2', 'Pune': 'Tier 2', 'Ahmedabad': 'Tier 2'}
    
    departments = ['Technology', 'Finance', 'Marketing', 'Sales', 'HR', 'Operations']
    education_levels = ['High School', 'Diploma', 'Bachelor', 'Master', 'PhD']
    company_sizes = ['Small (50-200)', 'Medium (200-1000)', 'Large (1000-5000)', 'Enterprise (5000+)']
    industries = ['IT Services', 'Banking', 'Healthcare', 'Manufacturing', 'E-commerce', 'Consulting']
    
    data = []
    
    for i in range(num_records):
        # Generate features
        age = max(22, min(65, int(np.random.normal(32, 8))))
        gender = np.random.choice(['Male', 'Female'], p=[0.65, 0.35])
        education = np.random.choice(education_levels, p=[0.1, 0.15, 0.4, 0.3, 0.05])
        
        # Experience
        min_exp = max(0, age - 22 - (4 if education in ['Bachelor', 'Master'] else 2))
        max_exp = age - 18
        years_experience = max(0, min(max_exp, int(np.random.normal(min_exp + 3, 3))))
        
        city = np.random.choice(cities)
        city_tier = city_tiers[city]
        department = np.random.choice(departments)
        company_size = np.random.choice(company_sizes)
        industry = np.random.choice(industries)
        performance_rating = np.random.choice([2, 3, 4, 5], p=[0.1, 0.4, 0.4, 0.1])
        
        # Technical skills and certifications
        base_skills = 40 + years_experience * 3 + {'High School': 0, 'Diploma': 5, 'Bachelor': 10, 'Master': 15, 'PhD': 20}[education]
        technical_skills_score = max(20, min(100, base_skills + np.random.normal(0, 10)))
        
        # Certifications
        cert_count = 0
        if years_experience > 3 and np.random.random() < 0.6:
            cert_count += 1
        if education in ['Master', 'PhD'] and np.random.random() < 0.4:
            cert_count += 1
        certifications_count = cert_count
        
        # Calculate salary
        base_salary = 400000
        
        # Multipliers
        edu_mult = {'High School': 0.7, 'Diploma': 0.8, 'Bachelor': 1.0, 'Master': 1.3, 'PhD': 1.6}
        dept_mult = {'Technology': 1.4, 'Finance': 1.2, 'Marketing': 1.0, 'Sales': 1.1, 'HR': 0.9, 'Operations': 1.0}
        city_mult = {'Tier 1': 1.3, 'Tier 2': 1.1, 'Tier 3': 1.0}
        
        salary = base_salary * edu_mult[education] * dept_mult[department] * city_mult[city_tier]
        salary *= (1 + years_experience * 0.08)
        salary *= (performance_rating / 3.0)
        salary *= (1 + technical_skills_score / 100 * 0.2)  # Skills factor
        salary *= (1 + certifications_count * 0.05)  # Certification bonus
        salary *= np.random.uniform(0.9, 1.1)
        
        # Add outliers
        if np.random.random() < 0.05:
            salary *= np.random.uniform(1.5, 3.0)
        
        data.append({
            'employee_id': f'EMP_{i+1:05d}',
            'age': age,
            'gender': gender,
            'education_level': education,
            'years_experience': years_experience,
            'city': city,
            'city_tier': city_tier,
            'department': department,
            'company_size': company_size,
            'industry': industry,
            'performance_rating': performance_rating,
            'technical_skills_score': round(technical_skills_score, 1),
            'certifications_count': certifications_count,
            'annual_salary': int(salary)
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    ensure_dataset_exists()
