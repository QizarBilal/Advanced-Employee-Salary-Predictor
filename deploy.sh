#!/bin/bash

# Automated deployment script for various platforms

echo "🚀 Employee Salary Prediction System - Deployment Assistant"
echo "==========================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to deploy to Streamlit Cloud
deploy_streamlit_cloud() {
    echo "📝 Deploying to Streamlit Community Cloud..."
    echo "1. Ensure your code is pushed to GitHub"
    echo "2. Go to https://share.streamlit.io"
    echo "3. Connect your GitHub repository"
    echo "4. Set main file path: webapp/app.py"
    echo "5. Click Deploy!"
    echo ""
    echo "✅ Manual steps required - visit the URL above"
}

# Function to deploy to Heroku
deploy_heroku() {
    echo "📝 Deploying to Heroku..."
    
    if ! command_exists heroku; then
        echo "❌ Heroku CLI not installed. Please install it first."
        echo "   Visit: https://devcenter.heroku.com/articles/heroku-cli"
        return 1
    fi
    
    echo "🔐 Logging into Heroku..."
    heroku login
    
    echo "📦 Creating Heroku app..."
    read -p "Enter your app name (or press Enter for auto-generated): " app_name
    
    if [ -z "$app_name" ]; then
        heroku create
    else
        heroku create "$app_name"
    fi
    
    echo "🚀 Deploying to Heroku..."
    git add .
    git commit -m "Deploy to Heroku"
    git push heroku main
    
    echo "✅ Deployment complete!"
    heroku open
}

# Function to deploy with Docker
deploy_docker() {
    echo "🐳 Deploying with Docker..."
    
    if ! command_exists docker; then
        echo "❌ Docker not installed. Please install Docker first."
        return 1
    fi
    
    echo "🔨 Building Docker image..."
    docker build -t employee-salary-prediction .
    
    echo "🚀 Running Docker container..."
    docker run -d -p 8501:8501 --name salary-app employee-salary-prediction
    
    echo "✅ Docker deployment complete!"
    echo "🌐 Access your app at: http://localhost:8501"
}

# Function to deploy locally for network access
deploy_local_network() {
    echo "🏠 Setting up local network deployment..."
    
    # Get local IP address
    if command_exists ip; then
        LOCAL_IP=$(ip route get 1 | sed -n 's/^.*src \([0-9.]*\) .*$/\1/p')
    elif command_exists ifconfig; then
        LOCAL_IP=$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -1)
    else
        LOCAL_IP="YOUR_LOCAL_IP"
    fi
    
    echo "🌐 Starting Streamlit on network..."
    echo "📱 Access from other devices at: http://$LOCAL_IP:8501"
    
    streamlit run webapp/app.py --server.address 0.0.0.0 --server.port 8501
}

# Function to deploy to Railway
deploy_railway() {
    echo "🚂 Deploying to Railway..."
    
    if ! command_exists railway; then
        echo "❌ Railway CLI not installed."
        echo "   Install with: npm install -g @railway/cli"
        return 1
    fi
    
    railway login
    railway init
    railway up
    
    echo "✅ Railway deployment complete!"
}

# Main menu
show_menu() {
    echo ""
    echo "Select deployment option:"
    echo "1. Streamlit Community Cloud (FREE - Recommended)"
    echo "2. Heroku"
    echo "3. Railway"
    echo "4. Docker (Local)"
    echo "5. Local Network Access"
    echo "6. View Deployment Guide"
    echo "7. Exit"
    echo ""
}

# Pre-deployment checks
pre_deployment_checks() {
    echo "🔍 Running pre-deployment checks..."
    
    # Check if required files exist
    required_files=("requirements.txt" "webapp/app.py" "data/raw/employee_salary_dataset.csv")
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            echo "❌ Missing required file: $file"
            return 1
        fi
    done
    
    # Check Python dependencies
    echo "📦 Checking Python dependencies..."
    if ! python -c "import streamlit, pandas, numpy, plotly" 2>/dev/null; then
        echo "❌ Missing Python dependencies. Run: pip install -r requirements.txt"
        return 1
    fi
    
    echo "✅ Pre-deployment checks passed!"
    return 0
}

# Main execution
main() {
    if ! pre_deployment_checks; then
        echo "❌ Pre-deployment checks failed. Please fix the issues above."
        exit 1
    fi
    
    while true; do
        show_menu
        read -p "Enter your choice (1-7): " choice
        
        case $choice in
            1)
                deploy_streamlit_cloud
                ;;
            2)
                deploy_heroku
                ;;
            3)
                deploy_railway
                ;;
            4)
                deploy_docker
                ;;
            5)
                deploy_local_network
                ;;
            6)
                if [ -f "DEPLOYMENT_GUIDE.md" ]; then
                    cat DEPLOYMENT_GUIDE.md
                else
                    echo "❌ Deployment guide not found"
                fi
                ;;
            7)
                echo "👋 Goodbye!"
                exit 0
                ;;
            *)
                echo "❌ Invalid choice. Please try again."
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run the script
main
