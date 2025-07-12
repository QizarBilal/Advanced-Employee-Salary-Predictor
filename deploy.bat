@echo off
REM Windows deployment script for Employee Salary Prediction System

echo 🚀 Employee Salary Prediction System - Windows Deployment Assistant
echo ==================================================================

:menu
echo.
echo Select deployment option:
echo 1. Local Development Server
echo 2. Local Network Access
echo 3. Docker Deployment
echo 4. Open Streamlit Cloud Guide
echo 5. Open Heroku Guide
echo 6. Exit
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto local_dev
if "%choice%"=="2" goto local_network
if "%choice%"=="3" goto docker_deploy
if "%choice%"=="4" goto streamlit_guide
if "%choice%"=="5" goto heroku_guide
if "%choice%"=="6" goto exit
goto invalid

:local_dev
echo 🏠 Starting local development server...
streamlit run webapp/app.py
goto menu

:local_network
echo 🌐 Starting server for network access...
echo Access from other devices on your network
for /f "delims=[] tokens=2" %%a in ('ping -4 -n 1 %ComputerName% ^| findstr [') do set NetworkIP=%%a
echo URL: http://%NetworkIP%:8501
streamlit run webapp/app.py --server.address 0.0.0.0 --server.port 8501
goto menu

:docker_deploy
echo 🐳 Docker deployment...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker not installed. Please install Docker Desktop first.
    goto menu
)
echo Building Docker image...
docker build -t employee-salary-prediction .
echo Running Docker container...
docker run -d -p 8501:8501 --name salary-app employee-salary-prediction
echo ✅ Docker deployment complete! Access at http://localhost:8501
goto menu

:streamlit_guide
echo 📖 Opening Streamlit Community Cloud guide...
echo.
echo Streamlit Community Cloud Deployment Steps:
echo 1. Push your code to GitHub
echo 2. Go to https://share.streamlit.io
echo 3. Sign in with GitHub
echo 4. Click "New app"
echo 5. Select your repository
echo 6. Set main file path: webapp/app.py
echo 7. Click "Deploy!"
echo.
start https://share.streamlit.io
goto menu

:heroku_guide
echo 📖 Opening Heroku deployment guide...
echo.
echo Heroku Deployment Steps:
echo 1. Install Heroku CLI
echo 2. Run: heroku login
echo 3. Run: heroku create your-app-name
echo 4. Run: git push heroku main
echo.
start https://devcenter.heroku.com/articles/heroku-cli
goto menu

:invalid
echo ❌ Invalid choice. Please try again.
goto menu

:exit
echo 👋 Goodbye!
pause
exit /b 0
