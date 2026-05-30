@echo off
echo.
echo ====================================
echo    Face Swap Stream Launcher
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.9+ from python.org
    pause
    exit /b 1
)

REM Check if venv exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Check if requirements are installed
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies (this may take 10-20 minutes)...
    pip install -r requirements.txt
)

REM Run the server
echo.
echo ✓ Starting Face Swap Stream server...
echo ✓ Open browser to: http://localhost:8000
echo.
python main.py

pause
