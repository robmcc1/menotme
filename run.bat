@echo off
setlocal EnableDelayedExpansion
echo.
echo ====================================
echo    Face Swap Stream Launcher
echo ====================================
echo.

set "PINNED_PYTHON=C:\v\menotme312\Scripts\python.exe"

REM Prefer pinned Python 3.12 venv to avoid interpreter regressions.
if exist "%PINNED_PYTHON%" (
    set "PYTHON_EXE=%PINNED_PYTHON%"
    echo Using pinned Python: !PYTHON_EXE!
) else (
    REM Fallback to project-local venv
    if not exist venv (
        echo Creating local virtual environment...
        python --version >nul 2>&1
        if errorlevel 1 (
            echo Error: Python is not installed or not in PATH
            echo Install Python 3.12 and re-run.
            pause
            exit /b 1
        )
        python -m venv venv
    )
    set "PYTHON_EXE=%CD%\venv\Scripts\python.exe"
    echo Using local Python: !PYTHON_EXE!
)

if not exist "!PYTHON_EXE!" (
    echo Error: Python executable not found: !PYTHON_EXE!
    pause
    exit /b 1
)

REM Check if requirements are installed
"!PYTHON_EXE!" -m pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies ^(this may take 10-20 minutes^)...
    "!PYTHON_EXE!" -m pip install -r requirements.txt
)

REM Run the server
echo.
echo ✓ Starting Face Swap Stream server...
echo ✓ Open browser to: http://localhost:8000
echo.
"!PYTHON_EXE!" main.py

pause
