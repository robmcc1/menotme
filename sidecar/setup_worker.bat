@echo off
setlocal EnableDelayedExpansion

REM Install a dedicated Python 3.10 runtime and create an isolated worker venv.
set ROOT=%~dp0..
set PY310=%LOCALAPPDATA%\Programs\Python\Python310\python.exe
set VENV=%ROOT%\venv310

echo [1/4] Checking Python 3.10...
if not exist "%PY310%" (
  echo Python 3.10 not found. Installing with winget...
  winget install -e --id Python.Python.3.10 --silent --accept-package-agreements --accept-source-agreements
  if errorlevel 1 (
    echo Failed to install Python 3.10 via winget.
    echo Install Python 3.10 manually, then rerun this script.
    exit /b 1
  )
)

if not exist "%PY310%" (
  echo Python 3.10 still not found at: %PY310%
  exit /b 1
)

echo [2/4] Creating worker venv at %VENV%
"%PY310%" -m venv "%VENV%"
if errorlevel 1 (
  echo Failed creating worker venv.
  exit /b 1
)

echo [3/4] Installing worker dependencies...
"%VENV%\Scripts\python.exe" -m pip install --upgrade "pip==24.0" setuptools wheel
if errorlevel 1 (
  echo Failed to upgrade pip tooling.
  exit /b 1
)

"%VENV%\Scripts\python.exe" -m pip install -r "%~dp0requirements-worker.txt"
if errorlevel 1 (
  echo Failed to install worker requirements.
  exit /b 1
)

echo [4/5] Installing CUDA-enabled PyTorch for worker...
"%VENV%\Scripts\python.exe" -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu130 torch torchaudio
if errorlevel 1 (
  echo Failed to install CUDA-enabled torch/torchaudio.
  exit /b 1
)

echo [5/5] Worker setup complete.
echo Worker python: %VENV%\Scripts\python.exe
echo You can set RVC_WORKER_PYTHON to override this path.
exit /b 0
