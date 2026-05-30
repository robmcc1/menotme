@echo off
echo.
echo =============================================
echo    Face Swap Stream - Full Uninstall
echo =============================================
echo.
echo This will remove:
echo   [1] Python venv + all pip packages
echo   [2] Python 3.10 sidecar venv + all pip packages
echo   [3] Extracted RVC voice models
echo   [4] InsightFace AI models (~1.5GB)
echo   [5] inswapper_128.onnx model
echo   [6] Uploaded face images
echo   [7] Project temp files
echo.
echo Optional uninstall (prompted later):
echo   - Python 3.10 (installed for sidecar)
echo   - Visual Studio 2022 Build Tools (C++ toolchain for fairseq)
echo.
echo This will NOT remove (uninstall manually if wanted):
echo   - Python 3.12 (Windows Store - uninstall from Apps)
echo   - CUDA 12.6 Toolkit (Control Panel - NVIDIA uninstaller)
echo   - CUDA 13.3 Toolkit (Control Panel - NVIDIA uninstaller)
echo   - cuDNN (just delete C:\cudnn\ or wherever you extracted it)
echo.
set /p CONFIRM=Type YES to continue: 
if /i not "%CONFIRM%"=="YES" (
    echo Cancelled.
    pause
    exit /b 0
)

echo.
echo --- Removing project files ---

REM Delete virtual environment (all pip packages)
if exist venv (
    echo Deleting venv ^(all pip packages^)...
    rmdir /s /q venv
    echo ✓ venv deleted
)

REM Delete sidecar worker virtual environment (all sidecar pip packages)
if exist venv310 (
    echo Deleting venv310 ^(sidecar pip packages^)...
    rmdir /s /q venv310
    echo ✓ venv310 deleted
)

REM Delete extracted voice models
if exist voice_models (
    echo Deleting extracted voice models...
    rmdir /s /q voice_models
    echo ✓ voice_models deleted
)

REM Delete inswapper model
if exist inswapper_128.onnx (
    echo Deleting inswapper_128.onnx...
    del /f inswapper_128.onnx
    echo ✓ inswapper_128.onnx deleted
)

REM Delete uploaded images
if exist uploads (
    echo Deleting uploaded images...
    rmdir /s /q uploads
    echo ✓ uploads deleted
)

REM Delete pycache
for /d /r . %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d"
)
echo ✓ pycache cleared

REM Clean sidecar pycache specifically
if exist sidecar\__pycache__ (
    rmdir /s /q sidecar\__pycache__
    echo ✓ sidecar pycache cleared
)

echo.
echo --- Removing InsightFace models ---

REM InsightFace downloads models to user profile
if exist "%USERPROFILE%\.insightface" (
    echo Deleting %USERPROFILE%\.insightface ^(~1.5GB of AI models^)...
    rmdir /s /q "%USERPROFILE%\.insightface"
    echo ✓ InsightFace models deleted
) else (
    echo - InsightFace models not found at %USERPROFILE%\.insightface
)

echo.
set /p REMOVE_GLOBAL=Also uninstall global sidecar prerequisites ^(Python 3.10 + VS Build Tools^) ? [y/N]: 
if /i "%REMOVE_GLOBAL%"=="Y" goto UNINSTALL_GLOBAL
if /i "%REMOVE_GLOBAL%"=="YES" goto UNINSTALL_GLOBAL
goto FINISH

:UNINSTALL_GLOBAL
echo.
echo --- Uninstalling global prerequisites ---

where winget >nul 2>&1
if errorlevel 1 (
    echo winget not found; skipping global uninstall.
    goto FINISH
)

echo Uninstalling Python 3.10 (if present)...
winget uninstall -e --id Python.Python.3.10 --silent --accept-source-agreements >nul 2>&1
if errorlevel 1 (
    echo - Python 3.10 not found or uninstall failed ^(check manually^)
) else (
    echo ✓ Python 3.10 uninstalled
)

echo Uninstalling Visual Studio 2022 Build Tools (if present)...
winget uninstall -e --id Microsoft.VisualStudio.2022.BuildTools --silent --accept-source-agreements >nul 2>&1
if errorlevel 1 (
    echo - VS Build Tools not found or uninstall failed ^(check manually^)
) else (
    echo ✓ VS Build Tools uninstalled
)

:FINISH

echo.
echo =============================================
echo ✓ Cleanup complete!
echo.
echo Still installed - remove manually if wanted:
echo.
echo  Python 3.12:
echo    Settings ^> Apps ^> search "Python" ^> Uninstall
echo.
echo  Optional sidecar prerequisites:
echo    Settings ^> Apps ^> uninstall "Python 3.10"
echo    Settings ^> Apps ^> uninstall "Visual Studio 2022 Build Tools"
echo.
echo  CUDA Toolkits:
echo    Control Panel ^> Programs ^> Uninstall:
echo    "NVIDIA CUDA Development 12.6"
echo    "NVIDIA CUDA Development 13.3"
echo.
echo  cuDNN:
echo    Just delete wherever you extracted it (e.g. C:\cudnn\)
echo.
echo  This project folder:
echo    Delete: C:\Users\robmc\OneDrive\Documents\development\face-swap-stream
echo =============================================
echo.
pause
