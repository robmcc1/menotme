@echo off
echo.
echo =============================================
echo    Face Swap Stream - Full Uninstall
echo =============================================
echo.
echo This will remove:
echo   [1] Python venv + all pip packages
echo   [2] InsightFace AI models (~1.5GB)
echo   [3] inswapper_128.onnx model
echo   [4] Uploaded face images
echo   [5] Project temp files
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
echo =============================================
echo ✓ Cleanup complete!
echo.
echo Still installed - remove manually if wanted:
echo.
echo  Python 3.12:
echo    Settings ^> Apps ^> search "Python" ^> Uninstall
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
