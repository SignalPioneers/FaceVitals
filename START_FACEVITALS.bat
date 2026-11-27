@echo off
setlocal
cd /d "%~dp0"
title FaceVitals Launcher

echo.
echo ==========================================
echo              FACEVITALS
echo ==========================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo ERROR: Python 3 was not found.
        echo Install Python 3 and enable "Add Python to PATH".
        echo Then run this file again.
        pause
        exit /b 1
    )
    set "PYTHON_CMD=python"
)

if not exist "facevitals_app.py" (
    echo ERROR: facevitals_app.py was not found.
    echo Please run this launcher from the extracted FaceVitals project folder.
    pause
    exit /b 1
)

if not exist "face_landmarker.task" (
    echo ERROR: face_landmarker.task is missing from this folder.
    echo Please download the complete FaceVitals project again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the FaceVitals Python environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto setup_failed
)

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if not exist ".venv\.facevitals-dependencies-installed" (
    echo.
    echo Installing FaceVitals packages. This is only needed on first launch.
    echo An internet connection is required and this may take a few minutes.
    "%VENV_PY%" -m pip install --upgrade pip
    if errorlevel 1 goto setup_failed
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto setup_failed
    echo ready>"%~dp0.venv\.facevitals-dependencies-installed"
)

echo.
echo Starting FaceVitals...
"%VENV_PY%" facevitals_app.py
if errorlevel 1 (
    echo.
    echo FaceVitals closed with an error. Please share this window with the project team.
    pause
)
exit /b 0

:setup_failed
echo.
echo SETUP FAILED. Check your internet connection and Python installation.
echo You can try again by double-clicking START_FACEVITALS.bat.
pause
exit /b 1
