@echo off
title AI Family Photo Sorter
cd /d "%~dp0"

echo.
echo ==========================================================
echo   AI Family Photo Sorter
echo   1-Click Automated Launcher
echo ==========================================================
echo.

:: Check Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in system PATH.
    echo.
    echo Please install Python 3.10 or newer:
    echo 1. Download from: https://www.python.org/downloads/
    echo 2. IMPORTANT: Check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: Run smart launcher if launcher.py exists
IF EXIST "launcher.py" (
    python launcher.py
    IF %ERRORLEVEL% NEQ 0 (
        echo.
        echo [ERROR] Launcher encountered an issue. See above for details.
        pause
        exit /b %ERRORLEVEL%
    )
    exit /b 0
)

:: Fallback direct start
IF NOT EXIST "main.py" (
    echo [ERROR] main.py not found. Please ensure you are running from the extracted folder.
    pause
    exit /b 1
)

start /b cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8000"
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --log-level info
pause
