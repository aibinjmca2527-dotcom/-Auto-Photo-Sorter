@echo off
title AI Family Photo Sorter
cd /d "%~dp0"

echo.
echo  Starting AI Family Photo Sorter...
echo  Open your browser at: http://127.0.0.1:8000
echo  Press Ctrl+C in this window to stop the server.
echo.

:: Check Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Please run setup.bat first.
    pause
    exit /b 1
)

:: Check if main.py exists
IF NOT EXIST "main.py" (
    echo [ERROR] main.py not found. Please ensure you are running from the correct folder.
    pause
    exit /b 1
)

:: Open browser after 3 seconds (small delay for server startup)
start /b cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8000"

:: Launch server
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --log-level info

pause
