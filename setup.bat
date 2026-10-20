@echo off
SETLOCAL EnableDelayedExpansion
title AI Family Photo Sorter - Setup

echo.
echo ==========================================================
echo   AI Family Photo Sorter - Windows Setup
echo   Installing dependencies for Python 3.10+
echo ==========================================================
echo.

:: Check Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Download Python 3.10+ from https://python.org
    pause
    exit /b 1
)

python --version
echo.

:: Upgrade pip
echo [1/6] Upgrading pip...
python -m pip install --upgrade pip --quiet
IF %ERRORLEVEL% NEQ 0 (
    echo [WARN] pip upgrade failed, continuing...
)

:: Install core dependencies (already present, harmless to reinstall)
echo.
echo [2/6] Installing core dependencies (FastAPI, uvicorn, SQLAlchemy, Pillow)...
pip install fastapi==0.143.3 uvicorn[standard]==0.32.0 python-multipart==0.0.18 aiofiles==25.1.0 SQLAlchemy==2.0.46 alembic==1.14.0 Pillow "numpy==1.26.4" scipy scikit-learn tqdm psutil python-dotenv requests --quiet
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Core dependency installation failed.
    pause
    exit /b 1
)

:: Install InsightFace
echo.
echo [3/6] Installing InsightFace (ArcFace face recognition)...
pip install insightface --quiet
IF %ERRORLEVEL% NEQ 0 (
    echo [WARN] InsightFace install failed. Trying alternative...
    pip install insightface --no-deps --quiet
)

:: Try ONNX Runtime GPU first (for GTX 1650 + CUDA 11.7)
echo.
echo [4/6] Installing ONNX Runtime (GPU preferred)...
python -c "import onnxruntime" >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo      ONNX Runtime already installed.
    goto onnx_done
)

pip install onnxruntime-gpu==1.16.3 --quiet
IF %ERRORLEVEL% NEQ 0 (
    echo [WARN] onnxruntime-gpu install failed. Installing CPU version...
    pip install onnxruntime==1.16.3 --quiet
)

:onnx_done

:: Install OpenCV (keep headless if already installed)
echo.
echo [5/6] Checking OpenCV installation...
python -c "import cv2" >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo      OpenCV already installed.
) ELSE (
    pip install opencv-python-headless --quiet
)

:: Download InsightFace models
echo.
echo [6/6] Pre-downloading InsightFace buffalo_l model (face detection + ArcFace)...
echo      This downloads ~300 MB. May take several minutes on first run.
echo.
python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=-1, det_size=(640,640)); print('Model download complete.')" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [WARN] Model download encountered an issue.
    echo       The application will retry on first launch.
) ELSE (
    echo [OK] Model ready.
)

:: Verify installation
echo.
echo ── Verifying installation ──
python setup_check.py
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARN] Some checks failed. See messages above.
    echo        The application may still work in CPU-only mode.
)

echo.
echo ==========================================================
echo   Setup complete!
echo   Run:  launch.bat   to start the application
echo ==========================================================
echo.
pause
