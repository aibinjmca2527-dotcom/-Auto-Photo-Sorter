"""
AI Family Photo Sorter - Smart 1-Click Launcher
Automatically handles:
1. Dependency checks & automated installation from requirements.txt
2. Model verification & automated pre-download (buffalo_l)
3. Dynamic port allocation (finds free port if 8000 is busy)
4. Automated browser launch
5. Graceful fallback & error diagnostics
"""
import os
import sys
import socket
import subprocess
import time
import threading
import webbrowser


def print_banner():
    print("=" * 60)
    print("   AI Family Photo Sorter - Smart 1-Click Launcher")
    print("   100% Local, Private & Zero-Touch Safe")
    print("=" * 60)
    print()


def check_and_install_dependencies():
    """Verify that all core dependencies are installed; install if missing."""
    print("[1/4] Checking Python environment & dependencies...")
    modules_to_check = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "uvicorn"),
        ("sqlalchemy", "SQLAlchemy"),
        ("PIL", "Pillow"),
        ("numpy", "numpy"),
        ("cv2", "OpenCV"),
        ("insightface", "InsightFace"),
        ("onnxruntime", "ONNX Runtime"),
    ]

    missing = []
    for mod_name, label in modules_to_check:
        try:
            __import__(mod_name)
        except ImportError:
            missing.append(label)

    if missing:
        print(f"      Missing packages detected: {', '.join(missing)}")
        print("      Automatically installing dependencies (this is a one-time process)...")
        print("      Please wait 1-3 minutes...")
        print()

        req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
        if os.path.exists(req_file):
            cmd = [sys.executable, "-m", "pip", "install", "-r", req_file]
        else:
            cmd = [
                sys.executable, "-m", "pip", "install",
                "fastapi", "uvicorn[standard]", "SQLAlchemy", "Pillow",
                "numpy<2.0.0", "opencv-python-headless", "insightface",
                "onnxruntime", "scipy", "scikit-learn", "tqdm", "psutil"
            ]

        try:
            subprocess.check_call(cmd)
            print("      [OK] Dependencies installed successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Automated installation failed (code {e.returncode}).")
            print("Please try running setup.bat manually or execute: pip install -r requirements.txt")
            sys.exit(1)
    else:
        print("      [OK] All core dependencies are installed.\n")


def check_and_download_model():
    """Verify InsightFace buffalo_l model exists; pre-download if missing."""
    print("[2/4] Checking AI face recognition model (buffalo_l)...")
    model_dir = os.path.expanduser(os.path.join("~", ".insightface", "models", "buffalo_l"))
    if not os.path.exists(model_dir):
        print("      Model weights not found locally.")
        print("      Pre-downloading InsightFace buffalo_l model (~300 MB)...")
        print("      This is a one-time download and ensures 100% offline capability.")
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=-1, det_size=(640, 640))
            print("      [OK] AI model ready.\n")
        except Exception as e:
            print(f"      [WARN] Model pre-download encountered: {e}")
            print("             The application will attempt to download on first scan.\n")
    else:
        print("      [OK] Model weights present locally.\n")


def find_available_port(start_port=8000, max_attempts=50):
    """Find a free port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port


def open_browser_delayed(url, delay=2.0):
    """Open user's default browser after server initialization."""
    def _open():
        time.sleep(delay)
        print(f"\n>>> Opening your browser at: {url}\n")
        webbrowser.open(url)

    t = threading.Thread(target=_open, daemon=True)
    t.start()


def main():
    print_banner()

    # 1. Dependency check & auto install
    check_and_install_dependencies()

    # 2. Model check & auto download
    check_and_download_model()

    # 3. Port allocation
    print("[3/4] Allocating server port...")
    target_port = 8000
    free_port = find_available_port(start_port=target_port)
    if free_port != target_port:
        print(f"      Port {target_port} is currently in use by another application.")
        print(f"      [OK] Assigned free port: {free_port}")
    else:
        print(f"      [OK] Assigned port: {free_port}")
    print()

    # 4. Launch server & browser
    url = f"http://127.0.0.1:{free_port}"
    print("[4/4] Starting AI Family Photo Sorter server...")
    print(f"      Server URL: {url}")
    print("      Press Ctrl+C in this terminal window to stop the server.\n")

    open_browser_delayed(url, delay=2.5)

    import uvicorn
    # Import app and start server
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=free_port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
