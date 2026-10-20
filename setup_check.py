"""
Setup verification script.
Checks all dependencies and hardware for the AI Photo Sorter.
"""
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

errors = []
warnings = []
ok = []

print("\n-- AI Family Photo Sorter - Dependency Check --\n")

# Python version
if sys.version_info < (3, 8):
    errors.append(f"Python 3.8+ required. Found: {sys.version}")
else:
    ok.append(f"Python {sys.version.split()[0]}")

# Core packages
packages = [
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("sqlalchemy", "SQLAlchemy"),
    ("PIL", "Pillow"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("cv2", "OpenCV"),
    ("aiofiles", "aiofiles"),
    ("psutil", "psutil"),
]

for mod, name in packages:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        ok.append(f"{name} {ver}")
    except ImportError:
        errors.append(f"{name} not installed. Run: pip install {mod}")

# InsightFace
try:
    import insightface
    ok.append(f"InsightFace {insightface.__version__}")
except ImportError:
    errors.append("InsightFace not installed. Run: pip install insightface")

# ONNX Runtime
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    has_gpu = "CUDAExecutionProvider" in providers
    ok.append(f"ONNX Runtime {ort.__version__} | GPU: {'YES' if has_gpu else 'NO (CPU only)'}")
    if not has_gpu:
        warnings.append("ONNX Runtime GPU not available. Will use CPU (slower but works).")
    else:
        ok.append("GPU inference via CUDAExecutionProvider: AVAILABLE")
except ImportError:
    errors.append("ONNX Runtime not installed. Run: pip install onnxruntime-gpu")

# GPU check (nvidia-smi)
import subprocess
try:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=5
    )
    if r.returncode == 0:
        ok.append(f"NVIDIA GPU: {r.stdout.strip()}")
    else:
        warnings.append("nvidia-smi failed. GPU may not be accessible.")
except Exception:
    warnings.append("nvidia-smi not found. GPU not accessible (will use CPU).")

# InsightFace model check
try:
    model_dir = os.path.expanduser("~/.insightface/models/buffalo_l")
    if os.path.isdir(model_dir):
        files = os.listdir(model_dir)
        ok.append(f"InsightFace buffalo_l model: {len(files)} files in {model_dir}")
    else:
        warnings.append(f"buffalo_l model not downloaded yet. Will download on first launch.")
except Exception as e:
    warnings.append(f"Model check failed: {e}")

# Print results
print("✓ OK:")
for msg in ok:
    print(f"    {msg}")

if warnings:
    print("\n⚠ Warnings:")
    for w in warnings:
        print(f"    {w}")

if errors:
    print("\n✗ Errors:")
    for e in errors:
        print(f"    {e}")
    print("\n[FAIL] Fix errors above before running the application.")
    sys.exit(1)
else:
    print("\n[PASS] All checks passed. Run launch.bat to start.\n")
    sys.exit(0)
