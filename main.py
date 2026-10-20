"""
AI Family Photo Sorter — FastAPI Backend
All endpoints for setup, scanning, gallery, review, and reports.
"""
import os
import re
import io
import json
import logging
import hashlib
import csv
import base64
import asyncio
from datetime import datetime
import mimetypes
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure correct MIME types on Windows (where .css often defaults to application/x-css)
mimetypes.init()
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("image/jpeg", ".jpg")
mimetypes.add_type("image/jpeg", ".jpeg")
mimetypes.add_type("image/png", ".png")

import numpy as np
import psutil
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import (
    FileResponse, JSONResponse, StreamingResponse, HTMLResponse
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from sqlalchemy.orm import Session

from database import (
    get_db, get_session_factory, get_engine, init_default_identities,
    ScanJob, ImageFile, FaceDetection, MatchResult, ImageClassification,
    Identity, ReferencePhoto, AppConfig, ErrorLog
)
from face_engine import FaceRecognitionEngine, IdentityManager, HardwareInfo
from scan_worker import ScanWorker, ScanState, compute_identity_config_hash
from file_ops import validate_folders, safe_copy_file, sha256_file

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("photo_sorter.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ─── Application state ────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "photo_sorter.db")
THUMBNAIL_DIR = os.path.join(os.path.dirname(__file__), "thumbnails")
REF_PHOTO_DIR = os.path.join(os.path.dirname(__file__), "reference_photos")

os.makedirs(THUMBNAIL_DIR, exist_ok=True)
os.makedirs(REF_PHOTO_DIR, exist_ok=True)

hw_info = HardwareInfo()
engine_instance = FaceRecognitionEngine(hw_info=hw_info)
identity_manager = IdentityManager()
scan_state = ScanState()
scan_worker: Optional[ScanWorker] = None

# Initialize DB
get_engine(DB_PATH)
SessionLocal = get_session_factory(DB_PATH)

# Initialize default identities in DB
_init_db = SessionLocal()
init_default_identities(_init_db)
_init_db.close()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup in background thread."""
    loop = asyncio.get_event_loop()

    async def _load():
        await loop.run_in_executor(None, engine_instance.load_model)
        await loop.run_in_executor(None, _reload_identities_from_db)

    asyncio.create_task(_load())
    yield

app = FastAPI(
    title="AI Family Photo Sorter",
    description="Local face recognition photo sorting application.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomStaticFiles(StaticFiles):
    """Ensure explicit, correct Content-Type headers on Windows."""
    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        path = args[0] if args else kwargs.get("full_path", "")
        p_str = str(path).lower()
        if p_str.endswith(".css"):
            resp.headers["content-type"] = "text/css; charset=utf-8"
        elif p_str.endswith(".js"):
            resp.headers["content-type"] = "application/javascript; charset=utf-8"
        elif p_str.endswith(".svg"):
            resp.headers["content-type"] = "image/svg+xml"
        return resp

# Serve static files (frontend)
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/static", CustomStaticFiles(directory=frontend_dir), name="static")

# Serve thumbnails
app.mount("/thumbnails", CustomStaticFiles(directory=THUMBNAIL_DIR), name="thumbnails")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>AI Family Photo Sorter</h1><p>Frontend not found.</p>")


def _reload_identities_from_db():
    """Reload all identity embeddings from DB into IdentityManager."""
    db = SessionLocal()
    try:
        identities = db.query(Identity).all()
        for ident in identities:
            identity_manager.add_identity(
                ident.identity_id,
                ident.display_name,
                ident.high_threshold,
                ident.review_threshold,
            )
            # Load individual reference embeddings
            refs = db.query(ReferencePhoto).filter_by(
                identity_id=ident.identity_id,
                is_active=True,
                face_detected=True,
            ).all()
            for ref in refs:
                if ref.embedding:
                    emb = np.frombuffer(ref.embedding, dtype=np.float32).copy()
                    identity_manager.add_embedding(
                        ident.identity_id, emb, ref.face_quality_score
                    )
        logger.info("Identities reloaded from DB.")
    finally:
        db.close()


def get_db_dep():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── Pydantic models ─────────────────────────────────────────────────────────

class FolderConfig(BaseModel):
    source_folder: str
    dest_folder_high: str
    dest_folder_review: str
    matching_rule: str = "any_one"
    recursive_scan: bool = True
    image_formats: List[str] = ["jpg", "jpeg", "png", "tiff", "tif", "bmp", "webp", "heic", "heif"]
    selected_identities: Optional[List[str]] = None  # None = search all ready; list = search only selected

class IdentityCreate(BaseModel):
    display_name: str
    identity_id: Optional[str] = None
    high_threshold: float = 0.45
    review_threshold: float = 0.35

class ThresholdUpdate(BaseModel):
    identity_id: str
    high_threshold: float
    review_threshold: float

class ReviewDecision(BaseModel):
    image_file_id: int
    decision: str  # approved / rejected / corrected
    corrected_identity: Optional[str] = None
    notes: Optional[str] = None


# ─── Hardware status ──────────────────────────────────────────────────────────

@app.get("/api/hardware")
async def get_hardware():
    """Return hardware and model status."""
    model_status = engine_instance.get_status()
    mem = psutil.virtual_memory()
    return {
        "hardware": hw_info.to_dict(),
        "model": model_status,
        "memory": {
            "total_mb": mem.total // (1024 * 1024),
            "available_mb": mem.available // (1024 * 1024),
            "percent_used": mem.percent,
        },
    }


# ─── Identity / Reference management ─────────────────────────────────────────

@app.get("/api/identities")
async def list_identities(db: Session = Depends(get_db_dep)):
    identities = db.query(Identity).all()
    result = []
    for ident in identities:
        ref_count = db.query(ReferencePhoto).filter_by(
            identity_id=ident.identity_id,
            is_active=True,
            face_detected=True,
        ).count()
        result.append({
            "identity_id": ident.identity_id,
            "display_name": ident.display_name,
            "high_threshold": ident.high_threshold,
            "review_threshold": ident.review_threshold,
            "reference_count": ref_count,
            "ready": ref_count > 0,
            "is_enabled": getattr(ident, "is_enabled", True),
        })
    return result


@app.post("/api/identities")
async def create_identity(data: IdentityCreate, db: Session = Depends(get_db_dep)):
    """Create a new custom family member identity."""
    name = data.display_name.strip()
    if not name:
        raise HTTPException(400, "Display name cannot be empty")

    raw_id = (data.identity_id or name).strip()
    clean_id = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_id.lower()).strip("_") or "member"

    # Ensure unique ID
    existing = db.query(Identity).filter_by(identity_id=clean_id).first()
    counter = 1
    base_id = clean_id
    while existing:
        clean_id = f"{base_id}_{counter}"
        existing = db.query(Identity).filter_by(identity_id=clean_id).first()
        counter += 1

    new_ident = Identity(
        identity_id=clean_id,
        display_name=name,
        high_threshold=data.high_threshold,
        review_threshold=data.review_threshold,
        reference_count=0,
        is_enabled=True,
    )
    db.add(new_ident)
    db.commit()

    # Register in IdentityManager
    identity_manager.add_identity(
        clean_id,
        new_ident.display_name,
        new_ident.high_threshold,
        new_ident.review_threshold,
    )
    logger.info(f"Created new identity: {clean_id} ({name})")

    return {
        "success": True,
        "identity": {
            "identity_id": new_ident.identity_id,
            "display_name": new_ident.display_name,
            "high_threshold": new_ident.high_threshold,
            "review_threshold": new_ident.review_threshold,
            "reference_count": 0,
            "ready": False,
            "is_enabled": True,
        }
    }


@app.delete("/api/identities/{identity_id}")
async def delete_identity(identity_id: str, db: Session = Depends(get_db_dep)):
    """Delete an identity and all its reference photos."""
    ident = db.query(Identity).filter_by(identity_id=identity_id).first()
    if not ident:
        raise HTTPException(404, f"Identity '{identity_id}' not found")

    # Remove references and files
    refs = db.query(ReferencePhoto).filter_by(identity_id=identity_id).all()
    for r in refs:
        try:
            if r.file_path and os.path.isfile(r.file_path):
                os.remove(r.file_path)
            if r.thumbnail_path and os.path.isfile(r.thumbnail_path):
                os.remove(r.thumbnail_path)
        except Exception:
            pass
        db.delete(r)

    db.delete(ident)
    db.commit()

    # Remove from in-memory engine
    identity_manager.remove_identity(identity_id)
    logger.info(f"Deleted identity: {identity_id}")

    return {"success": True, "deleted": identity_id}


@app.patch("/api/identities/{identity_id}/toggle")
async def toggle_identity(identity_id: str, db: Session = Depends(get_db_dep)):
    """Toggle whether an identity is active/enabled for search."""
    ident = db.query(Identity).filter_by(identity_id=identity_id).first()
    if not ident:
        raise HTTPException(404, f"Identity '{identity_id}' not found")

    ident.is_enabled = not getattr(ident, "is_enabled", True)
    db.commit()
    return {"success": True, "identity_id": identity_id, "is_enabled": ident.is_enabled}


@app.post("/api/identities/{identity_id}/references")
async def add_reference_photo(
    identity_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db_dep),
):
    """Upload a reference photo for an identity."""
    # Validate identity
    ident = db.query(Identity).filter_by(identity_id=identity_id).first()
    if not ident:
        raise HTTPException(404, f"Identity '{identity_id}' not found")

    if not engine_instance.is_ready():
        raise HTTPException(503, "Face recognition model not yet loaded. Please wait.")

    # Save uploaded file
    safe_name = f"{identity_id}_{hashlib.md5(file.filename.encode()).hexdigest()[:8]}_{file.filename}"
    save_path = os.path.join(REF_PHOTO_DIR, safe_name)

    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    # Detect face in reference
    try:
        faces, w, h = engine_instance.detect_and_embed(save_path)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(500, f"Face detection failed: {e}")

    if not faces:
        # Keep file but mark as failed detection
        ref = ReferencePhoto(
            identity_id=identity_id,
            file_path=save_path,
            file_size=len(contents),
            face_detected=False,
            quality_notes="No face detected in reference photo.",
        )
        db.add(ref)
        db.commit()
        return {
            "success": False,
            "message": "No face detected in the uploaded photo. Please use a clear, front-facing photo.",
            "ref_id": ref.id,
        }

    # Use the best quality face
    best_face = max(faces, key=lambda f: f.quality_score)

    quality_notes = []
    if best_face.is_blurry:
        quality_notes.append("Image appears blurry")
    if best_face.is_small:
        quality_notes.append("Face is small in the image")
    if best_face.quality_score < 0.3:
        quality_notes.append(f"Low quality score ({best_face.quality_score:.2f})")
    if len(faces) > 1:
        quality_notes.append(f"Multiple faces detected ({len(faces)}). Using highest-quality face.")

    # Save thumbnail
    thumb_path = None
    if best_face.aligned_face is not None:
        import cv2
        thumb_name = f"ref_{identity_id}_{hashlib.md5(save_path.encode()).hexdigest()[:8]}.jpg"
        thumb_path = os.path.join(THUMBNAIL_DIR, thumb_name)
        cv2.imwrite(thumb_path, best_face.aligned_face, [cv2.IMWRITE_JPEG_QUALITY, 90])

    ref = ReferencePhoto(
        identity_id=identity_id,
        file_path=save_path,
        file_size=len(contents),
        face_detected=True,
        detection_confidence=best_face.confidence,
        face_quality_score=best_face.quality_score,
        embedding=(
            best_face.embedding.tobytes() if best_face.embedding is not None else None
        ),
        embedding_model=engine_instance.MODEL_NAME,
        bbox=json.dumps(best_face.bbox),
        quality_notes="; ".join(quality_notes),
        thumbnail_path=thumb_path,
    )
    db.add(ref)
    db.commit()

    # Add to in-memory identity manager
    if best_face.embedding is not None:
        identity_manager.add_embedding(
            identity_id, best_face.embedding, best_face.quality_score
        )
        # Update centroid in DB
        centroid_bytes = identity_manager.get_centroid_bytes(identity_id)
        ident.aggregated_embedding = centroid_bytes
        ident.reference_count += 1
        db.commit()

    return {
        "success": True,
        "ref_id": ref.id,
        "face_count_in_image": len(faces),
        "quality_score": best_face.quality_score,
        "quality_notes": quality_notes,
        "face_detected": True,
    }


@app.delete("/api/identities/{identity_id}/references/{ref_id}")
async def delete_reference(
    identity_id: str, ref_id: int, db: Session = Depends(get_db_dep)
):
    ref = db.query(ReferencePhoto).filter_by(
        id=ref_id, identity_id=identity_id
    ).first()
    if not ref:
        raise HTTPException(404, "Reference not found")
    ref.is_active = False
    db.commit()

    # Rebuild identity embeddings from remaining active references
    identity_manager.clear_identity(identity_id)
    active_refs = db.query(ReferencePhoto).filter_by(
        identity_id=identity_id, is_active=True, face_detected=True
    ).all()
    for r in active_refs:
        if r.embedding:
            emb = np.frombuffer(r.embedding, dtype=np.float32).copy()
            identity_manager.add_embedding(identity_id, emb, r.face_quality_score)

    return {"success": True}


@app.get("/api/identities/{identity_id}/references")
async def list_references(identity_id: str, db: Session = Depends(get_db_dep)):
    refs = db.query(ReferencePhoto).filter_by(
        identity_id=identity_id, is_active=True
    ).order_by(ReferencePhoto.added_at).all()
    result = []
    for r in refs:
        thumb_url = None
        if r.thumbnail_path and os.path.isfile(r.thumbnail_path):
            thumb_url = f"/thumbnails/{Path(r.thumbnail_path).name}"
        result.append({
            "id": r.id,
            "file_path": r.file_path,
            "face_detected": r.face_detected,
            "quality_score": r.face_quality_score,
            "confidence": r.detection_confidence,
            "quality_notes": r.quality_notes,
            "thumbnail_url": thumb_url,
            "added_at": r.added_at.isoformat() if r.added_at else None,
        })
    return result


@app.put("/api/identities/{identity_id}/thresholds")
async def update_thresholds(
    identity_id: str,
    data: ThresholdUpdate,
    db: Session = Depends(get_db_dep),
):
    ident = db.query(Identity).filter_by(identity_id=identity_id).first()
    if not ident:
        raise HTTPException(404, "Identity not found")
    ident.high_threshold = data.high_threshold
    ident.review_threshold = data.review_threshold
    db.commit()
    identity_manager.update_thresholds(identity_id, data.high_threshold, data.review_threshold)
    return {"success": True}


# ─── Folder validation ────────────────────────────────────────────────────────

@app.post("/api/validate-folders")
async def validate_folders_endpoint(config: FolderConfig):
    valid, error = validate_folders(
        config.source_folder, config.dest_folder_high, config.dest_folder_review
    )
    return {"valid": valid, "error": error}


# ─── Folder browsing ──────────────────────────────────────────────────────────

class BrowseDirRequest(BaseModel):
    path: str = ""

class CreateFolderRequest(BaseModel):
    parent_path: str
    folder_name: str

@app.get("/api/browse/quick-paths")
async def get_quick_paths():
    """Return available drives and common user folder shortcuts."""
    drives = []
    try:
        for part in psutil.disk_partitions(all=False):
            dp = part.device
            try:
                usage = psutil.disk_usage(dp)
                free_gb = round(usage.free / (1024 ** 3), 1)
                total_gb = round(usage.total / (1024 ** 3), 1)
            except Exception:
                free_gb = 0
                total_gb = 0
            drives.append({
                "path": dp,
                "label": f"{dp} ({free_gb} GB free)" if free_gb else dp,
                "free_gb": free_gb,
            })
    except Exception as e:
        logger.warning(f"Error enumerating drives: {e}")
        if os.path.exists("C:\\"):
            drives.append({"path": "C:\\", "label": "C:\\", "free_gb": 0})

    home = os.path.expanduser("~")
    shortcuts = [
        {"name": "Pictures", "path": os.path.join(home, "Pictures"), "icon": "🖼️"},
        {"name": "Desktop", "path": os.path.join(home, "Desktop"), "icon": "🖥️"},
        {"name": "Downloads", "path": os.path.join(home, "Downloads"), "icon": "⬇️"},
        {"name": "Documents", "path": os.path.join(home, "Documents"), "icon": "📁"},
        {"name": "User Home", "path": home, "icon": "🏠"},
    ]
    existing_shortcuts = [s for s in shortcuts if os.path.isdir(s["path"])]

    default_path = home if os.path.isdir(home) else (drives[0]["path"] if drives else "C:\\")
    return {
        "drives": drives,
        "shortcuts": existing_shortcuts,
        "default_path": default_path,
    }


@app.post("/api/browse/list")
async def list_directory_contents(req: BrowseDirRequest):
    """List subfolders inside a given path for the in-browser folder picker."""
    target_path = req.path.strip() if req.path else ""
    if not target_path or not os.path.exists(target_path):
        target_path = os.path.expanduser("~")
        if not os.path.exists(target_path):
            target_path = "C:\\"

    target_path = os.path.abspath(target_path)
    parent = os.path.dirname(target_path)
    is_root = parent == target_path or not parent

    subfolders = []
    can_write = False
    try:
        # Check write permission
        test_file = os.path.join(target_path, f".test_perm_{os.getpid()}")
        try:
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
            can_write = True
        except Exception:
            can_write = False

        with os.scandir(target_path) as it:
            for entry in it:
                try:
                    if entry.is_dir():
                        name = entry.name
                        if not name.startswith((".", "$", "System Volume Information", "Recovery")):
                            subfolders.append({
                                "name": name,
                                "path": entry.path,
                            })
                except (PermissionError, OSError):
                    continue
        subfolders.sort(key=lambda x: x["name"].lower())
    except PermissionError:
        return {
            "current_path": target_path,
            "parent_path": parent if not is_root else None,
            "is_root": is_root,
            "subfolders": [],
            "can_write": False,
            "error": "Permission denied accessing this directory.",
        }
    except Exception as e:
        return {
            "current_path": target_path,
            "parent_path": parent if not is_root else None,
            "is_root": is_root,
            "subfolders": [],
            "can_write": False,
            "error": str(e),
        }

    return {
        "current_path": target_path,
        "parent_path": parent if not is_root else None,
        "is_root": is_root,
        "subfolders": subfolders,
        "can_write": can_write,
        "error": None,
    }


@app.post("/api/browse/create-folder")
async def create_new_folder(req: CreateFolderRequest):
    """Create a new directory for destination folder selection."""
    parent = req.parent_path.strip()
    name = req.folder_name.strip()
    if not parent or not os.path.isdir(parent):
        raise HTTPException(400, "Parent folder does not exist.")
    if not name or any(c in name for c in r'\/:*?"<>|'):
        raise HTTPException(400, 'Invalid folder name. Cannot contain \\ / : * ? " < > |')

    new_path = os.path.join(parent, name)
    try:
        os.makedirs(new_path, exist_ok=True)
        return {"success": True, "path": new_path}
    except Exception as e:
        raise HTTPException(500, f"Failed to create folder: {str(e)}")


def _run_native_picker_sync():
    """Run native Windows folder dialog with top-most activation in thread."""
    import subprocess
    script = """
[System.Reflection.Assembly]::LoadWithPartialName("System.windows.forms") | Out-Null
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = "Select Folder for AI Photo Sorter"
$dialog.ShowNewFolderButton = $true

$form = New-Object System.Windows.Forms.Form
$form.TopMost = $true
$form.WindowState = [System.Windows.Forms.FormWindowState]::Minimized
$form.Show()
$form.Activate()
$res = $dialog.ShowDialog($form)
$form.Dispose()

if ($res -eq [System.Windows.Forms.DialogResult]::OK) {
    Write-Output $dialog.SelectedPath
} else {
    Write-Output ""
}
"""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Sta", "-Command", script],
            capture_output=True, text=True, timeout=120
        )
        selected = result.stdout.strip()
        return {"path": selected, "cancelled": selected == ""}
    except subprocess.TimeoutExpired:
        return {"path": "", "cancelled": True, "error": "Dialog timed out"}
    except Exception as e:
        return {"path": "", "cancelled": True, "error": str(e)}


@app.post("/api/browse-folder")
async def browse_folder(data: dict = {}):
    """Open Windows folder picker dialog non-blocking in executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_native_picker_sync)


# ─── Scan management ──────────────────────────────────────────────────────────

@app.post("/api/scan/start")
async def start_scan(config: FolderConfig, db: Session = Depends(get_db_dep)):
    global scan_worker

    if scan_state.status == "running":
        raise HTTPException(400, "Scan already running.")

    if not engine_instance.is_ready():
        raise HTTPException(503, "Model not loaded. Please wait for model initialization.")

    # Validate folders
    valid, error = validate_folders(
        config.source_folder, config.dest_folder_high, config.dest_folder_review
    )
    if not valid:
        raise HTTPException(400, error)

    # Check identities are loaded and handle selected_identities
    ready_identities = [
        iid for iid in identity_manager.get_all_ids()
        if identity_manager.has_identity(iid)
    ]
    if config.selected_identities and len(config.selected_identities) > 0:
        target_list = [tid for tid in config.selected_identities if tid in ready_identities]
        if not target_list:
            raise HTTPException(400, "None of the selected family members have reference photos. Please upload photos first.")
    else:
        if not ready_identities:
            raise HTTPException(400, "No family member references loaded. Please add reference photos first.")
        target_list = ready_identities

    # Create destination folders
    os.makedirs(config.dest_folder_high, exist_ok=True)
    os.makedirs(config.dest_folder_review, exist_ok=True)

    # Compute identity config hash for cache invalidation
    identity_hash = compute_identity_config_hash(identity_manager, config.matching_rule)

    # Create or find resumable job
    existing_job = db.query(ScanJob).filter(
        ScanJob.source_folder == config.source_folder,
        ScanJob.status.in_(["paused", "pending"]),
    ).order_by(ScanJob.id.desc()).first()

    if existing_job and existing_job.identity_config_hash == identity_hash:
        job = existing_job
        job.status = "pending"
        job.target_identities = json.dumps(target_list)
        logger.info(f"Resuming job {job.id} for identities: {target_list}")
    else:
        job = ScanJob(
            source_folder=config.source_folder,
            dest_folder_high=config.dest_folder_high,
            dest_folder_review=config.dest_folder_review,
            matching_rule=config.matching_rule,
            target_identities=json.dumps(target_list),
            recursive_scan=config.recursive_scan,
            image_formats=json.dumps(config.image_formats),
            identity_config_hash=identity_hash,
        )
        db.add(job)
        logger.info(f"Created new scan job for identities: {target_list}")

    db.commit()

    # Reset state counters for new/resumed scan cleanly
    scan_state.reset(job_id=job.id)

    scan_worker = ScanWorker(
        engine=engine_instance,
        identity_manager=identity_manager,
        db_path=DB_PATH,
        state=scan_state,
        thumbnail_dir=THUMBNAIL_DIR,
    )
    scan_worker.start(job.id)

    return {"job_id": job.id, "status": "started"}


@app.post("/api/scan/pause")
async def pause_scan():
    if scan_state.status != "running":
        raise HTTPException(400, "No scan running.")
    scan_state.pause()
    return {"status": "paused"}


@app.post("/api/scan/resume")
async def resume_scan():
    if scan_state.status != "paused":
        raise HTTPException(400, "Scan is not paused.")
    scan_state.resume()
    return {"status": "running"}


@app.post("/api/scan/cancel")
async def cancel_scan():
    scan_state.cancel()
    return {"status": "cancelled"}


@app.get("/api/scan/status")
async def get_scan_status():
    state = scan_state.to_dict()
    mem = psutil.virtual_memory()
    state["memory_available_mb"] = mem.available // (1024 * 1024)
    state["memory_percent"] = mem.percent
    return state


# ─── Gallery endpoints ────────────────────────────────────────────────────────

def _build_image_card(img: ImageFile, cls: ImageClassification, db: Session) -> dict:
    """Build a gallery card dict for one image."""
    matched_ids = json.loads(cls.matched_identities) if cls.matched_identities else []

    # Thumbnail: find first face thumbnail for this image
    thumb_url = None
    face = db.query(FaceDetection).filter_by(image_file_id=img.id).first()
    if face:
        thumb_path = os.path.join(THUMBNAIL_DIR, f"{img.id}_0.jpg")
        if os.path.isfile(thumb_path):
            thumb_url = f"/thumbnails/{img.id}_0.jpg"

    return {
        "image_file_id": img.id,
        "file_path": img.file_path,
        "filename": Path(img.file_path).name,
        "classification": cls.classification,
        "matched_identities": matched_ids,
        "best_score": round(cls.best_score, 3),
        "has_quality_warnings": cls.has_quality_warnings,
        "quality_warning_details": cls.quality_warning_details,
        "export_status": cls.export_status,
        "export_path": cls.export_path,
        "export_verified": cls.export_verified,
        "face_count": img.face_count,
        "thumbnail_url": thumb_url,
        "image_width": img.image_width,
        "image_height": img.image_height,
        "manual_reviewed": cls.manual_reviewed,
        "manual_decision": cls.manual_decision,
    }


@app.get("/api/gallery/high-confidence")
async def gallery_high(
    page: int = 1, page_size: int = 24, job_id: Optional[int] = None,
    db: Session = Depends(get_db_dep)
):
    query = db.query(ImageClassification, ImageFile).join(
        ImageFile, ImageClassification.image_file_id == ImageFile.id
    ).filter(ImageClassification.classification == "high_confidence")

    if job_id:
        query = query.filter(ImageClassification.job_id == job_id)

    total = query.count()
    rows = query.order_by(ImageClassification.id.desc()).offset(
        (page - 1) * page_size
    ).limit(page_size).all()

    items = [_build_image_card(img, cls, db) for cls, img in rows]
    return {"total": total, "page": page, "page_size": page_size, "items": items}


@app.get("/api/gallery/review")
async def gallery_review(
    page: int = 1, page_size: int = 24, job_id: Optional[int] = None,
    db: Session = Depends(get_db_dep)
):
    query = db.query(ImageClassification, ImageFile).join(
        ImageFile, ImageClassification.image_file_id == ImageFile.id
    ).filter(
        ImageClassification.classification == "review_candidate",
        ImageClassification.manual_reviewed == False,
    )
    if job_id:
        query = query.filter(ImageClassification.job_id == job_id)

    total = query.count()
    rows = query.order_by(ImageClassification.id.desc()).offset(
        (page - 1) * page_size
    ).limit(page_size).all()

    items = []
    for cls, img in rows:
        card = _build_image_card(img, cls, db)
        # Add face-level details for review
        faces_detail = []
        face_records = db.query(FaceDetection, MatchResult).join(
            MatchResult, FaceDetection.id == MatchResult.face_detection_id, isouter=True
        ).filter(FaceDetection.image_file_id == img.id).all()

        seen_faces = set()
        for fd, mr in face_records:
            if fd.face_index not in seen_faces:
                seen_faces.add(fd.face_index)
                face_info = {
                    "face_index": fd.face_index,
                    "confidence": round(fd.detection_confidence, 3),
                    "quality_score": round(fd.face_quality_score, 3),
                    "is_small": fd.is_small_face,
                    "is_blurry": fd.is_blurry,
                    "bbox": [fd.bbox_x, fd.bbox_y, fd.bbox_w, fd.bbox_h],
                    "thumbnail_url": None,
                    "matches": [],
                }
                thumb_path = os.path.join(THUMBNAIL_DIR, f"{img.id}_{fd.face_index}.jpg")
                if os.path.isfile(thumb_path):
                    face_info["thumbnail_url"] = f"/thumbnails/{img.id}_{fd.face_index}.jpg"
                faces_detail.append(face_info)

            # Add match info
            if mr and faces_detail:
                target_face = next(
                    (f for f in faces_detail if f["face_index"] == fd.face_index), None
                )
                if target_face:
                    target_face["matches"].append({
                        "identity_id": mr.identity_id,
                        "similarity": round(mr.similarity_score, 3),
                        "classification": mr.classification,
                    })

        card["faces_detail"] = faces_detail
        items.append(card)

    return {"total": total, "page": page, "page_size": page_size, "items": items}


@app.post("/api/review/decide")
async def make_review_decision(
    decision: ReviewDecision, db: Session = Depends(get_db_dep)
):
    cls = db.query(ImageClassification).filter_by(
        image_file_id=decision.image_file_id
    ).first()
    if not cls:
        raise HTTPException(404, "Classification record not found")

    img = db.query(ImageFile).filter_by(id=decision.image_file_id).first()
    if not img:
        raise HTTPException(404, "Image record not found")

    cls.manual_reviewed = True
    cls.manual_decision = decision.decision
    cls.manual_corrected_identity = decision.corrected_identity
    cls.review_notes = decision.notes

    if decision.decision == "approved":
        # Already exported to review folder — no further action needed
        pass
    elif decision.decision == "rejected":
        # Mark as no_match going forward
        cls.classification = "no_match"
    elif decision.decision == "corrected":
        # Promote to high_confidence and copy to high folder if not already there
        cls.classification = "high_confidence"
        job = db.query(ScanJob).filter_by(id=cls.job_id).first()
        if job and cls.export_status != "exported" or cls.export_destination != "high":
            ok, dest_path, src_hash, dst_hash, err = safe_copy_file(
                img.file_path, job.dest_folder_high, verify_hash=True
            )
            if ok:
                cls.export_destination = "high"
                cls.export_path = dest_path
                cls.export_status = "exported"
                cls.export_verified = True
                cls.source_hash = src_hash
                cls.dest_hash = dst_hash
                cls.exported_at = datetime.utcnow()
            else:
                cls.export_error = err

    db.commit()
    return {"success": True}


# ─── Preview / thumbnail ──────────────────────────────────────────────────────

@app.get("/api/preview/{image_file_id}")
async def get_preview(image_file_id: int, db: Session = Depends(get_db_dep)):
    """Return a downscaled JPEG preview of an image for the browser."""
    img = db.query(ImageFile).filter_by(id=image_file_id).first()
    if not img:
        raise HTTPException(404, "Image not found")

    file_path = img.file_path
    if not os.path.isfile(file_path):
        raise HTTPException(404, "File not found on disk")

    import cv2
    from PIL import Image as PILImage

    try:
        with PILImage.open(file_path) as pil_img:
            pil_img.thumbnail((1200, 1200), PILImage.LANCZOS)
            buf = io.BytesIO()
            pil_img.convert("RGB").save(buf, format="JPEG", quality=85)
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(500, f"Preview failed: {e}")


# ─── Reports ──────────────────────────────────────────────────────────────────

@app.get("/api/reports/summary")
async def report_summary(job_id: Optional[int] = None, db: Session = Depends(get_db_dep)):
    query = db.query(ScanJob)
    if job_id:
        query = query.filter_by(id=job_id)
    jobs = query.order_by(ScanJob.id.desc()).limit(10).all()

    result = []
    for job in jobs:
        duration = None
        if job.start_time and job.end_time:
            duration = (job.end_time - job.start_time).total_seconds()

        result.append({
            "job_id": job.id,
            "status": job.status,
            "source_folder": job.source_folder,
            "total_files": job.total_files,
            "processed_files": job.processed_files,
            "high_confidence_matches": job.high_confidence_matches,
            "review_candidates": job.review_candidates,
            "skipped_files": job.skipped_files,
            "failed_files": job.failed_files,
            "matching_rule": job.matching_rule,
            "duration_seconds": duration,
            "created_at": job.created_at.isoformat() if job.created_at else None,
        })
    return result


@app.get("/api/reports/export-csv")
async def export_csv(job_id: Optional[int] = None, db: Session = Depends(get_db_dep)):
    """Export classification results as CSV."""
    query = db.query(ImageClassification, ImageFile).join(
        ImageFile, ImageClassification.image_file_id == ImageFile.id
    )
    if job_id:
        query = query.filter(ImageClassification.job_id == job_id)

    rows = query.all()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "image_file_id", "filename", "file_path", "classification",
        "matched_identities", "best_score", "export_status",
        "export_destination", "export_path", "export_verified",
        "manual_reviewed", "manual_decision", "has_quality_warnings"
    ])
    for cls, img in rows:
        writer.writerow([
            img.id, Path(img.file_path).name, img.file_path,
            cls.classification,
            json.loads(cls.matched_identities) if cls.matched_identities else [],
            round(cls.best_score, 4),
            cls.export_status, cls.export_destination, cls.export_path,
            cls.export_verified, cls.manual_reviewed, cls.manual_decision,
            cls.has_quality_warnings,
        ])

    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode("utf-8-sig")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=photo_sorter_results.csv"},
    )


@app.get("/api/reports/errors")
async def get_errors(job_id: Optional[int] = None, db: Session = Depends(get_db_dep)):
    query = db.query(ErrorLog)
    if job_id:
        query = query.filter_by(job_id=job_id)
    errors = query.order_by(ErrorLog.timestamp.desc()).limit(200).all()
    return [
        {
            "id": e.id,
            "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            "error_type": e.error_type,
            "message": e.message,
            "file_path": e.file_path,
            "is_recoverable": e.is_recoverable,
        }
        for e in errors
    ]


@app.get("/api/reports/jobs")
async def list_jobs(db: Session = Depends(get_db_dep)):
    jobs = db.query(ScanJob).order_by(ScanJob.id.desc()).limit(20).all()
    return [
        {
            "id": j.id,
            "status": j.status,
            "source_folder": j.source_folder,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "total_files": j.total_files,
            "processed_files": j.processed_files,
        }
        for j in jobs
    ]


# ─── Clear cached data ────────────────────────────────────────────────────────

@app.post("/api/cache/clear-embeddings")
async def clear_embeddings(db: Session = Depends(get_db_dep)):
    """Delete all cached face embeddings from DB (does not affect source or exported files)."""
    db.query(FaceDetection).update({"embedding": None})
    db.query(MatchResult).delete()
    db.query(ImageClassification).delete()
    db.query(ImageFile).update({"status": "pending"})
    db.commit()
    return {"success": True, "message": "All cached face data cleared. Re-scan to rebuild."}


# ─── Settings ─────────────────────────────────────────────────────────────────

@app.get("/api/settings")
async def get_settings(db: Session = Depends(get_db_dep)):
    configs = db.query(AppConfig).all()
    return {c.key: c.value for c in configs}


@app.post("/api/settings")
async def save_settings(data: dict, db: Session = Depends(get_db_dep)):
    for key, value in data.items():
        existing = db.query(AppConfig).filter_by(key=key).first()
        if existing:
            existing.value = str(value)
        else:
            db.add(AppConfig(key=key, value=str(value)))
    db.commit()
    return {"success": True}


# ─── Root — serve frontend ────────────────────────────────────────────────────

@app.get("/")
async def root():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>AI Photo Sorter Backend Running</h1><p>Frontend not found.</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
