"""
Background scan worker.
Handles the main scanning loop: discover → detect → embed → classify → export.
Runs in a separate thread. Communicates via shared state (ScanState).
"""
import os
import json
import logging
import threading
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable

import numpy as np

from database import (
    get_session_factory, ScanJob, ImageFile, FaceDetection,
    MatchResult, ImageClassification, ErrorLog, Identity
)
from face_engine import (
    FaceRecognitionEngine, IdentityManager, ImageClassifier,
    DetectedFace, IdentityResult
)
from file_ops import (
    safe_copy_file, discover_images, get_file_mtime, get_file_size
)

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "tiff", "tif", "bmp", "webp", "heic", "heif", "raw", "cr2", "nef", "arw"]


class ScanState:
    """Thread-safe shared state for the scan worker."""

    def __init__(self):
        self._lock = threading.Lock()
        self.status = "idle"  # idle/running/paused/completed/cancelled/error
        self.total_files = 0
        self.processed_files = 0
        self.high_confidence_matches = 0
        self.review_candidates = 0
        self.skipped_files = 0
        self.failed_files = 0
        self.current_file = ""
        self.current_job_id: Optional[int] = None
        self.speed_fps = 0.0
        self.eta_seconds: Optional[float] = None
        self.last_error = ""
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused initially
        self._cancel_event = threading.Event()
        self._cancel_event.clear()
        self._speed_window: List[float] = []  # timestamps for speed calc

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "total_files": self.total_files,
                "processed_files": self.processed_files,
                "high_confidence_matches": self.high_confidence_matches,
                "review_candidates": self.review_candidates,
                "skipped_files": self.skipped_files,
                "failed_files": self.failed_files,
                "current_file": Path(self.current_file).name if self.current_file else "",
                "current_job_id": self.current_job_id,
                "speed_fps": round(self.speed_fps, 2),
                "eta_seconds": self.eta_seconds,
                "last_error": self.last_error,
                "progress_pct": round(
                    100.0 * self.processed_files / max(self.total_files, 1), 1
                ),
            }

    def pause(self):
        with self._lock:
            if self.status == "running":
                self.status = "paused"
        self._pause_event.clear()

    def resume(self):
        with self._lock:
            if self.status == "paused":
                self.status = "running"
        self._pause_event.set()

    def cancel(self):
        self._cancel_event.set()
        self._pause_event.set()  # unblock if paused
        with self._lock:
            self.status = "cancelled"

    def reset(self, job_id: Optional[int] = None):
        """Reset state cleanly for a new or resumed scan job."""
        with self._lock:
            self.status = "idle"
            self.total_files = 0
            self.processed_files = 0
            self.high_confidence_matches = 0
            self.review_candidates = 0
            self.skipped_files = 0
            self.failed_files = 0
            self.current_file = ""
            self.current_job_id = job_id
            self.speed_fps = 0.0
            self.eta_seconds = None
            self.last_error = ""
            self._pause_event.set()
            self._cancel_event.clear()
            self._speed_window.clear()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def wait_if_paused(self):
        self._pause_event.wait()

    def update_speed(self):
        now = time.monotonic()
        with self._lock:
            self._speed_window.append(now)
            # Keep only last 30 timestamps
            if len(self._speed_window) > 30:
                self._speed_window = self._speed_window[-30:]
            # FPS over last 30 images
            if len(self._speed_window) >= 2:
                elapsed = self._speed_window[-1] - self._speed_window[0]
                if elapsed > 0:
                    self.speed_fps = (len(self._speed_window) - 1) / elapsed
                    remaining = self.total_files - self.processed_files
                    if self.speed_fps > 0 and remaining > 0:
                        self.eta_seconds = remaining / self.speed_fps
                    else:
                        self.eta_seconds = None


class ScanWorker:
    """
    Main background worker that processes images.
    Runs detect → embed → classify → export pipeline for each image.
    """

    def __init__(
        self,
        engine: FaceRecognitionEngine,
        identity_manager: IdentityManager,
        db_path: str,
        state: ScanState,
        thumbnail_dir: str = "thumbnails",
    ):
        self.engine = engine
        self.identity_manager = identity_manager
        self.db_path = db_path
        self.state = state
        self.thumbnail_dir = thumbnail_dir
        self.classifier = ImageClassifier()
        self._thread: Optional[threading.Thread] = None

    def start(self, job_id: int):
        """Start the scan in a background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Scan already running.")
            return
        self.state._cancel_event.clear()
        self.state._pause_event.set()
        self._thread = threading.Thread(
            target=self._run, args=(job_id,), daemon=True
        )
        self._thread.start()

    def _run(self, job_id: int):
        """Main scan loop."""
        SessionLocal = get_session_factory(self.db_path)
        db = SessionLocal()

        try:
            job = db.query(ScanJob).filter_by(id=job_id).first()
            if not job:
                logger.error(f"Job {job_id} not found.")
                return

            with self.state._lock:
                self.state.status = "running"
                self.state.current_job_id = job_id

            # Mark job as running
            job.status = "running"
            job.start_time = job.start_time or datetime.utcnow()
            db.commit()

            source_folder = job.source_folder
            dest_high = job.dest_folder_high
            dest_review = job.dest_folder_review
            matching_rule = job.matching_rule
            formats = json.loads(job.image_formats)
            recursive = job.recursive_scan
            identity_config_hash = job.identity_config_hash

            target_ids = None
            if hasattr(job, "target_identities") and job.target_identities:
                try:
                    parsed_targets = json.loads(job.target_identities)
                    if isinstance(parsed_targets, list) and len(parsed_targets) > 0:
                        target_ids = parsed_targets
                except Exception:
                    pass

            # Discover unprocessed images
            # First: find all image files on disk
            exclude = [dest_high, dest_review]
            all_files = discover_images(source_folder, formats, recursive, exclude)

            with self.state._lock:
                self.state.total_files = len(all_files)

            job.total_files = len(all_files)
            db.commit()

            logger.info(f"Job {job_id}: discovered {len(all_files)} images (target identities: {target_ids or 'ALL'}).")

            # Register all files in DB (or skip already processed)
            self._register_files(db, job_id, all_files, source_folder,
                                  identity_config_hash)
            db.commit()

            # Count existing processed files
            already_done = db.query(ImageFile).filter(
                ImageFile.job_id == job_id,
                ImageFile.status == "done",
                ImageFile.identity_config_hash == identity_config_hash,
            ).count()

            with self.state._lock:
                self.state.processed_files = already_done
                self.state.high_confidence_matches = db.query(ImageClassification).filter(
                    ImageClassification.job_id == job_id,
                    ImageClassification.classification == "high_confidence",
                ).count()
                self.state.review_candidates = db.query(ImageClassification).filter(
                    ImageClassification.job_id == job_id,
                    ImageClassification.classification == "review_candidate",
                ).count()
                self.state.skipped_files = db.query(ImageClassification).filter(
                    ImageClassification.job_id == job_id,
                    ImageClassification.classification == "no_match",
                ).count()
                self.state.failed_files = db.query(ImageFile).filter(
                    ImageFile.job_id == job_id,
                    ImageFile.status == "failed",
                ).count()

            # Get pending files
            pending_files = db.query(ImageFile).filter(
                ImageFile.job_id == job_id,
                ImageFile.status.in_(["pending", "failed"]),
            ).order_by(ImageFile.id).all()

            logger.info(f"Job {job_id}: {len(pending_files)} pending, {already_done} already done.")

            for img_record in pending_files:
                if self.state.is_cancelled():
                    break
                self.state.wait_if_paused()
                if self.state.is_cancelled():
                    break

                file_path = img_record.file_path
                with self.state._lock:
                    self.state.current_file = file_path

                success = self._process_image(
                    db, job, img_record, dest_high, dest_review, matching_rule, target_ids=target_ids
                )

                with self.state._lock:
                    self.state.processed_files += 1
                    if not success:
                        self.state.failed_files += 1

                self.state.update_speed()

                # Periodic checkpoint (every 25 files)
                if self.state.processed_files % 25 == 0:
                    job.processed_files = self.state.processed_files
                    job.high_confidence_matches = self.state.high_confidence_matches
                    job.review_candidates = self.state.review_candidates
                    job.skipped_files = self.state.skipped_files
                    job.failed_files = self.state.failed_files
                    db.commit()

            # Final update
            final_status = "completed" if not self.state.is_cancelled() else "cancelled"
            job.status = final_status
            job.processed_files = self.state.processed_files
            job.high_confidence_matches = self.state.high_confidence_matches
            job.review_candidates = self.state.review_candidates
            job.skipped_files = self.state.skipped_files
            job.failed_files = self.state.failed_files
            job.end_time = datetime.utcnow()
            db.commit()

            with self.state._lock:
                self.state.status = final_status

            logger.info(f"Job {job_id} {final_status}. Matches: {self.state.high_confidence_matches}, Review: {self.state.review_candidates}")

        except Exception as e:
            logger.error(f"Scan worker crashed: {e}", exc_info=True)
            with self.state._lock:
                self.state.status = "error"
                self.state.last_error = str(e)
            try:
                job = db.query(ScanJob).filter_by(id=job_id).first()
                if job:
                    job.status = "error"
                    db.commit()
            except Exception:
                pass
        finally:
            db.close()

    def _register_files(self, db, job_id: int, all_files: List[str],
                         source_folder: str, identity_config_hash: str):
        """Register discovered files in DB. Skip already registered."""
        existing = {
            r.file_path for r in db.query(ImageFile.file_path).filter_by(job_id=job_id)
        }
        new_records = []
        for fp in all_files:
            if fp in existing:
                continue
            rel = os.path.relpath(fp, source_folder)
            new_records.append(ImageFile(
                job_id=job_id,
                file_path=fp,
                relative_path=rel,
                file_size=get_file_size(fp),
                file_mtime=get_file_mtime(fp),
                status="pending",
                identity_config_hash=identity_config_hash,
            ))
        if new_records:
            db.bulk_save_objects(new_records)

    def _process_image(
        self, db, job: ScanJob, img_record: ImageFile,
        dest_high: str, dest_review: str, matching_rule: str,
        target_ids: Optional[List[str]] = None
    ) -> bool:
        """
        Full pipeline for one image.
        Returns True if successfully processed (even if no match).
        """
        file_path = img_record.file_path

        # Check file accessibility
        if not os.path.isfile(file_path):
            img_record.status = "failed"
            img_record.processing_error = "File not found (drive disconnected?)"
            self._log_error(db, job.id, img_record.id, file_path,
                            "FileNotFound", img_record.processing_error, True)
            db.commit()
            return False

        img_record.status = "processing"
        db.commit()

        try:
            # ── Detection and embedding ────────────────────────────────────
            faces, img_w, img_h = self.engine.detect_and_embed(file_path)
            img_record.image_width = img_w
            img_record.image_height = img_h
            img_record.face_count = len(faces)
            img_record.model_version = self.engine.MODEL_NAME
            img_record.identity_config_hash = job.identity_config_hash

            # Save face detections to DB
            saved_face_ids: Dict[int, int] = {}  # face_index -> DB id
            for face in faces:
                fd = FaceDetection(
                    image_file_id=img_record.id,
                    face_index=face.face_index,
                    bbox_x=face.bbox[0],
                    bbox_y=face.bbox[1],
                    bbox_w=face.bbox[2],
                    bbox_h=face.bbox[3],
                    detection_confidence=face.confidence,
                    face_quality_score=face.quality_score,
                    is_small_face=face.is_small,
                    is_blurry=face.is_blurry,
                    embedding=(
                        face.embedding.tobytes() if face.embedding is not None else None
                    ),
                    embedding_model=self.engine.MODEL_NAME,
                    landmark_data=(
                        json.dumps(face.landmark5.tolist()) if face.landmark5 is not None else None
                    ),
                )
                db.add(fd)
                db.flush()
                saved_face_ids[face.face_index] = fd.id

                # Save aligned thumbnail
                if face.aligned_face is not None:
                    self._save_thumbnail(face.aligned_face, img_record.id, face.face_index)

            # ── Identity matching ─────────────────────────────────────────
            identity_results_by_face: Dict[int, List[IdentityResult]] = {}

            for face in faces:
                if face.embedding is None:
                    continue
                results = self.identity_manager.match_all(face.embedding, target_ids=target_ids)
                identity_results_by_face[face.face_index] = results

                # Save match results
                face_db_id = saved_face_ids.get(face.face_index)
                for r in results:
                    mr = MatchResult(
                        image_file_id=img_record.id,
                        face_detection_id=face_db_id or 0,
                        identity_id=r.identity_id,
                        similarity_score=r.similarity,
                        classification=r.classification,
                        threshold_used=r.threshold_high,
                        review_threshold_used=r.threshold_review,
                    )
                    db.add(mr)

            # ── Classification ────────────────────────────────────────────
            classification, matched_ids, best_score, has_warnings, warning_detail = \
                self.classifier.classify(faces, identity_results_by_face, matching_rule, target_ids=target_ids)

            # ── Save classification ───────────────────────────────────────
            # Check if classification record already exists
            existing_cls = db.query(ImageClassification).filter_by(
                image_file_id=img_record.id
            ).first()
            if existing_cls:
                cls_record = existing_cls
            else:
                cls_record = ImageClassification(
                    image_file_id=img_record.id,
                    job_id=job.id,
                )
                db.add(cls_record)

            cls_record.classification = classification
            cls_record.matched_identities = json.dumps(matched_ids)
            cls_record.best_score = best_score
            cls_record.has_quality_warnings = has_warnings
            cls_record.quality_warning_details = warning_detail
            cls_record.classified_at = datetime.utcnow()

            # ── Export ────────────────────────────────────────────────────
            if classification == "high_confidence":
                dest = dest_high
                dest_key = "high"
            elif classification == "review_candidate":
                dest = dest_review
                dest_key = "review"
            else:
                dest = None
                dest_key = None

            if dest:
                ok, dest_path, src_hash, dst_hash, err = safe_copy_file(
                    file_path, dest, verify_hash=True
                )
                cls_record.export_destination = dest_key
                cls_record.export_path = dest_path
                cls_record.export_status = "exported" if ok else "failed"
                cls_record.export_verified = ok and dst_hash == src_hash
                cls_record.source_hash = src_hash
                cls_record.dest_hash = dst_hash
                cls_record.export_error = err
                cls_record.exported_at = datetime.utcnow()

                if ok:
                    with self.state._lock:
                        if dest_key == "high":
                            self.state.high_confidence_matches += 1
                        else:
                            self.state.review_candidates += 1
                else:
                    self._log_error(db, job.id, img_record.id, file_path,
                                    "CopyError", err or "Unknown copy error", True)
            else:
                cls_record.export_status = "skipped"
                with self.state._lock:
                    self.state.skipped_files += 1

            img_record.status = "done"
            img_record.processed_at = datetime.utcnow()
            db.commit()
            return True

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            img_record.status = "failed"
            img_record.processing_error = str(e)[:512]
            self._log_error(db, job.id, img_record.id, file_path,
                            type(e).__name__, str(e)[:512], True)
            db.commit()
            return False

    def _save_thumbnail(self, aligned_face_bgr: np.ndarray,
                         image_file_id: int, face_index: int):
        """Save face thumbnail for UI display."""
        try:
            import cv2
            os.makedirs(self.thumbnail_dir, exist_ok=True)
            thumb_path = os.path.join(
                self.thumbnail_dir, f"{image_file_id}_{face_index}.jpg"
            )
            cv2.imwrite(thumb_path, aligned_face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception as e:
            logger.debug(f"Thumbnail save failed: {e}")

    def _log_error(self, db, job_id: int, image_file_id: Optional[int],
                   file_path: str, error_type: str, message: str,
                   recoverable: bool):
        try:
            err = ErrorLog(
                job_id=job_id,
                image_file_id=image_file_id,
                error_type=error_type,
                message=message,
                file_path=file_path,
                is_recoverable=recoverable,
            )
            db.add(err)
        except Exception:
            pass


def compute_identity_config_hash(
    identity_manager: IdentityManager,
    matching_rule: str,
) -> str:
    """Compute a hash representing the current identity configuration."""
    parts = []
    for iid in sorted(identity_manager.get_all_ids()):
        centroid = identity_manager.get_centroid_bytes(iid)
        parts.append(f"{iid}:{centroid.hex() if centroid else 'none'}")
    parts.append(f"rule:{matching_rule}")
    combined = "|".join(parts)
    return hashlib.md5(combined.encode()).hexdigest()
