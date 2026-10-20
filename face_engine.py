"""
Face Recognition Engine
Uses InsightFace (buffalo_l: RetinaFace detector + ArcFace recognizer)
via ONNX Runtime. Supports GPU (CUDAExecutionProvider) with CPU fallback.
"""
import os
import io
import json
import logging
import hashlib
import struct
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class DetectedFace:
    bbox: List[float]            # [x, y, w, h] in pixels
    confidence: float            # detection score 0-1
    embedding: Optional[np.ndarray]  # 512-dim ArcFace embedding
    landmark5: Optional[np.ndarray]  # 5 keypoints [[x,y],...]
    quality_score: float         # 0-1 composite quality
    is_small: bool               # face area < threshold
    is_blurry: bool
    face_index: int
    image_width: int
    image_height: int
    aligned_face: Optional[np.ndarray] = None  # 112x112 aligned crop for thumbnail

@dataclass
class IdentityResult:
    identity_id: str
    similarity: float            # cosine similarity 0-1
    classification: str          # high_confidence / review_candidate / no_match
    threshold_high: float
    threshold_review: float

@dataclass
class ImageScanResult:
    file_path: str
    faces: List[DetectedFace]
    identity_matches: Dict[str, List[IdentityResult]]  # face_index -> list of results
    classification: str          # high_confidence / review_candidate / no_match
    matched_identities: List[str]
    best_score: float
    has_quality_warnings: bool
    quality_warning_details: str
    error: Optional[str] = None
    image_width: int = 0
    image_height: int = 0


# ─── Hardware detection ───────────────────────────────────────────────────────

class HardwareInfo:
    def __init__(self):
        self.gpu_name = "Unknown"
        self.gpu_vram_mb = 0
        self.cuda_version = "N/A"
        self.onnx_providers: List[str] = []
        self.active_provider = "CPUExecutionProvider"
        self.system_ram_mb = 0
        self._detect()

    def _detect(self):
        import subprocess, re
        # RAM
        try:
            import psutil
            self.system_ram_mb = psutil.virtual_memory().total // (1024 * 1024)
        except Exception:
            self.system_ram_mb = 8192

        # Try ONNX Runtime providers
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            self.onnx_providers = available
            if "CUDAExecutionProvider" in available:
                self.active_provider = "CUDAExecutionProvider"
            else:
                self.active_provider = "CPUExecutionProvider"
        except ImportError:
            self.onnx_providers = ["CPUExecutionProvider"]
            self.active_provider = "CPUExecutionProvider"

        # GPU info via nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 2:
                    self.gpu_name = parts[0].strip()
                    self.gpu_vram_mb = int(parts[1].strip())
        except Exception:
            pass

        # CUDA version from nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                self.cuda_version = result.stdout.strip()
        except Exception:
            pass

    def to_dict(self) -> dict:
        return {
            "gpu_name": self.gpu_name,
            "gpu_vram_mb": self.gpu_vram_mb,
            "cuda_version": self.cuda_version,
            "onnx_providers": self.onnx_providers,
            "active_provider": self.active_provider,
            "system_ram_mb": self.system_ram_mb,
            "gpu_available": "CUDAExecutionProvider" in self.onnx_providers,
        }


# ─── InsightFace wrapper ──────────────────────────────────────────────────────

SMALL_FACE_THRESHOLD_PX = 40     # face height < 40px = small face
BLUR_LAPLACIAN_THRESHOLD = 80    # lower = more blurry
MIN_DETECTION_CONFIDENCE = 0.5
MIN_QUALITY_FOR_EMBEDDING = 0.2  # skip embedding if quality too low


class FaceRecognitionEngine:
    """
    Wraps InsightFace buffalo_l (RetinaFace + ArcFace).
    Thread-safe for single-threaded sequential use.
    Model is loaded once and reused.
    """

    MODEL_NAME = "buffalo_l"

    def __init__(self, hw_info: Optional[HardwareInfo] = None,
                 model_root: str = "~/.insightface"):
        self.hw_info = hw_info or HardwareInfo()
        self.model_root = os.path.expanduser(model_root)
        self._app = None
        self._lock = threading.Lock()
        self._model_loaded = False
        self._load_error: Optional[str] = None
        self.ctx_id = 0 if "CUDAExecutionProvider" in self.hw_info.onnx_providers else -1
        # ctx_id=0 uses GPU (CUDA), ctx_id=-1 forces CPU in InsightFace

    def load_model(self) -> bool:
        """Load the InsightFace model. Returns True on success."""
        with self._lock:
            if self._model_loaded:
                return True
            try:
                import insightface
                from insightface.app import FaceAnalysis
                logger.info(f"Loading InsightFace model: {self.MODEL_NAME}, ctx_id={self.ctx_id}")
                self._app = FaceAnalysis(
                    name=self.MODEL_NAME,
                    root=self.model_root,
                    providers=(
                        ["CUDAExecutionProvider", "CPUExecutionProvider"]
                        if self.ctx_id == 0
                        else ["CPUExecutionProvider"]
                    )
                )
                # det_size: detection resolution — 640x640 is standard
                self._app.prepare(ctx_id=self.ctx_id, det_size=(640, 640))
                self._model_loaded = True
                self._load_error = None
                logger.info("InsightFace model loaded successfully.")
                return True
            except Exception as e:
                self._load_error = str(e)
                logger.error(f"Failed to load face recognition model: {e}", exc_info=True)
                # Try CPU fallback
                if self.ctx_id == 0:
                    logger.warning("GPU load failed. Retrying with CPU...")
                    self.ctx_id = -1
                    return self.load_model()
                return False

    def is_ready(self) -> bool:
        return self._model_loaded

    def get_status(self) -> dict:
        return {
            "model_name": self.MODEL_NAME,
            "loaded": self._model_loaded,
            "error": self._load_error,
            "ctx_id": self.ctx_id,
            "provider": "CUDA/GPU" if self.ctx_id == 0 else "CPU",
        }

    # ── Image loading ─────────────────────────────────────────────────────────

    def load_image_safe(self, file_path: str,
                        max_dim: int = 4096) -> Tuple[Optional[np.ndarray], Tuple[int, int], float]:
        """
        Load image as BGR numpy array using fast native OpenCV C++ loader.
        Falls back to PIL for TIFF/HEIC or uncommon formats.
        Resizes very large images for detection only (originals are never modified).
        Returns (img_bgr, (orig_w, orig_h), scale).
        """
        try:
            # 1. Fast path: OpenCV via np.fromfile (handles unicode file paths reliably on Windows)
            img_bgr = None
            try:
                buf = np.fromfile(file_path, dtype=np.uint8)
                if buf is not None and len(buf) > 0:
                    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception:
                img_bgr = None

            if img_bgr is not None:
                h, w = img_bgr.shape[:2]
                scale = 1.0
                if max(w, h) > max_dim:
                    scale = max_dim / max(w, h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                return img_bgr, (w, h), scale

            # 2. Fallback path: PIL for other formats
            with Image.open(file_path) as pil_img:
                pil_img = pil_img.convert("RGB")
                w, h = pil_img.size

                scale = 1.0
                if max(w, h) > max_dim:
                    scale = max_dim / max(w, h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

                img_rgb = np.array(pil_img, dtype=np.uint8)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                return img_bgr, (w, h), scale
        except Exception as e:
            logger.warning(f"Failed to load image {file_path}: {e}")
            return None, (0, 0), 1.0

    # ── Face quality assessment ───────────────────────────────────────────────

    def _compute_face_quality(self, face_crop_bgr: np.ndarray,
                               det_confidence: float,
                               bbox: List[float]) -> Tuple[float, bool, bool]:
        """
        Returns (quality_score 0-1, is_blurry, is_profile).
        quality_score combines: detection confidence, size, sharpness, symmetry.
        """
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return 0.0, True, False

        # Blur: Laplacian variance
        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = lap_var < BLUR_LAPLACIAN_THRESHOLD
        blur_score = min(lap_var / 500.0, 1.0)  # normalize

        # Size score
        face_h = bbox[3]
        size_score = min(face_h / 80.0, 1.0)

        # Composite quality
        quality = 0.4 * det_confidence + 0.35 * blur_score + 0.25 * size_score
        return float(np.clip(quality, 0.0, 1.0)), is_blurry, False

    # ── Core detection + embedding ────────────────────────────────────────────

    def detect_and_embed(self, file_path: str) -> Tuple[List[DetectedFace], int, int]:
        """
        Run face detection and embedding on a single image.
        Returns (faces, image_width, image_height).
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        img_bgr, (orig_w, orig_h), scale = self.load_image_safe(file_path)
        if img_bgr is None:
            return [], orig_w, orig_h

        try:
            with self._lock:
                raw_faces = self._app.get(img_bgr)
        except Exception as e:
            logger.warning(f"Detection failed for {file_path}: {e}")
            return [], orig_w, orig_h

        detected_faces: List[DetectedFace] = []
        for idx, face in enumerate(raw_faces):
            if face.det_score < MIN_DETECTION_CONFIDENCE:
                continue

            # bbox in detection-image coords → rescale to original
            bbox_det = face.bbox  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox_det
            x1_orig = x1 / scale
            y1_orig = y1 / scale
            w_orig = (x2 - x1) / scale
            h_orig = (y2 - y1) / scale
            bbox = [float(x1_orig), float(y1_orig), float(w_orig), float(h_orig)]

            is_small = h_orig < SMALL_FACE_THRESHOLD_PX

            # Face crop for quality check (in detection-resolution space)
            x1i, y1i, x2i, y2i = (
                max(0, int(x1)), max(0, int(y1)),
                min(img_bgr.shape[1], int(x2)),
                min(img_bgr.shape[0], int(y2))
            )
            face_crop = img_bgr[y1i:y2i, x1i:x2i]

            quality, is_blurry, is_profile = self._compute_face_quality(
                face_crop, float(face.det_score), bbox
            )

            # Embedding (512-dim ArcFace)
            embedding = None
            if hasattr(face, "embedding") and face.embedding is not None:
                if quality >= MIN_QUALITY_FOR_EMBEDDING:
                    embedding = np.array(face.embedding, dtype=np.float32)
                    # Normalize to unit vector (cosine similarity)
                    norm = np.linalg.norm(embedding)
                    if norm > 1e-6:
                        embedding = embedding / norm

            # Landmarks (5-point)
            landmark5 = None
            if hasattr(face, "kps") and face.kps is not None:
                landmark5 = np.array(face.kps, dtype=np.float32)

            # Aligned face thumbnail (112x112)
            aligned = None
            if hasattr(face, "embedding") and face.embedding is not None:
                try:
                    from insightface.utils import face_align
                    if landmark5 is not None:
                        aligned = face_align.norm_crop(img_bgr, landmark=landmark5)
                except Exception:
                    pass

            df = DetectedFace(
                bbox=bbox,
                confidence=float(face.det_score),
                embedding=embedding,
                landmark5=landmark5,
                quality_score=quality,
                is_small=is_small,
                is_blurry=is_blurry,
                face_index=idx,
                image_width=orig_w,
                image_height=orig_h,
                aligned_face=aligned,
            )
            detected_faces.append(df)

        return detected_faces, orig_w, orig_h


    # ── Embedding utilities ───────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two unit-norm vectors."""
        dot = float(np.dot(a, b))
        return float(np.clip(dot, -1.0, 1.0))

    @staticmethod
    def embedding_to_bytes(emb: np.ndarray) -> bytes:
        """Serialize float32 numpy array to bytes."""
        return emb.astype(np.float32).tobytes()

    @staticmethod
    def bytes_to_embedding(data: bytes) -> np.ndarray:
        """Deserialize float32 numpy array from bytes."""
        return np.frombuffer(data, dtype=np.float32).copy()


# ─── Identity manager ─────────────────────────────────────────────────────────

class IdentityManager:
    """
    Manages per-identity reference embeddings.
    Uses a robust aggregation: median-based centroid, rejects outliers.
    """

    def __init__(self):
        self._identities: Dict[str, Dict] = {}
        # {identity_id: {"embeddings": [...], "centroid": np.ndarray,
        #                "high_threshold": float, "review_threshold": float,
        #                "display_name": str}}

    def add_identity(self, identity_id: str, display_name: str,
                     high_threshold: float = 0.45,
                     review_threshold: float = 0.35):
        if identity_id not in self._identities:
            self._identities[identity_id] = {
                "display_name": display_name,
                "embeddings": [],
                "centroid": None,
                "high_threshold": high_threshold,
                "review_threshold": review_threshold,
            }
        else:
            self._identities[identity_id]["high_threshold"] = high_threshold
            self._identities[identity_id]["review_threshold"] = review_threshold
            self._identities[identity_id]["display_name"] = display_name

    def add_embedding(self, identity_id: str, embedding: np.ndarray,
                      quality_score: float = 1.0):
        """Add a reference embedding for an identity."""
        if identity_id not in self._identities:
            raise ValueError(f"Identity '{identity_id}' not registered.")
        if embedding is None:
            return
        # Store with quality score
        self._identities[identity_id]["embeddings"].append(
            (embedding.copy(), quality_score)
        )
        self._recompute_centroid(identity_id)

    def _recompute_centroid(self, identity_id: str):
        """
        Compute robust centroid from all reference embeddings.
        - Filter out very low quality embeddings.
        - Filter outliers (embeddings with low similarity to peers).
        - Compute mean of the filtered set.
        """
        identity = self._identities[identity_id]
        embs_with_quality = identity["embeddings"]

        if not embs_with_quality:
            identity["centroid"] = None
            return

        # Quality filter: keep if quality >= 0.2
        good_embs = [e for e, q in embs_with_quality if q >= 0.2]

        if len(good_embs) == 0:
            # Fall back to all
            good_embs = [e for e, q in embs_with_quality]

        if len(good_embs) == 1:
            identity["centroid"] = good_embs[0].copy()
            return

        # Outlier rejection: if >2 embeddings, remove those with
        # mean similarity to all others below 0.3
        if len(good_embs) > 2:
            mean_sims = []
            for i, e_i in enumerate(good_embs):
                others = [e_j for j, e_j in enumerate(good_embs) if j != i]
                sims = [FaceRecognitionEngine.cosine_similarity(e_i, e_j) for e_j in others]
                mean_sims.append(np.mean(sims))

            sim_threshold = max(0.25, np.mean(mean_sims) - 0.15)
            filtered = [
                e for e, ms in zip(good_embs, mean_sims) if ms >= sim_threshold
            ]
            if len(filtered) >= 1:
                good_embs = filtered

        # Mean centroid, renormalized
        centroid = np.mean(np.stack(good_embs, axis=0), axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-6:
            centroid = centroid / norm
        identity["centroid"] = centroid

    def match_face(self, identity_id: str,
                   embedding: np.ndarray) -> IdentityResult:
        """Compare an embedding against a registered identity."""
        identity = self._identities.get(identity_id)
        if identity is None or identity["centroid"] is None:
            return IdentityResult(
                identity_id=identity_id,
                similarity=0.0,
                classification="no_match",
                threshold_high=0.45,
                threshold_review=0.35,
            )

        # Compare against centroid AND individual references → take max
        centroid_sim = FaceRecognitionEngine.cosine_similarity(
            embedding, identity["centroid"]
        )
        # Also compare against each individual reference (handles outliers in centroid)
        ref_sims = [
            FaceRecognitionEngine.cosine_similarity(embedding, e)
            for e, q in identity["embeddings"]
            if q >= 0.2
        ]
        best_sim = max([centroid_sim] + ref_sims) if ref_sims else centroid_sim

        high_th = identity["high_threshold"]
        review_th = identity["review_threshold"]

        if best_sim >= high_th:
            classification = "high_confidence"
        elif best_sim >= review_th:
            classification = "review_candidate"
        else:
            classification = "no_match"

        return IdentityResult(
            identity_id=identity_id,
            similarity=float(best_sim),
            classification=classification,
            threshold_high=high_th,
            threshold_review=review_th,
        )

    def match_all(self, embedding: np.ndarray,
                  target_ids: Optional[List[str]] = None) -> List[IdentityResult]:
        """Match an embedding against registered identities (or target_ids if specified)."""
        ids_to_check = target_ids if target_ids is not None else list(self._identities.keys())
        return [
            self.match_face(iid, embedding)
            for iid in ids_to_check
            if iid in self._identities
        ]

    def get_centroid_bytes(self, identity_id: str) -> Optional[bytes]:
        identity = self._identities.get(identity_id)
        if identity and identity["centroid"] is not None:
            return identity["centroid"].astype(np.float32).tobytes()
        return None

    def has_identity(self, identity_id: str) -> bool:
        return (identity_id in self._identities and
                self._identities[identity_id]["centroid"] is not None)

    def get_all_ids(self) -> List[str]:
        return list(self._identities.keys())

    def update_thresholds(self, identity_id: str,
                          high_threshold: float, review_threshold: float):
        if identity_id in self._identities:
            self._identities[identity_id]["high_threshold"] = high_threshold
            self._identities[identity_id]["review_threshold"] = review_threshold

    def clear_identity(self, identity_id: str):
        if identity_id in self._identities:
            self._identities[identity_id]["embeddings"] = []
            self._identities[identity_id]["centroid"] = None

    def remove_identity(self, identity_id: str):
        if identity_id in self._identities:
            del self._identities[identity_id]


# ─── Image classifier ─────────────────────────────────────────────────────────

class ImageClassifier:
    """
    Classifies a full image based on its detected faces and identity matches.
    Applies the user-selected matching rule.
    """

    def classify(
        self,
        faces: List[DetectedFace],
        identity_results_by_face: Dict[int, List[IdentityResult]],
        matching_rule: str = "any_one",
        target_ids: Optional[List[str]] = None,
    ) -> Tuple[str, List[str], float, bool, str]:
        """
        Returns (classification, matched_identities, best_score, has_warnings, warning_detail).
        classification: high_confidence / review_candidate / no_match
        """
        if not faces:
            return "no_match", [], 0.0, False, ""

        # Collect best result per identity across all faces
        best_per_identity: Dict[str, IdentityResult] = {}
        for face_idx, results in identity_results_by_face.items():
            for r in results:
                # If target_ids is specified, only consider target identities
                if target_ids is not None and r.identity_id not in target_ids:
                    continue
                prev = best_per_identity.get(r.identity_id)
                if prev is None or r.similarity > prev.similarity:
                    best_per_identity[r.identity_id] = r

        # Matched identities (any non-"no_match" classification)
        high_conf_ids = [
            iid for iid, r in best_per_identity.items()
            if r.classification == "high_confidence"
        ]
        review_ids = [
            iid for iid, r in best_per_identity.items()
            if r.classification == "review_candidate"
        ]
        all_matched_ids = high_conf_ids + review_ids

        best_score = max(
            (r.similarity for r in best_per_identity.values()), default=0.0
        )

        # Quality warnings
        warnings = []
        for face in faces:
            if face.is_small:
                warnings.append(f"Face {face.face_index}: too small ({face.bbox[3]:.0f}px)")
            if face.is_blurry:
                warnings.append(f"Face {face.face_index}: blurry (low sharpness)")
            if face.quality_score < 0.3:
                warnings.append(f"Face {face.face_index}: poor quality ({face.quality_score:.2f})")
        has_warnings = len(warnings) > 0
        warning_detail = "; ".join(warnings[:5])

        # Apply matching rule to determine required identities
        RULES = {
            "any_one": 1,
            "any_two": 2,
            "all_four": 4,
            "all_selected": 4,
        }
        required_count = RULES.get(matching_rule, 1)
        if target_ids is not None and len(target_ids) > 0:
            if matching_rule in ("all_four", "all_selected"):
                required_count = len(target_ids)
            else:
                required_count = min(required_count, len(target_ids))

        # Classification based on high-confidence matches
        if len(high_conf_ids) >= required_count:
            return "high_confidence", high_conf_ids, best_score, has_warnings, warning_detail

        # Check review: high_conf + review combined meets requirement
        if len(all_matched_ids) >= required_count:
            return "review_candidate", all_matched_ids, best_score, has_warnings, warning_detail

        # Partial: some review candidates but not enough for rule
        if review_ids:
            return "review_candidate", review_ids, best_score, has_warnings, warning_detail

        return "no_match", [], best_score, has_warnings, warning_detail
