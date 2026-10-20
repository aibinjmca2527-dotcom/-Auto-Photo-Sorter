"""
File copy and verification utilities.
Ensures byte-for-byte preservation of original images.
"""
import os
import io
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1 MB chunks for streaming copy


def sha256_file(file_path: str) -> str:
    """Compute SHA-256 hash of a file, reading in chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_copy_file(
    source_path: str,
    dest_folder: str,
    verify_hash: bool = True,
) -> Tuple[bool, str, Optional[str], Optional[str], Optional[str]]:
    """
    Copy a file to dest_folder, preserving original bytes.
    Returns:
        (success, dest_path, source_hash, dest_hash, error_message)

    Rules:
    - Never overwrites existing files silently.
    - Uses collision-safe naming if filename already exists.
    - Verifies file size and SHA-256 hash.
    - Does NOT modify source.
    """
    source_path = str(source_path)
    dest_folder = str(dest_folder)

    # Ensure destination folder exists
    os.makedirs(dest_folder, exist_ok=True)

    source_name = Path(source_path).name
    dest_path = os.path.join(dest_folder, source_name)

    # Collision-safe naming: append _{n} before extension if file exists
    if os.path.exists(dest_path):
        stem = Path(source_name).stem
        suffix = Path(source_name).suffix
        counter = 1
        while os.path.exists(dest_path):
            dest_path = os.path.join(dest_folder, f"{stem}_{counter}{suffix}")
            counter += 1
            if counter > 9999:
                return False, dest_path, None, None, "Too many filename collisions"

    # Compute source hash before copy
    source_hash = None
    try:
        source_size = os.path.getsize(source_path)
        if verify_hash:
            source_hash = sha256_file(source_path)
    except OSError as e:
        return False, dest_path, None, None, f"Cannot read source: {e}"

    # Copy file (stream in chunks, do NOT use shutil.copy2 which may modify metadata)
    try:
        with open(source_path, "rb") as src, open(dest_path, "wb") as dst:
            written = 0
            for chunk in iter(lambda: src.read(CHUNK_SIZE), b""):
                dst.write(chunk)
                written += len(chunk)
    except OSError as e:
        # Clean up partial file
        try:
            if os.path.exists(dest_path):
                os.remove(dest_path)
        except Exception:
            pass
        return False, dest_path, source_hash, None, f"Copy failed: {e}"

    # Verify size
    try:
        dest_size = os.path.getsize(dest_path)
    except OSError as e:
        return False, dest_path, source_hash, None, f"Cannot read dest: {e}"

    if dest_size != source_size:
        error = f"Size mismatch: source={source_size}, dest={dest_size}"
        logger.error(f"Copy integrity error for {source_path}: {error}")
        return False, dest_path, source_hash, None, error

    # Verify hash
    dest_hash = None
    if verify_hash and source_hash:
        try:
            dest_hash = sha256_file(dest_path)
        except OSError as e:
            return False, dest_path, source_hash, None, f"Hash check failed: {e}"

        if dest_hash != source_hash:
            error = f"Hash mismatch! source={source_hash[:16]}… dest={dest_hash[:16]}…"
            logger.error(f"Copy hash error for {source_path}: {error}")
            return False, dest_path, source_hash, dest_hash, error

    logger.info(f"Copied: {Path(source_path).name} → {dest_path} (size={dest_size}, hash verified={verify_hash})")
    return True, dest_path, source_hash, dest_hash, None


def discover_images(
    source_folder: str,
    formats: list,
    recursive: bool = True,
    exclude_folders: Optional[list] = None,
) -> list:
    """
    Discover all image files in source_folder.
    Returns list of absolute file paths.
    exclude_folders: list of absolute paths to skip.
    """
    source_folder = str(source_folder)
    formats_lower = {f.lower().lstrip(".") for f in formats}
    exclude_set = {os.path.normcase(str(p)) for p in (exclude_folders or [])}

    results = []

    def _should_exclude(path: str) -> bool:
        norm = os.path.normcase(path)
        for excl in exclude_set:
            if norm.startswith(excl):
                return True
        return False

    if recursive:
        for root, dirs, files in os.walk(source_folder, topdown=True):
            # Prune excluded directories
            dirs[:] = [
                d for d in dirs
                if not _should_exclude(os.path.join(root, d))
            ]
            if _should_exclude(root):
                continue
            for fname in files:
                ext = Path(fname).suffix.lower().lstrip(".")
                if ext in formats_lower:
                    results.append(os.path.join(root, fname))
    else:
        for fname in os.listdir(source_folder):
            fpath = os.path.join(source_folder, fname)
            if os.path.isfile(fpath):
                ext = Path(fname).suffix.lower().lstrip(".")
                if ext in formats_lower:
                    results.append(fpath)

    return results


def validate_folders(
    source: str,
    dest_high: str,
    dest_review: str,
) -> Tuple[bool, str]:
    """
    Validate the three folder selections.
    Returns (is_valid, error_message).
    """
    source = os.path.normcase(os.path.abspath(source))
    high = os.path.normcase(os.path.abspath(dest_high))
    review = os.path.normcase(os.path.abspath(dest_review))

    if not os.path.isdir(source):
        return False, f"Source folder does not exist: {source}"

    if source == high:
        return False, "High-confidence destination cannot be the same as source folder."
    if source == review:
        return False, "Review destination cannot be the same as source folder."
    if high == review:
        return False, "High-confidence and review destinations must be different folders."

    # Check destinations are not inside source
    if high.startswith(source + os.sep):
        return False, "High-confidence destination is inside source folder."
    if review.startswith(source + os.sep):
        return False, "Review destination is inside source folder."

    # Check source is not inside destinations
    if source.startswith(high + os.sep):
        return False, "Source is inside high-confidence destination folder."
    if source.startswith(review + os.sep):
        return False, "Source is inside review destination folder."

    return True, ""


def get_file_mtime(file_path: str) -> float:
    try:
        return os.path.getmtime(file_path)
    except OSError:
        return 0.0


def get_file_size(file_path: str) -> int:
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0
