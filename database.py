"""
AI Family Photo Sorter
Database models and initialization using SQLAlchemy + SQLite
"""
import json
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, Text, LargeBinary, Index, event
)
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import StaticPool
import logging

logger = logging.getLogger(__name__)
Base = declarative_base()


class ScanJob(Base):
    """Represents one scanning job/session."""
    __tablename__ = "scan_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    source_folder = Column(String(2048), nullable=False)
    dest_folder_high = Column(String(2048), nullable=False)
    dest_folder_review = Column(String(2048), nullable=False)
    status = Column(String(32), default="pending")  # pending/running/paused/completed/cancelled
    total_files = Column(Integer, default=0)
    processed_files = Column(Integer, default=0)
    high_confidence_matches = Column(Integer, default=0)
    review_candidates = Column(Integer, default=0)
    skipped_files = Column(Integer, default=0)
    failed_files = Column(Integer, default=0)
    model_name = Column(String(128), default="")
    identity_config_hash = Column(String(64), default="")  # MD5 of identity config, for cache invalidation
    threshold_config = Column(Text, default="{}")  # JSON
    matching_rule = Column(String(32), default="any_one")  # any_one/any_two/all_four
    target_identities = Column(Text, default="[]")  # JSON list of identity IDs to search for
    recursive_scan = Column(Boolean, default=True)
    image_formats = Column(Text, default='["jpg","jpeg","png","tiff","tif","bmp","webp","heic","heif"]')
    paused_at_file = Column(String(2048), default=None)  # last file when paused
    error_log = Column(Text, default="")
    bytes_processed = Column(Integer, default=0)
    start_time = Column(DateTime, default=None)
    end_time = Column(DateTime, default=None)


class ImageFile(Base):
    """Tracks each discovered image file."""
    __tablename__ = "image_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, nullable=False, index=True)
    file_path = Column(String(2048), nullable=False)
    relative_path = Column(String(2048), default="")
    file_size = Column(Integer, default=0)
    file_mtime = Column(Float, default=0.0)
    file_hash = Column(String(64), default=None)  # SHA-256 of file (optional, expensive)
    status = Column(String(32), default="pending")
    # pending / processing / done / failed / skipped
    face_count = Column(Integer, default=0)
    processing_error = Column(Text, default=None)
    model_version = Column(String(64), default="")
    identity_config_hash = Column(String(64), default="")
    processed_at = Column(DateTime, default=None)
    image_width = Column(Integer, default=0)
    image_height = Column(Integer, default=0)

    __table_args__ = (
        Index("ix_image_files_job_path", "job_id", "file_path"),
        Index("ix_image_files_status", "job_id", "status"),
    )


class FaceDetection(Base):
    """Stores each detected face within an image."""
    __tablename__ = "face_detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_file_id = Column(Integer, nullable=False, index=True)
    face_index = Column(Integer, default=0)  # index within image
    bbox_x = Column(Float, default=0)
    bbox_y = Column(Float, default=0)
    bbox_w = Column(Float, default=0)
    bbox_h = Column(Float, default=0)
    detection_confidence = Column(Float, default=0.0)
    face_quality_score = Column(Float, default=0.0)
    is_small_face = Column(Boolean, default=False)
    is_blurry = Column(Boolean, default=False)
    is_profile = Column(Boolean, default=False)
    embedding = Column(LargeBinary, default=None)  # numpy float32 array, raw bytes
    embedding_model = Column(String(64), default="")
    landmark_data = Column(Text, default=None)  # JSON string of 5-point landmarks

    __table_args__ = (
        Index("ix_face_detections_image", "image_file_id"),
    )


class MatchResult(Base):
    """Stores face-to-identity match scores."""
    __tablename__ = "match_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_file_id = Column(Integer, nullable=False, index=True)
    face_detection_id = Column(Integer, nullable=False)
    identity_id = Column(String(64), nullable=False)  # "me" / "mother" / "father" / "sister"
    similarity_score = Column(Float, default=0.0)
    classification = Column(String(32), default="no_match")
    # high_confidence / review_candidate / no_match
    threshold_used = Column(Float, default=0.0)
    review_threshold_used = Column(Float, default=0.0)
    matched_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_match_results_image", "image_file_id"),
        Index("ix_match_results_identity", "identity_id", "classification"),
    )


class ImageClassification(Base):
    """Final per-image classification and export decision."""
    __tablename__ = "image_classifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_file_id = Column(Integer, nullable=False, unique=True, index=True)
    job_id = Column(Integer, nullable=False, index=True)
    classification = Column(String(32), default="no_match")
    # high_confidence / review_candidate / no_match
    matched_identities = Column(Text, default="[]")  # JSON list of matched identity IDs
    best_score = Column(Float, default=0.0)
    has_quality_warnings = Column(Boolean, default=False)
    quality_warning_details = Column(Text, default="")
    # Export state
    export_status = Column(String(32), default="pending")
    # pending / exported / failed / skipped
    export_destination = Column(String(32), default=None)  # "high" / "review"
    export_path = Column(String(2048), default=None)
    export_verified = Column(Boolean, default=False)
    source_hash = Column(String(64), default=None)
    dest_hash = Column(String(64), default=None)
    export_error = Column(Text, default=None)
    # Manual review
    manual_reviewed = Column(Boolean, default=False)
    manual_decision = Column(String(32), default=None)
    # approved / rejected / corrected
    manual_corrected_identity = Column(String(64), default=None)
    review_notes = Column(Text, default=None)
    classified_at = Column(DateTime, default=datetime.utcnow)
    exported_at = Column(DateTime, default=None)

    __table_args__ = (
        Index("ix_image_class_job_class", "job_id", "classification"),
        Index("ix_image_class_export", "job_id", "export_status"),
    )


class Identity(Base):
    """Stores registered identity (family member) references."""
    __tablename__ = "identities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    identity_id = Column(String(64), nullable=False, unique=True)
    display_name = Column(String(256), default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    high_threshold = Column(Float, default=0.45)
    review_threshold = Column(Float, default=0.35)
    reference_count = Column(Integer, default=0)
    is_enabled = Column(Boolean, default=True)  # whether to include in scans
    aggregated_embedding = Column(LargeBinary, default=None)  # mean embedding


class ReferencePhoto(Base):
    """Individual reference photographs for an identity."""
    __tablename__ = "reference_photos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    identity_id = Column(String(64), nullable=False, index=True)
    file_path = Column(String(2048), nullable=False)
    file_size = Column(Integer, default=0)
    added_at = Column(DateTime, default=datetime.utcnow)
    face_detected = Column(Boolean, default=False)
    detection_confidence = Column(Float, default=0.0)
    face_quality_score = Column(Float, default=0.0)
    embedding = Column(LargeBinary, default=None)
    embedding_model = Column(String(64), default="")
    bbox = Column(Text, default=None)  # JSON
    is_active = Column(Boolean, default=True)  # can be disabled if bad quality
    quality_notes = Column(Text, default="")
    thumbnail_path = Column(String(2048), default=None)


class AppConfig(Base):
    """Key-value store for application configuration."""
    __tablename__ = "app_config"

    key = Column(String(256), primary_key=True)
    value = Column(Text, default="")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ErrorLog(Base):
    """Persistent error log entries."""
    __tablename__ = "error_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, default=None, index=True)
    image_file_id = Column(Integer, default=None)
    timestamp = Column(DateTime, default=datetime.utcnow)
    error_type = Column(String(128), default="")
    message = Column(Text, default="")
    file_path = Column(String(2048), default=None)
    is_recoverable = Column(Boolean, default=True)
    retry_count = Column(Integer, default=0)


# ─── Engine / Session factory ────────────────────────────────────────────────

_engine = None
_SessionLocal = None


def get_engine(db_path: str = "photo_sorter.db"):
    global _engine
    if _engine is None:
        url = f"sqlite:///{db_path}"
        _engine = create_engine(
            url,
            connect_args={"check_same_thread": False, "timeout": 30},
            poolclass=StaticPool,
        )
        # Enable WAL mode for concurrent reads + writes (crash-safe)
        @event.listens_for(_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        Base.metadata.create_all(_engine)

        # Ensure added columns exist in existing database files
        try:
            with _engine.connect() as conn:
                cursor = conn.connection.cursor()
                cursor.execute("PRAGMA table_info(scan_jobs)")
                sj_cols = [row[1] for row in cursor.fetchall()]
                if "target_identities" not in sj_cols:
                    cursor.execute("ALTER TABLE scan_jobs ADD COLUMN target_identities TEXT DEFAULT '[]'")

                cursor.execute("PRAGMA table_info(identities)")
                id_cols = [row[1] for row in cursor.fetchall()]
                if "is_enabled" not in id_cols:
                    cursor.execute("ALTER TABLE identities ADD COLUMN is_enabled BOOLEAN DEFAULT 1")
                conn.connection.commit()
                cursor.close()
        except Exception as e:
            logger.warning(f"Database migration notice: {e}")

        logger.info(f"Database initialized at: {db_path}")
    return _engine


def get_session_factory(db_path: str = "photo_sorter.db") -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine(db_path)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SessionLocal


def get_db(db_path: str = "photo_sorter.db"):
    """FastAPI dependency for database sessions."""
    SessionLocal = get_session_factory(db_path)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_default_identities(db: Session):
    """Create the four default family member identities if not present."""
    defaults = [
        ("me", "Me", 0.45, 0.35),
        ("mother", "Mother", 0.45, 0.35),
        ("father", "Father", 0.45, 0.35),
        ("sister", "Sister", 0.45, 0.35),
    ]
    for identity_id, display_name, high_thresh, review_thresh in defaults:
        existing = db.query(Identity).filter_by(identity_id=identity_id).first()
        if not existing:
            identity = Identity(
                identity_id=identity_id,
                display_name=display_name,
                high_threshold=high_thresh,
                review_threshold=review_thresh,
            )
            db.add(identity)
    db.commit()
    logger.info("Default identities initialized.")
