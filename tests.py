"""
Automated test suite for AI Family Photo Sorter.
Tests face engine, file operations, database, and API endpoints.
"""
import os
import sys
import io
import json
import tempfile
import shutil
import hashlib
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestFaceEngine(unittest.TestCase):
    """Test the face recognition engine components."""

    def test_cosine_similarity_identical(self):
        from face_engine import FaceRecognitionEngine
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(FaceRecognitionEngine.cosine_similarity(a, a), 1.0, places=5)

    def test_cosine_similarity_orthogonal(self):
        from face_engine import FaceRecognitionEngine
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(FaceRecognitionEngine.cosine_similarity(a, b), 0.0, places=5)

    def test_cosine_similarity_opposite(self):
        from face_engine import FaceRecognitionEngine
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(FaceRecognitionEngine.cosine_similarity(a, b), -1.0, places=5)

    def test_embedding_serialization(self):
        from face_engine import FaceRecognitionEngine
        emb = np.random.randn(512).astype(np.float32)
        data = FaceRecognitionEngine.embedding_to_bytes(emb)
        restored = FaceRecognitionEngine.bytes_to_embedding(data)
        np.testing.assert_array_almost_equal(emb, restored, decimal=5)

    def test_embedding_serialization_roundtrip(self):
        from face_engine import FaceRecognitionEngine
        emb = np.ones(512, dtype=np.float32)
        data = FaceRecognitionEngine.embedding_to_bytes(emb)
        self.assertEqual(len(data), 512 * 4)
        restored = FaceRecognitionEngine.bytes_to_embedding(data)
        self.assertEqual(restored.dtype, np.float32)


class TestIdentityManager(unittest.TestCase):
    """Test identity registration and matching."""

    def setUp(self):
        from face_engine import IdentityManager
        self.manager = IdentityManager()
        self.manager.add_identity("me", "Me", 0.45, 0.35)
        self.manager.add_identity("mother", "Mother", 0.45, 0.35)

    def test_add_single_embedding(self):
        emb = np.ones(512, dtype=np.float32)
        emb /= np.linalg.norm(emb)
        self.manager.add_embedding("me", emb, quality_score=0.9)
        self.assertTrue(self.manager.has_identity("me"))

    def test_match_identical_high_confidence(self):
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        self.manager.add_embedding("me", emb, quality_score=0.9)
        result = self.manager.match_face("me", emb)
        self.assertEqual(result.classification, "high_confidence")
        self.assertGreater(result.similarity, 0.99)

    def test_match_orthogonal_no_match(self):
        emb_a = np.zeros(512, dtype=np.float32)
        emb_a[0] = 1.0
        emb_b = np.zeros(512, dtype=np.float32)
        emb_b[1] = 1.0
        self.manager.add_embedding("me", emb_a, quality_score=0.9)
        result = self.manager.match_face("me", emb_b)
        self.assertEqual(result.classification, "no_match")

    def test_no_cross_contamination(self):
        """Embeddings for 'me' should not affect 'mother'."""
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        self.manager.add_embedding("me", emb, quality_score=0.9)
        # Mother still has no embeddings
        self.assertFalse(self.manager.has_identity("mother"))

    def test_centroid_single_ref(self):
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        self.manager.add_embedding("me", emb, quality_score=0.9)
        centroid_bytes = self.manager.get_centroid_bytes("me")
        self.assertIsNotNone(centroid_bytes)
        centroid = np.frombuffer(centroid_bytes, dtype=np.float32)
        np.testing.assert_array_almost_equal(centroid, emb, decimal=5)

    def test_centroid_multiple_refs(self):
        embs = [np.random.randn(512).astype(np.float32) for _ in range(5)]
        embs = [e / np.linalg.norm(e) for e in embs]
        for e in embs:
            self.manager.add_embedding("me", e, quality_score=0.8)
        centroid_bytes = self.manager.get_centroid_bytes("me")
        self.assertIsNotNone(centroid_bytes)
        centroid = np.frombuffer(centroid_bytes, dtype=np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0, places=4)

    def test_threshold_update(self):
        self.manager.update_thresholds("me", 0.55, 0.40)
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        self.manager.add_embedding("me", emb, quality_score=0.9)
        result = self.manager.match_face("me", emb)
        self.assertEqual(result.threshold_high, 0.55)
        self.assertEqual(result.threshold_review, 0.40)

    def test_match_all_returns_all_identities(self):
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        self.manager.add_embedding("me", emb, quality_score=0.9)
        results = self.manager.match_all(emb)
        self.assertEqual(len(results), 2)  # me and mother
        ids = {r.identity_id for r in results}
        self.assertIn("me", ids)
        self.assertIn("mother", ids)

    def test_clear_identity(self):
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        self.manager.add_embedding("me", emb, quality_score=0.9)
        self.assertTrue(self.manager.has_identity("me"))
        self.manager.clear_identity("me")
        self.assertFalse(self.manager.has_identity("me"))


class TestImageClassifier(unittest.TestCase):
    """Test image classification logic."""

    def setUp(self):
        from face_engine import ImageClassifier, DetectedFace, IdentityResult
        self.classifier = ImageClassifier()
        self.DetectedFace = DetectedFace
        self.IdentityResult = IdentityResult

    def _make_face(self, idx=0, quality=0.8, is_small=False, is_blurry=False):
        return self.DetectedFace(
            bbox=[100, 100, 60, 80],
            confidence=0.95,
            embedding=np.ones(512, dtype=np.float32),
            landmark5=None,
            quality_score=quality,
            is_small=is_small,
            is_blurry=is_blurry,
            face_index=idx,
            image_width=1920,
            image_height=1080,
        )

    def _make_result(self, identity_id, sim, classification):
        return self.IdentityResult(
            identity_id=identity_id,
            similarity=sim,
            classification=classification,
            threshold_high=0.45,
            threshold_review=0.35,
        )

    def test_single_high_confidence_match(self):
        faces = [self._make_face(0)]
        results = {0: [self._make_result("me", 0.75, "high_confidence")]}
        cls, ids, score, _, _ = self.classifier.classify(faces, results, "any_one")
        self.assertEqual(cls, "high_confidence")
        self.assertIn("me", ids)

    def test_review_candidate(self):
        faces = [self._make_face(0)]
        results = {0: [self._make_result("me", 0.40, "review_candidate")]}
        cls, ids, score, _, _ = self.classifier.classify(faces, results, "any_one")
        self.assertEqual(cls, "review_candidate")

    def test_no_match(self):
        faces = [self._make_face(0)]
        results = {0: [self._make_result("me", 0.20, "no_match")]}
        cls, ids, score, _, _ = self.classifier.classify(faces, results, "any_one")
        self.assertEqual(cls, "no_match")
        self.assertEqual(ids, [])

    def test_no_faces(self):
        cls, ids, score, _, _ = self.classifier.classify([], {}, "any_one")
        self.assertEqual(cls, "no_match")

    def test_any_two_rule_requires_two(self):
        faces = [self._make_face(0), self._make_face(1)]
        results = {
            0: [self._make_result("me", 0.75, "high_confidence")],
            1: [self._make_result("mother", 0.50, "high_confidence")],
        }
        cls, ids, score, _, _ = self.classifier.classify(faces, results, "any_two")
        self.assertEqual(cls, "high_confidence")
        self.assertGreaterEqual(len(ids), 2)

    def test_any_two_rule_only_one_match(self):
        faces = [self._make_face(0)]
        results = {0: [self._make_result("me", 0.75, "high_confidence")]}
        cls, ids, score, _, _ = self.classifier.classify(faces, results, "any_two")
        # Only 1 high-confidence → should be review or no_match
        self.assertIn(cls, ["review_candidate", "no_match"])

    def test_quality_warnings_detected(self):
        faces = [self._make_face(0, quality=0.2, is_small=True, is_blurry=True)]
        results = {0: [self._make_result("me", 0.75, "high_confidence")]}
        cls, ids, score, has_warnings, detail = self.classifier.classify(faces, results)
        self.assertTrue(has_warnings)
        self.assertGreater(len(detail), 0)

    def test_multiple_faces_group_photo(self):
        """A group photo with one family member should still match."""
        faces = [self._make_face(i) for i in range(6)]
        results = {
            0: [self._make_result("me", 0.22, "no_match")],
            1: [self._make_result("me", 0.18, "no_match")],
            2: [self._make_result("me", 0.80, "high_confidence")],  # family member
            3: [self._make_result("me", 0.15, "no_match")],
            4: [self._make_result("me", 0.12, "no_match")],
            5: [self._make_result("me", 0.20, "no_match")],
        }
        cls, ids, score, _, _ = self.classifier.classify(faces, results, "any_one")
        self.assertEqual(cls, "high_confidence")


class TestFileOperations(unittest.TestCase):
    """Test file copying, verification, and discovery."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_file(self, name, content=b"test content 1234"):
        path = os.path.join(self.tmpdir, name)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def test_copy_and_verify(self):
        from file_ops import safe_copy_file
        src = self._make_file("test.jpg", b"JFIF" + b"\x00" * 1000)
        dest_dir = os.path.join(self.tmpdir, "dest")
        ok, dest, sh, dh, err = safe_copy_file(src, dest_dir, verify_hash=True)
        self.assertTrue(ok)
        self.assertIsNone(err)
        self.assertEqual(sh, dh)
        self.assertTrue(os.path.isfile(dest))
        with open(src, "rb") as a, open(dest, "rb") as b:
            self.assertEqual(a.read(), b.read())

    def test_collision_safe_naming(self):
        from file_ops import safe_copy_file
        content = b"A" * 1000
        src = self._make_file("photo.jpg", content)
        dest_dir = os.path.join(self.tmpdir, "dest")
        os.makedirs(dest_dir)

        # Copy twice — second should get a new name
        ok1, dest1, _, _, _ = safe_copy_file(src, dest_dir, verify_hash=False)
        ok2, dest2, _, _, _ = safe_copy_file(src, dest_dir, verify_hash=False)
        self.assertTrue(ok1)
        self.assertTrue(ok2)
        self.assertNotEqual(dest1, dest2)

    def test_source_unchanged_after_copy(self):
        from file_ops import safe_copy_file, sha256_file
        content = b"original bytes"
        src = self._make_file("original.png", content)
        before_hash = sha256_file(src)
        dest_dir = os.path.join(self.tmpdir, "dest2")
        safe_copy_file(src, dest_dir, verify_hash=True)
        after_hash = sha256_file(src)
        self.assertEqual(before_hash, after_hash)

    def test_sha256_consistency(self):
        from file_ops import sha256_file
        content = b"known content"
        path = self._make_file("hash_test.bin", content)
        expected = hashlib.sha256(content).hexdigest()
        self.assertEqual(sha256_file(path), expected)

    def test_discover_images_flat(self):
        from file_ops import discover_images
        self._make_file("a.jpg")
        self._make_file("b.jpeg")
        self._make_file("c.png")
        self._make_file("d.txt")  # should be excluded
        found = discover_images(self.tmpdir, ["jpg", "jpeg", "png"], recursive=False)
        names = {os.path.basename(f) for f in found}
        self.assertIn("a.jpg", names)
        self.assertIn("b.jpeg", names)
        self.assertIn("c.png", names)
        self.assertNotIn("d.txt", names)

    def test_discover_images_recursive(self):
        from file_ops import discover_images
        sub = os.path.join(self.tmpdir, "sub")
        os.makedirs(sub)
        self._make_file("root.jpg")
        path = os.path.join(sub, "nested.jpg")
        with open(path, "wb") as f:
            f.write(b"x")
        found = discover_images(self.tmpdir, ["jpg"], recursive=True)
        names = {os.path.basename(f) for f in found}
        self.assertIn("root.jpg", names)
        self.assertIn("nested.jpg", names)

    def test_discover_excludes_dest_folders(self):
        from file_ops import discover_images
        dest = os.path.join(self.tmpdir, "output")
        os.makedirs(dest)
        self._make_file("source.jpg")
        extra_path = os.path.join(dest, "copied.jpg")
        with open(extra_path, "wb") as f:
            f.write(b"x")
        found = discover_images(self.tmpdir, ["jpg"], recursive=True, exclude_folders=[dest])
        names = {os.path.basename(f) for f in found}
        self.assertIn("source.jpg", names)
        self.assertNotIn("copied.jpg", names)

    def test_validate_folders_same_source_dest(self):
        from file_ops import validate_folders
        d = self.tmpdir
        valid, error = validate_folders(d, d, os.path.join(self.tmpdir, "review"))
        self.assertFalse(valid)

    def test_validate_folders_dest_inside_source(self):
        from file_ops import validate_folders
        src = self.tmpdir
        high = os.path.join(src, "matched")
        review = os.path.join(self.tmpdir + "_review", "review")
        os.makedirs(review, exist_ok=True)
        valid, error = validate_folders(src, high, review)
        self.assertFalse(valid)

    def test_validate_folders_valid(self):
        from file_ops import validate_folders
        src = self.tmpdir
        high = os.path.join(self.tmpdir + "_high")
        review = os.path.join(self.tmpdir + "_review")
        os.makedirs(high, exist_ok=True)
        os.makedirs(review, exist_ok=True)
        valid, error = validate_folders(src, high, review)
        self.assertTrue(valid, error)

    def test_unicode_path(self):
        """Test that Unicode paths in filenames work."""
        from file_ops import safe_copy_file
        unicode_name = "фото_семья_2024.jpg"
        src = os.path.join(self.tmpdir, unicode_name)
        with open(src, "wb") as f:
            f.write(b"unicode test")
        dest_dir = os.path.join(self.tmpdir, "dest_unicode")
        ok, dest, _, _, err = safe_copy_file(src, dest_dir, verify_hash=True)
        self.assertTrue(ok, f"Failed: {err}")


class TestDatabase(unittest.TestCase):
    """Test database initialization and basic CRUD."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_database_init(self):
        from database import get_engine, get_session_factory, Identity, init_default_identities
        engine = get_engine(self.db_path)
        self.assertIsNotNone(engine)

    def test_default_identities_created(self):
        from database import get_session_factory, Identity, init_default_identities, get_engine
        get_engine(self.db_path)
        SessionLocal = get_session_factory(self.db_path)
        db = SessionLocal()
        try:
            init_default_identities(db)
            identities = db.query(Identity).all()
            ids = {i.identity_id for i in identities}
            self.assertIn("me", ids)
            self.assertIn("mother", ids)
            self.assertIn("father", ids)
            self.assertIn("sister", ids)
        finally:
            db.close()

    def test_scan_job_create(self):
        from database import get_engine, get_session_factory, ScanJob
        get_engine(self.db_path)
        SessionLocal = get_session_factory(self.db_path)
        db = SessionLocal()
        try:
            job = ScanJob(
                source_folder="C:\\Photos",
                dest_folder_high="D:\\Matched",
                dest_folder_review="D:\\Review",
            )
            db.add(job)
            db.commit()
            self.assertIsNotNone(job.id)
            retrieved = db.query(ScanJob).filter_by(id=job.id).first()
            self.assertEqual(retrieved.source_folder, "C:\\Photos")
        finally:
            db.close()


class TestLargeImageHandling(unittest.TestCase):
    """Test that large images don't cause OOM."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_large_synthetic_image(self):
        """Create a 4000x3000 synthetic image and verify it loads without error."""
        from face_engine import FaceRecognitionEngine
        engine = FaceRecognitionEngine.__new__(FaceRecognitionEngine)
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (4000, 3000), color=(200, 100, 50))
        path = os.path.join(self.tmpdir, "large_test.jpg")
        img.save(path, "JPEG")
        img_bgr, (w, h), scale = engine.load_image_safe(path, max_dim=4096)
        self.assertIsNotNone(img_bgr)
        self.assertLessEqual(max(img_bgr.shape[:2]), 4096)

    def test_corrupted_file_handled(self):
        """Corrupted file should return None, not crash."""
        from face_engine import FaceRecognitionEngine
        engine = FaceRecognitionEngine.__new__(FaceRecognitionEngine)
        path = os.path.join(self.tmpdir, "corrupted.jpg")
        with open(path, "wb") as f:
            f.write(b"NOTAJPEG" + b"\x00" * 100)
        img_bgr, (w, h), scale = engine.load_image_safe(path)
        self.assertIsNone(img_bgr)


class TestApiEndpoints(unittest.TestCase):
    """Test FastAPI endpoints for routes, status, and identities."""

    def setUp(self):
        from fastapi.testclient import TestClient
        from main import app
        self.client = TestClient(app)

    def test_root_index_html(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("AI Family Photo Sorter", r.text)

    def test_hardware_endpoint(self):
        r = self.client.get("/api/hardware")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("hardware", data)
        self.assertIn("model", data)

    def test_identities_endpoint(self):
        r = self.client.get("/api/identities")
        self.assertEqual(r.status_code, 200)
        identities = r.json()
        self.assertGreaterEqual(len(identities), 4)
        ids = {i["identity_id"] for i in identities}
        self.assertTrue({"me", "mother", "father", "sister"}.issubset(ids))

    def test_dynamic_identity_crud_and_toggle(self):
        # 1. Create a dynamic member (e.g., cousin)
        create_res = self.client.post("/api/identities", json={
            "display_name": "Cousin Alex",
            "high_threshold": 0.45,
            "review_threshold": 0.35,
        })
        self.assertEqual(create_res.status_code, 200)
        created = create_res.json()["identity"]
        self.assertEqual(created["display_name"], "Cousin Alex")
        ident_id = created["identity_id"]
        self.assertIn("cousin_alex", ident_id)

        # 2. Toggle active status
        toggle_res = self.client.patch(f"/api/identities/{ident_id}/toggle")
        self.assertEqual(toggle_res.status_code, 200)
        self.assertFalse(toggle_res.json()["is_enabled"])

        # Toggle back
        toggle_res2 = self.client.patch(f"/api/identities/{ident_id}/toggle")
        self.assertEqual(toggle_res2.status_code, 200)
        self.assertTrue(toggle_res2.json()["is_enabled"])

        # 3. Delete the dynamic member
        del_res = self.client.delete(f"/api/identities/{ident_id}")
        self.assertEqual(del_res.status_code, 200)
        self.assertTrue(del_res.json()["success"])

        # 4. Verify no longer present
        list_res = self.client.get("/api/identities")
        remaining_ids = {i["identity_id"] for i in list_res.json()}
        self.assertNotIn(ident_id, remaining_ids)

    def test_classifier_with_dynamic_targets(self):
        from face_engine import ImageClassifier, DetectedFace, IdentityResult
        classifier = ImageClassifier()
        face0 = DetectedFace(
            bbox=[10, 10, 50, 50],
            confidence=0.9,
            embedding=np.ones(512, dtype=np.float32),
            landmark5=None,
            quality_score=0.8,
            is_small=False,
            is_blurry=False,
            face_index=0,
            image_width=1000,
            image_height=1000,
        )
        # Mother matches, sister matches
        results = {
            0: [
                IdentityResult(identity_id="mother", similarity=0.7, classification="high_confidence", threshold_high=0.45, threshold_review=0.35),
                IdentityResult(identity_id="sister", similarity=0.6, classification="high_confidence", threshold_high=0.45, threshold_review=0.35),
            ]
        }
        # If user only searched for 'sister'
        cls, ids, _, _, _ = classifier.classify([face0], results, matching_rule="any_one", target_ids=["sister"])
        self.assertEqual(cls, "high_confidence")
        self.assertEqual(ids, ["sister"])

        # If user searched only for 'father' (not present)
        cls2, ids2, _, _, _ = classifier.classify([face0], results, matching_rule="any_one", target_ids=["father"])
        self.assertEqual(cls2, "no_match")
        self.assertEqual(ids2, [])

    def test_browse_quick_paths(self):
        r = self.client.get("/api/browse/quick-paths")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("drives", data)
        self.assertIn("shortcuts", data)
        self.assertIn("default_path", data)
        self.assertTrue(len(data["drives"]) > 0)

    def test_browse_list_and_create_folder(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create subfolders
            sub1 = os.path.join(tmp_dir, "SubfolderA")
            os.makedirs(sub1)

            # Test list
            list_res = self.client.post("/api/browse/list", json={"path": tmp_dir})
            self.assertEqual(list_res.status_code, 200)
            data = list_res.json()
            self.assertEqual(os.path.normpath(data["current_path"]), os.path.normpath(tmp_dir))
            names = [s["name"] for s in data["subfolders"]]
            self.assertIn("SubfolderA", names)

            # Test create folder
            create_res = self.client.post("/api/browse/create-folder", json={
                "parent_path": tmp_dir,
                "folder_name": "NewCreatedFolder",
            })
            self.assertEqual(create_res.status_code, 200)
            self.assertTrue(create_res.json()["success"])
            self.assertTrue(os.path.isdir(os.path.join(tmp_dir, "NewCreatedFolder")))


if __name__ == "__main__":
    print("Running AI Photo Sorter test suite...\n")
    unittest.main(verbosity=2)


