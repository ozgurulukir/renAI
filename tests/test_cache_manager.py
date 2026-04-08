import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

# Note: We don't need to mock everything globally here if we run with pytest
# But to be safe and allow running with python3, we can use patches in tests.

from renai.cache.manager import (
    invalidate_cache_for_provider_model,
    get_metadata_cache_path
)

class TestCacheManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.metadata_dir = self.tmp_dir / "metadata"
        self.metadata_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_invalidate_cache_for_provider_model_basic(self):
        # Setup: Create some cache files
        # Provider A, Model 1
        p1 = get_metadata_cache_path("hash1", "providerA", "model1", cache_dir=self.tmp_dir)
        p1.write_text("{}", encoding="utf-8")

        # Provider A, Model 2
        p2 = get_metadata_cache_path("hash2", "providerA", "model2", cache_dir=self.tmp_dir)
        p2.write_text("{}", encoding="utf-8")

        # Provider B, Model 1
        p3 = get_metadata_cache_path("hash3", "providerB", "model1", cache_dir=self.tmp_dir)
        p3.write_text("{}", encoding="utf-8")

        # Invalidate for Provider A, Model 1
        count = invalidate_cache_for_provider_model(cache_dir=self.tmp_dir)

        # Only p1 should be deleted
        self.assertEqual(count, 1)
        self.assertFalse(p1.exists())
        self.assertTrue(p2.exists())
        self.assertTrue(p3.exists())

    def test_invalidate_cache_for_provider_model_vision(self):
        # Regular and vision for same provider/model
        p1 = get_metadata_cache_path("hash1", "p", "m", cache_dir=self.tmp_dir, use_vision=False)
        p1.write_text("{}", encoding="utf-8")
        p2 = get_metadata_cache_path("hash1", "p", "m", cache_dir=self.tmp_dir, use_vision=True)
        p2.write_text("{}", encoding="utf-8")

        # Different hash, same provider/model
        p3 = get_metadata_cache_path("hash2", "p", "m", cache_dir=self.tmp_dir, use_vision=False)
        p3.write_text("{}", encoding="utf-8")

        count = invalidate_cache_for_provider_model(cache_dir=self.tmp_dir)

        # All 3 should be deleted
        self.assertEqual(count, 3)
        self.assertFalse(p1.exists())
        self.assertFalse(p2.exists())
        self.assertFalse(p3.exists())

    def test_invalidate_cache_for_provider_model_nonexistent(self):
        # Directory doesn't exist
        non_existent = self.tmp_dir / "nonexistent"
        count = invalidate_cache_for_provider_model(cache_dir=non_existent)
        self.assertEqual(count, 0)

    def test_invalidate_cache_for_provider_model_empty(self):
        count = invalidate_cache_for_provider_model(cache_dir=self.tmp_dir)
        self.assertEqual(count, 0)

    def test_invalidate_cache_for_provider_model_partial_match(self):
        # Test that we don't accidentally delete files where the key is a substring
        with patch("renai.cache.manager.get_provider_model_key") as mock_key:
            mock_key.side_effect = lambda p, m: f"{p}_{m}"

            target_key = "p_m"
            other_key1 = "p_m2"
            other_key2 = "other_p_m"

            p1 = self.metadata_dir / f"hash1_{target_key}.json"
            p1.write_text("{}", encoding="utf-8")

            p2 = self.metadata_dir / f"hash2_{target_key}_vision.json"
            p2.write_text("{}", encoding="utf-8")

            p3 = self.metadata_dir / f"hash3_{other_key1}.json"
            p3.write_text("{}", encoding="utf-8")

            p4 = self.metadata_dir / f"hash4_{other_key2}.json"
            p4.write_text("{}", encoding="utf-8")

            count = invalidate_cache_for_provider_model(cache_dir=self.tmp_dir)

            self.assertEqual(count, 2)
            self.assertFalse(p1.exists())
            self.assertFalse(p2.exists())
            self.assertTrue(p3.exists())
            self.assertTrue(p4.exists())
