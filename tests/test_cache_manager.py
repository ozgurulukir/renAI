import sys
from unittest.mock import MagicMock, patch
import json
import logging
import tempfile
import shutil
from pathlib import Path

# Mock missing dependencies before importing renai
def pre_mock(module_name):
    if module_name not in sys.modules:
        mock = MagicMock()
        mock.__path__ = []
        mock.__spec__ = None
        sys.modules[module_name] = mock

# Essential mocks for settings initialization
pre_mock("platformdirs")
pre_mock("pydantic")
pre_mock("pydantic_settings")
pre_mock("structlog")
pre_mock("rich")
pre_mock("rich.progress")
pre_mock("rich.console")
pre_mock("rich.table")
pre_mock("rich.logging")
pre_mock("typer")
pre_mock("pymupdf")
pre_mock("fitz")

import pytest
import unittest

from renai.cache.manager import (
    read_metadata_cache,
    write_metadata_cache,
    invalidate_cache_for_provider_model,
    get_metadata_cache_path,
    get_provider_model_key
)

# --- Pytest based tests for read/write ---

def test_read_metadata_cache_invalid_provider_model(tmp_path, caplog):
    """Test cache invalidation when provider or model doesn't match."""
    cache_path = tmp_path / "test_metadata.json"
    metadata = {"title": "Test Book", "author": "Test Author"}
    provider = "test_provider"
    model = "test_model"

    # Write initial cache
    write_metadata_cache(cache_path, metadata, provider, model)

    # Test provider mismatch
    with caplog.at_level(logging.INFO):
        result = read_metadata_cache(cache_path, "different_provider", model)
        assert result is None
        assert "Cache invalidated" in caplog.text
        assert "was test_provider/test_model" in caplog.text
        assert "now different_provider/test_model" in caplog.text

    # Test model mismatch
    caplog.clear()
    with caplog.at_level(logging.INFO):
        result = read_metadata_cache(cache_path, provider, "different_model")
        assert result is None
        assert "Cache invalidated" in caplog.text
        assert "was test_provider/test_model" in caplog.text
        assert "now test_provider/different_model" in caplog.text

def test_read_metadata_cache_success(tmp_path):
    """Test successful metadata cache read."""
    cache_path = tmp_path / "success_metadata.json"
    metadata = {"title": "Success Book", "author": "Success Author"}
    provider = "p1"
    model = "m1"

    write_metadata_cache(cache_path, metadata, provider, model)

    result = read_metadata_cache(cache_path, provider, model)
    assert result == metadata
    # Internal fields should be removed
    assert "_provider" not in result
    assert "_model" not in result

def test_read_metadata_cache_nonexistent(tmp_path):
    """Test reading from non-existent cache file."""
    cache_path = tmp_path / "nonexistent.json"
    result = read_metadata_cache(cache_path, "p", "m")
    assert result is None

def test_read_metadata_cache_invalid_json(tmp_path, caplog):
    """Test reading cache file with invalid JSON."""
    cache_path = tmp_path / "invalid.json"
    cache_path.write_text("invalid json content", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        result = read_metadata_cache(cache_path, "p", "m")
        assert result is None
        assert "Failed to read metadata cache" in caplog.text

def test_read_metadata_cache_exception(tmp_path, caplog):
    """Test reading cache when an exception occurs."""
    cache_path = tmp_path / "error.json"
    cache_path.write_text("{}", encoding="utf-8")

    with patch.object(Path, "read_text", side_effect=Exception("Read error")):
        with caplog.at_level(logging.WARNING):
            result = read_metadata_cache(cache_path, "p", "m")
            assert result is None
            assert "Failed to read metadata cache" in caplog.text
            assert "Read error" in caplog.text

# --- Unittest based tests for cache invalidation ---

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
        count = invalidate_cache_for_provider_model("providerA", "model1", cache_dir=self.tmp_dir)

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

        count = invalidate_cache_for_provider_model("p", "m", cache_dir=self.tmp_dir)

        # All 3 should be deleted
        self.assertEqual(count, 3)
        self.assertFalse(p1.exists())
        self.assertFalse(p2.exists())
        self.assertFalse(p3.exists())

    def test_invalidate_cache_for_provider_model_nonexistent(self):
        # Directory doesn't exist
        non_existent = self.tmp_dir / "nonexistent"
        count = invalidate_cache_for_provider_model("p", "m", cache_dir=non_existent)
        self.assertEqual(count, 0)

    def test_invalidate_cache_for_provider_model_empty(self):
        count = invalidate_cache_for_provider_model("p", "m", cache_dir=self.tmp_dir)
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

            count = invalidate_cache_for_provider_model("p", "m", cache_dir=self.tmp_dir)

            self.assertEqual(count, 2)
            self.assertFalse(p1.exists())
            self.assertFalse(p2.exists())
            self.assertTrue(p3.exists())
            self.assertTrue(p4.exists())
