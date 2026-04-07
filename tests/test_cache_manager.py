import sys
from unittest.mock import MagicMock

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

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from renai.cache.manager import (
    read_metadata_cache,
    write_metadata_cache,
)

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
