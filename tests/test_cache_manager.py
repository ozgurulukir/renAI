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

import logging  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402

from renai.cache.manager import (  # noqa: E402
    read_metadata_cache,
    write_metadata_cache,
    invalidate_cache_for_provider_model,
    get_metadata_cache_path,
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
    assert result is not None
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

def test_get_metadata_cache_path():
    """Test generating metadata cache path."""
    file_hash = "abc123"
    provider = "openai"
    model = "gpt-4"
    cache_dir = Path("/tmp/cache")
    
    path = get_metadata_cache_path(file_hash, provider, model, cache_dir=cache_dir)
    assert "abc123_" in path.name
    assert ".json" in path.name
    assert path.parent.name == "metadata"

def test_invalidate_cache_for_provider_model_basic(tmp_path):
    """Test basic clearing of metadata cache files for provider/model."""
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    
    (metadata_dir / "test1.json").write_text("{}")
    (metadata_dir / "test2.json").write_text("{}")
    (metadata_dir / "not_json.txt").write_text("")
    
    count = invalidate_cache_for_provider_model(cache_dir=tmp_path)
    assert count == 2
    assert not (metadata_dir / "test1.json").exists()
    assert not (metadata_dir / "test2.json").exists()
    assert (metadata_dir / "not_json.txt").exists()

def test_invalidate_cache_for_provider_model_nonexistent(tmp_path):
    """Test clearing metadata cache when directory doesn't exist."""
    non_existent = tmp_path / "nonexistent"
    count = invalidate_cache_for_provider_model(cache_dir=non_existent)
    assert count == 0

def test_invalidate_cache_for_provider_model_empty(tmp_path):
    """Test clearing metadata cache when directory is empty."""
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    count = invalidate_cache_for_provider_model(cache_dir=tmp_path)
    assert count == 0
