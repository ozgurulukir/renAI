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
    clear_all_cache,
    get_metadata_cache_path,
    invalidate_cache_for_provider_model,
    read_metadata_cache,
    read_metadata_cache_with_source,
    read_text_cache,
    write_metadata_cache,
    write_text_cache,
)
from renai.models.schemas import BookMetadata  # noqa: E402


def test_read_metadata_cache_invalid_provider_model(tmp_path, caplog):
    """Test cache invalidation when provider or model doesn't match."""
    cache_path = tmp_path / "test_metadata.json"
    metadata = BookMetadata(
        title="Test Book",
        author="Test Author",
        category="Other",
        language="en",
        subtitle=None,
        edition=None,
        year=None,
    )
    provider = "test_provider"
    model = "test_model"

    # Write initial cache (supports both models and dicts)
    write_metadata_cache(cache_path, metadata, provider, model)

    # Test provider mismatch
    with caplog.at_level(logging.DEBUG):
        result = read_metadata_cache(cache_path, "different_provider", model)
        assert result is None
        assert "Cache invalidated" in caplog.text
        assert "was test_provider/test_model" in caplog.text


def test_read_metadata_cache_success(tmp_path):
    """Test successful metadata cache read with model validation."""
    cache_path = tmp_path / "success_metadata.json"
    metadata = BookMetadata(
        title="Success Book",
        author="Success Author",
        category="Other",
        language="en",
        subtitle=None,
        edition=None,
        year=None,
    )
    provider = "p1"
    model = "m1"

    write_metadata_cache(cache_path, metadata, provider, model)

    # Test reading with model validation - SSoT focus
    result = read_metadata_cache(cache_path, provider, model, model_class=BookMetadata)
    assert isinstance(result, BookMetadata)
    assert result.title == "Success Book"


def test_metadata_cache_source_marker_roundtrip(tmp_path):
    """The persisted _is_vision_doc marker round-trips and stays out of the model."""
    cache_path = tmp_path / "marker_metadata.json"
    metadata = BookMetadata(title="Marker Book", author="A", category="Other", language="en")
    provider = "p1"
    model = "m1"

    write_metadata_cache(cache_path, metadata, provider, model, is_vision_doc=False)

    # The source-aware reader returns the marker alongside the metadata.
    result, is_vision_doc = read_metadata_cache_with_source(
        cache_path, provider, model, model_class=BookMetadata
    )
    assert isinstance(result, BookMetadata)
    assert result.title == "Marker Book"
    assert is_vision_doc is False

    # The plain reader strips the reserved key; validation still succeeds.
    plain = read_metadata_cache(cache_path, provider, model, model_class=BookMetadata)
    assert isinstance(plain, BookMetadata)
    assert plain.title == "Marker Book"


def test_metadata_cache_without_marker_reports_none_source(tmp_path):
    """Records written without the marker read back as None (legacy semantics)."""
    cache_path = tmp_path / "unmarked_metadata.json"
    metadata = BookMetadata(title="Unmarked", author="A", category="Other", language="en")

    write_metadata_cache(cache_path, metadata, "p1", "m1")

    result, is_vision_doc = read_metadata_cache_with_source(
        cache_path, "p1", "m1", model_class=BookMetadata
    )
    assert isinstance(result, BookMetadata)
    assert is_vision_doc is None


def test_read_metadata_cache_non_object_json_degrades_to_miss(tmp_path, caplog):
    """A row holding valid-but-non-object JSON must degrade to a miss, not crash.

    Regression guard: an unguarded dict-only operation (e.g. .pop) on such a row
    used to raise out of the reader and fail the whole file.
    """
    cache_path = tmp_path / "corrupt_metadata.json"
    db_dir = cache_path.parent
    db_dir.mkdir(parents=True, exist_ok=True)

    from renai.cache.manager import _get_db_connection

    with _get_db_connection(db_dir) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO metadata_cache (cache_key, provider, model, metadata_json) "
            "VALUES (?, ?, ?, ?)",
            (cache_path.name, "p", "m", "[1, 2, 3]"),
        )

    with caplog.at_level(logging.WARNING):
        result = read_metadata_cache(cache_path, "p", "m")
        with_source = read_metadata_cache_with_source(
            cache_path, "p", "m", model_class=BookMetadata
        )

    assert result is None
    assert with_source == (None, None)
    assert any("expected a JSON object" in r.message for r in caplog.records)

    # The corrupt row is dropped so subsequent reads fall through cleanly.
    with _get_db_connection(db_dir) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM metadata_cache WHERE cache_key = ?", (cache_path.name,)
        ).fetchone()
    assert row[0] == 0


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

    with (
        patch.object(Path, "read_text", side_effect=Exception("Read error")),
        caplog.at_level(logging.WARNING),
    ):
        result = read_metadata_cache(cache_path, "p", "m")
        assert result is None
        assert "Failed to read metadata cache" in caplog.text
        assert "Read error" in caplog.text


def test_get_metadata_cache_path():
    """Test generating metadata cache path with mode awareness."""
    file_hash = "abc123"
    provider = "openai"
    model = "gpt-4"
    cache_dir = Path("/tmp/cache")

    # Test default mode (doc)
    path = get_metadata_cache_path(file_hash, provider, model, cache_dir=cache_dir)
    assert "doc_abc123_" in path.name
    assert ".json" in path.name
    assert path.parent.name == "metadata"

    # Test image mode
    path = get_metadata_cache_path(file_hash, provider, model, mode="img", cache_dir=cache_dir)
    assert "img_abc123_" in path.name
    assert ".json" in path.name


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


def test_clear_all_cache(tmp_path):
    """Test clearing all cache types including ocr_debug."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    text_dir = cache_dir / "text"
    text_dir.mkdir()
    (text_dir / "t1.txt").write_text("text")

    metadata_dir = cache_dir / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "m1.json").write_text("{}")

    # Mock DEBUG_DIR in settings to use tmp_path
    debug_dir = tmp_path / "ocr_debug"
    debug_dir.mkdir()
    (debug_dir / "d1.png").write_text("debug")

    with patch("renai.cache.manager.DEBUG_DIR", debug_dir):
        results = clear_all_cache(cache_dir=cache_dir)

        assert results["text"] == 1
        assert results["metadata"] == 1
        assert results["ocr_debug"] == 1

        assert not text_dir.exists()
        assert not metadata_dir.exists()
        assert not debug_dir.exists()


def test_read_text_cache_success(tmp_path, caplog):
    """Test reading existing text cache returns content."""
    cache_file = tmp_path / "cached.txt"
    cache_file.write_text("cached text content", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        result = read_text_cache(cache_file)

    assert result == "cached text content"
    assert "Failed to read" not in caplog.text


def test_read_text_cache_nonexistent(tmp_path):
    """Test reading non-existent cache file returns None."""
    missing = tmp_path / "does_not_exist.txt"
    result = read_text_cache(missing)
    assert result is None


def test_read_text_cache_io_error(tmp_path, caplog):
    """Test that IO errors during read are caught and return None."""
    cache_file = tmp_path / "bad.txt"
    cache_file.write_text("content", encoding="utf-8")

    with (
        patch.object(Path, "read_text", side_effect=OSError("Permission denied")),
        caplog.at_level(logging.WARNING),
    ):
        result = read_text_cache(cache_file)

    assert result is None
    assert "Failed to read text cache" in caplog.text
    assert "Permission denied" in caplog.text


def test_write_text_cache_success(tmp_path, caplog):
    """Test writing text cache succeeds and returns True."""
    cache_file = tmp_path / "output" / "cached.txt"
    content = "new cached content"

    with caplog.at_level(logging.WARNING):
        result = write_text_cache(cache_file, content)

    assert result is True
    assert read_text_cache(cache_file) == content
    assert "Failed to write" not in caplog.text


def test_write_text_cache_io_error(tmp_path, caplog):
    """Test that IO errors during write are caught and return False."""
    cache_file = tmp_path / "readonly" / "cached.txt"

    with (
        patch.object(Path, "mkdir", side_effect=OSError("Disk full")),
        caplog.at_level(logging.WARNING),
    ):
        result = write_text_cache(cache_file, "content")

    assert result is False
    assert "Failed to write text cache" in caplog.text
    assert "Disk full" in caplog.text


def test_read_metadata_cache_sqlite_error(tmp_path, caplog):
    """Test that SQLite errors during metadata read are logged and fallback occurs."""
    cache_path = tmp_path / "metadata" / "test.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        '{"title": "Fallback Book", "_provider": "p", "_model": "m"}', encoding="utf-8"
    )

    with (
        patch("renai.cache.manager._get_db_connection", side_effect=Exception("DB lock error")),
        caplog.at_level(logging.WARNING),
    ):
        result = read_metadata_cache(cache_path, "p", "m")
        assert result == {"title": "Fallback Book"}
        assert "Failed to read metadata cache" in caplog.text
        assert "DB lock error" in caplog.text


def test_read_metadata_cache_validation_error(tmp_path, caplog):
    """Test that metadata schema validation failure logs warning and returns None."""
    cache_path = tmp_path / "metadata" / "invalid_schema.json"
    # Write invalid data for BookMetadata (missing required fields)
    write_metadata_cache(cache_path, {"invalid_field": 123}, "p", "m")

    with caplog.at_level(logging.WARNING):
        result = read_metadata_cache(cache_path, "p", "m", model_class=BookMetadata)
        assert result is None
        assert "Failed to validate cached metadata against BookMetadata" in caplog.text


def test_clear_debug_cache_exception(tmp_path, caplog):
    """Test that exceptions during OCR debug cache clearing are handled safely."""
    from renai.cache.manager import _clear_debug_cache

    debug_dir = tmp_path / "ocr_debug"
    debug_dir.mkdir()
    (debug_dir / "sample.png").write_text("dummy")

    with (
        patch("renai.cache.manager.DEBUG_DIR", debug_dir),
        patch("shutil.rmtree", side_effect=PermissionError("Cannot remove debug dir")),
        caplog.at_level(logging.WARNING),
    ):
        count = _clear_debug_cache()
        assert count == 0
        assert "Failed to clear ocr_debug cache" in caplog.text
        assert "Cannot remove debug dir" in caplog.text


def test_clear_models_dev_cache_exception(tmp_path, caplog):
    """Test that exceptions during models.dev cache unlink are handled safely."""
    from renai.cache.manager import _clear_models_dev_cache

    models_file = tmp_path / "models_dev_cache.json"
    models_file.write_text("{}")

    with (
        patch("renai.config.constants.APP_DATA_DIR", tmp_path),
        patch.object(Path, "unlink", side_effect=PermissionError("Cannot unlink")),
        caplog.at_level(logging.WARNING),
    ):
        count = _clear_models_dev_cache()
        assert count == 0
        assert "Failed to clear models.dev cache" in caplog.text
        assert "Cannot unlink" in caplog.text
