import sqlite3
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
    MetadataCacheKey,
    TextCacheKey,
    _encode_legacy_metadata_key,
    _encode_metadata_key,
    _encode_text_key,
    _get_db_connection,
    clear_all_cache,
    read_metadata_cache,
    read_text_cache,
    write_metadata_cache,
    write_text_cache,
)
from renai.models.schemas import BookMetadata  # noqa: E402


def _meta_key(file_hash="hash1", provider="p1", model="m1", mode="doc", use_vision=False):
    return MetadataCacheKey(
        file_hash=file_hash,
        provider=provider,
        model=model,
        mode=mode,
        use_vision=use_vision,
    )


def _book(title="Test Book"):
    return BookMetadata(
        title=title,
        author="Test Author",
        category="Other",
        language="en",
        subtitle=None,
        edition=None,
        year=None,
    )


def test_key_encoding_matches_grammar():
    """The manager owns the on-disk key grammar; pin it exactly.

    The CURRENT metadata key is identity-free (#193 provenance): provider/model
    must NOT appear. The pre-provenance grammar (which embedded the
    provider/model sha256 pair) is pinned separately so legacy rows written by
    earlier versions stay findable.
    """
    assert _encode_text_key(TextCacheKey(file_hash="abc123")) == "abc123.txt"
    assert _encode_text_key(TextCacheKey(file_hash="abc123", ocr=True)) == "abc123_ocr.txt"
    assert _encode_text_key(TextCacheKey(file_hash="abc123", mutool=True)) == "abc123_mutool.txt"
    assert (
        _encode_text_key(TextCacheKey(file_hash="abc123", ocr=True, mutool=True))
        == "abc123_ocr_mutool.txt"
    )

    # Current: identity-free. The requested identity must not perturb the key.
    assert _encode_metadata_key(_meta_key(file_hash="abc123")) == "doc_abc123.json"
    assert _encode_metadata_key(_meta_key(file_hash="abc123", mode="img", use_vision=True)) == (
        "img_abc123_vision.json"
    )
    assert _encode_metadata_key(_meta_key(file_hash="abc123", provider="p2")) == "doc_abc123.json"

    # Legacy: provider/model pair is embedded via the same sha256("|".join)
    # scheme the removed get_provider_model_key helper used.
    import hashlib

    assert _encode_legacy_metadata_key(_meta_key(file_hash="abc123")) == (
        f"doc_abc123_{hashlib.sha256(b'p1|m1').hexdigest()}.json"
    )
    assert (
        _encode_legacy_metadata_key(_meta_key(file_hash="abc123", mode="img", use_vision=True))
        == f"img_abc123_{hashlib.sha256(b'p1|m1').hexdigest()}_vision.json"
    )


def test_read_metadata_cache_invalid_provider_model(tmp_path, caplog):
    """Legacy-format rows (no _prod_* markers) keep the compare-and-invalidate
    behavior: an identity mismatch on the row columns invalidates the row."""
    key = _meta_key(provider="test_provider", model="test_model")
    cache_key = _encode_metadata_key(key)

    # Seed a legacy-format row (no _prod_* markers) under the current key.
    with _get_db_connection(tmp_path) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO metadata_cache (cache_key, provider, model, metadata_json)"
            " VALUES (?, ?, ?, ?)",
            (
                cache_key,
                "test_provider",
                "test_model",
                '{"title": "Legacy Book", "language": "en", "category": "Other"}',
            ),
        )

    # Identity mismatch on a legacy row invalidates it (the key is
    # identity-free, so the same row is found under the changed identity).
    with caplog.at_level(logging.DEBUG):
        result, _ = read_metadata_cache(
            _meta_key(provider="different_provider", model="test_model"),
            cache_dir=tmp_path,
        )
        assert result is None
        assert "Cache invalidated" in caplog.text
        assert "was test_provider/test_model" in caplog.text


def test_provenance_rows_are_identity_agnostic_at_manager_level(tmp_path):
    """Rows carrying a producing identity are returned regardless of the
    requested identity: validation against the producing engine is the
    middleware's job (it knows which engine slot produced the record)."""
    key = _meta_key()
    metadata = _book("Provenance Book")

    write_metadata_cache(
        key,
        metadata,
        cache_dir=tmp_path,
        produced_provider="producing_p",
        produced_model="producing_m",
    )

    result, source = read_metadata_cache(
        _meta_key(provider="requested_p", model="requested_m"),
        BookMetadata,
        cache_dir=tmp_path,
    )
    assert isinstance(result, BookMetadata)
    assert source is not None
    assert source.produced_by_provider == "producing_p"
    assert source.produced_by_model == "producing_m"


def test_legacy_key_format_still_readable(tmp_path):
    """Rows written under the pre-provenance key (provider/model embedded) are
    found via the legacy-key fallback so old caches survive the upgrade."""
    from renai.cache.manager import _encode_legacy_metadata_key, _get_db_connection

    key = _meta_key(file_hash="abc123", provider="p", model="m", use_vision=True)
    legacy_key = _encode_legacy_metadata_key(key)

    insert_sql = "INSERT OR REPLACE INTO metadata_cache (cache_key, provider, model, metadata_json) VALUES (?, ?, ?, ?)"
    with _get_db_connection(tmp_path) as conn, conn:
        conn.execute(
            insert_sql,
            (
                legacy_key,
                "p",
                "m",
                '{"title": "Old Key Book", "language": "en", "category": "Other"}',
            ),
        )

    result, source = read_metadata_cache(key, BookMetadata, cache_dir=tmp_path)
    assert isinstance(result, BookMetadata)
    assert result.title == "Old Key Book"
    # No producing identity was stored -> legacy semantics.
    assert source is not None
    assert source.produced_by_provider is None
    assert source.produced_by_model is None


def test_read_metadata_cache_success(tmp_path):
    """Test successful metadata cache read with model validation."""
    key = _meta_key()
    metadata = _book("Success Book")

    assert write_metadata_cache(key, metadata, cache_dir=tmp_path) is True

    result, source = read_metadata_cache(key, BookMetadata, cache_dir=tmp_path)
    assert isinstance(result, BookMetadata)
    assert result.title == "Success Book"
    assert source is not None
    # No provenance markers written -> legacy semantics.
    assert source.is_vision_doc is None
    assert source.rule_signature == ""


def test_metadata_cache_isolated_by_key_components(tmp_path):
    """Structural key components (hash, mode, vision namespace) isolate rows.

    Requested identity (provider/model) deliberately does NOT isolate: the
    key is identity-free (#193 provenance) and producing-engine mismatch is
    judged at the middleware level against the persisted producing identity.
    """
    metadata = _book("Component Book")
    write_metadata_cache(_meta_key(), metadata, cache_dir=tmp_path)

    for variant in [
        _meta_key(file_hash="other"),
        _meta_key(mode="img"),
        _meta_key(use_vision=True),
    ]:
        result, _ = read_metadata_cache(variant, BookMetadata, cache_dir=tmp_path)
        assert result is None, f"expected miss for {variant}"

    # Requested-identity variants still find the row (identity-free key) and
    # surface the row's persisted producing identity.
    for variant in [_meta_key(provider="p2"), _meta_key(model="m2")]:
        result, source = read_metadata_cache(variant, BookMetadata, cache_dir=tmp_path)
        assert result is not None, f"expected hit for {variant}"
        assert source is not None
        assert source.produced_by_provider == "p1"
        assert source.produced_by_model == "m1"

    result, _ = read_metadata_cache(_meta_key(), BookMetadata, cache_dir=tmp_path)
    assert result is not None


def test_metadata_cache_source_marker_roundtrip(tmp_path):
    """The persisted _is_vision_doc marker round-trips and stays out of the model."""
    key = _meta_key()
    metadata = _book("Marker Book")

    write_metadata_cache(
        key, metadata, is_vision_doc=False, rule_signature="abc123", cache_dir=tmp_path
    )

    result, source = read_metadata_cache(key, BookMetadata, cache_dir=tmp_path)
    assert isinstance(result, BookMetadata)
    assert result.title == "Marker Book"
    assert source is not None
    assert source.is_vision_doc is False
    assert source.rule_signature == "abc123"
    # The producing-engine identity round-trips too (#193).
    assert source.produced_by_provider == "p1"
    assert source.produced_by_model == "m1"

    # The dict path (model_class=None) strips the reserved keys too.
    raw, _ = read_metadata_cache(key, cache_dir=tmp_path)
    assert raw == metadata.model_dump()


def test_metadata_cache_without_marker_reports_none_source(tmp_path):
    """Records written without the marker read back as None (legacy semantics)."""
    key = _meta_key()
    write_metadata_cache(key, _book("Unmarked"), cache_dir=tmp_path)

    result, source = read_metadata_cache(key, BookMetadata, cache_dir=tmp_path)
    assert isinstance(result, BookMetadata)
    assert source is not None
    assert source.is_vision_doc is None
    assert source.rule_signature == ""


def test_read_metadata_cache_non_object_json_degrades_to_miss(tmp_path, caplog):
    """A row holding valid-but-non-object JSON must degrade to a miss, not crash.

    Regression guard: an unguarded dict-only operation (e.g. .pop) on such a row
    used to raise out of the reader and fail the whole file.
    """
    cache_key = _encode_metadata_key(_meta_key(file_hash="corrupt", provider="p", model="m"))

    with _get_db_connection(tmp_path) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO metadata_cache (cache_key, provider, model, metadata_json) "
            "VALUES (?, ?, ?, ?)",
            (cache_key, "p", "m", "[1, 2, 3]"),
        )

    key = _meta_key(file_hash="corrupt", provider="p", model="m")
    with caplog.at_level(logging.WARNING):
        result, _ = read_metadata_cache(key, cache_dir=tmp_path)
        with_source = read_metadata_cache(key, BookMetadata, cache_dir=tmp_path)

    assert result is None
    assert with_source == (None, None)
    assert any("expected a JSON object" in r.message for r in caplog.records)

    # The corrupt row is dropped so subsequent reads fall through cleanly.
    with _get_db_connection(tmp_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM metadata_cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
    assert row[0] == 0


def test_read_metadata_cache_nonexistent(tmp_path):
    """Test reading from an empty cache directory."""
    result, _ = read_metadata_cache(_meta_key(file_hash="missing"), cache_dir=tmp_path)
    assert result is None


def test_read_metadata_cache_legacy_json_file_fallback(tmp_path):
    """A legacy JSON file (pre-SQLite store) is still readable."""
    key = _meta_key(file_hash="legacy", provider="p", model="m")
    legacy_dir = tmp_path / "metadata"
    legacy_dir.mkdir()
    (legacy_dir / _encode_legacy_metadata_key(key)).write_text(
        '{"title": "Legacy Book", "language": "en", "category": "Other", '
        '"_provider": "p", "_model": "m"}',
        encoding="utf-8",
    )

    result, _ = read_metadata_cache(key, BookMetadata, cache_dir=tmp_path)
    assert isinstance(result, BookMetadata)
    assert result.title == "Legacy Book"


def test_read_metadata_cache_legacy_json_stale_provider_deleted(tmp_path, caplog):
    """A legacy JSON file whose payload disagrees with the key is removed."""
    key = _meta_key(file_hash="stale", provider="p", model="m")
    legacy_path = tmp_path / "metadata" / _encode_legacy_metadata_key(key)
    legacy_path.parent.mkdir()
    legacy_path.write_text(
        '{"title": "Stale", "language": "en", "category": "Other", '
        '"_provider": "other", "_model": "m"}',
        encoding="utf-8",
    )

    with caplog.at_level(logging.DEBUG):
        result, _ = read_metadata_cache(key, cache_dir=tmp_path)
    assert result is None
    assert "Cache invalidated" in caplog.text
    assert not legacy_path.exists()


def test_read_metadata_cache_invalid_json(tmp_path, caplog):
    """Test reading legacy cache file with invalid JSON."""
    key = _meta_key(file_hash="invalid", provider="p", model="m")
    legacy_path = tmp_path / "metadata" / _encode_legacy_metadata_key(key)
    legacy_path.parent.mkdir()
    legacy_path.write_text("invalid json content", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        result, _ = read_metadata_cache(key, cache_dir=tmp_path)
        assert result is None
        assert "Failed to read metadata cache" in caplog.text


def test_read_metadata_cache_exception(tmp_path, caplog):
    """Test reading cache when an exception occurs."""
    key = _meta_key(file_hash="error", provider="p", model="m")
    legacy_path = tmp_path / "metadata" / _encode_legacy_metadata_key(key)
    legacy_path.parent.mkdir()
    legacy_path.write_text("{}", encoding="utf-8")

    with (
        patch.object(Path, "read_text", side_effect=Exception("Read error")),
        caplog.at_level(logging.WARNING),
    ):
        result, _ = read_metadata_cache(key, cache_dir=tmp_path)
        assert result is None
        assert "Failed to read metadata cache" in caplog.text
        assert "Read error" in caplog.text


def test_clear_all_cache(tmp_path):
    """Test clearing all cache types including ocr_debug, verifying counting semantics for hidden files and subdirectories."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    text_dir = cache_dir / "text"
    text_dir.mkdir()
    (text_dir / "t1.txt").write_text("text")
    (text_dir / ".hidden").write_text("hidden")
    (text_dir / ".DS_Store").write_text("ds_store")
    sub_dir = text_dir / "sub"
    sub_dir.mkdir()
    (sub_dir / "nested.txt").write_text("nested")

    metadata_dir = cache_dir / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "m1.json").write_text("{}")

    # Mock DEBUG_DIR in settings to use tmp_path
    debug_dir = tmp_path / "ocr_debug"
    debug_dir.mkdir()
    (debug_dir / "d1.png").write_text("debug")

    with patch("renai.cache.manager.DEBUG_DIR", debug_dir):
        results = clear_all_cache(cache_dir=cache_dir)

        assert results["text"] == 3  # t1.txt, .hidden, .DS_Store (sub/nested.txt excluded)
        assert results["metadata"] == 1
        assert results["ocr_debug"] == 1

        assert not text_dir.exists()
        assert not metadata_dir.exists()
        assert not debug_dir.exists()


def test_read_text_cache_success(tmp_path):
    """Test writing then reading a text cache entry round-trips via SQLite."""
    key = TextCacheKey(file_hash="abc123")
    content = "cached text content"

    assert write_text_cache(key, content, cache_dir=tmp_path) is True
    assert read_text_cache(key, cache_dir=tmp_path) == content


def test_read_text_cache_legacy_file_fallback(tmp_path):
    """A legacy text cache file on disk (pre-SQLite store) is still readable."""
    key = TextCacheKey(file_hash="abc123", ocr=True)
    legacy_dir = tmp_path / "text"
    legacy_dir.mkdir()
    (legacy_dir / _encode_text_key(key)).write_text("legacy text", encoding="utf-8")

    assert read_text_cache(key, cache_dir=tmp_path) == "legacy text"


def test_read_text_cache_nonexistent(tmp_path):
    """Test reading non-existent cache returns None."""
    result = read_text_cache(TextCacheKey(file_hash="does_not_exist"), cache_dir=tmp_path)
    assert result is None


def test_read_text_cache_io_error(tmp_path, caplog):
    """Test that IO errors during read are caught and return None."""
    key = TextCacheKey(file_hash="bad")
    legacy_dir = tmp_path / "text"
    legacy_dir.mkdir()
    (legacy_dir / _encode_text_key(key)).write_text("content", encoding="utf-8")

    with (
        patch.object(Path, "read_text", side_effect=OSError("Permission denied")),
        caplog.at_level(logging.WARNING),
    ):
        result = read_text_cache(key, cache_dir=tmp_path)

    assert result is None
    assert "Failed to read text cache" in caplog.text
    assert "Permission denied" in caplog.text


def test_write_text_cache_io_error(tmp_path, caplog):
    """Test that IO errors during write are caught and return False."""
    key = TextCacheKey(file_hash="readonly")

    with (
        patch.object(Path, "mkdir", side_effect=OSError("Disk full")),
        caplog.at_level(logging.WARNING),
    ):
        result = write_text_cache(key, "content", cache_dir=tmp_path)

    assert result is False
    assert "Failed to write text cache" in caplog.text
    assert "Disk full" in caplog.text


def test_read_metadata_cache_sqlite_error(tmp_path, caplog):
    """Test that generic SQLite errors during metadata read fall back to legacy files."""
    key = _meta_key(file_hash="fallback", provider="p", model="m")
    legacy_path = tmp_path / "metadata" / _encode_legacy_metadata_key(key)
    legacy_path.parent.mkdir()
    legacy_path.write_text(
        '{"title": "Fallback Book", "language": "en", "category": "Other", '
        '"_provider": "p", "_model": "m"}',
        encoding="utf-8",
    )

    with (
        patch("renai.cache.manager._get_db_connection", side_effect=Exception("DB lock error")),
        caplog.at_level(logging.WARNING),
    ):
        result, _ = read_metadata_cache(key, cache_dir=tmp_path)
        assert result == {"title": "Fallback Book", "language": "en", "category": "Other"}
        assert "Failed to read metadata cache" in caplog.text
        assert "DB lock error" in caplog.text


def test_read_metadata_cache_corrupt_db_skips_legacy_fallback(tmp_path, caplog):
    """A corrupt primary store must not be papered over by stale legacy JSON files.

    sqlite3.DatabaseError means the DB file itself is broken: log loud (error
    level, naming cache.db so the user can act on the right artifact), degrade
    to a miss, and do NOT serve records from the superseded legacy store.
    """
    key = _meta_key(file_hash="corruptdb", provider="p", model="m")
    legacy_path = tmp_path / "metadata" / _encode_legacy_metadata_key(key)
    legacy_path.parent.mkdir()
    legacy_path.write_text(
        '{"title": "Stale Legacy Book", "language": "en", "category": "Other", '
        '"_provider": "p", "_model": "m"}',
        encoding="utf-8",
    )

    with (
        patch(
            "renai.cache.manager._get_db_connection",
            side_effect=sqlite3.DatabaseError("file is not a database"),
        ),
        caplog.at_level(logging.ERROR),
    ):
        result, _ = read_metadata_cache(key, cache_dir=tmp_path)

    assert result is None
    assert "corrupt or unreadable" in caplog.text
    assert "cache.db" in caplog.text


def test_write_metadata_cache_corrupt_db_is_loud_but_not_fatal(tmp_path, caplog):
    """Write failures on a corrupt store log an error naming cache.db, never raise.

    Caching is best-effort: extraction already succeeded, so the run must
    continue — but the failure must not look like a routine miss.
    """
    key = _meta_key(file_hash="wfail", provider="p", model="m")

    with (
        patch(
            "renai.cache.manager._get_db_connection",
            side_effect=sqlite3.DatabaseError("file is not a database"),
        ),
        caplog.at_level(logging.ERROR),
    ):
        result = write_metadata_cache(key, {"title": "Uncached Book"}, cache_dir=tmp_path)

    assert result is False
    assert "corrupt or unreadable" in caplog.text
    assert "NOT cached" in caplog.text


def test_read_metadata_cache_validation_error(tmp_path, caplog):
    """Test that metadata schema validation failure logs warning and returns None."""
    key = _meta_key(file_hash="invalid")
    write_metadata_cache(key, {"invalid_field": 123}, cache_dir=tmp_path)

    with caplog.at_level(logging.WARNING):
        result, _ = read_metadata_cache(key, BookMetadata, cache_dir=tmp_path)
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


def test_legacy_rows_without_rule_sig_read_as_legacy_provenance(tmp_path):
    """Rows predating _rule_sig must read with rule_signature=None.

    The middleware treats a None stored signature as global: global runs keep
    hitting, rule-scoped runs re-extract once and rewrite the row. Both a
    pre-marker row (no reserved keys at all) and a #175-era row (_is_vision_doc
    set, no _rule_sig) must surface rule_signature=None, not "".
    """
    key_a = _meta_key(file_hash="pre-marker", provider="p", model="m")
    key_b = _meta_key(file_hash="175-era", provider="p", model="m")
    rows = {
        _encode_legacy_metadata_key(
            key_a
        ): '{"title": "Legacy Book", "language": "en", "category": "Other"}',
        _encode_legacy_metadata_key(
            key_b
        ): '{"title": "175-era Book", "language": "en", "category": '
        '"Other", "_is_vision_doc": true}',
    }
    for cache_key, metadata_json in rows.items():
        with _get_db_connection(tmp_path) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO metadata_cache (cache_key, provider, model, metadata_json)"
                " VALUES (?, ?, ?, ?)",
                (cache_key, "p", "m", metadata_json),
            )

    result, source = read_metadata_cache(key_a, BookMetadata, cache_dir=tmp_path)
    assert isinstance(result, BookMetadata)
    assert source is not None
    assert source.is_vision_doc is None
    assert source.rule_signature is None

    result_era, source_era = read_metadata_cache(key_b, BookMetadata, cache_dir=tmp_path)
    assert isinstance(result_era, BookMetadata)
    assert source_era is not None
    assert source_era.is_vision_doc is True
    # _rule_sig absent -> legacy (None), distinct from "" (global).
    assert source_era.rule_signature is None
