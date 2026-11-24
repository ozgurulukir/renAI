import contextlib
import hashlib
import json
import logging
import shutil
import sqlite3
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any, NamedTuple, overload

from pydantic import BaseModel

from renai.config import CACHE_DIR, DEBUG_DIR
from renai.utils import ensure_secure_dir

logger = logging.getLogger(__name__)

# Thread-safe lock for all cache database operations
_cache_lock = threading.Lock()
_initialized_dbs: set[Path] = set()


class TextCacheKey(NamedTuple):
    """Structured identity of a cached text-extraction result.

    Mirrors every component the on-disk key grammar encodes
    (``{hash}_ocr_mutool.txt``): the file hash plus the two OCR fallback flags
    that change the produced text.
    """

    file_hash: str
    ocr: bool = False
    mutool: bool = False


class MetadataCacheKey(NamedTuple):
    """Structured identity of a cached metadata record.

    ``provider``/``model`` are the identity of the engine REQUESTED for this
    lookup. They deliberately do NOT participate in the current key grammar
    (``{mode}_{hash}{_vision}.json``): since provenance (#193), the key is
    identity-free and the engine that actually PRODUCED a record is persisted
    inside the row (``_prod_provider``/``_prod_model``) and validated on read.
    The provider/model pair stays on the key so the pre-provenance key format
    (which embedded it) remains reconstructable for legacy reads, and so the
    row columns can be populated for observability.
    """

    file_hash: str
    provider: str
    model: str
    mode: str = "doc"
    use_vision: bool = False


def _provider_model_key(provider: str, model: str) -> str:
    """Encode a provider/model pair for the legacy cache-key grammar."""
    combined = "|".join(str(arg) for arg in (provider, model) if arg)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _encode_text_key(key: TextCacheKey) -> str:
    """Encode a :class:`TextCacheKey` to its on-disk/cache-row key."""
    ocr_suffix = "_ocr" if key.ocr else ""
    mutool_suffix = "_mutool" if key.mutool else ""
    suffix = f"{ocr_suffix}{mutool_suffix}" if ocr_suffix or mutool_suffix else ""
    return f"{key.file_hash}{suffix}.txt"


def _encode_metadata_key(key: MetadataCacheKey) -> str:
    """Encode a :class:`MetadataCacheKey` to its current (identity-free) key.

    The key is deliberately independent of provider/model: the identity of the
    engine that actually produced a record is persisted *inside* the row
    (``_prod_provider``/``_prod_model``) and validated on read. Keying by the
    requested mode instead of the requested engine is what lets a text-LLM
    record survive a vision-model change under ``--fallback vision`` (and vice
    versa) while a change to the producing engine itself still invalidates.
    """
    vision_suffix = "_vision" if key.use_vision else ""
    return f"{key.mode}_{key.file_hash}{vision_suffix}.json"


def _encode_legacy_metadata_key(key: MetadataCacheKey) -> str:
    """Encode the pre-provenance key, which embedded provider/model.

    Old keys had the form ``{mode}_{file_hash}_{provider_model}{vision}.json``;
    the current key omits the ``{provider_model}`` segment. Used only to keep
    reading rows written by earlier versions.
    """
    provider_model_key = _provider_model_key(key.provider, key.model)
    vision_suffix = "_vision" if key.use_vision else ""
    return f"{key.mode}_{key.file_hash}_{provider_model_key}{vision_suffix}.json"


def _resolve_cache_dir(cache_dir: Path | None) -> Path:
    """Default the storage root to the package CACHE_DIR when not given."""
    return CACHE_DIR if cache_dir is None else cache_dir


@contextlib.contextmanager
def _get_db_connection(db_dir: Path) -> Generator[sqlite3.Connection, None, None]:
    """Create, initialize, and yield a SQLite database connection with WAL mode enabled."""
    ensure_secure_dir(db_dir)
    db_path = db_dir / "cache.db"
    resolved_db = db_path.resolve()
    needs_init = not db_path.exists() or resolved_db not in _initialized_dbs
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    try:
        if needs_init:
            conn.execute("PRAGMA journal_mode=WAL;")
            _init_db(conn)
            _initialized_dbs.add(resolved_db)
        conn.execute("PRAGMA synchronous=NORMAL;")
        yield conn
    finally:
        conn.close()


def _init_db(conn: sqlite3.Connection) -> None:
    """Initialize database tables for text and metadata caching."""
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS text_cache (
                cache_key TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata_cache (
                cache_key TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)


def read_text_cache(key: TextCacheKey, cache_dir: Path | None = None) -> str | None:
    """Read cached text content from SQLite (with legacy file fallback)."""
    with _cache_lock:
        db_dir = _resolve_cache_dir(cache_dir)
        cache_key = _encode_text_key(key)
        try:
            with _get_db_connection(db_dir) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT content FROM text_cache WHERE cache_key = ?", (cache_key,))
                row = cursor.fetchone()
                if row:
                    return row[0]
        except Exception as e:
            logger.warning(f"Failed to read text cache {cache_key}: {e}")

        # Fallback to legacy file reading if present (guarded against path traversal)
        legacy_root = (db_dir / "text").resolve()
        legacy_path = (legacy_root / cache_key).resolve()
        if not legacy_path.is_relative_to(legacy_root):
            logger.warning("Blocked legacy text-cache path traversal: %s", cache_key)
            return None
        try:
            if legacy_path.exists():
                return legacy_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to read text cache {legacy_path}: {e}")
    return None


def write_text_cache(key: TextCacheKey, content: str, cache_dir: Path | None = None) -> bool:
    """Write text content to SQLite cache atomically."""
    with _cache_lock:
        db_dir = _resolve_cache_dir(cache_dir)
        cache_key = _encode_text_key(key)
        try:
            with _get_db_connection(db_dir) as conn, conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO text_cache (cache_key, content)
                    VALUES (?, ?)
                    """,
                    (cache_key, content),
                )
            return True
        except Exception as e:
            logger.warning(f"Failed to write text cache {cache_key}: {e}")
            return False


def _read_from_legacy_file(legacy_path: Path, provider: str, model: str) -> dict[str, Any] | None:
    """Helper to read and validate stale/legacy metadata cache file."""
    if not legacy_path.exists():
        return None
    cache_content = legacy_path.read_text(encoding="utf-8")
    cache_data = json.loads(cache_content)

    cached_provider = cache_data.get("_provider", "")
    cached_model = cache_data.get("_model", "")

    if cached_provider == provider and cached_model == model:
        cache_data.pop("_provider", None)
        cache_data.pop("_model", None)
        return cache_data

    msg = f"Cache invalidated (was {cached_provider}/{cached_model}, now {provider}/{model})"
    logger.debug(msg)
    with contextlib.suppress(Exception):
        legacy_path.unlink()
        logger.debug(f"Removed stale cache file: {legacy_path}")
    return None


def _parse_metadata_row(
    conn: sqlite3.Connection,
    cache_key: str,
    row: tuple[Any, Any, Any],
    provider: str,
    model: str,
) -> tuple[dict[str, Any], tuple[str, str] | None] | None:
    """Interpret one metadata_cache row; ``None`` means skip (miss for this key).

    Rows carrying ``_prod_provider``/``_prod_model`` are returned with that
    identity and are NOT judged here — the caller owns the producing-identity
    validation. Legacy rows without the marker keep the old behavior: their
    provider/model columns are compared against ``provider``/``model`` and
    mismatches are deleted. The caller holds ``_cache_lock`` and an open
    connection.
    """
    cached_provider, cached_model, metadata_json = row

    data: Any = None
    try:
        data = json.loads(metadata_json)
    except (TypeError, ValueError) as e:
        logger.warning(
            f"Corrupt metadata cache row {cache_key}: invalid JSON ({e}); treating as a cache miss"
        )
    if not isinstance(data, dict):
        if data is not None:
            # Valid JSON but not an object: corrupt row. Degrade to a
            # miss (and drop the row) instead of crashing on dict-only
            # operations downstream.
            logger.warning(
                f"Corrupt metadata cache row for {cache_key}: expected a JSON "
                f"object, got {type(data).__name__}; treating as a cache miss"
            )
        with conn:
            conn.execute("DELETE FROM metadata_cache WHERE cache_key = ?", (cache_key,))
        return None

    produced_provider = data.pop("_prod_provider", None)
    produced_model = data.pop("_prod_model", None)
    if isinstance(produced_provider, str) and isinstance(produced_model, str):
        return data, (produced_provider, produced_model)
    if produced_provider is not None or produced_model is not None:
        logger.warning(
            f"Corrupt producing-identity marker in metadata cache row {cache_key}; "
            "treating row as legacy"
        )

    # Legacy row (pre-provenance): the row columns are the single
    # stored identity; keep the historical compare-and-invalidate.
    if cached_provider == provider and cached_model == model:
        return data, None
    msg = f"Cache invalidated (was {cached_provider}/{cached_model}, now {provider}/{model})"
    logger.debug(msg)
    with conn:
        conn.execute("DELETE FROM metadata_cache WHERE cache_key = ?", (cache_key,))
    return None


def _read_provenance_cache_data(
    cache_dir: Path,
    cache_keys: list[str],
    legacy_path: Path,
    provider: str,
    model: str,
) -> tuple[dict[str, Any], tuple[str, str] | None] | None:
    """Read raw cache data together with its producing-engine identity.

    The current key (``{mode}_{file_hash}{vision_suffix}.json``) is tried first;
    on a miss the pre-provenance key (which embedded provider/model) is tried so
    rows written by earlier versions stay readable. Returns
    ``(cache_data, producing_identity)`` with ``producing_identity=None`` for
    legacy rows, or ``None`` on a miss. The caller must hold ``_cache_lock``.
    Reserved keys other than the producing identity (e.g. ``_is_vision_doc``)
    are left in the dict for the caller.
    """
    db_result: tuple[dict[str, Any], tuple[str, str] | None] | None = None

    try:
        with _get_db_connection(cache_dir) as conn:
            for cache_key in cache_keys:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT provider, model, metadata_json FROM metadata_cache WHERE cache_key = ?",
                    (cache_key,),
                )
                row = cursor.fetchone()
                if row is None:
                    continue
                parsed = _parse_metadata_row(conn, cache_key, row, provider, model)
                if parsed is not None:
                    db_result = parsed
                    break
    except sqlite3.DatabaseError as e:
        # Corrupt/locked/unreadable primary store. Log loud, name the artifact
        # the user can act on (the DB file, not the .json cache key), and do NOT
        # fall back to legacy JSON files: serving records from a superseded
        # store while the primary is broken hides the failure. Degrade to a
        # miss so processing continues at full LLM cost.
        logger.error(
            f"Metadata cache database {cache_dir / 'cache.db'} is corrupt or unreadable ({e}). "
            "Delete or rename it to rebuild the cache; skipping legacy fallback."
        )
        return None
    except Exception as e:
        logger.warning(f"Failed to read metadata cache {cache_keys[0]}: {e}")

    if db_result is not None:
        return db_result

    try:
        legacy_file_data = _read_from_legacy_file(legacy_path, provider, model)
    except Exception as e:
        logger.warning(f"Failed to read metadata cache {legacy_path}: {e}")
        return None
    if legacy_file_data is None:
        return None
    return legacy_file_data, None


def _validate_cached_metadata[V: BaseModel](
    cache_data: dict[str, Any],
    model_class: type[V],
) -> V | None:
    """Validate raw cache data against a Pydantic model, degrading to None on failure."""
    try:
        return model_class(**cache_data)
    except Exception as e:
        logger.warning(f"Failed to validate cached metadata against {model_class.__name__}: {e}")
        return None


class CacheSource(NamedTuple):
    """Extraction provenance persisted alongside a cached metadata record.

    ``is_vision_doc`` records the engine path that actually produced the
    metadata (True = vision fallback). ``rule_signature`` is the category
    universe the record was extracted under (``config.folder_rules.
    rule_cache_signature``; empty string = global categories).
    ``produced_by_provider``/``produced_by_model`` identify the engine that
    actually produced the record — which may differ from the engine the cache
    namespace was requested for (e.g. a text-LLM record written during a
    ``--fallback vision`` run). All fields are ``None`` for legacy records
    written before the keys existed; callers own the fallback decision for
    those.
    """

    is_vision_doc: bool | None
    rule_signature: str | None
    produced_by_provider: str | None = None
    produced_by_model: str | None = None


@overload
def read_metadata_cache[V: BaseModel](
    key: MetadataCacheKey,
    model_class: type[V],
    cache_dir: Path | None = None,
    expected_rule_signature: str | None = None,
) -> tuple[V | None, CacheSource | None]: ...


@overload
def read_metadata_cache(
    key: MetadataCacheKey,
    model_class: None = None,
    cache_dir: Path | None = None,
    expected_rule_signature: str | None = None,
) -> tuple[dict[str, Any] | None, CacheSource | None]: ...


def read_metadata_cache[V: BaseModel](
    key: MetadataCacheKey,
    model_class: type[V] | None = None,
    cache_dir: Path | None = None,
    expected_rule_signature: str | None = None,
) -> tuple[Any | None, CacheSource | None]:
    """Read cached metadata together with its persisted extraction provenance.

    Returns ``(metadata, source)``; ``source`` is ``None`` on a miss. With
    ``model_class=None`` the raw dict is returned unvalidated (the reserved
    bookkeeping keys are still stripped). See :class:`CacheSource` for the
    meaning of the individual fields.

    ``key.provider``/``key.model`` do not select or validate current rows —
    the cache key is identity-free and the producing identity is validated by
    the caller. They are used only for rows written before provenance was
    stored: locating their legacy keys and the historical compare-and-invalidate.
    """
    with _cache_lock:
        cache_dir = _resolve_cache_dir(cache_dir)
        cache_key = _encode_metadata_key(key)
        legacy_key = _encode_legacy_metadata_key(key)
        legacy_path = cache_dir / "metadata" / legacy_key
        provenance = _read_provenance_cache_data(
            cache_dir, [cache_key, legacy_key], legacy_path, key.provider, key.model
        )
        if provenance is not None:
            cache_data, produced = provenance
        else:
            cache_data, produced = None, None

        raw_vision = cache_data.pop("_is_vision_doc", None) if cache_data else None
        if raw_vision is not None and not isinstance(raw_vision, bool):
            # Present but corrupt (external tampering; the writer only stores
            # bools). Warn so the silent template switch is explainable, then
            # let the caller apply its legacy fallback instead of trusting it.
            logger.warning(
                f"Corrupt vision marker in metadata cache row {cache_key}: expected "
                f"a bool, got {type(raw_vision).__name__}; treating as legacy"
            )
        is_vision_doc = raw_vision if isinstance(raw_vision, bool) else None
        rule_signature = cache_data.pop("_rule_sig", None) if cache_data else None
        if rule_signature is not None and not isinstance(rule_signature, str):
            logger.warning(
                f"Corrupt rule signature in metadata cache row {cache_key}: expected "
                f"a string, got {type(rule_signature).__name__}; treating as legacy"
            )
            rule_signature = None
        if (
            expected_rule_signature is not None
            and (rule_signature or "") != expected_rule_signature
        ):
            return None, None

        source = CacheSource(
            is_vision_doc=is_vision_doc,
            rule_signature=rule_signature,
            produced_by_provider=produced[0] if produced else None,
            produced_by_model=produced[1] if produced else None,
        )
        if cache_data is None:
            return None, None
        if model_class is None:
            return cache_data, source
        validated = _validate_cached_metadata(cache_data, model_class)
        if validated is None:
            return None, None
        return validated, source


def write_metadata_cache(
    key: MetadataCacheKey,
    metadata: BaseModel | dict[str, Any],
    *,
    is_vision_doc: bool | None = None,
    rule_signature: str | None = None,
    cache_dir: Path | None = None,
    produced_provider: str | None = None,
    produced_model: str | None = None,
) -> bool:
    """Write metadata to SQLite cache atomically.

    The row is written under the structured ``key``'s current (identity-free)
    encoding. ``produced_provider``/``produced_model`` carry the identity of
    the engine that PRODUCED the record — which may differ from
    ``key.provider``/``key.model`` (the requested engine) under
    ``--fallback vision``; they default to the requested identity. The
    producing identity is written to the row columns and persisted under the
    reserved ``_prod_provider``/``_prod_model`` keys so cache hits can
    validate against the producing engine. When ``is_vision_doc`` is not None, the observed
    extraction source is persisted under the reserved ``_is_vision_doc`` key so
    cache hits can replay the fact instead of reconstructing it from request
    flags. The category-universe ``rule_signature`` is always persisted (empty
    string for global categories) so rule-scoped runs can tell their records
    apart from global ones.
    """
    with _cache_lock:
        db_dir = _resolve_cache_dir(cache_dir)
        cache_key = _encode_metadata_key(key)
        row_provider = produced_provider if produced_provider is not None else key.provider
        row_model = produced_model if produced_model is not None else key.model
        try:
            data = metadata.model_dump() if isinstance(metadata, BaseModel) else dict(metadata)
            data["_prod_provider"] = row_provider
            data["_prod_model"] = row_model
            if is_vision_doc is not None:
                data["_is_vision_doc"] = is_vision_doc
            data["_rule_sig"] = rule_signature or ""
            metadata_json = json.dumps(data, ensure_ascii=False, indent=4)

            with _get_db_connection(db_dir) as conn, conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO metadata_cache
                    (cache_key, provider, model, metadata_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (cache_key, row_provider, row_model, metadata_json),
                )
            return True
        except sqlite3.DatabaseError as e:
            # Caching is best-effort: extraction succeeded, so never fail the
            # run — but a broken store must not look like a routine miss.
            logger.error(
                f"Metadata cache database {db_dir / 'cache.db'} is corrupt or unreadable ({e}); "
                "metadata was extracted but NOT cached. Delete or rename the file to rebuild."
            )
            return False
        except Exception as e:
            logger.warning(f"Failed to write metadata cache {cache_key}: {e}")
            return False


def clear_all_cache(cache_dir: Path | None = None) -> dict[str, int]:
    """Clear all caches (text, metadata, and ocr_debug) without double-counting."""
    with _cache_lock:
        if cache_dir is None:
            cache_dir = CACHE_DIR

        text_keys: set[str] = set()
        metadata_keys: set[str] = set()

        db_path = cache_dir / "cache.db"
        if db_path.exists():
            try:
                with _get_db_connection(cache_dir) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT cache_key FROM text_cache")
                    text_keys.update(row[0] for row in cursor.fetchall())

                    cursor.execute("SELECT cache_key FROM metadata_cache")
                    metadata_keys.update(row[0] for row in cursor.fetchall())

                    with conn:
                        conn.execute("DELETE FROM text_cache")
                        conn.execute("DELETE FROM metadata_cache")
            except Exception as e:
                logger.warning(f"Failed to clear DB cache: {e}")

        targets = [
            ("text", cache_dir / "text", text_keys),
            ("metadata", cache_dir / "metadata", metadata_keys),
        ]

        for _, target_path, keys_set in targets:
            if target_path.exists() and target_path.is_dir():
                try:
                    keys_set.update(f.name for f in target_path.iterdir() if f.is_file())
                    shutil.rmtree(target_path)
                except Exception as e:
                    logger.warning(f"Failed to clear cache directory at {target_path}: {e}")

        ocr_count = _clear_debug_cache()
        models_dev_count = _clear_models_dev_cache()

        return {
            "text": len(text_keys),
            "metadata": len(metadata_keys),
            "ocr_debug": ocr_count,
            "models_dev": models_dev_count,
        }


def _clear_debug_cache() -> int:
    """Clear OCR debug directory."""
    if not DEBUG_DIR.exists():
        return 0
    try:
        if DEBUG_DIR.is_dir():
            count = len(list(DEBUG_DIR.glob("*")))
            shutil.rmtree(DEBUG_DIR)
            return count
        DEBUG_DIR.unlink()
        return 1
    except Exception as e:
        logger.warning(f"Failed to clear ocr_debug cache at {DEBUG_DIR}: {e}")
        return 0


def _clear_models_dev_cache() -> int:
    """Clear models.dev cache file."""
    from renai.config.constants import APP_DATA_DIR

    models_dev_file = APP_DATA_DIR / "models_dev_cache.json"
    if models_dev_file.exists():
        try:
            models_dev_file.unlink()
            return 1
        except Exception as e:
            logger.warning(f"Failed to clear models.dev cache at {models_dev_file}: {e}")
    return 0
