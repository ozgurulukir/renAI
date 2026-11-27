import contextlib
import hashlib
import json
import logging
import shutil
import sqlite3
import threading
from collections.abc import Generator
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple, overload

from pydantic import BaseModel

from renai.config import CACHE_DIR, DEBUG_DIR
from renai.utils import ensure_secure_dir

logger = logging.getLogger(__name__)

# Thread-safe lock for all cache database operations. Held by the public
# read/write/clear operations; ``_get_db_connection`` expects it to be held
# by its caller and must not acquire it again (Lock is not reentrant).
_cache_lock = threading.Lock()


class _CacheStore:
    """Owns the persistent SQLite connection for one cache directory.

    One connection is opened per cache root (lazily, on first use) and reused
    for every operation, instead of opening a fresh connection and re-running
    the secure-dir setup, DDL, and PRAGMAs on each call. All use is serialized
    by ``_cache_lock`` held by the public operations, so a single connection
    with ``check_same_thread=False`` is safe across worker threads. A broken
    connection is discarded so the next operation opens a fresh one rather
    than poisoning the store for the rest of the run.
    """

    def __init__(self, db_dir: Path) -> None:
        self._db_dir = db_dir
        self._conn: sqlite3.Connection | None = None

    @contextlib.contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        if self._conn is None:
            self._conn = self._open()
        try:
            yield self._conn
        except sqlite3.DatabaseError:
            self._close_quietly()
            raise

    def _open(self) -> sqlite3.Connection:
        ensure_secure_dir(self._db_dir)
        # Short busy timeout: a foreign exclusive lock must not stall workers
        # for tens of seconds; failing fast degrades to a (logged) cache miss.
        conn = sqlite3.connect(
            str(self._db_dir / "cache.db"),
            timeout=_BUSY_TIMEOUT_SECONDS,
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        _init_db(conn)
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _close_quietly(self) -> None:
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None


# One persistent store per resolved cache root for the life of the process.
_stores: dict[Path, _CacheStore] = {}

# Short busy timeout (seconds) for the SQLite connection so a foreign lock
# cannot stall worker threads; see #248.
_BUSY_TIMEOUT_SECONDS = 5.0

# Bumped whenever the on-disk row grammar or invalidation contract changes in a
# way old rows cannot honor (see #246/#247). Stored via ``PRAGMA user_version``.
CACHE_SCHEMA_VERSION = 2


class TextCacheKey(NamedTuple):
    """Structured identity of a cached text-extraction result.

    The on-disk key grammar (``{hash}_v2_{fallback}_{fingerprint}.txt``) encodes:

    - ``file_hash``: content identity of the source file.
    - ``fallback``: the requested :class:`FallbackMode` value, ALWAYS part of
      the key — OCR, vision and move-to-folder runs produce different text for
      the same file and must not share a row (previously the fallback was only
      encoded via the OCR enhancement flags, which default to off, so an
      OCR-produced row was served to vision requests; #246).
    - ``settings_fingerprint``: digest of the extraction settings that change
      the produced text (target length, extraction mode, head ratio, page
      limit, extractor backend, OCR language/DPI/PSM). See
      :func:`text_settings_fingerprint`.

    The ``v2`` marker makes rows written by earlier versions (keyed
    ``{hash}.txt`` / ``{hash}_ocr_mutool.txt``) natural misses that are simply
    repopulated on the next extraction — no migration needed.
    """

    file_hash: str
    fallback: str = "ocr"
    settings_fingerprint: str = ""


def text_settings_fingerprint(
    *,
    text_length: int,
    extraction_mode: str,
    head_ratio: float,
    extraction_pages: int,
    extractor: str,
    ocr_lang: str,
    ocr_dpi: int,
    ocr_psm: int,
    ocr_enhanced: bool = False,
    use_mutool: bool = False,
    ocr_head_pages: int = 0,
    ocr_tail_pages: int = 0,
    subprocess_timeout: int = 0,
    mutool_path: str = "",
) -> str:
    """Stable fingerprint of the settings that change extracted text (#246).

    Any change to one of these settings must produce a text-cache miss so the
    previously trimmed/cached text is re-extracted under the new settings.
    """
    payload = json.dumps(
        [
            int(text_length),
            str(extraction_mode),
            float(head_ratio),
            int(extraction_pages),
            str(extractor),
            str(ocr_lang or ""),
            int(ocr_dpi),
            int(ocr_psm),
            bool(ocr_enhanced),
            bool(use_mutool),
            int(ocr_head_pages),
            int(ocr_tail_pages),
            int(subprocess_timeout),
            str(mutool_path or ""),
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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
    """Encode a :class:`TextCacheKey` to its on-disk/cache-row key.

    The ``v2`` segment means pre-#246 rows (``{hash}.txt`` etc.) can never
    match a current key: they are invalid by construction and repopulated on
    the next extraction.
    """
    return f"{key.file_hash}_v2_{key.fallback}_{key.settings_fingerprint}.txt"


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


@lru_cache(maxsize=1)
def metadata_version_fingerprint() -> str:
    """Version fingerprint binding metadata rows to prompts, schemas and app (#247).

    Digest of (application version, sorted ``prompts/*.txt`` contents,
    ``BookMetadata``/``ImageMetadata`` ``model_json_schema()``). Stored with
    every metadata row; a mismatch on read is treated as a cache miss, so a
    prompt edit, schema change, or app upgrade invalidates affected rows even
    when the Pydantic validation would still pass.
    """
    import renai
    from renai.models.schemas import BookMetadata, ImageMetadata

    digest = hashlib.sha256()
    digest.update(renai.__version__.encode("utf-8"))
    prompts_dir = Path(__file__).parent.parent / "prompts"
    for prompt_path in sorted(prompts_dir.glob("*.txt")):
        digest.update(prompt_path.name.encode("utf-8"))
        digest.update(prompt_path.read_bytes())
    for schema_model in (BookMetadata, ImageMetadata):
        digest.update(json.dumps(schema_model.model_json_schema(), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:16]


def _resolve_cache_dir(cache_dir: Path | None) -> Path:
    """Default the storage root to the package CACHE_DIR when not given."""
    return CACHE_DIR if cache_dir is None else cache_dir


def _get_db_connection(db_dir: Path) -> contextlib.AbstractContextManager[sqlite3.Connection]:
    """Return the persistent-connection context manager for ``db_dir``.

    The connection is opened (secure dir, WAL, DDL, synchronous pragma) on
    first use and reused for every subsequent operation against the same
    cache root. The caller must hold ``_cache_lock``.
    """
    key = db_dir.resolve()
    store = _stores.get(key)
    if store is None:
        store = _CacheStore(db_dir)
        _stores[key] = store
    return store.connection()


def _init_db(conn: sqlite3.Connection) -> None:
    """Initialize database tables for text and metadata caching.

    The physical schema version is recorded in ``PRAGMA user_version`` so the
    store self-describes its layout (#247). The cached-content validity is
    handled separately per row (key grammar + ``_version_fp`` fingerprint).
    """
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
        conn.execute(f"PRAGMA user_version = {CACHE_SCHEMA_VERSION};")


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
    version_fingerprint: str | None = None


@overload
def read_metadata_cache[V: BaseModel](
    key: MetadataCacheKey,
    model_class: type[V],
    cache_dir: Path | None = None,
) -> tuple[V | None, CacheSource | None]: ...


@overload
def read_metadata_cache(
    key: MetadataCacheKey,
    model_class: None = None,
    cache_dir: Path | None = None,
) -> tuple[dict[str, Any] | None, CacheSource | None]: ...


def read_metadata_cache[V: BaseModel](
    key: MetadataCacheKey,
    model_class: type[V] | None = None,
    cache_dir: Path | None = None,
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

    Validity judgments (rule signature, producing identity, version
    fingerprint) live with the caller — the cache middleware is the single
    place that applies them (#247); this function only surfaces the stored
    provenance.
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
        version_fingerprint = cache_data.pop("_version_fp", None) if cache_data else None
        if version_fingerprint is not None and not isinstance(version_fingerprint, str):
            logger.warning(
                f"Corrupt version fingerprint in metadata cache row {cache_key}: expected "
                f"a string, got {type(version_fingerprint).__name__}; treating as legacy"
            )
            version_fingerprint = None

        source = CacheSource(
            is_vision_doc=is_vision_doc,
            rule_signature=rule_signature,
            produced_by_provider=produced[0] if produced else None,
            produced_by_model=produced[1] if produced else None,
            version_fingerprint=version_fingerprint,
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
    version_fingerprint: str | None = None,
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
    apart from global ones. ``version_fingerprint`` (see
    :func:`metadata_version_fingerprint`) is persisted under ``_version_fp``;
    it defaults to the current fingerprint so rows always carry the prompts,
    schemas and app version they were extracted under (#247).
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
            data["_version_fp"] = (
                version_fingerprint
                if version_fingerprint is not None
                else metadata_version_fingerprint()
            )
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
    """Clear all caches (text, metadata, and ocr_debug) without double-counting.

    On a corrupt database the ``cache.db`` (plus WAL/SHM sidecars) is removed
    and an empty store recreated, reported honestly via the ``db_reset`` count
    instead of silently reporting "already empty" (#248). After a successful
    delete a ``VACUUM`` reclaims the freed pages so clearing shrinks the file.
    """
    with _cache_lock:
        if cache_dir is None:
            cache_dir = CACHE_DIR

        text_keys: set[str] = set()
        metadata_keys: set[str] = set()
        db_reset = 0

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
                    conn.execute("VACUUM;")
            except sqlite3.DatabaseError as e:
                logger.warning(f"Cache database is corrupt ({e}); resetting it from scratch")
                db_reset = _reset_corrupt_db(cache_dir)
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
        models_dev_count = _clear_models_dev_cache(cache_dir)

        return {
            "text": len(text_keys),
            "metadata": len(metadata_keys),
            "ocr_debug": ocr_count,
            "models_dev": models_dev_count,
            "db_reset": db_reset,
        }


def _reset_corrupt_db(cache_dir: Path) -> int:
    """Delete a corrupt ``cache.db`` (with sidecars) and recreate an empty store.

    The caller holds ``_cache_lock``. Returns 1 on a successful reset, 0 when
    the corrupt files could not be removed (the failure is reported honestly
    by the caller surfacing ``db_reset = 0``).
    """
    key = cache_dir.resolve()
    store = _stores.pop(key, None)
    if store is not None:
        store._close_quietly()
    removed = True
    for name in ("cache.db", "cache.db-wal", "cache.db-shm"):
        path = cache_dir / name
        if path.exists():
            try:
                path.unlink()
            except OSError as e:
                logger.warning(f"Failed to remove corrupt cache file {path}: {e}")
                removed = False
    if not removed:
        return 0
    try:
        with _get_db_connection(cache_dir):
            pass  # recreate the store (DDL + PRAGMA user_version) immediately
    except Exception as e:
        logger.warning(f"Failed to recreate cache database after reset: {e}")
        return 0
    return 1


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


def _clear_models_dev_cache(cache_dir: Path) -> int:
    """Clear the models.dev cache file.

    The path is derived from the same root as the passed ``cache_dir``
    (``CACHE_DIR`` lives directly under ``APP_DATA_DIR``) instead of the
    global ``APP_DATA_DIR`` constant, so tests pointing ``cache_dir`` at a
    temporary directory cannot delete the developer's real cache (#248).
    """
    models_dev_file = cache_dir.parent / "models_dev_cache.json"
    if models_dev_file.exists():
        try:
            models_dev_file.unlink()
            return 1
        except Exception as e:
            logger.warning(f"Failed to clear models.dev cache at {models_dev_file}: {e}")
    return 0
