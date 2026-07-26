import contextlib
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Any, TypeVar, overload

from pydantic import BaseModel

from renai.config import CACHE_DIR, DEBUG_DIR
from renai.utils import (
    get_provider_model_key,
)

T = TypeVar("T", bound=BaseModel)

# Thread-safe lock for all cache database operations
_cache_lock = threading.Lock()


def _get_db_dir(cache_path: Path | None = None, cache_dir: Path | None = None) -> Path:
    """Determine the database directory given a cache path or explicit cache dir."""
    if cache_dir is not None:
        return cache_dir
    if cache_path is not None:
        parent = cache_path.parent
        if parent.name in ("text", "metadata"):
            return parent.parent
        return parent
    return CACHE_DIR


def _get_db_connection(db_dir: Path) -> sqlite3.Connection:
    """Create and initialize a SQLite database connection with WAL mode enabled."""
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "cache.db"
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    _init_db(conn)
    return conn


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


def get_text_cache_path(
    file_hash: str,
    *,
    ocr_enabled: bool = False,
    use_mutool: bool = False,
    cache_dir: Path | None = None,
) -> Path:
    """Get the cache file path key for extracted text."""
    if cache_dir is None:
        cache_dir = CACHE_DIR

    ocr_suffix = "_ocr" if ocr_enabled else ""
    mutool_suffix = "_mutool" if use_mutool else ""
    suffix = f"{ocr_suffix}{mutool_suffix}" if ocr_suffix or mutool_suffix else ""

    cache_key = f"{file_hash}{suffix}.txt"
    return cache_dir / "text" / cache_key


def get_metadata_cache_path(
    file_hash: str,
    provider: str,
    model: str,
    *,
    mode: str = "doc",
    cache_dir: Path | None = None,
    use_vision: bool = False,
) -> Path:
    """Get the cache file path key for metadata."""
    if cache_dir is None:
        cache_dir = CACHE_DIR

    provider_model_key = get_provider_model_key(provider, model)
    vision_suffix = "_vision" if use_vision else ""

    cache_key = f"{mode}_{file_hash}_{provider_model_key}{vision_suffix}.json"
    return cache_dir / "metadata" / cache_key


def read_text_cache(file_path: Path) -> str | None:
    """Read cached text content from SQLite (with legacy file fallback)."""
    with _cache_lock:
        db_dir = _get_db_dir(cache_path=file_path)
        cache_key = file_path.name
        try:
            with _get_db_connection(db_dir) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT content FROM text_cache WHERE cache_key = ?", (cache_key,)
                )
                row = cursor.fetchone()
                if row:
                    return row[0]
        except Exception as e:
            logging.warning(f"Failed to read text cache {file_path}: {e}")
            return None

        # Fallback to legacy file reading if present
        try:
            if file_path.exists():
                return file_path.read_text(encoding="utf-8")
        except Exception as e:
            logging.warning(f"Failed to read text cache {file_path}: {e}")
    return None


def write_text_cache(file_path: Path, content: str) -> bool:
    """Write text content to SQLite cache atomically."""
    with _cache_lock:
        db_dir = _get_db_dir(cache_path=file_path)
        cache_key = file_path.name
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=file_path.parent, prefix=".renai_tmp_", suffix=".txt"
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    f.write(content)
                Path(tmp_path).replace(file_path)
            except Exception:
                with contextlib.suppress(OSError):
                    Path(tmp_path).unlink()
                raise

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
            logging.warning(f"Failed to write text cache {file_path}: {e}")
            return False


@overload
def read_metadata_cache(
    cache_path: Path,
    provider: str,
    model: str,
    model_class: None = None,
) -> dict[str, Any] | None: ...


@overload
def read_metadata_cache[V: BaseModel](
    cache_path: Path,
    provider: str,
    model: str,
    model_class: type[V],
) -> V | None: ...


def _read_from_legacy_file(cache_path: Path, provider: str, model: str) -> dict[str, Any] | None:
    """Helper to read and validate stale/legacy metadata cache file."""
    if not cache_path.exists():
        return None
    cache_content = cache_path.read_text(encoding="utf-8")
    cache_data = json.loads(cache_content)

    cached_provider = cache_data.get("_provider", "")
    cached_model = cache_data.get("_model", "")

    if cached_provider == provider and cached_model == model:
        cache_data.pop("_provider", None)
        cache_data.pop("_model", None)
        return cache_data

    msg = f"Cache invalidated (was {cached_provider}/{cached_model}, now {provider}/{model})"
    logging.debug(msg)
    with contextlib.suppress(Exception):
        cache_path.unlink()
        logging.debug(f"Removed stale cache file: {cache_path}")
    return None


def read_metadata_cache(
    cache_path: Path,
    provider: str,
    model: str,
    model_class: type[BaseModel] | None = None,
) -> Any | None:
    """Read cached metadata from SQLite (with legacy file fallback)."""
    with _cache_lock:
        db_dir = _get_db_dir(cache_path=cache_path)
        cache_key = cache_path.name
        cache_data = None

        try:
            with _get_db_connection(db_dir) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT provider, model, metadata_json "
                    "FROM metadata_cache WHERE cache_key = ?",
                    (cache_key,),
                )
                row = cursor.fetchone()
                if row:
                    cached_provider, cached_model, metadata_json = row
                    if cached_provider == provider and cached_model == model:
                        cache_data = json.loads(metadata_json)
                    else:
                        msg = (
                            f"Cache invalidated (was {cached_provider}/{cached_model}, "
                            f"now {provider}/{model})"
                        )
                        logging.debug(msg)
                        conn.execute(
                            "DELETE FROM metadata_cache WHERE cache_key = ?", (cache_key,)
                        )
        except Exception as e:
            logging.warning(f"Failed to read metadata cache {cache_path}: {e}")

        if cache_data is None:
            try:
                cache_data = _read_from_legacy_file(cache_path, provider, model)
            except Exception as e:
                logging.warning(f"Failed to read metadata cache {cache_path}: {e}")

        if cache_data is None:
            return None

        if model_class:
            try:
                return model_class(**cache_data)
            except Exception as e:
                logging.warning(
                    f"Failed to validate cached metadata against {model_class.__name__}: {e}"
                )
                return None
        return cache_data


def write_metadata_cache(
    cache_path: Path,
    metadata: BaseModel | dict[str, Any],
    provider: str,
    model: str,
) -> bool:
    """Write metadata to SQLite cache atomically."""
    with _cache_lock:
        db_dir = _get_db_dir(cache_path=cache_path)
        cache_key = cache_path.name
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = metadata.model_dump() if isinstance(metadata, BaseModel) else metadata
            metadata_json = json.dumps(data, ensure_ascii=False, indent=4)

            # Write legacy file for compatibility
            cache_data = {**data, "_provider": provider, "_model": model}
            content = json.dumps(cache_data, ensure_ascii=False, indent=4)
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=cache_path.parent, prefix=".renai_tmp_", suffix=".json"
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    f.write(content)
                Path(tmp_path).replace(cache_path)
            except Exception:
                with contextlib.suppress(OSError):
                    Path(tmp_path).unlink()
                raise

            with _get_db_connection(db_dir) as conn, conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO metadata_cache
                    (cache_key, provider, model, metadata_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (cache_key, provider, model, metadata_json),
                )
            return True
        except Exception as e:
            logging.warning(f"Failed to write metadata cache {cache_path}: {e}")
            return False


def invalidate_cache_for_provider_model(cache_dir: Path | None = None) -> int:
    """Clear metadata cache when switching providers/models."""
    with _cache_lock:
        if cache_dir is None:
            cache_dir = CACHE_DIR

        db_path = cache_dir / "cache.db"
        count = 0
        if db_path.exists():
            try:
                with _get_db_connection(cache_dir) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM metadata_cache")
                    count = cursor.fetchone()[0]
                    with conn:
                        conn.execute("DELETE FROM metadata_cache")
            except Exception as e:
                logging.warning(f"Failed to clear metadata cache from DB: {e}")

        metadata_dir = cache_dir / "metadata"
        if metadata_dir.exists():
            try:
                for cache_file in metadata_dir.glob("*.json"):
                    cache_file.unlink()
                    count += 1
            except Exception as e:
                logging.warning(f"Failed to clear metadata cache files: {e}")
        return count


def clear_all_cache(cache_dir: Path | None = None) -> dict[str, int]:
    """Clear all caches (text, metadata, and ocr_debug)."""
    with _cache_lock:
        if cache_dir is None:
            cache_dir = CACHE_DIR

        results = {"text": 0, "metadata": 0, "ocr_debug": 0}

        db_path = cache_dir / "cache.db"
        if db_path.exists():
            try:
                with _get_db_connection(cache_dir) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM text_cache")
                    results["text"] = cursor.fetchone()[0]
                    cursor.execute("SELECT COUNT(*) FROM metadata_cache")
                    results["metadata"] = cursor.fetchone()[0]
                    with conn:
                        conn.execute("DELETE FROM text_cache")
                        conn.execute("DELETE FROM metadata_cache")
            except Exception as e:
                logging.warning(f"Failed to clear DB cache: {e}")

        targets = [
            ("text", cache_dir / "text"),
            ("metadata", cache_dir / "metadata"),
            ("ocr_debug", DEBUG_DIR),
        ]

        for cache_type, target_path in targets:
            if target_path.exists():
                try:
                    if target_path.is_dir():
                        file_count = len(list(target_path.glob("*")))
                        shutil.rmtree(target_path)
                        results[cache_type] = max(results[cache_type], file_count)
                    else:
                        results[cache_type] = max(results[cache_type], 1)
                        target_path.unlink()
                except Exception as e:
                    logging.warning(f"Failed to clear {cache_type} cache at {target_path}: {e}")

        return results
