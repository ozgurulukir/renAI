import contextlib
import json
import logging
import os
import shutil
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

# Thread-safe lock for all cache operations (read + write + invalidate)
_cache_lock = threading.Lock()


def get_text_cache_path(
    file_hash: str,
    *,
    ocr_enabled: bool = False,
    use_mutool: bool = False,
    cache_dir: Path | None = None,
) -> Path:
    """Get the cache file path for extracted text."""
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
    """Get the cache file path for metadata."""
    if cache_dir is None:
        cache_dir = CACHE_DIR

    provider_model_key = get_provider_model_key(provider, model)
    vision_suffix = "_vision" if use_vision else ""

    cache_key = f"{mode}_{file_hash}_{provider_model_key}{vision_suffix}.json"
    return cache_dir / "metadata" / cache_key


def read_text_cache(file_path: Path) -> str | None:
    """Read cached text content."""
    with _cache_lock:
        try:
            if file_path.exists():
                return file_path.read_text(encoding="utf-8")
        except Exception as e:
            logging.warning(f"Failed to read text cache {file_path}: {e}")
    return None


def write_text_cache(file_path: Path, content: str) -> bool:
    """Write text content to cache atomically."""
    with _cache_lock:
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


def read_metadata_cache(
    cache_path: Path,
    provider: str,
    model: str,
    model_class: type[BaseModel] | None = None,
) -> Any | None:
    """
    Read cached metadata with provider-model validation.
    If model_class is provided, returns a validated Pydantic model instance.
    """
    with _cache_lock:
        try:
            if not cache_path.exists():
                return None

            cache_content = cache_path.read_text(encoding="utf-8")
            cache_data = json.loads(cache_content)

            # Validate provider-model match
            cached_provider = cache_data.get("_provider", "")
            cached_model = cache_data.get("_model", "")

            if cached_provider == provider and cached_model == model:
                # Remove internal cache fields
                cache_data.pop("_provider", None)
                cache_data.pop("_model", None)

                if model_class:
                    try:
                        return model_class(**cache_data)
                    except Exception as e:
                        logging.warning(
                            f"Failed to validate cached metadata "
                            f"against {model_class.__name__}: {e}"
                        )
                        return None
                return cache_data
            logging.info(
                f"Cache invalidated (was {cached_provider}/{cached_model}, now {provider}/{model})"
            )
            try:
                cache_path.unlink()
                logging.debug(f"Removed stale cache file: {cache_path}")
            except Exception:
                pass
            return None

        except Exception as e:
            logging.warning(f"Failed to read metadata cache {cache_path}: {e}")
            return None


def write_metadata_cache(
    cache_path: Path,
    metadata: BaseModel | dict[str, Any],
    provider: str,
    model: str,
) -> bool:
    """
    Write metadata to cache atomically.
    Supports both Pydantic models (recommended) and dictionaries.
    """
    with _cache_lock:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            data = metadata.model_dump() if isinstance(metadata, BaseModel) else metadata

            cache_data = {
                **data,
                "_provider": provider,
                "_model": model,
            }

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
            return True
        except Exception as e:
            logging.warning(f"Failed to write metadata cache {cache_path}: {e}")
            return False


def invalidate_cache_for_provider_model(cache_dir: Path | None = None) -> int:
    """Clear metadata cache when switching providers/models."""
    with _cache_lock:
        if cache_dir is None:
            cache_dir = CACHE_DIR

        metadata_dir = cache_dir / "metadata"
        if not metadata_dir.exists():
            return 0

        count = 0
        try:
            for cache_file in metadata_dir.glob("*.json"):
                cache_file.unlink()
                count += 1
        except Exception as e:
            logging.warning(f"Failed to clear metadata cache: {e}")
        return count


def clear_all_cache(cache_dir: Path | None = None) -> dict[str, int]:
    """Clear all caches (text, metadata, and ocr_debug)."""
    with _cache_lock:
        if cache_dir is None:
            cache_dir = CACHE_DIR

        results = {"text": 0, "metadata": 0, "ocr_debug": 0}

        # Define targets
        targets = [
            ("text", cache_dir / "text"),
            ("metadata", cache_dir / "metadata"),
            ("ocr_debug", DEBUG_DIR),
        ]

        for cache_type, target_path in targets:
            if target_path.exists():
                try:
                    # Count files for reporting
                    if target_path.is_dir():
                        count = len(list(target_path.glob("*")))
                        shutil.rmtree(target_path)
                    else:
                        count = 1
                        target_path.unlink()
                    results[cache_type] = count
                except Exception as e:
                    logging.warning(f"Failed to clear {cache_type} cache at {target_path}: {e}")

        return results
