"""
Cache management module - handles all caching operations for text and metadata.

This module was extracted from file_processor.py and utils.py as part of the 
refactoring to improve code organization and testability.

Cache Strategy Overview:
=======================

1. Text Cache (get_text_cache_path)
   - Location: CACHE_DIR/text/
   - Key: {file_hash}[_ocr][_mutool].txt
   - Invalidated when: OCR settings change
   
2. Metadata Cache (get_metadata_cache_path)
   - Location: CACHE_DIR/metadata/
   - Key: {file_hash}_{provider_model_hash}[_vision].json
   - Invalidated when: provider or model changes
   - Separate cache for vision mode

Why Provider-Model Aware Caching?
=================================
Different LLMs may produce different metadata for the same document.
By including the provider-model pair in the cache key, we ensure:
- Switching providers creates new cache entries (old preserved)
- Switching models creates new cache entries
- Using --update-metadata bypasses cache entirely

Example Cache Behavior:
=======================
# First run with DeepInfra - caches metadata
renai "C:/Books" --provider deepinfra

# Same files with OpenRouter - NEW cache entry (old one preserved)
renai "C:/Books" --provider openrouter

# Same provider/model - uses cached metadata
renai "C:/Books" --provider deepinfra

# Force refresh - bypasses cache
renai "C:/Books" --provider deepinfra --update-metadata

Module Functions:
=================
- get_text_cache_path(): Get path for text cache file
- get_metadata_cache_path(): Get path for metadata cache file  
- read_text_cache(): Read cached text content
- write_text_cache(): Write text content to cache
- read_metadata_cache(): Read cached metadata with validation
- write_metadata_cache(): Write metadata to cache
- clear_all_cache(): Clear all caches
- invalidate_cache_for_provider_model(): Clear metadata cache

Usage:
======
    from renai.cache.manager import (
        get_text_cache_path,
        read_text_cache,
        write_text_cache,
    )
    
    # Get cache path
    cache_path = get_text_cache_path(file_hash, ocr_enabled=False)
    
    # Read from cache
    cached_text = read_text_cache(cache_path)
    
    # Write to cache
    write_text_cache(cache_path, extracted_text)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ..settings import CACHE_DIR
from ..utils import (
    get_provider_model_key,
)


def get_text_cache_path(
    file_hash: str,
    ocr_enabled: bool = False,
    use_mutool: bool = False,
    cache_dir: Optional[Path] = None,
) -> Path:
    """
    Get the cache file path for extracted text.

    The cache key includes OCR settings to ensure cache invalidation when
    OCR mode changes (e.g., switching from regular to enhanced OCR).

    Args:
        file_hash: SHA256 hash of the source file. This uniquely identifies
                   the file content for cache lookup.
        ocr_enabled: Whether OCR was enabled during extraction. If True,
                    cache key includes "_ocr" suffix.
        use_mutool: Whether mutool was used for extraction. If True,
                   cache key includes "_mutool" suffix.
        cache_dir: Override the default cache directory. Defaults to CACHE_DIR
                  from config module.

    Returns:
        Path to the text cache file. The file may or may not exist.
        Path format: {cache_dir}/text/{file_hash}[_ocr][_mutool].txt

    Example:
        >>> path = get_text_cache_path("a" * 64)
        >>> # Returns: Path("C:/Users/user/AppData/Local/renAI/Cache/text/" + "a" * 64 + ".txt")
        >>> 
        >>> path = get_text_cache_path("a" * 64, ocr_enabled=True)
        >>> # Returns: Path(".../text/abc123_ocr.txt")

    Note:
        The cache directory and subdirectories are created automatically
        when writing to the cache, not when generating the path.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR

    # Include OCR settings in cache key to invalidate when OCR mode changes
    ocr_suffix = "_ocr" if ocr_enabled else ""
    mutool_suffix = "_mutool" if use_mutool else ""
    suffix = f"{ocr_suffix}{mutool_suffix}" if ocr_suffix or mutool_suffix else ""

    cache_key = f"{file_hash}{suffix}.txt"
    return cache_dir / "text" / cache_key


def get_metadata_cache_path(
    file_hash: str,
    provider: str,
    model: str,
    cache_dir: Optional[Path] = None,
    use_vision: bool = False,
) -> Path:
    """
    Get the cache file path for metadata.

    The cache key includes provider and model to ensure separate cache entries
    for different LLM providers/models. This allows switching between providers
    without losing cached results.

    Args:
        file_hash: SHA256 hash of the source file.
        provider: LLM provider name (e.g., "deepinfra", "openrouter", "openai").
        model: LLM model name (e.g., "Qwen/Qwen2.5-72B-Instruct").
        cache_dir: Override the default cache directory.
        use_vision: Whether vision mode was used. If True, includes "_vision"
                    suffix for separate cache.

    Returns:
        Path to the metadata cache file.
        Path format: {cache_dir}/metadata/{file_hash}_{provider_model_key}[_vision].json

    Example:
        >>> path = get_metadata_cache_path("a" * 64, "deepinfra", "llama-3.1-70b")
        >>> # Returns: Path(".../metadata/" + "a" * 64 + "_hashed_provider_model.json")
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR

    # Include provider-model in cache key for provider-model aware caching
    provider_model_key = get_provider_model_key(provider, model)
    vision_suffix = "_vision" if use_vision else ""

    cache_key = f"{file_hash}_{provider_model_key}{vision_suffix}.json"
    return cache_dir / "metadata" / cache_key


def read_text_cache(file_path: Path) -> Optional[str]:
    """
    Read cached text content.

    Args:
        file_path: Path to the cache file to read.

    Returns:
        Cached text content as string, or None if:
        - File doesn't exist
        - File is empty
        - Read error occurs (logs warning)

    Example:
        >>> content = read_text_cache(Path("cache/text/abc123.txt"))
        >>> if content:
        ...     print(f"Found cached text: {content[:100]}...")
    """
    try:
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
    except Exception as e:
        logging.warning(f"Failed to read text cache {file_path}: {e}")
    return None


def write_text_cache(file_path: Path, content: str) -> bool:
    """
    Write text content to cache.

    Creates the cache directory if it doesn't exist.

    Args:
        file_path: Path where to save the cached text.
        content: Text content to cache.

    Returns:
        True if successful, False if write failed (logs warning).

    Example:
        >>> success = write_text_cache(
        ...     Path("cache/text/abc123.txt"),
        ...     "Extracted text content..."
        ... )
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return True
    except Exception as e:
        logging.warning(f"Failed to write text cache {file_path}: {e}")
        return False


def read_metadata_cache(
    cache_path: Path,
    provider: str,
    model: str,
) -> Optional[Dict[str, Any]]:
    """
    Read cached metadata with provider-model validation.

    Validates that the cached metadata was generated by the same
    provider-model pair currently in use. If different, returns None
    to force re-generation.

    Args:
        cache_path: Path to the metadata cache file.
        provider: Current LLM provider name.
        model: Current LLM model name.

    Returns:
        Cached metadata as dictionary with keys like:
        - title, subtitle, author, year, category
        - _provider, _model (these are removed before return)
        
        Returns None if:
        - Cache file doesn't exist
        - JSON parse error
        - Provider/model doesn't match cached values

    Example:
        >>> metadata = read_metadata_cache(
        ...     Path("cache/metadata/abc123.json"),
        ...     "deepinfra",
        ...     "llama-3.1-70b"
        ... )
        >>> if metadata:
        ...     print(f"Title: {metadata.get('title')}")
    """
    try:
        if not cache_path.exists():
            return None

        cache_content = cache_path.read_text(encoding="utf-8")
        cache_data = json.loads(cache_content)

        # Validate provider-model match
        cached_provider = cache_data.get("_provider", "")
        cached_model = cache_data.get("_model", "")

        if cached_provider == provider and cached_model == model:
            # Remove internal cache fields before returning
            cache_data.pop("_provider", None)
            cache_data.pop("_model", None)
            return cache_data
        else:
            # Provider or model changed - cache invalid
            logging.info(
                f"Cache invalidated (was {cached_provider}/{cached_model}, "
                f"now {provider}/{model})"
            )
            return None

    except Exception as e:
        logging.warning(f"Failed to read metadata cache {cache_path}: {e}")
        return None


def write_metadata_cache(
    cache_path: Path,
    metadata: Dict[str, Any],
    provider: str,
    model: str,
) -> bool:
    """
    Write metadata to cache with provider-model info.

    Stores the provider and model used to generate the metadata
    so future reads can validate cache freshness.

    Args:
        cache_path: Path where to save the metadata.
        metadata: Metadata dictionary with keys like
                  title, subtitle, author, year, category.
        provider: LLM provider name.
        model: LLM model name.

    Returns:
        True if successful, False if write failed (logs warning).

    Example:
        >>> success = write_metadata_cache(
        ...     Path("cache/metadata/abc123.json"),
        ...     {"title": "Book Title", "author": "Author"},
        ...     "deepinfra",
        ...     "llama-3.1-70b"
        ... )
    """
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Add provider-model info for validation
        cache_data = {
            **metadata,
            "_provider": provider,
            "_model": model,
        }

        cache_path.write_text(
            json.dumps(cache_data, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        return True
    except Exception as e:
        logging.warning(f"Failed to write metadata cache {cache_path}: {e}")
        return False


def invalidate_cache_for_provider_model(
    cache_dir: Optional[Path] = None,
) -> int:
    """
    Clear metadata cache when switching providers/models.

    This can be used when you want to force regeneration of all
    metadata by clearing the metadata cache entirely.

    Args:
        cache_dir: Override the default cache directory.

    Returns:
        Number of cache files removed. 0 if no cache exists or error.

    Example:
        >>> count = invalidate_cache_for_provider_model()
        >>> print(f"Cleared {count} cached entries")
    """
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


def clear_all_cache(cache_dir: Optional[Path] = None) -> Dict[str, int]:
    """
    Clear all caches (text and metadata).

    Removes both text and metadata cache directories and all their contents.

    Args:
        cache_dir: Override the default cache directory.

    Returns:
        Dictionary with counts of cleared files per cache type:
        {"text": <count>, "metadata": <count>}

    Example:
        >>> results = clear_all_cache()
        >>> print(f"Cleared {results['text']} text files, {results['metadata']} metadata files")
    """
    import shutil

    if cache_dir is None:
        cache_dir = CACHE_DIR

    results = {"text": 0, "metadata": 0}

    for cache_type in ["text", "metadata"]:
        cache_path = cache_dir / cache_type
        if cache_path.exists():
            try:
                count = len(list(cache_path.glob("*")))
                shutil.rmtree(cache_path)
                results[cache_type] = count
            except Exception as e:
                logging.warning(f"Failed to clear {cache_type} cache: {e}")

    return results
