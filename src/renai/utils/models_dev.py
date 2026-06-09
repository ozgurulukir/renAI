"""
models_dev.py - Integration with models.dev directory for AI model capabilities.
"""

import contextlib
import json
import logging
import time

import requests

from renai.config.constants import APP_DATA_DIR

CACHE_FILE = APP_DATA_DIR / "models_dev_cache.json"
CACHE_EXPIRY = 86400  # 24 hours in seconds
HTTP_STATUS_OK = 200


def get_models_catalog() -> dict:
    """Fetch and return the models.dev catalog, caching it locally."""
    # Ensure cache directory exists
    with contextlib.suppress(Exception):
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Try loading from cache if it exists and is fresh
    if CACHE_FILE.exists():
        try:
            mtime = CACHE_FILE.stat().st_mtime
            if time.time() - mtime < CACHE_EXPIRY:
                with CACHE_FILE.open("r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logging.debug(f"Failed to read models.dev cache: {e}")

    # Fetch fresh data from models.dev
    try:
        response = requests.get("https://models.dev/models.json", timeout=3.0)
        if response.status_code == HTTP_STATUS_OK:
            data = response.json()
            try:
                with CACHE_FILE.open("w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
            except Exception as e:
                logging.debug(f"Failed to write models.dev cache: {e}")
            return data
    except Exception as e:
        logging.debug(f"Failed to fetch models.dev catalog: {e}")

    # Fallback to expired cache if fetch failed
    if CACHE_FILE.exists():
        with contextlib.suppress(Exception), CACHE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def is_vision_supported(model_name: str) -> bool | None:
    """Check if the given model name supports vision modalities.

    Returns:
        True: if supported
        False: if explicitly not supported
        None: if the model is not found in the catalog (unknown)
    """
    catalog = get_models_catalog()
    if not catalog:
        return None

    # Canonicalize model name (e.g. openai/gpt-4o -> openai/gpt-4o, gpt-4o -> openai/gpt-4o)
    model_name_lower = model_name.lower()

    # Try direct match
    if model_name_lower in catalog:
        model_info = catalog[model_name_lower]
        modalities = model_info.get("modalities", {})
        inputs = modalities.get("input", [])
        return "image" in inputs

    # Try suffix match (e.g. "gpt-4o" matching "openai/gpt-4o" or similar)
    for key, info in catalog.items():
        parts = key.lower().split("/")
        if len(parts) > 1 and parts[1] == model_name_lower:
            modalities = info.get("modalities", {})
            inputs = modalities.get("input", [])
            return "image" in inputs

    return None
