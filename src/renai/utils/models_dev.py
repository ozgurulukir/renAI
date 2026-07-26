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


def get_models_catalog(force_refresh: bool = False) -> dict:
    """Fetch and return the models.dev catalog, caching it locally."""
    # Ensure cache directory exists
    with contextlib.suppress(Exception):
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Try loading from cache if it exists, is fresh, and force_refresh is False
    if not force_refresh and CACHE_FILE.exists():
        try:
            mtime = CACHE_FILE.stat().st_mtime
            if time.time() - mtime < CACHE_EXPIRY:
                with CACHE_FILE.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and data:
                        return data
        except Exception as e:
            logging.debug(f"Failed to read models.dev cache: {e}")

    # Fetch fresh data from models.dev
    try:
        headers = {
            "User-Agent": "renAI/2.2.0 (https://github.com/ozgurulukir/renAI)",
            "Accept": "application/json",
        }
        response = requests.get("https://models.dev/models.json", headers=headers, timeout=6.0)
        if response.status_code == HTTP_STATUS_OK:
            data = response.json()
            if isinstance(data, dict) and data:
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
            data = json.load(f)
            if isinstance(data, dict) and data:
                return data

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

    raw = model_name.strip()
    if ":" in raw and not raw.startswith("http://") and not raw.startswith("https://"):
        raw = raw.split(":")[0]

    model_name_lower = raw.lower()

    # 1. Direct match
    if model_name_lower in catalog:
        return _check_vision_modalities(catalog[model_name_lower])

    # Standardize provider prefix aliases
    normalized = model_name_lower
    for prefix in ("openrouter/", "deepinfra/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]

    if normalized.startswith("meta-llama/"):
        normalized = "meta/" + normalized[len("meta-llama/") :]
    elif normalized.startswith("mistralai/"):
        normalized = "mistral/" + normalized[len("mistralai/") :]

    if normalized in catalog:
        return _check_vision_modalities(catalog[normalized])

    # 2. Try matching by model basename (suffix)
    target_basename = normalized.split("/")[-1]
    for key, info in catalog.items():
        key_lower = key.lower()
        key_basename = key_lower.split("/")[-1]
        if key_basename == target_basename:
            return _check_vision_modalities(info)

    return None


def _check_vision_modalities(model_info: dict) -> bool:
    modalities = model_info.get("modalities", {})
    inputs = modalities.get("input", [])
    return "image" in inputs

