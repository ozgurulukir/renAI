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


def _load_cached_catalog(force_refresh: bool) -> dict | None:
    """Try loading catalog data from local cache file."""
    if force_refresh or not CACHE_FILE.exists():
        return None
    try:
        mtime = CACHE_FILE.stat().st_mtime
        if time.time() - mtime < CACHE_EXPIRY:
            with CACHE_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and (data.get("models") or data.get("providers")):
                    return data
    except Exception as e:
        logging.debug(f"Failed to read models.dev cache: {e}")
    return None


def _fetch_remote_json(endpoint: str) -> dict | None:
    """Fetch remote JSON from models.dev endpoint."""
    headers = {
        "User-Agent": "renAI/2.2.0 (https://github.com/ozgurulukir/renAI)",
        "Accept": "application/json",
    }
    url = f"https://models.dev/{endpoint}"
    try:
        response = requests.get(url, headers=headers, timeout=6.0)
        if response.status_code == HTTP_STATUS_OK:
            data = response.json()
            if isinstance(data, dict) and data:
                return data
    except Exception as e:
        logging.debug(f"Failed to fetch {url}: {e}")
    return None


def get_full_catalog(force_refresh: bool = False) -> dict:
    """Fetch and return the complete models.dev catalog (models + providers)."""
    with contextlib.suppress(Exception):
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    cached = _load_cached_catalog(force_refresh)
    if cached:
        return cached

    # Try fetching catalog.json first
    data = _fetch_remote_json("catalog.json")
    if data and (data.get("models") or data.get("providers")):
        with contextlib.suppress(Exception), CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return data

    # Fallback: try fetching models.json
    models_data = _fetch_remote_json("models.json")
    if models_data:
        data = {"models": models_data, "providers": {}}
        with contextlib.suppress(Exception), CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return data

    # Fallback to expired cache if fetch failed
    if CACHE_FILE.exists():
        with contextlib.suppress(Exception), CACHE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and (data.get("models") or data.get("providers")):
                return data

    return {"models": {}, "providers": {}}



def get_models_catalog(force_refresh: bool = False) -> dict:
    """Fetch and return canonical models catalog dictionary."""
    full = get_full_catalog(force_refresh=force_refresh)
    return full.get("models", {})


def get_providers_catalog(force_refresh: bool = False) -> dict:
    """Fetch and return providers catalog dictionary."""
    full = get_full_catalog(force_refresh=force_refresh)
    return full.get("providers", {})


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


