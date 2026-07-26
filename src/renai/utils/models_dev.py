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
            logging.debug(f"Endpoint {url} returned empty or non-dict response: {data}")
        else:
            logging.debug(f"Endpoint {url} returned HTTP status {response.status_code}")
    except Exception as e:
        logging.debug(f"Failed to fetch {url}: {e}")
    return None


KNOWN_OPENAI_PROVIDERS = {
    "openai",
    "openrouter",
    "deepinfra",
    "groq",
    "cerebras",
    "togetherai",
    "together",
    "mistral",
    "xai",
    "deepseek",
    "google",
    "perplexity",
    "perplexity-agent",
}


def is_openai_supported_provider(provider_id: str, provider_info: dict) -> bool:
    """Check whether a provider supports an OpenAI-compatible endpoint.

    Returns True if:
      - Provider ID is in known OpenAI-compatible providers
      - npm package indicates OpenAI compatibility (@ai-sdk/openai-compatible, @ai-sdk/openai)
      - api endpoint is provided and is an OpenAI-compatible path (not /anthropic/)
      - Neither npm nor api is set (minimal test mock data fallback)
    """
    p_id = provider_id.lower()
    if p_id in KNOWN_OPENAI_PROVIDERS:
        return True
    npm = provider_info.get("npm")
    api = provider_info.get("api")
    if not npm and not api:
        return True
    if npm in ("@ai-sdk/openai-compatible", "@ai-sdk/openai"):
        return True
    api_str = api or ""
    has_openai_path = (
        "/v1" in api_str or "/openai" in api_str or "/compatible" in api_str or "/paas/" in api_str
    )
    return bool(api_str and "/anthropic/" not in api_str and has_openai_path)


def _get_raw_catalog(force_refresh: bool = False) -> dict:
    """Fetch raw catalog from cache or remote endpoint."""
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


def get_full_catalog(force_refresh: bool = False, openai_only: bool = True) -> dict:
    """Fetch and return the complete models.dev catalog (models + providers).

    When openai_only is True (default), only providers that offer OpenAI-compatible endpoints
    and their associated models are returned.
    """
    raw = _get_raw_catalog(force_refresh=force_refresh)
    if not openai_only:
        return raw

    providers = raw.get("providers", {})
    models = raw.get("models", {})

    if providers:
        filtered_providers = {
            k: v for k, v in providers.items() if is_openai_supported_provider(k, v)
        }
    else:
        filtered_providers = providers

    if models and filtered_providers:
        filtered_models = {k: v for k, v in models.items() if k.split("/")[0] in filtered_providers}
    else:
        filtered_models = models

    return {"models": filtered_models, "providers": filtered_providers}


def get_models_catalog(force_refresh: bool = False, openai_only: bool = True) -> dict:
    """Fetch and return canonical models catalog dictionary."""
    full = get_full_catalog(force_refresh=force_refresh, openai_only=openai_only)
    return full.get("models", {})


def get_providers_catalog(force_refresh: bool = False, openai_only: bool = True) -> dict:
    """Fetch and return providers catalog dictionary."""
    full = get_full_catalog(force_refresh=force_refresh, openai_only=openai_only)
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


def filter_vision_models(model_names: list[str], catalog: dict | None = None) -> list[str]:
    """Filter a list of catalog model IDs to only those supporting vision/image input.

    Models that are explicitly known NOT to support vision (is_vision_supported is False)
    are excluded. Unknown models (None) or vision-capable models (True) are retained.
    """
    catalog = catalog or get_models_catalog()
    if not catalog:
        return model_names

    filtered = [m for m in model_names if is_vision_supported(m) is not False]
    return filtered or model_names
