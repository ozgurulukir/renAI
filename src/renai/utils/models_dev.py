"""
models_dev.py - Integration with models.dev directory for AI model capabilities.
"""

import contextlib
import functools
import json
import logging
import time
from pathlib import Path

import httpx

from renai import __version__
from renai.config.constants import APP_DATA_DIR

logger = logging.getLogger(__name__)

CACHE_FILE = APP_DATA_DIR / "models_dev_cache.json"
# Catalog refresh cadence: model catalogs change slowly (new releases, not
# pricing), so a 24h TTL keeps the wizard fresh without a network hit per run.
CACHE_EXPIRY = 86400  # 24 hours in seconds
# Short fetch timeout: the catalog is a multi-MB JSON on a fixed host and the
# caller degrades gracefully (stale cache) when it fails.
HTTP_TIMEOUT_SEC = 6.0
HTTP_STATUS_OK = 200


def _remote_headers() -> dict[str, str]:
    """Headers for models.dev requests."""
    return {
        "User-Agent": f"renAI/{__version__} (https://github.com/ozgurulukir/renAI)",
        "Accept": "application/json",
    }


def _decode_remote_json(response: httpx.Response) -> dict | None:
    """Decode a models.dev JSON response, logging anomalies."""
    if response.status_code == HTTP_STATUS_OK:
        data = response.json()
        if isinstance(data, dict) and data:
            return data
        logger.debug(f"Endpoint returned empty or non-dict response: {data}")
    else:
        logger.debug(f"Endpoint returned HTTP status {response.status_code}")
    return None


def _fetch_catalog_json() -> dict | None:
    """Fetch the models.dev catalog document.

    The URL is a compile-time literal on a fixed remote host (no
    caller-supplied endpoint interpolation), so no SSRF surface exists.
    """
    try:
        response = httpx.get(
            "https://models.dev/catalog.json",
            headers=_remote_headers(),
            timeout=HTTP_TIMEOUT_SEC,
        )
        return _decode_remote_json(response)
    except Exception as e:
        logger.debug(f"Failed to fetch https://models.dev/catalog.json: {e}")
        return None


def _fetch_models_json() -> dict | None:
    """Fetch the models.dev legacy models document (literal URL, fixed host)."""
    try:
        response = httpx.get(
            "https://models.dev/models.json",
            headers=_remote_headers(),
            timeout=HTTP_TIMEOUT_SEC,
        )
        return _decode_remote_json(response)
    except Exception as e:
        logger.debug(f"Failed to fetch https://models.dev/models.json: {e}")
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


@functools.lru_cache(maxsize=8)
def _load_cached_catalog_parsed(cache_path: str) -> tuple:
    """Parse the on-disk catalog cache once per process per cache file.

    The models.dev catalog JSON is multi-megabyte; re-parsing it on every
    ``is_vision_supported`` / ``get_full_catalog`` call (which happens hundreds
    of times during the setup wizard) dominates runtime. Memoizing the parse
    here collapses that churn to a single ``json.load`` per process. The cache
    is keyed by the cache-file path so tests that patch ``CACHE_FILE`` get their
    own isolated entry rather than stale data. The result is wrapped in a tuple
    (to satisfy ``lru_cache`` hashability); callers copy what they need so no
    shared mutable state escapes.
    """
    path = Path(cache_path)
    if path.exists():
        with contextlib.suppress(Exception), path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and (data.get("models") or data.get("providers")):
                return (data,)
    return ({},)


def _cache_is_fresh() -> bool:
    """Check the on-disk cache age against CACHE_EXPIRY.

    The TTL was previously defined but never consulted, so a cached catalog
    lived forever; enforcing mtime-based expiry lets model metadata (vision
    support, new providers) actually refresh while keeping the stale copy as
    an offline fallback.
    """
    try:
        return (time.time() - CACHE_FILE.stat().st_mtime) < CACHE_EXPIRY
    except OSError:
        return False


def _get_raw_catalog(force_refresh: bool = False) -> dict:
    """Fetch raw catalog from cache or remote endpoint."""
    with contextlib.suppress(Exception):
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    if force_refresh or not _cache_is_fresh():
        # ``force_refresh`` bypasses the memoized read entirely (used only for an
        # explicit refresh, never in the wizard path) and falls through to the
        # fresh-load/remote-fetch logic below. An expired cache is treated the
        # same way so a refresh is attempted; the final fallback below still
        # serves the stale file when the network is unavailable.
        cached: dict = {}
    else:
        cached = _load_cached_catalog_parsed(str(CACHE_FILE))[0]

    if cached:
        return cached

    # Try fetching catalog.json first
    data = _fetch_catalog_json()
    if data and (data.get("models") or data.get("providers")):
        with contextlib.suppress(Exception), CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        _load_cached_catalog_parsed.cache_clear()
        return data

    # Fallback: try fetching models.json
    models_data = _fetch_models_json()
    if models_data:
        data = {"models": models_data, "providers": {}}
        with contextlib.suppress(Exception), CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        _load_cached_catalog_parsed.cache_clear()
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


def _match_provider_scoped_basename(catalog: dict, normalized: str) -> bool | None:
    """Basename (suffix) match scoped to the same provider.

    Two providers' variants that share a basename must not shadow each other:
    only entries whose provider prefix equals the query's prefix are considered.
    A bare query (no provider prefix) keeps the historical global fallback.
    """
    target_basename = normalized.rsplit("/", maxsplit=1)[-1]
    target_provider = normalized.split("/", maxsplit=1)[0] if "/" in normalized else None
    for key, info in catalog.items():
        key_lower = key.lower()
        key_basename = key_lower.split("/")[-1]
        if key_basename != target_basename:
            continue
        if target_provider is not None and key_lower.split("/")[0] != target_provider:
            continue
        return _check_vision_modalities(info)
    return None


def _lookup_provider_model_info(
    model_name: str, provider: str | None, providers_catalog: dict | None
) -> dict | None:
    """Look up a model's metadata entry inside a provider's own models table."""
    if not provider or not providers_catalog or provider not in providers_catalog:
        return None
    p_models = providers_catalog[provider].get("models")
    if not isinstance(p_models, dict):
        return None

    raw = model_name.strip()
    stripped = (
        raw.split(":", 1)[0] if ":" in raw and not raw.startswith(("http://", "https://")) else raw
    )
    prefix = f"{provider}/"
    candidates = [raw, stripped]
    if stripped.startswith(prefix):
        candidates.append(stripped[len(prefix) :])

    for candidate in candidates:
        entry = p_models.get(candidate)
        if isinstance(entry, dict):
            return entry

    target_lower = stripped.lower()
    for key, info in p_models.items():
        if key.lower() == target_lower and isinstance(info, dict):
            return info
    return None


def is_text_output_model(model_info: dict | None) -> bool | None:
    """Return whether a model entry lists ``"text"`` in its output modalities.

    Returns ``None`` when modality metadata is absent so callers can distinguish
    unknown models from explicitly non-text models (embeddings, TTS, image/video).
    """
    if not isinstance(model_info, dict):
        return None
    modalities = model_info.get("modalities")
    if not isinstance(modalities, dict) or "output" not in modalities:
        return None
    outputs = modalities.get("output")
    if not isinstance(outputs, list):
        return None
    return "text" in outputs


def is_vision_supported(
    model_name: str,
    provider: str | None = None,
    providers_catalog: dict | None = None,
    catalog: dict | None = None,
) -> bool | None:
    """Check if the given model name supports vision modalities.

    Consults ``providers_catalog[provider]["models"][model_name]`` first when
    available, falling back to the top-level ``models`` catalog.

    Returns:
        True: if supported
        False: if explicitly not supported
        None: if the model is not found in the catalog (unknown)
    """
    provider_entry = _lookup_provider_model_info(model_name, provider, providers_catalog)
    if provider_entry is not None:
        modalities = provider_entry.get("modalities")
        if isinstance(modalities, dict) and "input" in modalities:
            return _check_vision_modalities(provider_entry)

    models_catalog = catalog if catalog is not None else get_models_catalog()
    if not models_catalog:
        return None

    raw = model_name.strip()
    if ":" in raw and not raw.startswith("http://") and not raw.startswith("https://"):
        raw = raw.split(":")[0]

    model_name_lower = raw.lower()

    # 1. Direct match
    if model_name_lower in models_catalog:
        return _check_vision_modalities(models_catalog[model_name_lower])

    # Standardize provider prefix aliases
    normalized = model_name_lower
    for prefix in ("openrouter/", "deepinfra/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]

    if normalized.startswith("meta-llama/"):
        normalized = "meta/" + normalized[len("meta-llama/") :]
    elif normalized.startswith("mistralai/"):
        normalized = "mistral/" + normalized[len("mistralai/") :]

    if normalized in models_catalog:
        return _check_vision_modalities(models_catalog[normalized])

    # 2. Try matching by model basename (suffix), scoped to the same provider so
    #    two providers' variants that share a basename cannot shadow each other.
    return _match_provider_scoped_basename(models_catalog, normalized)


def _check_vision_modalities(model_info: dict) -> bool:
    modalities = model_info.get("modalities", {})
    inputs = modalities.get("input", [])
    return "image" in inputs


def filter_vision_models(
    model_names: list[str],
    catalog: dict | None = None,
    provider: str | None = None,
    providers_catalog: dict | None = None,
) -> list[str]:
    """Filter a list of catalog model IDs to only those supporting vision/image input.

    Consults the provider's own ``modalities`` table first, falling back to the
    top-level ``models`` catalog. When catalog modality information is available,
    text-only models are excluded without falling back to the unfiltered list.
    """
    has_provider_catalog = bool(provider and providers_catalog and provider in providers_catalog)
    models_catalog = catalog
    if models_catalog is None and not has_provider_catalog:
        models_catalog = get_models_catalog()

    if not models_catalog and not has_provider_catalog:
        return model_names

    return [
        m
        for m in model_names
        if is_vision_supported(
            m,
            provider=provider,
            providers_catalog=providers_catalog,
            catalog=models_catalog or {},
        )
        is not False
    ]
