"""
model_catalog.py - Pure provider→model-name mapping over the models.dev catalog.

Prompt-free and I/O-free by design: importing this module must not drag in
Rich, questionary, or typer, so the mapping is directly testable.
"""

from typing import TypedDict


class ConfiguredProvider(TypedDict):
    """One provider entry collected by the init wizard's ``configured`` dict.

    ``base_url`` and ``api_key`` are populated for every provider before
    ``_build_final_config_and_secrets`` reads them (wizard always collects
    them on both the custom and standard paths), so they are required.
    """

    models: list[str]
    base_url: str
    api_key: str


ConfiguredProviders = dict[str, ConfiguredProvider]


DIRECT_DEV_PROVIDERS = {
    "openai",
    "google",
    "anthropic",
    "deepseek",
    "mistral",
    "cohere",
    "xai",
    "alibaba",
}


def _map_openrouter_model(catalog_id: str) -> str:
    model = catalog_id
    if model.startswith("openrouter/"):
        model = model[len("openrouter/") :]
    if model.startswith("meta/"):
        return "meta-llama/" + model[len("meta/") :]
    if model.startswith("mistral/"):
        return "mistralai/" + model[len("mistral/") :]
    return model


def _map_deepinfra_model(catalog_id: str, catalog_info: dict) -> str:
    for w in catalog_info.get("weights", []):
        if w.get("label") == "Hugging Face" and "huggingface.co/" in w.get("url", ""):
            parts = w["url"].split("huggingface.co/")
            if len(parts) > 1:
                return parts[1].strip("/")
    model = catalog_id
    if model.startswith("deepinfra/"):
        model = model[len("deepinfra/") :]
    if model.startswith("meta/"):
        return "meta-llama/" + model[len("meta/") :]
    if model.startswith("mistral/"):
        return "mistralai/" + model[len("mistral/") :]
    return model


def get_provider_model_name(catalog_id: str, catalog_info: dict, provider: str) -> str:
    """Map the models.dev catalog ID to the actual model name expected by the provider."""
    if provider == "openrouter":
        return _map_openrouter_model(catalog_id)
    if provider == "deepinfra":
        return _map_deepinfra_model(catalog_id, catalog_info)

    if catalog_id.startswith(f"{provider}/"):
        return catalog_id[len(f"{provider}/") :]

    if "/" in catalog_id:
        dev, name = catalog_id.split("/", 1)
        if dev in DIRECT_DEV_PROVIDERS:
            return name

    return catalog_id


def _filter_catalog_models(
    selected_provider: str,
    catalog: dict,
    providers_catalog: dict | None = None,
    vision_only: bool = False,
) -> list[str]:
    """Filter models from catalog based on provider and optional vision capability.

    Excludes non-text-output models (embedding, image-generation, audio/TTS, video)
    whenever modality metadata is available.
    """
    from renai.utils.models_dev import filter_vision_models, is_text_output_model

    p_models: dict | None = None
    models: list[str] = []
    if providers_catalog and selected_provider in providers_catalog:
        raw_p_models = providers_catalog[selected_provider].get("models")
        if raw_p_models and isinstance(raw_p_models, dict):
            p_models = raw_p_models
            models = list(raw_p_models.keys())

    if not models:
        if not catalog:
            models = []
        elif selected_provider in ["deepinfra", "openrouter"]:
            models = list(catalog.keys())
        else:
            models = [k for k in catalog if k.startswith(f"{selected_provider}/")]

    if models:
        text_capable = []
        for m in models:
            info = p_models.get(m) if p_models and m in p_models else catalog.get(m)
            if is_text_output_model(info) is not False:
                text_capable.append(m)
        models = text_capable

    if vision_only and models:
        models = filter_vision_models(
            models,
            catalog=catalog,
            provider=selected_provider,
            providers_catalog=providers_catalog,
        )

    return models


def select_default_model(
    provider: str,
    catalog: dict,
    providers_catalog: dict | None = None,
    *,
    vision_only: bool = False,
    packaged_defaults: dict[str, str] | None = None,
) -> str:
    """Choose a valid default model for *provider* from text/vision-capable catalog models.

    Prefers the packaged default from ``settings.default.toml`` when it is eligible.
    """
    eligible = _filter_catalog_models(provider, catalog, providers_catalog, vision_only=vision_only)
    display_choices = _get_display_choices(provider, eligible, catalog)

    if packaged_defaults and provider in packaged_defaults:
        packaged = packaged_defaults[provider]
        if not display_choices or packaged in display_choices:
            return packaged

    if display_choices:
        return display_choices[0]

    return f"{provider}-model" if not vision_only else ""


def _get_display_choices(
    selected_provider: str, model_names: list[str], catalog: dict | None = None
) -> list[str]:
    """Map full catalog model IDs to display choices."""
    catalog = catalog or {}
    display_choices = []
    for m in model_names:
        info = catalog.get(m, {})
        display_name = get_provider_model_name(m, info, selected_provider)
        display_choices.append(display_name)
    return display_choices
