"""
model_catalog.py - Pure provider→model-name mapping over the models.dev catalog.

Prompt-free and I/O-free by design: importing this module must not drag in
Rich, questionary, or typer, so the mapping is directly testable.
"""

from typing import NotRequired, TypedDict


class ConfiguredProvider(TypedDict):
    """One provider entry collected by the init wizard's ``configured`` dict."""

    models: list[str]
    base_url: NotRequired[str]
    api_key: NotRequired[str]


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
    """Filter models from catalog based on provider and optional vision capability."""
    from renai.utils.models_dev import filter_vision_models

    models = []
    if providers_catalog and selected_provider in providers_catalog:
        p_models = providers_catalog[selected_provider].get("models")
        if p_models and isinstance(p_models, dict) and p_models:
            models = list(p_models.keys())

    if not models:
        if not catalog:
            models = []
        elif selected_provider in ["deepinfra", "openrouter"]:
            models = list(catalog.keys())
        else:
            models = [k for k in catalog if k.startswith(f"{selected_provider}/")]

    if vision_only and models:
        models = filter_vision_models(models, catalog=catalog)

    return models


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
