"""
setup_steps.py - Step implementations for renAI interactive initialization wizard.
Contains questionary-based interactive prompts for provider and model configuration.
"""

import questionary
from questionary import Style
from rich.console import Console
from rich.theme import Theme

from renai.messages import setup as msg

from ..settings import app_settings

# Custom theme for setup messages
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "success": "green",
        "highlight": "magenta",
    }
)
console = Console(theme=custom_theme)

# Premium, high-contrast, terminal-friendly questionary style
custom_style = Style(
    [
        ("qmark", "fg:#00ffff bold"),  # Cyan prefix
        ("question", "bold"),  # Bold question text
        ("answer", "fg:#00ff00 bold"),  # Bright green selected answer
        ("pointer", "fg:#00ffff bold"),  # Cyan pointer
        ("highlighted", "fg:#000000 bg:#00ffff bold"),  # Black text on Cyan background
        ("selected", "fg:#00ff00"),  # Selected checkbox items
        ("instruction", "fg:#888888 italic"),  # Gray instruction
        # Autocomplete completion menu dropdown
        ("completion-menu", "bg:#2b2b2b fg:#eeeeee"),
        ("completion-menu.completion", "bg:#2b2b2b fg:#eeeeee"),
        ("completion-menu.completion.current", "bg:#00ffff fg:#000000 bold noreverse"),
        ("scrollbar.background", "bg:#3a3a3a"),
        ("scrollbar.button", "bg:#00ffff"),
    ]
)


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
    if catalog_id.startswith("meta/"):
        return "meta-llama/" + catalog_id[len("meta/") :]
    if catalog_id.startswith("mistral/"):
        return "mistralai/" + catalog_id[len("mistral/") :]
    return catalog_id


def _map_deepinfra_model(catalog_id: str, catalog_info: dict) -> str:
    for w in catalog_info.get("weights", []):
        if w.get("label") == "Hugging Face" and "huggingface.co/" in w.get("url", ""):
            parts = w["url"].split("huggingface.co/")
            if len(parts) > 1:
                return parts[1].strip("/")
    if catalog_id.startswith("meta/"):
        return "meta-llama/" + catalog_id[len("meta/") :]
    if catalog_id.startswith("mistral/"):
        return "mistralai/" + catalog_id[len("mistral/") :]
    return catalog_id


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



def _build_provider_options(catalog: dict, all_devs: list[str]) -> list[dict]:
    """Build list of provider option dicts."""
    provider_options = [
        {
            "name": "deepinfra",
            "desc": msg.Provider.DEEPINFRA_DESC,
            "extra": "meta-llama/Llama-3.3-70B-Instruct",
        },
        {
            "name": "openrouter",
            "desc": msg.Provider.OPENROUTER_DESC,
            "extra": "google/gemini-2.0-flash-exp:free",
        },
    ]

    popular_devs = ["openai", "deepseek", "google", "anthropic", "mistral"]
    sorted_devs = sorted(
        all_devs,
        key=lambda d: (
            d not in popular_devs,
            popular_devs.index(d) if d in popular_devs else d,
        ),
    )

    for dev in sorted_devs:
        desc_map = {
            "openai": "Official OpenAI models (GPT-4o, GPT-4o-mini)",
            "deepseek": "DeepSeek API endpoints (R1, Chat)",
            "google": "Google Gemini API via OpenAI compatibility",
            "anthropic": "Anthropic Claude API endpoints",
            "mistral": "Mistral AI API endpoints",
        }
        desc = desc_map.get(dev, f"Models by {dev.capitalize()} from models.dev")
        dev_models = [k for k in catalog if k.startswith(f"{dev}/")] if catalog else []
        default_model = (
            get_provider_model_name(dev_models[0], catalog.get(dev_models[0], {}), dev)
            if dev_models
            else f"{dev}-model"
        )

        provider_options.append({"name": dev, "desc": desc, "extra": default_model})

    provider_options.append(
        {
            "name": "custom",
            "desc": msg.Provider.CUSTOM_DESC,
            "extra": "llama3",
        }
    )
    return provider_options


def _filter_catalog_models(selected_provider: str, catalog: dict) -> list[str]:
    """Filter models from catalog based on provider."""
    if not catalog:
        return []
    if selected_provider in ["deepinfra", "openrouter"]:
        return list(catalog.keys())
    return [k for k in catalog if k.startswith(f"{selected_provider}/")]


def _get_display_choices(selected_provider: str, model_names: list[str]) -> list[str]:
    """Map full catalog model IDs to display choices."""
    display_choices = []
    prefix = f"{selected_provider}/"
    for m in model_names:
        display_name = m
        if m.startswith(prefix):
            display_name = m[len(prefix) :]
        elif "/" in m:
            dev = m.split("/")[0]
            if dev in DIRECT_DEV_PROVIDERS:
                display_name = m[len(dev) + 1 :]
        display_choices.append(display_name)
    return display_choices


def _prompt_model_selection(selected_provider: str, catalog: dict, provider_options: list) -> str:
    """Prompt the user to select a model for the given provider."""
    model_names = _filter_catalog_models(selected_provider, catalog)

    popular_models = [
        "meta/llama-3.3-70b-instruct",
        "google/gemini-2.0-flash",
        "google/gemini-2.5-flash",
        "deepseek/deepseek-r1",
        "deepseek/deepseek-chat",
        "anthropic/claude-3-7-sonnet-20250219",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
    ]
    model_names.sort(
        key=lambda m: (
            m not in popular_models,
            popular_models.index(m) if m in popular_models else m,
        )
    )

    display_choices = _get_display_choices(selected_provider, model_names)

    default_model = ""
    for opt in provider_options:
        if opt["name"] == selected_provider:
            default_model = opt["extra"]
            break

    prompt_msg = msg.Provider.MODEL_PROMPT.format(selected_provider.capitalize())
    selected_model = questionary.autocomplete(
        prompt_msg,
        choices=display_choices,
        default=default_model,
        style=custom_style,
    ).ask()

    if selected_model is None:
        raise KeyboardInterrupt

    catalog_id = None
    prefix = f"{selected_provider}/"
    for m in model_names:
        d_name = m[len(prefix) :] if m.startswith(prefix) else m
        if selected_model in (d_name, m):
            catalog_id = m
            break


    if catalog_id:
        info = catalog.get(catalog_id, {})
        actual_model = get_provider_model_name(catalog_id, info, selected_provider)
    else:
        actual_model = selected_model

    return actual_model


def _prompt_base_url(selected_provider: str) -> str:
    """Prompt user for dynamic provider base URL."""
    known_base_urls = {
        "deepseek": "https://api.deepseek.com/v1",
        "google": "https://generativelanguage.googleapis.com/v1beta/openai",
        "groq": "https://api.groq.com/openai/v1",
        "mistral": "https://api.mistral.ai/v1",
        "together": "https://api.together.xyz/v1",
        "perplexity": "https://api.perplexity.ai",
    }
    default_base = known_base_urls.get(selected_provider, "http://localhost:8000/v1")
    base_url = questionary.text(
        msg.Provider.BASE_URL_PROMPT.format(selected_provider.capitalize()),
        default=default_base,
        style=custom_style,
    ).ask()
    if base_url is None:
        raise KeyboardInterrupt
    return base_url


async def _validate_api_key_async(base_url: str, api_key: str) -> tuple[bool, str]:
    """Perform a lightweight API call to validate the API key."""
    from openai import APIConnectionError, AsyncOpenAI, AuthenticationError

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    try:
        # We use a very low timeout (e.g., 5 seconds) to avoid freezing the setup wizard
        await client.models.list(timeout=5.0)
        return True, ""
    except AuthenticationError as e:
        # Invalid API key
        return False, str(e)
    except APIConnectionError as e:
        return True, msg.Provider.CONNECTION_WARN.format(e)
    except Exception as e:
        return True, msg.Provider.VALID_GENERIC_WARN.format(e)


def _validate_api_key(base_url: str, api_key: str) -> tuple[bool, str]:
    """Synchronous wrapper to run the API key validation."""
    import asyncio
    import concurrent.futures

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _validate_api_key_async(base_url, api_key))
                return future.result()
        else:
            return asyncio.run(_validate_api_key_async(base_url, api_key))
    except Exception as e:
        return True, msg.Provider.VALID_WRAPPER_WARN.format(e)


def _prompt_and_validate_api_key(
    selected_provider: str, base_url: str, is_custom: bool = False
) -> str:
    """Prompt the user for their API key and validate it against the provider's endpoint."""
    while True:
        if is_custom:
            api_key = questionary.text(
                msg.Provider.CUSTOM_KEY_PROMPT, default="not-required", style=custom_style
            ).ask()
        else:
            api_key = questionary.password(
                msg.Provider.API_KEY_PROMPT.format(selected_provider.capitalize()),
                style=custom_style,
            ).ask()

        if api_key is None:
            raise KeyboardInterrupt

        # If they leave it empty, or if custom and default "not-required", skip validation
        if not api_key.strip() or (is_custom and api_key == "not-required"):
            return api_key

        console.print(msg.Provider.VALIDATING.format(selected_provider.capitalize()))
        is_valid, error_msg = _validate_api_key(base_url, api_key)
        if is_valid:
            if error_msg:
                console.print(msg.Provider.VALID_WARNING.format(error_msg))
            else:
                console.print(msg.Provider.KEY_VALID)
            return api_key

        console.print(msg.Provider.KEY_INVALID.format(error_msg))


def _configure_standard_provider(
    selected_provider: str, configured: dict, provider_options: list, catalog: dict
) -> None:
    """Configure credentials and models for a standard provider."""
    base_url = ""
    if selected_provider not in ["deepinfra", "openrouter", "openai"]:
        base_url = _prompt_base_url(selected_provider)
    elif selected_provider == "openai":
        base_url = "https://api.openai.com/v1"
    elif selected_provider == "deepinfra":
        base_url = "https://api.deepinfra.com/v1/openai"
    elif selected_provider == "openrouter":
        base_url = "https://openrouter.ai/api/v1"

    api_key = _prompt_and_validate_api_key(selected_provider, base_url)

    configured[selected_provider]["base_url"] = base_url
    configured[selected_provider]["api_key"] = api_key

    while True:
        actual_model = _prompt_model_selection(selected_provider, catalog, provider_options)
        if actual_model not in configured[selected_provider]["models"]:
            configured[selected_provider]["models"].append(actual_model)

        add_another_model = questionary.confirm(
            msg.Provider.ADD_MODEL.format(selected_provider.capitalize()),
            default=False,
            style=custom_style,
        ).ask()

        if add_another_model is None:
            raise KeyboardInterrupt
        if not add_another_model:
            break


def _configure_single_provider(
    selected_provider: str, configured: dict, provider_options: list, catalog: dict
) -> None:
    """Configure credentials and models for a single provider."""
    if selected_provider not in configured:
        configured[selected_provider] = {"models": []}

    if selected_provider == "custom":
        base_url = questionary.text(
            msg.Provider.CUSTOM_URL_PROMPT,
            default="http://localhost:8000/v1",
            style=custom_style,
        ).ask()
        custom_model = questionary.text(
            msg.Provider.CUSTOM_MODEL_PROMPT, default="llama3", style=custom_style
        ).ask()

        if base_url is None or custom_model is None:
            raise KeyboardInterrupt

        api_key = _prompt_and_validate_api_key("custom", base_url, is_custom=True)

        configured["custom"] = {"base_url": base_url, "models": [custom_model], "api_key": api_key}
    else:
        _configure_standard_provider(selected_provider, configured, provider_options, catalog)


def _build_final_config_and_secrets(
    configured: dict, default_provider: str, default_model: str
) -> tuple[dict, dict]:
    """Build the final configuration and secrets dictionaries from user selections."""
    secrets = {}
    config = {"provider": default_provider, "model": default_model, "providers": {}}

    for name, data in configured.items():
        secrets[f"{name}_api_key"] = data["api_key"]

        if name == "custom":
            config["custom_api_base_url"] = data["base_url"]
            config["custom_model"] = data["models"][0]
            config["custom_api_key"] = data["api_key"]
        else:
            config["providers"][name] = {
                "base_url": data["base_url"],
                "default_model": data["models"][0] if name != default_provider else default_model,
            }
            config["providers"][name]["api_key"] = data["api_key"]

            if not data["api_key"].strip():
                console.print(msg.Provider.NO_KEY_WARN.format(name.upper()))

    return config, secrets


def _configure_provider() -> tuple[dict[str, str | int], dict[str, str]]:
    """Prompt the user for provider selection and credentials."""
    try:
        from renai.utils.models_dev import get_models_catalog

        catalog = get_models_catalog()
    except Exception:
        catalog = {}

    if catalog:
        all_devs = sorted({k.split("/")[0] for k in catalog if "/" in k})
    else:
        all_devs = ["openai", "deepseek", "google", "anthropic", "mistral"]

    provider_options = _build_provider_options(catalog, all_devs)

    configured = {}

    while True:
        provider_choices = []
        for opt in provider_options:
            suffix = " (configured)" if opt["name"] in configured else ""
            provider_choices.append(
                questionary.Choice(
                    title=f"{opt['name']}{suffix} - {opt['desc']}", value=opt["name"]
                )
            )

        selected_provider = questionary.select(
            msg.Provider.SELECT_PROMPT,
            choices=provider_choices,
            default="openai" if "openai" not in configured else None,
            style=custom_style,
        ).ask()

        if selected_provider is None:
            raise KeyboardInterrupt

        _configure_single_provider(selected_provider, configured, provider_options, catalog)

        add_another_provider = questionary.confirm(
            msg.Provider.ANOTHER, default=False, style=custom_style
        ).ask()

        if add_another_provider is None:
            raise KeyboardInterrupt
        if not add_another_provider:
            break

    configured_names = list(configured.keys())
    default_provider = questionary.select(
        msg.Provider.DEFAULT_SELECT,
        choices=configured_names,
        default="openai" if "openai" in configured_names else configured_names[0],
        style=custom_style,
    ).ask()

    if default_provider is None:
        raise KeyboardInterrupt

    default_models = configured[default_provider]["models"]
    default_model = questionary.select(
        msg.Provider.DEFAULT_MODEL.format(default_provider.capitalize()),
        choices=default_models,
        default=default_models[0],
        style=custom_style,
    ).ask()

    if default_model is None:
        raise KeyboardInterrupt

    return _build_final_config_and_secrets(configured, default_provider, default_model)


def _configure_advanced_settings() -> dict[str, str | int]:
    """Prompt the user for advanced application settings."""
    workers_str = questionary.text(
        msg.Settings.WORKERS_PROMPT,
        default=str(app_settings.workers),
        validate=lambda text: text.isdigit() or msg.Settings.WORKERS_INVALID,
        style=custom_style,
    ).ask()
    if workers_str is None:
        raise KeyboardInterrupt
    workers = int(workers_str)

    fallback_mode = questionary.select(
        msg.Settings.FALLBACK_PROMPT,
        choices=["none", "ocr", "vision", "move"],
        default=app_settings.fallback_mode,
        style=custom_style,
    ).ask()
    if fallback_mode is None:
        raise KeyboardInterrupt

    return {
        "workers": workers,
        "fallback_mode": fallback_mode,
    }
