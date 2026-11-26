"""
setup_steps.py - Step implementations for renAI interactive initialization wizard.
Contains questionary-based interactive prompts for provider and model configuration.
"""

import logging
from collections.abc import Callable

import questionary
from questionary import Style
from rich.console import Console
from rich.theme import Theme

from renai.messages import setup as msg

from ..config import app_settings
from ..config.settings import AppSettings
from ..exceptions import ConfigurationError

# Pure catalog mapping lives in model_catalog.py; re-exported here so existing
# import sites (tests, callers) keep working unchanged.
from .model_catalog import (  # noqa: F401
    DIRECT_DEV_PROVIDERS,
    ConfiguredProviders,
    _filter_catalog_models,
    _get_display_choices,
    _map_deepinfra_model,
    _map_openrouter_model,
    get_provider_model_name,
)

logger = logging.getLogger(__name__)

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

# Deliberately short: the wizard validates keys with a lightweight models.list
# call and interactive users bail out on multi-second freezes. This is a
# setup-time convenience timeout, independent of llm_timeout (runtime calls).
API_VALIDATION_TIMEOUT_SEC = 5.0

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


def _build_provider_options(
    catalog: dict, all_devs: list[str], providers_catalog: dict | None = None
) -> list[dict]:
    """Build list of provider option dicts without popular/non-popular priority separation."""
    providers_catalog = providers_catalog or {}
    combined_devs = set(all_devs)
    if providers_catalog:
        combined_devs.update(providers_catalog.keys())

    sorted_devs = sorted(combined_devs)

    provider_options = []
    desc_map = {
        "openai": "Official OpenAI models (GPT-4o, GPT-4o-mini)",
        "deepseek": "DeepSeek API endpoints (R1, Chat)",
        "google": "Google Gemini API via OpenAI compatibility",
        "anthropic": "Anthropic Claude API endpoints",
        "mistral": "Mistral AI API endpoints",
        "deepinfra": msg.Provider.DEEPINFRA_DESC,
        "openrouter": msg.Provider.OPENROUTER_DESC,
    }

    for dev in sorted_devs:
        p_info = providers_catalog.get(dev, {})
        p_name = p_info.get("name", dev.capitalize())
        desc = desc_map.get(dev, f"Models by {p_name} from models.dev")
        dev_models = []
        if dev in providers_catalog:
            p_models = providers_catalog[dev].get("models")
            if isinstance(p_models, dict):
                dev_models = list(p_models.keys())
        if not dev_models and catalog:
            dev_models = [k for k in catalog if k.startswith(f"{dev}/")]

        if dev == "deepinfra":
            default_model = "meta-llama/Llama-3.3-70B-Instruct"
        elif dev == "openrouter":
            default_model = "google/gemini-2.0-flash-exp:free"
        elif dev_models:
            default_model = get_provider_model_name(
                dev_models[0], catalog.get(dev_models[0], {}), dev
            )
        else:
            default_model = f"{dev}-model"

        provider_options.append({"name": dev, "desc": desc, "extra": default_model})

    if "custom" not in combined_devs:
        provider_options.append(
            {
                "name": "custom",
                "desc": msg.Provider.CUSTOM_DESC,
                "extra": "llama3",
            }
        )
    return provider_options


def _prompt_model_selection(
    selected_provider: str,
    catalog: dict,
    provider_options: list,
    providers_catalog: dict | None = None,
    vision_only: bool = False,
) -> str:
    """Prompt the user to select a model for the given provider."""
    model_names = _filter_catalog_models(
        selected_provider, catalog, providers_catalog, vision_only=vision_only
    )

    model_names.sort(key=lambda m: (catalog.get(m, {}).get("name", m).lower(), m.lower()))

    display_choices = _get_display_choices(selected_provider, model_names, catalog)

    default_model = ""
    for opt in provider_options:
        if opt["name"] == selected_provider:
            default_model = opt["extra"]
            break

    prompt_template = msg.Provider.VISION_MODEL_PROMPT if vision_only else msg.Provider.MODEL_PROMPT
    prompt_msg = prompt_template.format(selected_provider.capitalize())
    selected_model = questionary.autocomplete(
        prompt_msg,
        choices=display_choices,
        default=default_model,
        style=custom_style,
    ).ask()

    if selected_model is None:
        raise KeyboardInterrupt

    catalog_id = None
    for m in model_names:
        info = catalog.get(m, {})
        disp = get_provider_model_name(m, info, selected_provider)

        # 1. Exact match on display name or catalog ID
        if selected_model in (disp, m):
            catalog_id = m
            break

        # 2. Normalized match ignoring tags (:free) or provider prefixes
        selected_clean = selected_model.split(":")[0].strip()
        disp_clean = disp.split(":")[0].strip()
        m_clean = m.split(":")[0].strip()

        if selected_clean in (disp_clean, m_clean):
            catalog_id = m
            break

        # 3. Match by model basename (e.g. "gpt-4o" typed for "openai/gpt-4o")
        if "/" in disp_clean and selected_clean == disp_clean.split("/")[-1]:
            catalog_id = m
            break
        if "/" in m_clean and selected_clean == m_clean.split("/")[-1]:
            catalog_id = m
            break

    if catalog_id:
        info = catalog.get(catalog_id, {})
        actual_model = get_provider_model_name(catalog_id, info, selected_provider)
    else:
        actual_model = selected_model

    return actual_model


def _prompt_base_url(selected_provider: str, default_url: str | None = None) -> str:
    """Prompt user for dynamic provider base URL."""
    known_base_urls = {
        "deepseek": "https://api.deepseek.com/v1",
        "google": "https://generativelanguage.googleapis.com/v1beta/openai",
        "groq": "https://api.groq.com/openai/v1",
        "mistral": "https://api.mistral.ai/v1",
        "together": "https://api.together.xyz/v1",
        "perplexity": "https://api.perplexity.ai",
    }
    default_base = default_url or known_base_urls.get(selected_provider, "http://localhost:8000/v1")
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

    try:
        async with AsyncOpenAI(api_key=api_key, base_url=base_url) as client:
            # Very low timeout to avoid freezing the setup wizard
            await client.models.list(timeout=API_VALIDATION_TIMEOUT_SEC)
            return True, ""
    except AuthenticationError as e:
        # Invalid API key
        return False, str(e)
    except APIConnectionError as e:
        logger.warning("API key validation connection error: %s", e)
        return True, msg.Provider.CONNECTION_WARN.format(e)
    except Exception as e:
        logger.warning("Unexpected error during API key validation: %s", e)
        return False, msg.Provider.VALID_GENERIC_WARN.format(e)


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
        logger.warning("API key validation wrapper failed: %s", e)
        return False, msg.Provider.VALID_WRAPPER_WARN.format(e)


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
    selected_provider: str,
    configured: ConfiguredProviders,
    provider_options: list,
    catalog: dict,
    providers_catalog: dict | None = None,
) -> None:
    """Configure credentials and models for a standard provider."""
    base_url = ""
    if selected_provider not in ["deepinfra", "openrouter", "openai"]:
        catalog_api = None
        if providers_catalog and selected_provider in providers_catalog:
            catalog_api = providers_catalog[selected_provider].get("api")
        base_url = _prompt_base_url(selected_provider, default_url=catalog_api)
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
        actual_model = _prompt_model_selection(
            selected_provider, catalog, provider_options, providers_catalog
        )
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
    selected_provider: str,
    configured: ConfiguredProviders,
    provider_options: list,
    catalog: dict,
    providers_catalog: dict | None = None,
) -> None:
    """Configure credentials and models for a single provider."""
    if selected_provider not in configured:
        configured[selected_provider] = {"models": [], "base_url": "", "api_key": ""}

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
        _configure_standard_provider(
            selected_provider, configured, provider_options, catalog, providers_catalog
        )


def _build_final_config_and_secrets(
    configured: ConfiguredProviders,
    default_provider: str,
    default_model: str,
    vision_provider: str | None = None,
    vision_model: str | None = None,
) -> tuple[dict, dict]:
    """Build the final configuration and secrets dictionaries from user selections.

    API keys are written ONLY to the secrets dict; settings.toml never contains
    credentials. ``None`` values (e.g. a declined vision configuration) are
    omitted so the result stays TOML-serialisable.
    """
    secrets = {}
    config: dict = {
        "provider": default_provider,
        "model": default_model,
        "providers": {},
    }

    if vision_provider is not None:
        config["vision_provider"] = vision_provider
    if vision_model is not None:
        config["vision_model"] = vision_model

    for name, data in configured.items():
        secrets[f"{name}_api_key"] = data["api_key"]
        config["providers"][name] = {
            "base_url": data["base_url"],
            "default_model": data["models"][0] if name != default_provider else default_model,
        }
        if not data["api_key"].strip():
            console.print(msg.Provider.NO_KEY_WARN.format(name.upper()))

    return config, secrets


def _configure_vision_selection(
    default_provider: str,
    default_model: str,
    configured_names: list[str],
    configured: ConfiguredProviders,
    catalog: dict,
    provider_options: list,
    providers_catalog: dict | None,
    is_vision_supported_func: Callable[[str], bool | None],
) -> tuple[str | None, str | None]:
    """Prompt user to configure vision provider and vision model."""
    text_supports_vision = is_vision_supported_func(default_model) is not False
    use_text_for_vision = False

    if text_supports_vision:
        use_text_for_vision = questionary.confirm(
            msg.Provider.VISION_USE_TEXT_MODEL.format(default_model),
            default=True,
            style=custom_style,
        ).ask()
        if use_text_for_vision is None:
            raise KeyboardInterrupt

    if use_text_for_vision:
        return default_provider, default_model

    configure_vision = questionary.confirm(
        msg.Provider.VISION_PROMPT,
        default=True,
        style=custom_style,
    ).ask()
    if configure_vision is None or not configure_vision:
        return None, None

    vision_provider = questionary.select(
        msg.Provider.VISION_PROVIDER_SELECT,
        choices=configured_names,
        default=default_provider,
        style=custom_style,
    ).ask()
    if vision_provider is None:
        raise KeyboardInterrupt

    vision_model = _prompt_model_selection(
        vision_provider,
        catalog,
        provider_options,
        providers_catalog,
        vision_only=True,
    )
    if vision_model and vision_model not in configured[vision_provider]["models"]:
        configured[vision_provider]["models"].append(vision_model)

    return vision_provider, vision_model


def _configure_provider() -> tuple[dict[str, str | int], dict[str, str]]:
    """Prompt the user for provider selection, credentials, text model, and vision model."""
    try:
        from renai.utils.models_dev import get_full_catalog, is_vision_supported

        full_cat = get_full_catalog()
        catalog = full_cat.get("models", {})
        providers_catalog = full_cat.get("providers", {})
    except Exception:
        catalog = {}
        providers_catalog = {}

        def is_vision_supported(model_name: str) -> bool | None:  # noqa: ARG001
            """Placeholder matching ``models_dev.is_vision_supported`` so pyright
            sees one consistent declaration for the try/except; the argument is
            intentionally unused because catalog availability is unknown."""
            """Vision support is unknown when the catalog is unavailable."""
            return None

    if catalog:
        all_devs = sorted({k.split("/")[0] for k in catalog if "/" in k})
    else:
        all_devs = ["openai", "deepseek", "google", "anthropic", "mistral"]

    provider_options = _build_provider_options(
        catalog, all_devs, providers_catalog=providers_catalog
    )
    configured: ConfiguredProviders = {}

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

        _configure_single_provider(
            selected_provider, configured, provider_options, catalog, providers_catalog
        )

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

    vision_provider, vision_model = _configure_vision_selection(
        default_provider,
        default_model,
        configured_names,
        configured,
        catalog,
        provider_options,
        providers_catalog,
        is_vision_supported,
    )

    return _build_final_config_and_secrets(
        configured, default_provider, default_model, vision_provider, vision_model
    )


def _workers_bounds() -> tuple[int, int]:
    """Return the allowed (min, max) worker range from AppSettings constraints."""
    metadata = AppSettings.model_fields["workers"].metadata
    lower = 1
    upper = 32
    for constraint in metadata:
        ge = getattr(constraint, "ge", None)
        le = getattr(constraint, "le", None)
        if ge is not None:
            lower = ge
        if le is not None:
            upper = le
    return lower, upper


def _validate_workers_input(text: str) -> bool | str:
    """Validate the workers prompt answer against the configured allowed range."""
    if not text.isdigit():
        return msg.Settings.WORKERS_INVALID
    lower, upper = _workers_bounds()
    if not lower <= int(text) <= upper:
        return msg.Settings.WORKERS_RANGE_INVALID.format(lower, upper)
    return True


def _configure_advanced_settings() -> dict[str, str | int]:
    """Prompt the user for advanced application settings."""
    workers_str = questionary.text(
        msg.Settings.WORKERS_PROMPT,
        default=str(app_settings.workers),
        validate=_validate_workers_input,
        style=custom_style,
    ).ask()
    if workers_str is None:
        raise KeyboardInterrupt
    try:
        workers = int(workers_str)
    except (TypeError, ValueError) as err:
        raise ConfigurationError(
            f"Invalid workers count: {workers_str!r}. Please enter a whole number."
        ) from err

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
