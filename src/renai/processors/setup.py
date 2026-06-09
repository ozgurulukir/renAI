"""
setup.py - Interactive initialization wizard for renAI.
Handles environment setup, provider configuration, and configuration file creation.
"""

import contextlib
import importlib.util
import shutil

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.theme import Theme

from ..settings import (
    CONFIG_DIR,
    PACKAGE_CONFIG_DIR,
    PROVIDERS,
    SECRETS_FILE,
    SETTINGS_USER_FILE,
    app_settings,
    initialize_directories,
)

# Custom theme for a premium look
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


def run_init_wizard() -> None:
    """Run the interactive setup wizard."""
    # 0. Early dependency check (fail-fast)
    if importlib.util.find_spec("tomli_w") is None:
        console.print(
            "[error]Error: Missing 'tomli-w' dependency. "
            "Please run 'uv sync' or install it manually.[/error]"
        )
        return

    console.print(
        Panel.fit(
            "[bold cyan]renAI Configuration Wizard[/bold cyan]\n"
            "[italic]Setting up your environment for AI-powered document organization.[/italic]",
            border_style="cyan",
        )
    )

    # 1. Ensure directories exist
    initialize_directories()
    console.print(f"[info]Config directory:[/info] {CONFIG_DIR}")

    # 2. Check if user config already exists
    if (SETTINGS_USER_FILE.exists() or SECRETS_FILE.exists()) and not Confirm.ask(
        "[warning]User configuration files already exist. Overwrite?[/warning]", default=False
    ):
        console.print("[info]Setup cancelled.[/info]")
        return

    # 3. Provider Selection & Configuration
    console.print("\n[highlight]Step 1: Provider Setup[/highlight]")
    config, secrets = _configure_provider()

    # 4. Advanced Settings (Optional)
    console.print("\n[highlight]Step 2: General Settings[/highlight]")
    _print_dependency_diagnostics()

    if Confirm.ask("Would you like to configure advanced settings?", default=False):
        config.update(_configure_advanced_settings())
    else:
        console.print("[info]Using factory defaults for general settings.[/info]")

    # 5. Save Files
    console.print("\n[highlight]Step 3: Saving Configuration[/highlight]")
    try:
        _save_configuration(config, secrets)
    except Exception:
        return

    console.print("\n[bold green]renAI is now ready to use![/bold green]")
    console.print("Try: [cyan]renai process <directory>[/cyan]")


def _configure_provider() -> tuple[dict[str, str | int], dict[str, str]]:
    """Prompt the user for provider selection and credentials."""
    available_providers = [*list(PROVIDERS.keys()), "custom"]
    provider = Prompt.ask(
        "Select your preferred LLM provider", choices=available_providers, default="deepinfra"
    )

    secrets: dict[str, str] = {}
    config: dict[str, str | int] = {"provider": provider}

    if provider == "custom":
        config["custom_api_base_url"] = Prompt.ask(
            "Custom API Base URL", default="http://localhost:8000/v1"
        )
        config["custom_model"] = Prompt.ask("Custom Model Name", default="llama3")
        config["custom_api_key"] = Prompt.ask("Custom API Key", default="not-required")
    else:
        default_model = PROVIDERS.get(provider).default_model if provider in PROVIDERS else ""

        # Query models.dev catalog for popular models
        try:
            from renai.utils.models_dev import get_models_catalog

            catalog = get_models_catalog()
            if catalog:
                suggested_names = []
                for k in catalog:
                    parts = k.split("/")
                    if len(parts) > 1 and parts[0].lower() == provider.lower():
                        suggested_names.append(parts[1])

                if provider == "openai" and suggested_names:
                    console.print(
                        f"[info]Popular models for OpenAI from models.dev:[/info]\n"
                        f"  {', '.join(suggested_names[:6])}"
                    )
                elif provider == "openrouter":
                    or_suggestions = [
                        "openai/gpt-4o",
                        "openai/gpt-4o-mini",
                        "anthropic/claude-3-7-sonnet-20250219",
                        "google/gemini-2.0-flash",
                    ]
                    console.print(
                        f"[info]Popular models for OpenRouter from models.dev:[/info]\n"
                        f"  {', '.join(or_suggestions)}"
                    )
                elif provider == "deepinfra":
                    di_suggestions = [
                        "meta-llama/Llama-3.2-90B-Vision-Instruct",
                        "meta-llama/Llama-3.3-70b-instruct",
                        "Qwen/Qwen2.5-72B-Instruct",
                    ]
                    console.print(
                        f"[info]Popular models for DeepInfra from models.dev:[/info]\n"
                        f"  {', '.join(di_suggestions)}"
                    )
        except Exception:
            pass

        config["model"] = Prompt.ask(
            f"Default Model for {provider.capitalize()}", default=default_model
        )

        api_key = Prompt.ask(f"Enter your {provider.capitalize()} API Key", password=True)
        secrets[f"{provider}_api_key"] = api_key
        if not api_key.strip():
            console.print(
                "[warning]Warning: No API key entered. You must set the\n"
                f"RENAI_{provider.upper()}_API_KEY env var to use this provider.[/warning]"
            )

    return config, secrets


def _print_dependency_diagnostics() -> None:
    """Check and print status of external binary dependencies (tesseract, mutool)."""
    tesseract_path = shutil.which("tesseract")
    mutool_path = shutil.which("mutool")

    console.print("[info]Pre-flight dependency check:[/info]")
    if tesseract_path:
        console.print(f"  - Tesseract OCR: [success]Found[/success] ({tesseract_path})")
    else:
        console.print("  - Tesseract OCR: [warning]Not Found (OCR fallback unavailable)[/warning]")

    if mutool_path:
        console.print(f"  - mutool: [success]Found[/success] ({mutool_path})")
    else:
        console.print("  - mutool: [info]Not Found (Optional, used for PDF page rendering)[/info]")
    console.print("")


def _configure_advanced_settings() -> dict[str, str | int]:
    """Prompt the user for advanced application settings."""
    return {
        "workers": IntPrompt.ask("Concurrent workers", default=app_settings.workers),
        "fallback_mode": Prompt.ask(
            "Extraction fallback strategy",
            choices=["none", "ocr", "vision", "move"],
            default=app_settings.fallback_mode,
        ),
    }


def _save_configuration(config: dict[str, str | int], secrets: dict[str, str]) -> None:
    """Save user configuration and secrets files, copy defaults."""
    try:
        import tomli_w

        # A. Propagate default files (Physical copy of all package defaults)
        console.print("[info]Propagating default configuration files...[/info]")
        for default_file in PACKAGE_CONFIG_DIR.glob("*.default.toml"):
            dest = CONFIG_DIR / default_file.name
            shutil.copy2(default_file, dest)
            console.print(f"  - Propagated: {dest.name}")

        # B. Save settings.toml (User overrides)
        with SETTINGS_USER_FILE.open("wb") as f:
            tomli_w.dump(config, f)

        # C. Save secrets.toml
        with SECRETS_FILE.open("wb") as f:
            tomli_w.dump(secrets, f)

        # Ensure secure permissions on secrets
        if hasattr(SECRETS_FILE, "chmod"):
            with contextlib.suppress(Exception):
                SECRETS_FILE.chmod(0o600)

        console.print("[success]✔ Configuration finalized successfully![/success]")
        console.print(f"[info]  - User Settings: {SETTINGS_USER_FILE.name}[/info]")
        console.print(f"[info]  - Secrets (Keys): {SECRETS_FILE.name}[/info]")
        console.print("[info]  - Default files are also available for reference.[/info]")

    except Exception as e:
        console.print(f"[error]Failed to save configuration: {e}[/error]")
        raise
