"""
setup.py - Interactive initialization wizard for renAI.
Handles environment setup, provider configuration, and configuration file creation.
"""

import contextlib
import importlib.util
import shutil

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme

from ..settings import (
    CONFIG_DIR,
    PACKAGE_CONFIG_DIR,
    SECRETS_FILE,
    SETTINGS_USER_FILE,
    initialize_directories,
)
from .setup_steps import _configure_advanced_settings, _configure_provider, custom_style

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
    if SETTINGS_USER_FILE.exists() or SECRETS_FILE.exists():
        overwrite = questionary.confirm(
            "User configuration files already exist. Overwrite?", default=False, style=custom_style
        ).ask()
        if not overwrite:
            console.print("[info]Setup cancelled.[/info]")
            return

    # 3. Provider Selection & Configuration
    console.print("\n[highlight]Step 1: Provider Setup[/highlight]")
    try:
        config, secrets = _configure_provider()
    except KeyboardInterrupt:
        console.print("[info]Setup cancelled.[/info]")
        return

    # 4. Advanced Settings (Optional)
    console.print("\n[highlight]Step 2: General Settings[/highlight]")
    _print_dependency_diagnostics()

    try:
        configure_advanced = questionary.confirm(
            "Would you like to configure advanced settings?", default=False, style=custom_style
        ).ask()
        if configure_advanced:
            config.update(_configure_advanced_settings())
        else:
            console.print("[info]Using factory defaults for general settings.[/info]")
    except KeyboardInterrupt:
        console.print("[info]Setup cancelled.[/info]")
        return

    # 5. Save Files
    console.print("\n[highlight]Step 3: Saving Configuration[/highlight]")
    try:
        _save_configuration(config, secrets)
    except Exception:
        return

    console.print("\n[bold green]renAI is now ready to use![/bold green]")
    console.print("Try: [cyan]renai process <directory>[/cyan]")


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
