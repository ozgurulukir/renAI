"""
setup.py - Interactive initialization wizard for renAI.
Handles environment setup, provider configuration, and configuration file creation.
"""

import importlib.util
import shutil

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme

from renai.messages import setup as msg

from ..config import (
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
        console.print(msg.Wizard.MISSING_TOMLI_W)
        return

    console.print(
        Panel.fit(
            f"{msg.Wizard.WELCOME_TITLE}\n{msg.Wizard.WELCOME_SUBTITLE}",
            border_style="cyan",
        )
    )

    # 1. Ensure directories exist
    initialize_directories()
    console.print(msg.Wizard.CONFIG_DIR.format(CONFIG_DIR))

    # 2. Check if user config already exists
    if SETTINGS_USER_FILE.exists() or SECRETS_FILE.exists():
        overwrite = questionary.confirm(
            msg.Wizard.OVERWRITE_PROMPT, default=False, style=custom_style
        ).ask()
        if not overwrite:
            console.print(msg.Wizard.CANCELLED)
            return

    # 3. Provider Selection & Configuration
    console.print(msg.Wizard.STEP_1)
    try:
        config, secrets = _configure_provider()
    except KeyboardInterrupt:
        console.print(msg.Wizard.CANCELLED)
        return

    console.print(msg.Wizard.STEP_2)
    _print_dependency_diagnostics()

    try:
        configure_advanced = questionary.confirm(
            msg.Wizard.ADVANCED_PROMPT, default=False, style=custom_style
        ).ask()
        if configure_advanced:
            config.update(_configure_advanced_settings())
        else:
            console.print(msg.Wizard.USING_DEFAULTS)
    except KeyboardInterrupt:
        console.print(msg.Wizard.CANCELLED)
        return

    console.print(msg.Wizard.STEP_3)
    try:
        _save_configuration(config, secrets)
    except Exception:
        console.print(msg.Save.ABORTED)
        return

    console.print(msg.Wizard.COMPLETE)
    console.print(msg.Wizard.USAGE_HINT)


def _print_dependency_diagnostics() -> None:
    """Check and print status of external binary dependencies (tesseract, mutool)."""
    tesseract_path = shutil.which("tesseract")
    mutool_path = shutil.which("mutool")

    console.print(msg.Preflight.HEADER)
    if tesseract_path:
        console.print(msg.Preflight.TESSERACT_FOUND.format(tesseract_path))
    else:
        console.print(msg.Preflight.TESSERACT_MISSING)

    if mutool_path:
        console.print(msg.Preflight.MUTOOL_FOUND.format(mutool_path))
    else:
        console.print(msg.Preflight.MUTOOL_MISSING)
    console.print("")


def _save_configuration(config: dict[str, str | int], secrets: dict[str, str]) -> None:
    """Save user configuration and secrets files, copy defaults."""
    try:
        import tomli_w

        console.print(msg.Save.PROPAGATING)
        for default_file in PACKAGE_CONFIG_DIR.glob("*.default.toml"):
            dest = CONFIG_DIR / default_file.name
            shutil.copy2(default_file, dest)
            console.print(msg.Save.PROPAGATED.format(dest.name))

        with SETTINGS_USER_FILE.open("wb") as f:
            tomli_w.dump(config, f)

        with SECRETS_FILE.open("wb") as f:
            tomli_w.dump(secrets, f)

        if hasattr(SECRETS_FILE, "chmod"):
            try:
                SECRETS_FILE.chmod(0o600)
            except OSError as e:
                console.print(msg.Save.SECRETS_PERM_WARN.format(e))

        console.print(msg.Save.FINALIZED)
        console.print(msg.Save.USER_SETTINGS.format(SETTINGS_USER_FILE.name))
        console.print(msg.Save.SECRETS.format(SECRETS_FILE.name))
        console.print(msg.Save.DEFAULTS_NOTE)

    except Exception as e:
        console.print(msg.Save.FAILED.format(e))
        raise
