"""
setup.py - Interactive initialization wizard for renAI.
Handles environment setup, provider configuration, and configuration file creation.
"""

import importlib.util
import logging
import os
import shutil
import tempfile
import tomllib
from pathlib import Path

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
logger = logging.getLogger(__name__)


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


def _load_existing_toml(path: Path) -> dict:
    """Load an existing TOML file if present, returning an empty dict on absence/error."""
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Could not read existing %s for merging: %s", path.name, e)
        return {}


def _merge_settings_config(existing: dict, updates: dict) -> dict:
    """Overlay wizard-owned keys onto existing settings, preserving unrelated keys and providers."""
    merged = dict(existing)
    existing_providers = existing.get("providers")
    merged_providers = dict(existing_providers) if isinstance(existing_providers, dict) else {}

    for key, value in updates.items():
        if key == "providers" and isinstance(value, dict):
            merged_providers.update(value)
        elif value is not None:
            merged[key] = value

    if merged_providers:
        merged["providers"] = merged_providers
    return merged


def _atomic_write_toml(path: Path, data: dict, *, mode: int | None = None) -> None:
    """Serialise *data* to TOML and atomically replace *path*.

    ``None`` values are dropped first (TOML has no null type), and the payload
    is fully serialised to a string before any file is touched, so a failure
    can never leave a truncated file behind. Temp files are created via
    ``tempfile.mkstemp`` (atomic ``O_CREAT | O_EXCL`` with ``0o600`` mode) in
    the target directory so credentials are never world-readable and fixed-name
    symlinks cannot be followed.
    """
    import tomli_w

    clean = {key: value for key, value in data.items() if value is not None}
    payload = tomli_w.dumps(clean)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        if mode is not None:
            tmp_path.chmod(mode)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _save_configuration(config: dict, secrets: dict[str, str]) -> None:
    """Save user configuration and secrets files, merging with existing files."""
    try:
        console.print(msg.Save.PROPAGATING)
        for default_file in PACKAGE_CONFIG_DIR.glob("*.default.toml"):
            dest = CONFIG_DIR / default_file.name
            shutil.copy2(default_file, dest)
            console.print(msg.Save.PROPAGATED.format(dest.name))

        existing_settings = _load_existing_toml(SETTINGS_USER_FILE)
        merged_settings = _merge_settings_config(existing_settings, config)
        _atomic_write_toml(SETTINGS_USER_FILE, merged_settings)

        existing_secrets = _load_existing_toml(SECRETS_FILE)
        merged_secrets = {**existing_secrets, **secrets} if existing_secrets else secrets
        _atomic_write_toml(SECRETS_FILE, merged_secrets, mode=0o600)

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
        logger.exception("Failed to save renAI configuration: %s", e)
        console.print(msg.Save.FAILED.format(e))
        raise
