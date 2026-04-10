"""
setup.py - Interactive initialization wizard for renAI.
Handles environment setup, provider configuration, and configuration file creation.
"""


from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.theme import Theme

from ..settings import (
    CONFIG_DIR,
    SETTINGS_USER_FILE,
    SECRETS_FILE,
    PACKAGE_CONFIG_DIR,
    FACTORY_DEFAULTS,
    PROVIDERS,
    app_settings,
    initialize_directories,
)
import shutil

# Custom theme for a premium look
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "success": "green",
    "highlight": "magenta",
})
console = Console(theme=custom_theme)

def run_init_wizard():
    """Run the interactive setup wizard."""
    console.print(Panel.fit(
        "[bold cyan]renAI Configuration Wizard[/bold cyan]\n"
        "[italic]Setting up your environment for AI-powered document organization.[/italic]",
        border_style="cyan"
    ))

    # 1. Ensure directories exist
    initialize_directories()
    console.print(f"[info]Config directory:[/info] {CONFIG_DIR}")

    # 2. Check if user config already exists
    if SETTINGS_USER_FILE.exists() or SECRETS_FILE.exists():
        if not Confirm.ask("[warning]User configuration files already exist. Overwrite?[/warning]", default=False):
            console.print("[info]Setup cancelled.[/info]")
            return

    # 3. Provider Selection
    console.print("\n[highlight]Step 1: Provider Setup[/highlight]")
    available_providers = list(PROVIDERS.keys()) + ["custom"]
    provider = Prompt.ask(
        "Select your preferred LLM provider",
        choices=available_providers,
        default="deepinfra"
    )

    secrets = {}
    config = {"provider": provider}

    if provider == "custom":
        config["custom_api_base_url"] = Prompt.ask("Custom API Base URL", default="http://localhost:8000/v1")
        config["custom_model"] = Prompt.ask("Custom Model Name", default="llama3")
        config["custom_api_key"] = Prompt.ask("Custom API Key", default="not-required")
    else:
        # Suggest default model from provider pool
        default_model = PROVIDERS.get(provider).default_model if provider in PROVIDERS else ""
        config["model"] = Prompt.ask(f"Default Model for {provider.capitalize()}", default=default_model)
        
        api_key = Prompt.ask(f"Enter your {provider.capitalize()} API Key", password=True)
        secrets[f"{provider}_api_key"] = api_key

    # 4. Advanced Settings (Optional)
    console.print("\n[highlight]Step 2: General Settings[/highlight]")
    if Confirm.ask("Would you like to configure advanced settings?", default=False):
        config["workers"] = int(Prompt.ask("Concurrent workers", default=str(app_settings.workers)))
        config["fallback_mode"] = Prompt.ask(
            "Extraction fallback strategy",
            choices=["none", "ocr", "vision", "move"],
            default=app_settings.fallback_mode
        )
    else:
        console.print("[info]Using factory defaults for general settings.[/info]")

    # 5. Save Files
    console.print("\n[highlight]Step 3: Saving Configuration[/highlight]")
    
    try:
        import tomli_w # type: ignore
        
        # A. Propagate default files (Physical copy of all package defaults)
        console.print("[info]Propagating default configuration files...[/info]")
        for default_file in PACKAGE_CONFIG_DIR.glob("*.default.toml"):
            dest = CONFIG_DIR / default_file.name
            shutil.copy2(default_file, dest)
            console.print(f"  - Propagated: {dest.name}")

        # B. Save settings.toml (User overrides)
        with open(SETTINGS_USER_FILE, "wb") as f:
            tomli_w.dump(config, f)
        
        # C. Save secrets.toml
        with open(SECRETS_FILE, "wb") as f:
            tomli_w.dump(secrets, f)
            
        # Ensure secure permissions on secrets
        if hasattr(SECRETS_FILE, "chmod"):
            SECRETS_FILE.chmod(0o600)

        console.print("[success]✔ Configuration finalized successfully![/success]")
        console.print(f"[info]  - User Settings: {SETTINGS_USER_FILE.name}[/info]")
        console.print(f"[info]  - Secrets (Keys): {SECRETS_FILE.name}[/info]")
        console.print("[info]  - Default files are also available for reference.[/info]")
        
    except ImportError:
        console.print("[error]Missing 'tomli-w' dependency. Please install it using 'uv sync' or 'pip install tomli-w'.[/error]")
        return
    except Exception as e:
        console.print(f"[error]Failed to save configuration: {e}[/error]")
        return

    console.print("\n[bold green]renAI is now ready to use![/bold green]")
    console.print("Try: [cyan]renai process <directory>[/cyan]")
