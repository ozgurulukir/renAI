"""
Setup wizard user-facing messages.
"""


class Wizard:
    MISSING_TOMLI_W = (
        "[error]Error: Missing 'tomli-w' dependency. "
        "Please run 'uv sync' or install it manually.[/error]"
    )
    WELCOME_TITLE = "[bold cyan]renAI Configuration Wizard[/bold cyan]"
    WELCOME_SUBTITLE = (
        "[italic]Setting up your environment for AI-powered document organization.[/italic]"
    )
    CONFIG_DIR = "[info]Config directory:[/info] {}"
    OVERWRITE_PROMPT = (
        "User configuration files already exist. Update and merge with existing configuration?"
    )
    CANCELLED = "[info]Setup cancelled.[/info]"
    STEP_1 = "\n[highlight]Step 1: Provider Setup[/highlight]"
    STEP_2 = "\n[highlight]Step 2: General Settings[/highlight]"
    STEP_3 = "\n[highlight]Step 3: Saving Configuration[/highlight]"
    COMPLETE = "\n[bold green]renAI is now ready to use![/bold green]"
    USAGE_HINT = "Try: [cyan]renai process <directory>[/cyan]"
    USING_DEFAULTS = "[info]Using factory defaults for general settings.[/info]"
    ADVANCED_PROMPT = "Would you like to configure advanced settings?"


class Preflight:
    HEADER = "[info]Pre-flight dependency check:[/info]"
    TESSERACT_FOUND = "  - Tesseract OCR: [success]Found[/success] ({})"
    TESSERACT_MISSING = "  - Tesseract OCR: [warning]Not Found (OCR fallback unavailable)[/warning]"
    MUTOOL_FOUND = "  - mutool: [success]Found[/success] ({})"
    MUTOOL_MISSING = "  - mutool: [info]Not Found (Optional, used for PDF page rendering)[/info]"


class Save:
    PROPAGATING = "[info]Propagating default configuration files...[/info]"
    PROPAGATED = "  - Propagated: {}"
    FINALIZED = "[success]✔ Configuration finalized successfully![/success]"
    USER_SETTINGS = "[info]  - User Settings: {}[/info]"
    SECRETS = "[info]  - Secrets (Keys): {}[/info]"
    DEFAULTS_NOTE = "[info]  - Default files are also available for reference.[/info]"
    FAILED = "[error]Failed to save configuration: {}[/error]"
    ABORTED = "[error]Setup aborted: configuration could not be saved.[/error]"
    SECRETS_PERM_WARN = (
        "[warning]Could not restrict permissions on the secrets file; "
        "it may be readable by other users on this machine: {}[/warning]"
    )


class Provider:
    DEEPINFRA_DESC = "Fast and highly cost-effective cloud endpoints"
    OPENROUTER_DESC = "Unified API accessing almost all commercial/open LLMs"
    CUSTOM_DESC = "Locally hosted LLMs (Ollama, LM Studio, etc.)"
    SELECT_PROMPT = "Select an LLM provider to configure:"
    ANOTHER = "Would you like to configure another LLM provider?"
    DEFAULT_SELECT = "Select the default LLM provider for renAI:"
    DEFAULT_MODEL = "Select the default model for {}:"
    MODEL_PROMPT = "Select or type a model for {}\n  (press Tab to show choices):"
    BASE_URL_PROMPT = "Base URL for {}:"
    API_KEY_PROMPT = "Enter your {} API Key:"
    CUSTOM_KEY_PROMPT = "Custom API Key (leave empty if not required):"
    CUSTOM_URL_PROMPT = "Custom API Base URL:"
    CUSTOM_MODEL_PROMPT = "Custom Model Name:"
    FETCHING_CATALOG = "Fetching model catalog..."
    INVALID_BASE_URL = "Base URL must be a non-empty http:// or https:// URL."
    INVALID_MODEL = "Model name cannot be empty."
    VALIDATING = "[info]Validating {} API Key...[/info]"
    VALID_WARNING = "[warning]{}[/warning]"
    KEY_VALID = "[success]✔ API Key is valid![/success]"
    KEY_INVALID = "[error]API Key validation failed: {}[/error]"
    KEY_ENDPOINT_UNSUPPORTED = "Endpoint does not support key listing (HTTP {})."
    KEY_VALIDATION_ACTION = "How would you like to proceed?"
    KEY_ACTION_RETRY = "Retry"
    KEY_ACTION_SKIP = "Skip validation"
    KEY_ACTION_CANCEL = "Cancel"
    CONNECTION_WARN = "Connection warning: {}"
    VALID_GENERIC_WARN = "Warning: Could not verify key status: {}"
    VALID_WRAPPER_WARN = "Warning: Could not run validation: {}"
    ADD_MODEL = "Would you like to add another model for {}?"
    NO_KEY_WARN = (
        "[warning]Warning: No API key entered. You must set the\n"
        "{}_API_KEY env var to use this provider.[/warning]"
    )
    VISION_PROMPT = "Would you like to configure a Vision model for image/PDF vision tasks?"
    VISION_USE_TEXT_MODEL = "Text model '{}' supports vision. Use it for Vision operations too?"
    VISION_USE_TEXT_MODEL_UNKNOWN = (
        "Vision support for '{}' is unknown. Use it for Vision operations anyway?"
    )
    VISION_PROVIDER_SELECT = "Select the LLM provider for Vision operations:"
    VISION_MODEL_SELECT = "Select the Vision model for {}:"
    VISION_MODEL_PROMPT = "Select or type a Vision model for {}\n  (press Tab to show choices):"


class Settings:
    WORKERS_PROMPT = "Concurrent workers:"
    WORKERS_INVALID = "Please enter a valid integer"
    WORKERS_RANGE_INVALID = "Please enter a value between {} and {}"
    FALLBACK_PROMPT = "Extraction fallback strategy:"
