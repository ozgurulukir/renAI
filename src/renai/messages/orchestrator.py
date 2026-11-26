"""
Orchestrator user-facing messages: pipeline status, progress, summaries.
"""


class Status:
    PROCESSING = "[cyan]Processing...[/]"
    SCANNING = "Scanning {}"
    NO_FILES = "[yellow]No supported files found to process.[/yellow]"
    BENCHMARK_NOTICE = "[yellow]Benchmark mode: only processing first {} files.[/yellow]"
    FILES_COUNT = "[bold blue]🚀 Files:[/bold blue] {}"
    WORKERS_COUNT = "[bold cyan]⚙ Workers:[/bold cyan] {}"
    RPM_LIMIT = "[bold yellow]⏱ RPM Limit:[/bold yellow] {}"


class Progress:
    TITLE_TOTAL = "[bold green]Total Progress[/]"
    TITLE_ACTIVITY = "[bold blue]Recent Activity[/]"
    STATS_LINE = " [bold cyan]Files:[/] {}  [bold green]Success:[/] {}  [bold red]Failed:[/] {}"


class Summary:
    TITLE = "[bold blue]Processing Result Summary[/]"
    SUCCESS_BADGE = "[bold green]✅ Successful[/bold green]"
    SUCCESS_COUNT = "[bold green]{}[/bold green]"
    FAILED_BADGE = "[bold red]❌ Failed[/bold red]"
    FAILED_COUNT = "[bold red]{}[/bold red]"


class Error:
    # Plain-text strings: these become exception messages carried by
    # ConfigurationError and rendered by the CLI, so no Rich markup here.
    UNKNOWN_PROVIDER = "Unknown provider: '{}'. Available providers: {}."
    API_KEY_MISSING = (
        "No API key configured for provider '{}'. "
        "Run 'renai init' or add the API key to your settings file."
    )
    VISION_NOT_CONFIGURED = (
        "Vision processing was requested but no vision provider/model is configured. "
        "Run 'renai init' or pass --vision-model."
    )
    SETTINGS_INVALID = (
        "Your settings file is invalid and could not be loaded ({}). "
        "Fix or remove the offending entry and run the command again."
    )
    INVALID_FALLBACK_MODE = "Invalid fallback_mode '{}': expected one of none, ocr, vision, move."


class Benchmark:
    TESTING = "[bold yellow]BENCHMARK[/] Testing [cyan]{}[/]..."
    COMPLETED = "[green]Completed {} in {:.2f}s[/]"
    TABLE_TITLE = "PDF Extractor Benchmark Results"
    COL_EXTRACTOR = "Extractor"
    COL_TIME = "Time (s)"
    COL_SUCCESS_RATE = "Success Rate"


class Eval:
    NO_MODELS = "[red]No evaluation models found in eval_models.toml.[/red]"
    HEADER = "🔍 [bold blue]Evaluation Mode[/bold blue]"
    TARGET = "Target: [cyan]{}[/cyan]"
    MODEL_COUNT = "Models: [green]{} total[/green]\n"
    PER_PROVIDER = "--- [bold yellow]Evaluating: {}/{}[/bold yellow] ---"
    RESULTS_LOGGED = "[dim]Results logged to: {}[/dim]\n"
    COMPARISON_TITLE = "\n📊 [bold green]Evaluation Comparison Summary[/bold green]\n"
    FILE_TABLE_TITLE = "File: [cyan]{}[/cyan]"
    STATUS_SUCCESS = "[green]SUCCESS[/green]"
    STATUS_FAILED = "[red]FAILED[/red]"
    ERROR_PREVIEW = "[dim red]Error: {}[/dim red]"
