"""
CLI user-facing messages: warnings, prompts, headers, info lines.
"""

PANEL_TITLE_REVIEW = "[bold yellow]⚠️  Review Before Proceeding[/bold yellow]"


class Warn:
    BADGE_RENAME = "[bold yellow]🏷️  RENAME[/bold yellow]"
    DESC_RENAME = (
        "Files will be [bold]renamed in-place[/bold] with AI-generated names.\n"
        "  Original filenames will be lost."
    )
    BADGE_ORGANIZE = "[bold yellow]📁 ORGANIZE[/bold yellow]"
    DESC_ORGANIZE = (
        "Files will be [bold]renamed and moved[/bold] into category subfolders.\n"
        "  Original filenames and locations will change."
    )
    BADGE_RECURSIVE = "[bold cyan]🔄 RECURSIVE[/bold cyan]"
    DESC_RECURSIVE = "Files in subdirectories will also be processed."
    BADGE_VISION = "[bold red]⚠️  VISION MODEL[/bold red]"

    @staticmethod
    def desc_vision(model: str) -> str:
        return (
            f"Model [bold yellow]'{model}'[/bold yellow] "
            "might not support vision inputs on models.dev.\n"
            "  Running in vision fallback mode or renaming images may fail."
        )


class Prompt:
    PROCEED = (
        "\n[dim]Press [bold]ENTER[/bold] to proceed, or [bold red]Ctrl+C[/] to cancel...[/dim]"
    )
    CANCELLED = "\n[yellow]Operation cancelled by user.[/yellow]"
    CANCELLED_EXIT = "\n[yellow]Cancelled by user. Exiting...[/yellow]"
    EVAL_CANCELLED = "\n[yellow]Evaluation cancelled.[/yellow]"
    CACHE_CONFIRM = "This will delete all cached text and metadata. Continue? [y/N]: "
    CACHE_CANCELLED = "[cyan]Cancelled.[/cyan]"


class Info:
    VERSION = "renAI version [bold cyan]{}[/bold cyan]"
    CACHE_CLEARED = "[green]Cache cleared: {} items removed ({})[/green]"
    CACHE_EMPTY = "[green]Cache is already empty. Nothing to clear.[/green]"
    DIR_NOT_FOUND = "[red]Error: Directory not found: {}[/red]"
    HEADER_TITLE = "[bold cyan]renAI[/bold cyan] [dim]v{}[/dim]"
    HEADER_DIR = "[blue]📂 Dir:[/blue] {}"
    HEADER_MODE = "[green]🛠 Mode:[/green] {}"
    HEADER_PROVIDER = "[yellow]☁ Provider:[/yellow] {}/{}"
    PROVIDER_TABLE_TITLE = "Available LLM Providers"
    COL_PROVIDER = "Provider"
    COL_MODEL = "Model Default"


class Help:
    PROCESS = "Process files in a directory."
    EVALUATE = "Evaluate multiple models and providers."
    LIST_PROVIDERS = "List available LLM providers."
    CLEAR_CACHE = "Clear the cache directory."
    INIT = "Initialize renAI setup wizard."
    DIRECTORY = "Directory to process"
    MODE = (
        "Action: 'rename' (renames files), 'organize' (moves to category) "
        "or 'evaluate' (no changes, just extraction test)"
    )
    PROVIDER = "LLM Provider (deepinfra, openrouter, openai, custom)"
    MODEL = "Specific model name override"
    WORKERS = "Concurrency level"
    OCR_ENHANCED = "Advanced OCR preprocessing"
    DEBUG_OCR = "Save OCR debug images"
    USE_MUTOOL = "Use mutool for PDF extraction"
    MUTOOL_PATH = "Path to mutool"
    EXTRACTOR = "PDF extractor: pymupdf, pdfplumber, pypdf, pypdfium2"
    TEXT_LENGTH = "Max characters for LLM"
    TEXT_EXTRACTION = "Extraction mode: head-tail or regular"
    EXTRACTION_PAGES = "Pages for text extraction"
    RECURSIVE = "Process subdirectories"
    UPDATE_METADATA = "Ignore cache"
    FALLBACK_MODE = "Fallback: none, ocr, vision, move"
    MOVE_SCAN_FAILURES = "Move failed files to Scan_Failures"
    RENAME_IMAGES = "Enable vision-based image renaming"
    EXIF = "Use EXIF date in filenames"
    BENCHMARK = "Benchmarking on first 3 files"
    VISION_MAX_PAGES = "Max pages to sample for vision PDF extraction"
    VISION_PROVIDER = "LLM Provider for vision operations"
    VISION_MODEL = "Model name for vision operations"
    EVAL_DIR = "Directory containing sample book files for evaluation"
    EVAL_PROVIDER = "LLM Provider to evaluate"
    EVAL_WORKERS = "Number of concurrent workers"
