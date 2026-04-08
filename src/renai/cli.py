"""
renAI CLI - Modern Typer-based command line interface.

This is the main CLI entry point for renAI. All functionality from the
original main.py has been migrated here.
"""

import sys
from pathlib import Path
from typing import Tuple, Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from renai.logging_config import configure_logging
from renai.processors.orchestrator import run_pipeline, run_benchmark, run_model_eval
from renai.processors.text_extractor import FallbackMode
from renai.settings import (
    PROVIDERS,
    app_settings,
    display_warnings,
    initialize_directories,
)
from renai.utils import ensure_secure_dir

# Initialize logging
configure_logging()

console = Console()

# Create Typer app
app = typer.Typer(
    name="renai",
    help="AI-Powered Book Renamer and Organizer",
    add_completion=False,
)


@app.callback()
def main_callback():
    """Global initializations for renAI CLI."""
    setup_environment()
    initialize_directories()


# =============================================================================
# Helper Functions (migrated from main.py)
# =============================================================================


def setup_environment() -> None:
    """Setup cross-platform encoding and terminal settings."""
    if sys.platform == "win32":
        import codecs

        if sys.stdout.encoding != "utf-8":
            sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
            sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

    try:
        from colorama import init

        init()
    except ImportError:
        pass





def handle_clear_cache() -> None:
    """Clear extracted text cache and metadata."""
    from renai.settings import CACHE_DIR

    ensure_secure_dir(CACHE_DIR)
    cache_files = list(CACHE_DIR.iterdir())
    if cache_files:
        for cache_file in cache_files:
            try:
                if cache_file.is_file():
                    cache_file.unlink()
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Failed to delete {cache_file.name}: {e}[/yellow]"
                )
        console.print(f"[green]Cache cleared: {len(cache_files)} files removed[/green]")
    else:
        console.print("[green]Cache directory is empty. Nothing to clear.[/green]")





# =============================================================================
# CLI Commands
# =============================================================================


@app.command()
def process(
    directory: Annotated[Optional[str], typer.Argument(
        help="Directory containing book files (required unless using --clear-cache)",
    )] = None,
    # Mode options
    mode: Annotated[str, typer.Option(
        "--mode",
        help="Action to perform (rename/organize)",
    )] = "rename",
    # OCR options
    ocr_enhanced: Annotated[bool, typer.Option(
        "--ocr-enhanced",
        help="Enable high-quality OCR with enhancements",
    )] = app_settings.ocr_enhanced,
    debug_ocr: Annotated[bool, typer.Option(
        "--debug-ocr",
        help="Save OCR debug images",
    )] = app_settings.debug_ocr,
    use_mutool: Annotated[bool, typer.Option(
        "--use-mutool",
        help="Use mutool for PDF-to-image conversion (recommended for scanned PDFs)",
    )] = app_settings.use_mutool,
    mutool_path: Annotated[str, typer.Option(
        "--mutool-path",
        help="Path to mutool executable",
    )] = app_settings.mutool_path,
    # Provider options
    provider: Annotated[str, typer.Option(
        "--provider",
        help="LLM Provider (deepinfra/openrouter/openai/custom)",
    )] = "deepinfra",
    model: Annotated[Optional[str], typer.Option(
        "--model",
        help="LLM Model name (overrides default)",
    )] = None,
    # Processing options
    extractor: Annotated[str, typer.Option(
        "--extractor",
        help="PDF Text Extractor library (pymupdf/pdfplumber/pypdf/pypdfium2)",
    )] = "pymupdf",
    workers: Annotated[int, typer.Option(
        "--workers",
        help="Number of concurrent workers",
    )] = app_settings.workers,
    text_length: Annotated[int, typer.Option(
        "--text-length",
        help="Target text length to send to LLM",
    )] = app_settings.text_length,
    recursive: Annotated[bool, typer.Option(
        "--recursive/--no-recursive",
        help="Process subdirectories recursively",
    )] = app_settings.recursive,
    # Output options
    update_metadata: Annotated[bool, typer.Option(
        "--update-metadata",
        help="Force update metadata from LLM (ignore cache)",
    )] = False,
    # Fallback options
    fallback_mode: Annotated[FallbackMode, typer.Option(
        "--fallback-mode",
        help="Strategy to use when text extraction fails (none/ocr/vision/move). 'move' shifts files to Needs_Scan/ folder.",
    )] = FallbackMode(app_settings.fallback_mode),
    move_scan_failures: Annotated[bool, typer.Option(
        "--move-scan-failures",
        help="Move files that fail all extraction attempts (including fallback) to Scan_Failures folder",
    )] = app_settings.move_scan_failures,
    rename_images: Annotated[bool, typer.Option(
        "--rename-images",
        help="Rename image files using vision model. Also processes non-images with selected fallback.",
    )] = app_settings.rename_images,
    exif: Annotated[bool, typer.Option(
        "--exif",
        help="Add EXIF date to image filenames (use with --rename-images)",
    )] = app_settings.exif,
    # Special modes
    benchmark: Annotated[bool, typer.Option(
        "--benchmark",
        help="Run extraction benchmark on first 3 files",
    )] = False,
    eval_model: Annotated[bool, typer.Option(
        "--eval-model",
        help="Evaluate multiple models for the selected provider",
    )] = False,
    # Cache option
    clear_cache: Annotated[bool, typer.Option(
        "--clear-cache",
        help="Clear cache and exit",
    )] = False,
) -> None:
    """Process files in a directory."""
    # Handle clear-cache first
    if clear_cache:
        handle_clear_cache()
        raise typer.Exit()

    # Validate directory
    if not directory:
        console.print(
            "[red]Error: Directory is required (use --clear-cache to clear cache only)[/red]"
        )
        raise typer.Exit(code=1)

    root_dir = Path(directory).resolve()
    if not root_dir.is_dir():
        console.print(f"[red]Error: Directory not found: {root_dir}[/red]")
        raise typer.Exit(code=1)



    # Display header
    console.print("\n[bold cyan]renAI[/bold cyan] - AI-Powered Book Renamer")
    console.print(f"[blue]Directory:[/blue] {root_dir}")
    console.print(f"[green]Mode:[/green] {mode}")
    console.print(f"[yellow]Provider:[/yellow] {provider}")
    if model:
        console.print(f"[magenta]Model:[/magenta] {model}")
    console.print()

    # Vision mode warnings
    if fallback_mode == FallbackMode.VISION or rename_images:
        console.print(
            "[yellow]Note: vision fallback and --rename-images require a vision-capable model. Please ensure your model supports vision.[/yellow]\n"
        )

    # Setup engines and run pipeline
    run_pipeline(
        root_dir=root_dir,
        mode=mode,
        provider=provider,
        model=model,
        ocr_enhanced=ocr_enhanced,
        debug_ocr=debug_ocr,
        use_mutool=use_mutool,
        mutool_path=mutool_path,
        extractor=extractor,
        workers=workers,
        text_length=text_length,
        recursive=recursive,
        update_metadata=update_metadata,
        fallback_mode=fallback_mode,
        move_scan_failures=move_scan_failures,
        rename_images=rename_images,
        exif=exif,
        benchmark=benchmark,
        eval_model=eval_model,
    )


@app.command("providers")
def list_providers() -> None:
    """List available LLM providers."""
    table = Table(title="Available LLM Providers")
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Default Model", style="green")

    for name, config in PROVIDERS.items():
        default = config.default_model if hasattr(config, "default_model") else "N/A"
        base_url = config.base_url if hasattr(config, "base_url") else "N/A"
        table.add_row(name, default, base_url)

    console.print(table)


@app.command("cache")
def clear_cache_cmd(
    confirm: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Clear the cache directory."""
    if not confirm:
        console.print("[yellow]This will delete all cached text and metadata.[/yellow]")
        response = input("Continue? [y/N]: ")
        if response.lower() not in ("y", "yes"):
            console.print("[cyan]Cancelled.[/cyan]")
            raise typer.Exit()

    handle_clear_cache()


@app.command("version")
def show_version() -> None:
    """Show renAI version."""
    try:
        from renai import __version__

        console.print(f"[bold cyan]renAI[/bold cyan] version {__version__}")
    except ImportError:
        console.print("[bold cyan]renAI[/bold cyan] version unknown (not installed)")





# Backward compatibility: main function for tests
def main():
    """Main entry point for backward compatibility."""
    app()


if __name__ == "__main__":
    app()
