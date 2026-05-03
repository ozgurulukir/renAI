"""
renAI CLI - Modern Typer-based command line interface.

This is the main CLI entry point for renAI. All functionality from the
original main.py has been migrated here.
"""

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

from renai import __version__
from renai.config import (
    PROVIDERS,
    app_settings,
    initialize_directories,
)
from renai.logging_config import configure_logging
from renai.models.schemas import FallbackMode, ProcessingConfig
from renai.processors.orchestrator import run_evaluation_pipeline, run_pipeline
from renai.processors.setup import run_init_wizard
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


def version_callback(value: bool):
    if value:
        console.print(f"renAI version [bold cyan]{__version__}[/bold cyan]")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Annotated[  # noqa: ARG001
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Global initializations for renAI CLI."""
    setup_environment()
    initialize_directories()


# =============================================================================
# Helper Functions
# =============================================================================


def setup_environment() -> None:
    """Setup cross-platform encoding and terminal settings."""
    if sys.platform == "win32":
        import codecs

        try:
            if sys.stdout.encoding != "utf-8":
                sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
                sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")
        except Exception:
            pass

    try:
        from colorama import init  # type: ignore[import-untyped]

        init()
    except ImportError:
        pass


def handle_clear_cache() -> None:
    """Clear extracted text cache, metadata, and OCR debug files."""
    from renai.cache.manager import clear_all_cache
    from renai.config import CACHE_DIR

    ensure_secure_dir(CACHE_DIR)
    results = clear_all_cache(CACHE_DIR)

    total = sum(results.values())
    if total > 0:
        details = ", ".join([f"{count} {ctype}" for ctype, count in results.items() if count > 0])
        console.print(f"[green]Cache cleared: {total} items removed ({details})[/green]")
    else:
        console.print("[green]Cache is already empty. Nothing to clear.[/green]")


def display_warnings(mode: str, recursive: bool) -> None:
    """Display important warnings to the user before processing."""
    if not app_settings.show_warnings:
        return

    from rich.panel import Panel
    from rich.table import Table

    # 1. Create Warning Table
    table = Table.grid(padding=(0, 1))

    if mode == "rename":
        table.add_row(
            "[bold yellow]🏷️  RENAMING[/bold yellow]", "Files WILL be renamed in place or moved."
        )
    elif mode == "organize":
        table.add_row(
            "[bold yellow]📁 ORGANIZE[/bold yellow]",
            "Files will be moved to category-based subfolders.",
        )

    if recursive:
        table.add_row(
            "[bold cyan]🔄 RECURSIVE[/bold cyan]", "Files in subdirectories will also be processed."
        )

    # 2. Show Panel
    console.print(
        Panel(
            table,
            title="[bold red]⚠️  IMPORTANT WARNINGS[/bold red]",
            border_style="red",
            expand=False,
            padding=(1, 2),
        )
    )

    # 3. Interactive confirmation
    try:
        console.print(
            "\n[dim]Press [bold]ENTER[/bold] to proceed, or [bold red]Ctrl+C[/] to cancel...[/dim]",
            end="",
        )
        input("")
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(0)


# =============================================================================
# CLI Commands
# =============================================================================


@app.command("process")
def process_cmd(
    directory: Annotated[
        str,
        typer.Argument(
            help="Directory to process",
        ),
    ] = ".",
    mode: Annotated[
        Literal["rename", "organize", "evaluate"],
        typer.Option(
            "--mode",
            "-m",
            help=(
                "Action: 'rename' (renames files), 'organize' (moves to category) "
                "or 'evaluate' (no changes, just extraction test)"
            ),
        ),
    ] = "rename",
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            "-p",
            help="LLM Provider (deepinfra, openrouter, openai, custom)",
        ),
    ] = app_settings.provider,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Specific model name override",
        ),
    ] = app_settings.model,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            help="Concurrency level",
        ),
    ] = app_settings.workers,
    ocr_enhanced: Annotated[
        bool,
        typer.Option(
            "--ocr-enhanced/--no-ocr-enhanced",
            help="Advanced OCR preprocessing",
        ),
    ] = app_settings.ocr_enhanced,
    debug_ocr: Annotated[
        bool,
        typer.Option(
            "--debug-ocr",
            help="Save OCR debug images",
        ),
    ] = app_settings.debug_ocr,
    use_mutool: Annotated[
        bool,
        typer.Option(
            "--use-mutool/--no-mutool",
            help="Use mutool for PDF extraction",
        ),
    ] = app_settings.use_mutool,
    mutool_path: Annotated[
        str,
        typer.Option(
            "--mutool-path",
            help="Path to mutool",
        ),
    ] = app_settings.mutool_path,
    extractor: Annotated[
        str,
        typer.Option(
            "--extractor",
            help="PDF extractor: pymupdf, pdfplumber, pypdf, pypdfium2",
        ),
    ] = app_settings.extractor,
    text_length: Annotated[
        int,
        typer.Option(
            "--text-length",
            help="Max characters for LLM",
        ),
    ] = app_settings.text_length,
    text_extraction: Annotated[
        str,
        typer.Option(
            "--text-extraction",
            help="Extraction mode: mixed or regular",
        ),
    ] = app_settings.text_extraction,
    extraction_pages: Annotated[
        int,
        typer.Option(
            "--extraction-pages",
            help="Pages for text extraction",
        ),
    ] = app_settings.extraction_pages,
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive/--no-recursive",
            help="Process subdirectories",
        ),
    ] = app_settings.recursive,
    update_metadata: Annotated[
        bool,
        typer.Option(
            "--update-metadata",
            help="Ignore cache",
        ),
    ] = False,
    fallback_mode: Annotated[
        FallbackMode | None,
        typer.Option(
            "--fallback-mode",
            help="Fallback: none, ocr, vision, move",
        ),
    ] = None,
    move_scan_failures: Annotated[
        bool,
        typer.Option(
            "--move-scan-failures",
            help="Move failed files to Scan_Failures",
        ),
    ] = app_settings.move_scan_failures,
    rename_images: Annotated[
        bool,
        typer.Option(
            "--rename-images",
            help="Enable vision-based image renaming",
        ),
    ] = app_settings.rename_images,
    exif: Annotated[
        bool,
        typer.Option(
            "--exif",
            help="Use EXIF date in filenames",
        ),
    ] = app_settings.exif,
    benchmark: Annotated[
        bool,
        typer.Option(
            "--benchmark",
            help="Benchmarking on first 3 files",
        ),
    ] = False,
) -> None:
    """Process files in a directory."""
    final_fallback = fallback_mode or FallbackMode(app_settings.fallback_mode)
    root_dir = Path(directory).resolve()

    if not root_dir.is_dir():
        console.print(f"[red]Error: Directory not found: {root_dir}[/red]")
        raise typer.Exit(code=1)

    from rich.columns import Columns
    from rich.panel import Panel

    # 1. Header Information
    title = f"[bold cyan]renAI[/bold cyan] [dim]v{__version__}[/dim]"

    info_cols = [
        f"[blue]📂 Dir:[/blue] {root_dir.name}",
        f"[green]🛠 Mode:[/green] {mode}",
        f"[yellow]☁ Provider:[/yellow] {provider}/{model or 'default'}",
    ]

    console.print(
        Panel(Columns(info_cols, expand=True), title=title, border_style="cyan", padding=(0, 1))
    )

    display_warnings(mode, recursive)

    try:
        asyncio.run(
            run_pipeline(
                root_dir=root_dir,
                provider=provider,
                model=model,
                config=ProcessingConfig(
                    mode=mode,
                    workers=workers,
                    recursive=recursive,
                    ocr_enhanced=ocr_enhanced,
                    debug_ocr=debug_ocr,
                    use_mutool=use_mutool,
                    mutool_path=mutool_path,
                    extractor=extractor,
                    text_length=text_length,
                    text_extraction=text_extraction,
                    extraction_pages=extraction_pages,
                    skip_ocr_checks=False,  # default for process cmd
                    move_scan_failures=move_scan_failures,
                    fallback_mode=final_fallback,
                    rename_images=rename_images,
                    add_exif_date=exif,
                    update_metadata=update_metadata,
                    benchmark=benchmark,
                ),
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user. Exiting...[/yellow]")
        raise typer.Exit(code=0) from None


@app.command("evaluate")
def evaluate_cmd(
    directory: Annotated[
        str | None,
        typer.Argument(
            help="Directory containing sample book files for evaluation",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help="LLM Provider to evaluate",
        ),
    ] = None,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            help="Number of concurrent workers",
        ),
    ] = app_settings.workers,
) -> None:
    """Evaluate multiple models and providers."""
    target_dir = Path(directory or ".").resolve()
    if not target_dir.is_dir():
        console.print(f"[red]Error: Directory not found: {target_dir}[/red]")
        raise typer.Exit(code=1)

    try:
        asyncio.run(
            run_evaluation_pipeline(
                root_dir=target_dir,
                provider_filter=provider,
                workers=workers,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation cancelled.[/yellow]")
        raise typer.Exit(code=0) from None


@app.command("providers")
def list_providers() -> None:
    """List available LLM providers."""
    table = Table(title="Available LLM Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Model Default", style="green")

    for name, config in PROVIDERS.items():
        table.add_row(name, config.default_model)

    console.print(table)


@app.command("cache")
def clear_cache_cmd(
    confirm: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Clear the cache directory."""
    if not confirm:
        response = input("This will delete all cached text and metadata. Continue? [y/N]: ")
        if response.lower() not in ("y", "yes"):
            console.print("[cyan]Cancelled.[/cyan]")
            raise typer.Exit()

    handle_clear_cache()


@app.command("init")
def init_cmd() -> None:
    """Initialize renAI setup wizard."""
    run_init_wizard()


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
