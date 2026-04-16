"""
renAI CLI - Modern Typer-based command line interface.

This is the main CLI entry point for renAI. All functionality from the
original main.py has been migrated here.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from renai.logging_config import configure_logging
from renai.processors.orchestrator import run_evaluation_pipeline, run_pipeline
from renai.processors.setup import run_init_wizard
from renai.processors.text_extractor import FallbackMode
from renai.settings import (
    PROVIDERS,
    app_settings,
    initialize_directories,
)
from renai.utils import ensure_secure_dir
from renai import __version__

# Initialize logging
configure_logging()

console = Console()

# Create Typer app
app = typer.Typer(
    name="renai",
    help="AI-Powered Book Renamer and Organizer",
    add_completion=False,
)


@app.command("version")
def version_cmd() -> None:
    """Show renAI version."""
    console.print(f"renAI version [bold cyan]{__version__}[/bold cyan]")


def version_callback(value: bool):
    if value:
        console.print(f"renAI version [bold cyan]{__version__}[/bold cyan]")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version", "-v", callback=version_callback, is_eager=True, help="Show version and exit"
        ),
    ] = None,
):
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
    from renai.settings import CACHE_DIR

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

    import sys

    # 1. Generate message
    msg = ["\n" + "=" * 70, "⚠️  IMPORTANT WARNINGS", "=" * 70]

    if mode == "rename":
        msg.extend(
            [
                "\n🏷️  RENAMING WARNING:",
                "   Source files WILL be renamed in place or moved to subfolders.",
            ]
        )
    elif mode == "organize":
        msg.extend(
            [
                "\n📁 DIRECTORY LAYOUT WARNING:",
                "   Files will be moved into category-based subfolders.",
            ]
        )

    if recursive:
        msg.extend(
            [
                "\n🔄 RECURSIVE MODE:",
                "   Files in subdirectories will also be processed.",
            ]
        )

    msg.append("=" * 70 + "\n")
    warning_text = "\n".join(msg)

    # 2. Show via logging (for tests/logs) and console (for UX)
    logging.warning(warning_text)
    console.print(warning_text)

    # 3. Interactive confirmation
    try:
        input("\nPress ENTER to proceed, or Ctrl+C to cancel...")
    except KeyboardInterrupt:
        logging.info("\nOperation cancelled by user.")
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except EOFError:
        logging.info("\nOperation cancelled by user.")
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
        str,
        typer.Option(
            "--mode",
            "-m",
            help="Action: 'rename' (renames files) or 'evaluate' (no changes, just extraction test)",
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

    console.print("\n[bold cyan]renAI[/bold cyan] - AI-Powered Book Renamer")
    console.print(f"[blue]Directory:[/blue] {root_dir}")
    console.print(f"[green]Mode:[/green] {mode}")
    console.print(f"[yellow]Provider:[/yellow] {provider}")

    display_warnings(mode, recursive)

    try:
        asyncio.run(
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
                text_extraction=text_extraction,
                extraction_pages=extraction_pages,
                recursive=recursive,
                update_metadata=update_metadata,
                fallback_mode=final_fallback,
                move_scan_failures=move_scan_failures,
                rename_images=rename_images,
                add_exif_date=exif,
                benchmark=benchmark,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user. Exiting...[/yellow]")
        raise typer.Exit(code=0)


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
        raise typer.Exit(code=0)


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
