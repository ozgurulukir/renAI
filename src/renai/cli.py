"""
renAI CLI - Modern Typer-based command line interface.

This is the main CLI entry point for renAI. All functionality from the
original main.py has been migrated here.
"""

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

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
from renai.messages import cli as msg
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
        console.print(msg.Info.VERSION.format(__version__))
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
        console.print(msg.Info.CACHE_CLEARED.format(total, details))
    else:
        console.print(msg.Info.CACHE_EMPTY)


def display_warnings(
    mode: str,
    recursive: bool,
    provider: str | None = None,
    model: str | None = None,
    fallback_mode: FallbackMode | None = None,
    rename_images: bool = False,
) -> None:
    """Display contextual warnings to the user before processing."""
    if not app_settings.show_warnings:
        return

    from rich.panel import Panel
    from rich.table import Table

    from renai.config import get_effective_model

    has_warnings = False
    table = Table.grid(padding=(0, 1))

    if mode == "rename":
        table.add_row(msg.Warn.BADGE_RENAME, msg.Warn.DESC_RENAME)
        has_warnings = True
    elif mode == "organize":
        table.add_row(msg.Warn.BADGE_ORGANIZE, msg.Warn.DESC_ORGANIZE)
        has_warnings = True

    if recursive:
        table.add_row(msg.Warn.BADGE_RECURSIVE, msg.Warn.DESC_RECURSIVE)
        has_warnings = True

    effective_model = model or (get_effective_model(provider) if provider else None)
    is_vision_needed = fallback_mode == FallbackMode.VISION or rename_images
    if is_vision_needed and effective_model:
        from renai.utils.models_dev import is_vision_supported

        vision_ok = is_vision_supported(effective_model)
        if vision_ok is False:
            table.add_row(
                msg.Warn.BADGE_VISION,
                msg.Warn.desc_vision(effective_model),
            )
            has_warnings = True

    if not has_warnings:
        return

    console.print(
        Panel(
            table,
            title=msg.PANEL_TITLE_REVIEW,
            border_style="yellow",
            expand=False,
            padding=(1, 2),
        )
    )

    try:
        console.print(msg.Prompt.PROCEED, end="")
        input("")
    except (KeyboardInterrupt, EOFError):
        console.print(msg.Prompt.CANCELLED)
        sys.exit(0)


# =============================================================================
# CLI Commands
# =============================================================================


def _print_process_header(root_dir: Path, mode: str, provider: str, model: str | None) -> None:
    """Print header panel information for file processing."""
    from rich.columns import Columns
    from rich.panel import Panel

    title = msg.Info.HEADER_TITLE.format(__version__)
    info_cols = [
        msg.Info.HEADER_DIR.format(root_dir.name),
        msg.Info.HEADER_MODE.format(mode),
        msg.Info.HEADER_PROVIDER.format(provider, model or "default"),
    ]
    console.print(
        Panel(Columns(info_cols, expand=True), title=title, border_style="cyan", padding=(0, 1))
    )


def _validate_cli_flags(root_dir: Path) -> None:
    """Validate directory existence and CLI inputs."""
    if not root_dir.is_dir():
        console.print(msg.Info.DIR_NOT_FOUND.format(root_dir))
        raise typer.Exit(code=1)


def _build_processing_config(**kwargs: Any) -> ProcessingConfig:
    """Build ProcessingConfig instance from CLI parameters using kwargs forwarding."""
    if "exif" in kwargs:
        kwargs["add_exif_date"] = kwargs.pop("exif")
    kwargs.setdefault("skip_ocr_checks", False)
    return ProcessingConfig(**kwargs)


def check_vision_model_guard(
    fallback_mode: FallbackMode,
    rename_images: bool,
    vision_provider: str | None = None,
    vision_model: str | None = None,
) -> None:
    """Validate vision model configuration before processing starts.

    If vision operations are requested, display a clear warning/info panel
    and abort execution if a valid vision model is not configured.
    """
    is_vision_needed = fallback_mode == FallbackMode.VISION or rename_images
    if not is_vision_needed:
        return

    from renai.config import get_effective_vision_model, get_effective_vision_provider
    from renai.utils.models_dev import is_vision_supported

    eff_v_provider = vision_provider or get_effective_vision_provider()
    eff_v_model = vision_model or get_effective_vision_model(eff_v_provider)

    if not eff_v_model:
        _display_vision_missing_error(
            "Vision model is not configured. Image processing or vision fallback\n"
            "requires a vision model."
        )
        raise typer.Exit(code=1)

    vis_ok = is_vision_supported(eff_v_model)
    if vis_ok is False:
        _display_vision_missing_error(
            f"Selected vision model '{eff_v_model}' does not support vision modalities."
        )
        raise typer.Exit(code=1)


def _display_vision_missing_error(reason: str) -> None:
    from rich.panel import Panel

    message = (
        f"[bold red]⚠️ VISION MODEL CONFIGURATION REQUIRED[/bold red]\n\n"
        f"[yellow]{reason}[/yellow]\n\n"
        f"[cyan]Please run 'renai init' to configure a Vision Provider & Model,\n"
        f"or specify '--vision-model <model_name>' when running process.[/cyan]"
    )
    console.print(
        Panel(
            message,
            title="[bold yellow]Vision Configuration Guard[/bold yellow]",
            border_style="red",
            padding=(1, 2),
        )
    )


def handle_process_command(
    directory: str,
    mode: Literal["rename", "organize", "evaluate"],
    provider: str,
    model: str | None,
    workers: int,
    ocr_enhanced: bool,
    debug_ocr: bool,
    use_mutool: bool,
    mutool_path: str,
    extractor: str,
    text_length: int,
    text_extraction: str,
    extraction_pages: int,
    recursive: bool,
    update_metadata: bool,
    fallback_mode: FallbackMode | None,
    move_scan_failures: bool,
    rename_images: bool,
    exif: bool,
    benchmark: bool,
    vision_max_pages: int,
    vision_provider: str | None = None,
    vision_model: str | None = None,
    dry_run: bool = False,
) -> None:
    """Orchestrate and execute the file processing pipeline."""
    final_fallback = fallback_mode or FallbackMode(app_settings.fallback_mode)
    root_dir = Path(directory).resolve()
    _validate_cli_flags(root_dir)

    # Check Vision Model Guard before running pipeline
    check_vision_model_guard(final_fallback, rename_images, vision_provider, vision_model)

    _print_process_header(root_dir, mode, provider, model)
    display_warnings(mode, recursive, provider, model, final_fallback, rename_images)

    config = _build_processing_config(
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
        fallback_mode=final_fallback,
        move_scan_failures=move_scan_failures,
        rename_images=rename_images,
        exif=exif,
        update_metadata=update_metadata,
        benchmark=benchmark,
        vision_max_pages=vision_max_pages,
        vision_provider=vision_provider,
        vision_model=vision_model,
        dry_run=dry_run,
    )

    try:
        asyncio.run(
            run_pipeline(
                root_dir=root_dir,
                provider=provider,
                model=model,
                config=config,
            )
        )
    except KeyboardInterrupt:
        console.print(msg.Prompt.CANCELLED_EXIT)
        raise typer.Exit(code=0) from None


@app.command("process")
def process_cmd(
    directory: Annotated[
        str,
        typer.Argument(
            help=msg.Help.DIRECTORY,
        ),
    ] = ".",
    mode: Annotated[
        Literal["rename", "organize", "evaluate"],
        typer.Option(
            "--mode",
            "-m",
            help=msg.Help.MODE,
        ),
    ] = "rename",
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            "-p",
            help=msg.Help.PROVIDER,
        ),
    ] = app_settings.provider,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help=msg.Help.MODEL,
        ),
    ] = app_settings.model,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            help=msg.Help.WORKERS,
        ),
    ] = app_settings.workers,
    ocr_enhanced: Annotated[
        bool,
        typer.Option(
            "--ocr-enhanced/--no-ocr-enhanced",
            help=msg.Help.OCR_ENHANCED,
        ),
    ] = app_settings.ocr_enhanced,
    debug_ocr: Annotated[
        bool,
        typer.Option(
            "--debug-ocr",
            help=msg.Help.DEBUG_OCR,
        ),
    ] = app_settings.debug_ocr,
    use_mutool: Annotated[
        bool,
        typer.Option(
            "--use-mutool/--no-mutool",
            help=msg.Help.USE_MUTOOL,
        ),
    ] = app_settings.use_mutool,
    mutool_path: Annotated[
        str,
        typer.Option(
            "--mutool-path",
            help=msg.Help.MUTOOL_PATH,
        ),
    ] = app_settings.mutool_path,
    extractor: Annotated[
        str,
        typer.Option(
            "--extractor",
            help=msg.Help.EXTRACTOR,
        ),
    ] = app_settings.extractor,
    text_length: Annotated[
        int,
        typer.Option(
            "--text-length",
            help=msg.Help.TEXT_LENGTH,
        ),
    ] = app_settings.text_length,
    text_extraction: Annotated[
        str,
        typer.Option(
            "--text-extraction",
            help=msg.Help.TEXT_EXTRACTION,
        ),
    ] = app_settings.text_extraction,
    extraction_pages: Annotated[
        int,
        typer.Option(
            "--extraction-pages",
            help=msg.Help.EXTRACTION_PAGES,
        ),
    ] = app_settings.extraction_pages,
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive/--no-recursive",
            help=msg.Help.RECURSIVE,
        ),
    ] = app_settings.recursive,
    update_metadata: Annotated[
        bool,
        typer.Option(
            "--update-metadata",
            help=msg.Help.UPDATE_METADATA,
        ),
    ] = False,
    fallback_mode: Annotated[
        FallbackMode | None,
        typer.Option(
            "--fallback-mode",
            help=msg.Help.FALLBACK_MODE,
        ),
    ] = None,
    move_scan_failures: Annotated[
        bool,
        typer.Option(
            "--move-scan-failures",
            help=msg.Help.MOVE_SCAN_FAILURES,
        ),
    ] = app_settings.move_scan_failures,
    rename_images: Annotated[
        bool,
        typer.Option(
            "--rename-images",
            help=msg.Help.RENAME_IMAGES,
        ),
    ] = app_settings.rename_images,
    exif: Annotated[
        bool,
        typer.Option(
            "--exif",
            help=msg.Help.EXIF,
        ),
    ] = app_settings.exif,
    benchmark: Annotated[
        bool,
        typer.Option(
            "--benchmark",
            help=msg.Help.BENCHMARK,
        ),
    ] = False,
    vision_max_pages: Annotated[
        int,
        typer.Option(
            "--vision-max-pages",
            help=msg.Help.VISION_MAX_PAGES,
        ),
    ] = app_settings.vision_max_pages,
    vision_provider: Annotated[
        str | None,
        typer.Option(
            "--vision-provider",
            help=msg.Help.VISION_PROVIDER,
        ),
    ] = app_settings.vision_provider,
    vision_model: Annotated[
        str | None,
        typer.Option(
            "--vision-model",
            help=msg.Help.VISION_MODEL,
        ),
    ] = app_settings.vision_model,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Simulate processing without renaming or moving files.",
        ),
    ] = False,
) -> None:
    """Process files in a directory."""
    handle_process_command(
        directory=directory,
        mode=mode,
        provider=provider,
        model=model,
        workers=workers,
        ocr_enhanced=ocr_enhanced,
        debug_ocr=debug_ocr,
        use_mutool=use_mutool,
        mutool_path=mutool_path,
        extractor=extractor,
        text_length=text_length,
        text_extraction=text_extraction,
        extraction_pages=extraction_pages,
        recursive=recursive,
        update_metadata=update_metadata,
        fallback_mode=fallback_mode,
        move_scan_failures=move_scan_failures,
        rename_images=rename_images,
        exif=exif,
        benchmark=benchmark,
        vision_max_pages=vision_max_pages,
        vision_provider=vision_provider,
        vision_model=vision_model,
        dry_run=dry_run,
    )


@app.command("evaluate")
def evaluate_cmd(
    directory: Annotated[
        str | None,
        typer.Argument(
            help=msg.Help.EVAL_DIR,
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help=msg.Help.EVAL_PROVIDER,
        ),
    ] = None,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            help=msg.Help.EVAL_WORKERS,
        ),
    ] = app_settings.workers,
) -> None:
    """Evaluate multiple models and providers."""
    target_dir = Path(directory or ".").resolve()
    if not target_dir.is_dir():
        console.print(msg.Info.DIR_NOT_FOUND.format(target_dir))
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
        console.print(msg.Prompt.EVAL_CANCELLED)
        raise typer.Exit(code=0) from None


@app.command("providers")
def list_providers() -> None:
    """List available LLM providers."""
    table = Table(title=msg.Info.PROVIDER_TABLE_TITLE)
    table.add_column(msg.Info.COL_PROVIDER, style="cyan")
    table.add_column(msg.Info.COL_MODEL, style="green")

    for name, config in PROVIDERS.items():
        table.add_row(name, config.default_model)

    console.print(table)


@app.command("cache")
def clear_cache_cmd(
    confirm: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Clear the cache directory."""
    if not confirm:
        response = input(msg.Prompt.CACHE_CONFIRM)
        if response.lower() not in ("y", "yes"):
            console.print(msg.Prompt.CACHE_CANCELLED)
            raise typer.Exit()

    handle_clear_cache()


@app.command("init")
def init_cmd() -> None:
    """Initialize renAI setup wizard."""
    run_init_wizard()


# Preserve 'undo' as a convenient short alias alongside 'undo-last' for
# user convenience and backward compatibility
@app.command("undo")
@app.command("undo-last")
def undo_cmd() -> None:
    """Undo operations from the most recent renaming or organizing run."""
    from renai.file_ops.operations import undo_last_operation

    reverted = undo_last_operation()
    if reverted > 0:
        console.print(f"[green]Successfully reverted {reverted} file operations.[/green]")
    else:
        console.print("[yellow]No recent operations to undo.[/yellow]")


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
