"""
renAI CLI - Modern Typer-based command line interface.

This is the main CLI entry point for renAI. All functionality from the
original main.py has been migrated here.
"""

import asyncio
import logging
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
    config_load_error,
    initialize_directories,
)
from renai.exceptions import ConfigurationError
from renai.logging_config import configure_logging
from renai.messages import cli as msg, orchestrator as orch_msg
from renai.models.schemas import FallbackMode, ProcessingConfig
from renai.processors.orchestrator import run_evaluation_pipeline, run_pipeline
from renai.processors.setup import run_init_wizard
from renai.utils import ensure_secure_dir

# Initialize logging
configure_logging()

logger = logging.getLogger(__name__)
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
        except Exception as e:
            logging.getLogger(__name__).debug("stdout UTF-8 re-wrap failed: %s", e)

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

    # A corrupt DB reset is reported honestly (#248): it is an action taken,
    # not "already empty", even though it reclaims no counted rows.
    db_reset = results.pop("db_reset", 0)
    total = sum(results.values())
    if total > 0:
        details = ", ".join([f"{count} {ctype}" for ctype, count in results.items() if count > 0])
        console.print(msg.Info.CACHE_CLEARED.format(total, details))
    elif not db_reset:
        console.print(msg.Info.CACHE_EMPTY)
    if db_reset:
        console.print(msg.Info.CACHE_DB_RESET)


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

        try:
            vision_ok = is_vision_supported(effective_model)
        except Exception:
            # A catalog lookup failure must not crash the run; treat the
            # model as unknown so no vision warning is shown.
            logger.debug(f"Vision capability lookup failed for {effective_model}", exc_info=True)
            vision_ok = None
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

    # Non-interactive runs (cron, CI, wrappers with stdin at EOF or a pipe)
    # must not be blocked by — or silently cancelled through — the prompt:
    # proceed without asking so unattended runs behave like explicit consent.
    if not sys.stdin.isatty():
        return

    try:
        console.print(msg.Prompt.PROCEED, end="")
        answer = input("")
    except (KeyboardInterrupt, EOFError):
        console.print(msg.Prompt.CANCELLED)
        # Cancelled is a failure, not success: exit non-zero (issue #257).
        # typer.Exit is an Exception (not BaseException like sys.exit), so it
        # MUST NOT be raised where a broad `except Exception` upstream could
        # swallow it and turn a deliberate cancel into an apparent crash.
        raise typer.Exit(code=1) from None
    # Only Enter or y/yes proceeds; any other answer cancels.
    if answer.strip().lower() not in ("", "y", "yes"):
        console.print(msg.Prompt.CANCELLED)
        raise typer.Exit(code=1)


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


def _validate_cli_flags(root_dir: Path, fallback_mode: FallbackMode | None = None) -> None:
    """Validate directory existence and CLI inputs."""
    if not root_dir.is_dir():
        console.print(msg.Info.DIR_NOT_FOUND.format(root_dir))
        raise typer.Exit(code=1)

    # FallbackMode.NONE is an internal sentinel remapped to OCR by BookProcessor;
    # it must not be selectable from the command line.
    if fallback_mode == FallbackMode.NONE:
        console.print(msg.Error.FALLBACK_NONE_REJECTED)
        raise typer.Exit(code=1)


def _build_processing_config(**kwargs: Any) -> ProcessingConfig:
    """Forward raw CLI flags to the single population path.

    Kept as a thin named seam for tests/readability: all config logic lives in
    ``ProcessingConfig.from_app_settings`` — the ``exif`` -> ``add_exif_date``
    rename and every settings-derived default happen there, not here.
    """
    return ProcessingConfig.from_app_settings(**kwargs)


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
        _display_vision_missing_error(msg.Error.VISION_REASON_NOT_CONFIGURED)
        raise typer.Exit(code=1)

    try:
        vis_ok = is_vision_supported(eff_v_model)
    except Exception:
        # A catalog lookup failure must not end the run with a traceback;
        # treat the model as unknown and let processing continue.
        logger.debug(f"Vision capability lookup failed for {eff_v_model}", exc_info=True)
        vis_ok = None
    if vis_ok is False:
        _display_vision_missing_error(msg.Error.VISION_REASON_NOT_SUPPORTED.format(eff_v_model))
        raise typer.Exit(code=1)


def _display_vision_missing_error(reason: str) -> None:
    from rich.panel import Panel

    console.print(
        Panel(
            msg.Error.VISION_GUARD_BODY.format(reason),
            title=msg.Error.VISION_GUARD_TITLE,
            border_style="red",
            padding=(1, 2),
        )
    )


def _run_process_pipeline(
    config: ProcessingConfig,
    root_dir: Path,
    provider: str,
    model: str | None,
) -> None:
    """Run the processing pipeline with the CLI error mapping.

    Setup failure maps to exit 1, user cancellation to exit 0.
    """
    try:
        asyncio.run(
            run_pipeline(
                root_dir=root_dir,
                provider=provider,
                model=model,
                config=config,
            )
        )
    except ConfigurationError as exc:
        console.print(msg.Error.SETUP_FAILED.format(exc))
        raise typer.Exit(code=1) from None
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
        Literal["rename", "organize", "copy", "evaluate"],
        typer.Option(
            "--mode",
            "-m",
            help=msg.Help.MODE,
        ),
    ] = "rename",
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help=msg.Help.PROVIDER,
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help=msg.Help.MODEL,
        ),
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-w",
            help=msg.Help.WORKERS,
        ),
    ] = None,
    ocr_enhanced: Annotated[
        bool | None,
        typer.Option(
            "--ocr-enhanced/--no-ocr-enhanced",
            help=msg.Help.OCR_ENHANCED,
        ),
    ] = None,
    debug_ocr: Annotated[
        bool | None,
        typer.Option(
            "--debug-ocr",
            help=msg.Help.DEBUG_OCR,
        ),
    ] = None,
    use_mutool: Annotated[
        bool | None,
        typer.Option(
            "--use-mutool/--no-mutool",
            help=msg.Help.USE_MUTOOL,
        ),
    ] = None,
    mutool_path: Annotated[
        str | None,
        typer.Option(
            "--mutool-path",
            help=msg.Help.MUTOOL_PATH,
        ),
    ] = None,
    extractor: Annotated[
        str | None,
        typer.Option(
            "--extractor",
            help=msg.Help.EXTRACTOR,
        ),
    ] = None,
    text_length: Annotated[
        int | None,
        typer.Option(
            "--text-length",
            help=msg.Help.TEXT_LENGTH,
        ),
    ] = None,
    text_extraction: Annotated[
        str | None,
        typer.Option(
            "--text-extraction",
            help=msg.Help.TEXT_EXTRACTION,
        ),
    ] = None,
    extraction_pages: Annotated[
        int | None,
        typer.Option(
            "--extraction-pages",
            help=msg.Help.EXTRACTION_PAGES,
        ),
    ] = None,
    recursive: Annotated[
        bool | None,
        typer.Option(
            "--recursive/--no-recursive",
            help=msg.Help.RECURSIVE,
        ),
    ] = None,
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
        bool | None,
        typer.Option(
            "--move-scan-failures",
            help=msg.Help.MOVE_SCAN_FAILURES,
        ),
    ] = None,
    rename_images: Annotated[
        bool | None,
        typer.Option(
            "--rename-images",
            help=msg.Help.RENAME_IMAGES,
        ),
    ] = None,
    exif: Annotated[
        bool | None,
        typer.Option(
            "--exif",
            help=msg.Help.EXIF,
        ),
    ] = None,
    benchmark: Annotated[
        bool,
        typer.Option(
            "--benchmark",
            help=msg.Help.BENCHMARK,
        ),
    ] = False,
    vision_max_pages: Annotated[
        int | None,
        typer.Option(
            "--vision-max-pages",
            help=msg.Help.VISION_MAX_PAGES,
        ),
    ] = None,
    vision_provider: Annotated[
        str | None,
        typer.Option(
            "--vision-provider",
            help=msg.Help.VISION_PROVIDER,
        ),
    ] = None,
    vision_model: Annotated[
        str | None,
        typer.Option(
            "--vision-model",
            help=msg.Help.VISION_MODEL,
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=msg.Help.DRY_RUN,
        ),
    ] = False,
) -> None:
    """Process files in a directory."""
    # Resolve omitted CLI options from app_settings at runtime rather than
    # at import time so malformed config or post-import overrides work cleanly.
    resolved_provider = provider if provider is not None else app_settings.provider
    resolved_model = model if model is not None else app_settings.model
    resolved_recursive = recursive if recursive is not None else app_settings.recursive
    resolved_rename_images = (
        rename_images if rename_images is not None else app_settings.rename_images
    )
    resolved_vision_provider = (
        vision_provider if vision_provider is not None else app_settings.vision_provider
    )
    resolved_vision_model = vision_model if vision_model is not None else app_settings.vision_model

    # An invalid settings.toml must stop the run with a clear message naming the
    # offending field instead of silently continuing on factory defaults.
    if config_load_error is not None:
        console.print(
            msg.Error.SETUP_FAILED.format(orch_msg.Error.SETTINGS_INVALID.format(config_load_error))
        )
        raise typer.Exit(code=1)

    # Reject only the explicit CLI flag; settings/env "none" stays an accepted
    # internal sentinel that BookProcessor remaps to OCR.
    try:
        final_fallback = fallback_mode or FallbackMode(app_settings.fallback_mode)
    except ValueError as err:
        console.print(
            msg.Error.SETUP_FAILED.format(
                orch_msg.Error.INVALID_FALLBACK_MODE.format(app_settings.fallback_mode)
            )
        )
        raise typer.Exit(code=1) from err
    root_dir = Path(directory).resolve()
    _validate_cli_flags(root_dir, fallback_mode)

    # Check Vision Model Guard before running pipeline
    check_vision_model_guard(
        final_fallback, resolved_rename_images, resolved_vision_provider, resolved_vision_model
    )

    overrides: dict[str, Any] = {
        "mode": mode,
        "fallback_mode": final_fallback,
        "update_metadata": update_metadata,
        "benchmark": benchmark,
        "dry_run": dry_run,
    }
    optional_overrides = {
        "workers": workers,
        "recursive": recursive,
        "ocr_enhanced": ocr_enhanced,
        "debug_ocr": debug_ocr,
        "use_mutool": use_mutool,
        "mutool_path": mutool_path,
        "extractor": extractor,
        "text_length": text_length,
        "text_extraction": text_extraction,
        "extraction_pages": extraction_pages,
        "move_scan_failures": move_scan_failures,
        "rename_images": rename_images,
        "exif": exif,
        "vision_max_pages": vision_max_pages,
        "vision_provider": vision_provider,
        "vision_model": vision_model,
    }
    for key, val in optional_overrides.items():
        if val is not None:
            overrides[key] = val

    config = _build_processing_config(**overrides)

    _print_process_header(root_dir, mode, resolved_provider, resolved_model)
    display_warnings(
        mode,
        resolved_recursive,
        resolved_provider,
        resolved_model,
        final_fallback,
        resolved_rename_images,
    )
    _run_process_pipeline(config, root_dir, resolved_provider, resolved_model)


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
        int | None,
        typer.Option(
            "--workers",
            help=msg.Help.EVAL_WORKERS,
        ),
    ] = None,
) -> None:
    """Evaluate multiple models and providers."""
    resolved_workers = workers if workers is not None else app_settings.workers
    target_dir = Path(directory or ".").resolve()
    if not target_dir.is_dir():
        console.print(msg.Info.DIR_NOT_FOUND.format(target_dir))
        raise typer.Exit(code=1)

    try:
        asyncio.run(
            run_evaluation_pipeline(
                root_dir=target_dir,
                provider_filter=provider,
                workers=resolved_workers,
            )
        )
    except ConfigurationError as exc:
        console.print(msg.Error.SETUP_FAILED.format(exc))
        raise typer.Exit(code=1) from None
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
    # Non-interactive runs proceed without prompting (see display_warnings):
    # unattended invocations must not silently cancel with exit code 0.
    if not confirm and sys.stdin.isatty():
        try:
            response = input(msg.Prompt.CACHE_CONFIRM)
        except (KeyboardInterrupt, EOFError):
            console.print(msg.Prompt.CACHE_CANCELLED)
            # A cancelled run is a failure, not success: exit non-zero.
            raise typer.Exit(code=1) from None
        # Only y/yes proceeds; any other answer cancels.
        if response.strip().lower() not in ("y", "yes"):
            console.print(msg.Prompt.CACHE_CANCELLED)
            raise typer.Exit(code=1)

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

    result = undo_last_operation()
    if result.run_id is None:
        console.print(msg.Info.UNDO_NOTHING)
        return
    attempted = result.reverted + result.skipped + result.failed
    console.print(msg.Info.UNDO_RUN.format(result.run_id, attempted))
    if result.reverted > 0:
        console.print(msg.Info.UNDO_REVERTED.format(result.reverted))
    if result.failed > 0:
        console.print(msg.Info.UNDO_FAILED.format(result.failed))
    if result.skipped > 0:
        console.print(msg.Info.UNDO_SKIPPED.format(result.skipped))


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
