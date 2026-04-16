import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from renai.engines.llm import LLMEngine
from renai.engines.ocr import OCREngine
from renai.processors.book import BookProcessor
from renai.processors.text_extractor import FallbackMode
from renai.settings import (
    EVAL_LOG_FILE,
    PROVIDERS,
    app_settings,
    get_api_key,
    get_effective_model,
    get_image_extensions,
    load_all_eval_models,
    load_eval_models,
)
from renai.utils.rate_limiter import AsyncRateLimiter

console = Console()


async def setup_engines(
    provider: str,
    model: str | None,
    ocr_enhanced: bool = False,
    debug_ocr: bool = False,
    use_mutool: bool = False,
    mutool_path: str = "mutool",
    extractor: str = "pymupdf",
    text_length: int = 10000,
    text_extraction: str = "mixed",
    extraction_pages: int = 10,
) -> tuple[OCREngine, LLMEngine, BookProcessor]:
    """Initialize OCR, LLM, and book processor engines asynchronously."""
    # Initialize OCR engine
    ocr_engine = OCREngine(
        enhanced=ocr_enhanced,
        debug=debug_ocr,
        use_mutool=use_mutool,
        mutool_path=mutool_path,
        silent=False,
    )

    # Get provider configuration
    if provider not in PROVIDERS:
        raise typer.Exit(code=1)

    provider_config = PROVIDERS[provider]
    api_key = provider_config.api_key or get_api_key(provider)
    if not api_key:
        raise typer.Exit(code=1)

    # Determine model to use
    model_name = model or get_effective_model(provider)

    # Initialize Rate Limiter
    rate_limiter = AsyncRateLimiter(rpm=app_settings.llm_rpm_limit)

    # Initialize LLM engine
    llm_engine = LLMEngine(
        provider=provider,
        api_key=api_key,
        base_url=provider_config.base_url,
        model_name=model_name,
        rate_limiter=rate_limiter,
    )

    # Initialize processor
    processor = BookProcessor(
        ocr_engine,
        llm_engine,
        extractor_name=extractor,
        provider=provider,
        model=model_name,
        target_length=text_length,
        text_extraction=text_extraction,
        extraction_pages=extraction_pages,
    )

    return ocr_engine, llm_engine, processor


def collect_files(root_dir: Path, processor: BookProcessor, recursive: bool) -> list[Path]:
    """Collect supported files from directory."""
    file_paths = []
    iterator = root_dir.rglob("*") if recursive else root_dir.iterdir()
    for entry in iterator:
        if (
            entry.is_file()
            and (
                entry.suffix.lower() in processor.supported_extensions
                or entry.suffix.lower() in get_image_extensions()
            )
            and "Needs_Scan" not in str(entry)
            and "Scan_Failures" not in str(entry)
        ):
            file_paths.append(entry)
    return file_paths


class StatusDisplay:
    """Tracks stats for renAI processing."""

    def __init__(self, total_files: int):
        self.total = total_files
        self.completed = 0
        self.success = 0
        self.failed = 0
        self.last_results = []

    def update(self, result=None):
        if result:
            self.completed += 1
            if result.success:
                self.success += 1
                status = "[green]SUCCESS[/green]"
            else:
                self.failed += 1
                status = "[red]FAILED[/red]"

            # Keep last 5 results
            display_name = getattr(result, "name", "Unknown")
            res_text = f"{status} {display_name}"
            self.last_results.append(res_text)
            if len(self.last_results) > 5:
                self.last_results.pop(0)


async def log_evaluation_result(file_path: Path, provider: str, model: str, metadata: Any) -> None:
    """Log evaluation result to a file asynchronously."""
    entry = {
        "file": file_path.name,
        "provider": provider,
        "model": model,
        "metadata": metadata,
    }
    try:

        def _write():
            with Path(EVAL_LOG_FILE).open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        await asyncio.to_thread(_write)
    except Exception as e:
        logging.error(f"Failed to log evaluation result: {e}")


async def run_pipeline(
    root_dir: Path,
    provider: str,
    model: str | None = None,
    mode: str = "rename",
    workers: int = 4,
    recursive: bool = False,
    ocr_enhanced: bool = False,
    debug_ocr: bool = False,
    use_mutool: bool = False,
    mutool_path: str = "mutool",
    extractor: str = "pymupdf",
    text_length: int = 10000,
    text_extraction: str = "mixed",
    extraction_pages: int = 10,
    skip_ocr_checks: bool = False,
    move_scan_failures: bool = False,
    fallback_mode: FallbackMode = FallbackMode.NONE,
    rename_images: bool = False,
    update_metadata: bool = False,
    add_exif_date: bool = False,
    benchmark: bool = False,
) -> None:
    """Core processing pipeline orchestration."""
    # 1. Setup
    _ocr_engine, _llm_engine, processor = await setup_engines(
        provider,
        model,
        ocr_enhanced,
        debug_ocr,
        use_mutool,
        mutool_path,
        extractor,
        text_length,
        text_extraction,
        extraction_pages,
    )

    await processor.setup_move_folders(
        root_dir,
        skip_ocr_checks=skip_ocr_checks,
        move_scan_failures=move_scan_failures,
    )

    # 2. Collection
    try:
        file_paths = collect_files(root_dir, processor, recursive)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user. Exiting...[/yellow]")
        raise typer.Exit(code=0) from None

    if not file_paths:
        console.print("[yellow]No supported files found to process.[/yellow]")
        return

    if benchmark:
        file_paths = file_paths[:3]
        console.print(
            f"[yellow]Benchmark mode: only processing first {len(file_paths)} files.[/yellow]"
        )

    # Initial Info Summary
    summary_cols = [
        f"[bold blue]🚀 Files:[/bold blue] {len(file_paths)}",
        f"[bold cyan]⚙ Workers:[/bold cyan] {workers}",
        f"[bold yellow]⏱ RPM Limit:[/bold yellow] {app_settings.llm_rpm_limit}",
    ]
    console.print(Columns(summary_cols, expand=True))
    console.print()

    # 3. Execution
    status_display = StatusDisplay(len(file_paths))
    semaphore = asyncio.Semaphore(workers)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    ) as progress:
        main_task = progress.add_task("[cyan]Processing...", total=len(file_paths))

        async def _process_task(f_path):
            async with semaphore:
                progress.update(main_task, description=f"[cyan]Processing {f_path.name}...")
                res = await processor.process_file(
                    f_path,
                    mode=mode,
                    fallback_mode=fallback_mode,
                    rename_images=rename_images,
                    update_metadata=update_metadata,
                    add_exif_date=add_exif_date,
                )
                if mode == "evaluate" and res.success:
                    await log_evaluation_result(f_path, provider, model or "unknown", res.metadata)

                status_display.update(result=res)
                progress.advance(main_task)

                # Print status briefly above progress bar
                status_color = "green" if res.success else "red"
                status_text = "SUCCESS" if res.success else "FAILED"
                progress.console.print(f"[{status_color}]{status_text:>7}[/] {f_path.name}")

                return res

        pending_tasks = [asyncio.ensure_future(_process_task(f)) for f in file_paths]
        results = await asyncio.gather(*pending_tasks)

    # 4. Final Summary Card
    success_count = sum(1 for r in results if r.success)
    error_count = len(results) - success_count

    summary_table = Table.grid(padding=(0, 2))
    summary_table.add_row(
        "[bold green]✅ Successful[/bold green]", f"[bold green]{success_count}[/bold green]"
    )
    if error_count > 0:
        summary_table.add_row(
            "[bold red]❌ Failed[/bold red]", f"[bold red]{error_count}[/bold red]"
        )

    console.print()
    console.print(
        Panel(
            summary_table,
            title="[bold]Processing Complete[/bold]",
            border_style="green" if error_count == 0 else "yellow",
            padding=(1, 2),
            expand=False,
        )
    )


async def run_evaluation_pipeline(
    root_dir: Path,
    provider_filter: str | None = None,
    workers: int = 4,
) -> None:
    """Run evaluation for multiple models asynchronously."""
    # Resolve models to a list of (provider, model_name) pairs
    target_models: list[tuple[str, str]] = []
    if provider_filter:
        models = load_eval_models(provider_filter)
        for m in models:
            target_models.append((provider_filter, m))
    else:
        all_models = load_all_eval_models()
        for prov, models in all_models.items():
            for m in models:
                target_models.append((prov, m))

    if not target_models:
        console.print("[red]No evaluation models found in eval_models.toml.[/red]")
        return

    console.print("🔍 [bold blue]Evaluation Mode[/bold blue]")
    console.print(f"Target: [cyan]{root_dir}[/cyan]")
    console.print(f"Models: [green]{len(target_models)} total[/green]\n")

    for provider, model_name in target_models:
        console.print(f"--- [bold yellow]Evaluating: {provider}/{model_name}[/bold yellow] ---")

        await run_pipeline(
            root_dir=root_dir,
            provider=provider,
            model=model_name,
            mode="evaluate",
            workers=workers,
            recursive=False,
            ocr_enhanced=False,
            debug_ocr=False,
            use_mutool=False,
            mutool_path="mutool",
            extractor=app_settings.extractor,
            text_length=app_settings.text_length,
            text_extraction=app_settings.text_extraction,
            extraction_pages=app_settings.extraction_pages,
            fallback_mode=FallbackMode.OCR,
        )
        console.print(f"[dim]Results logged to: {EVAL_LOG_FILE}[/dim]\n")
