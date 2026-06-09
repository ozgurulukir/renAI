import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import typer
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
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

from renai.config import (
    PROVIDERS,
    app_settings,
    get_api_key,
    get_effective_model,
    get_image_extensions,
    load_all_eval_models,
    load_eval_models,
)
from renai.config.constants import (
    EVAL_LOG_FILE,
    MAX_DISPLAY_NAME_LENGTH,
    MAX_RECENT_ACTIVITIES,
)
from renai.engines.llm import LLMEngine
from renai.engines.ocr import OCREngine
from renai.models import ProcessingResult
from renai.models.schemas import FallbackMode, ProcessingConfig
from renai.processors.book import BookProcessor
from renai.processors.text_extractor import TextExtractor
from renai.utils.rate_limiter import AsyncRateLimiter

console = Console()

_eval_log_buffer: list[dict[str, Any]] = []
_eval_log_lock = asyncio.Lock()
_EVAL_LOG_BATCH_SIZE = 50


async def setup_engines(
    provider: str,
    model: str | None,
    config: ProcessingConfig,
) -> tuple[OCREngine, LLMEngine, BookProcessor]:
    """Initialize OCR, LLM, and book processor engines asynchronously."""
    # Initialize OCR engine
    ocr_engine = OCREngine(
        enhanced=config.ocr_enhanced,
        debug=config.debug_ocr,
        use_mutool=config.use_mutool,
        mutool_path=config.mutool_path,
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
        extractor_name=config.extractor,
        provider=provider,
        model=model_name,
        target_length=config.text_length,
        text_extraction=config.text_extraction,
        extraction_pages=config.extraction_pages,
        move_scan_failures=config.move_scan_failures,
    )

    return ocr_engine, llm_engine, processor


def collect_files(
    root_dir: Path, processor: BookProcessor, recursive: bool, rename_images: bool = False
) -> list[Path]:
    """Collect supported files from directory."""
    file_paths = []
    iterator = root_dir.rglob("*") if recursive else root_dir.iterdir()
    for entry in iterator:
        if (
            entry.is_file()
            and (
                entry.suffix.lower() in processor.supported_extensions
                or (rename_images and entry.suffix.lower() in get_image_extensions())
            )
            and "Needs_Scan" not in str(entry)
            and "Scan_Failures" not in str(entry)
        ):
            file_paths.append(entry)
    return file_paths


class StatusDisplay:
    """Tracks stats for renAI processing."""

    def __init__(self, progress: Progress):
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.activities = []
        self.max_activities = MAX_RECENT_ACTIVITIES
        self.progress = progress

    def update(self, result: ProcessingResult | str):
        if isinstance(result, ProcessingResult):
            self.processed += 1
            if result.success:
                self.successful += 1
            else:
                self.failed += 1

            status = "[green]SUCCESS[/]" if result.success else "[red]FAILED[/]"
            display_name = getattr(result, "name", None) or getattr(result, "file_path", None)
            display_name = str(display_name) if display_name else "Unknown"
        else:
            status = "[dim]•[/]"
            display_name = str(result)

        # Aggressive truncation to prevent terminal row wrapping corruption
        if len(display_name) > MAX_DISPLAY_NAME_LENGTH:
            display_name = display_name[: MAX_DISPLAY_NAME_LENGTH - 3] + "..."

        res_text = f"{status} {display_name}"
        self.activities.append(res_text)

        if len(self.activities) > MAX_RECENT_ACTIVITIES:
            self.activities.pop(0)

    def __rich__(self) -> Table:
        # 1. Activity Panel
        activity_table = Table.grid(expand=True)
        activity_table.add_column(no_wrap=True, overflow="ellipsis")
        for act in self.activities:
            activity_table.add_row(act)

        # 2. Stats and Progress combined
        layout_table = Table.grid(expand=True)
        layout_table.add_row(
            Panel(
                activity_table,
                title="[bold blue]Recent Activity[/]",
                border_style="blue",
                padding=(1, 2),
            )
        )

        stats_line = (
            f" [bold cyan]Files:[/] {self.processed}  "
            f"[bold green]Success:[/] {self.successful}  "
            f"[bold red]Failed:[/] {self.failed}"
        )
        layout_table.add_row(
            Panel(
                Group(stats_line, "\n", self.progress),
                title="[bold green]Total Progress[/]",
                border_style="green",
            )
        )

        return layout_table


async def log_evaluation_result(file_path: Path, provider: str, model: str, metadata: Any) -> None:
    """Buffer evaluation result for batched async file writing."""
    entry = {
        "file": file_path.name,
        "provider": provider,
        "model": model,
        "metadata": metadata,
    }
    async with _eval_log_lock:
        _eval_log_buffer.append(entry)
        if len(_eval_log_buffer) >= _EVAL_LOG_BATCH_SIZE:
            await _flush_eval_log()


async def _flush_eval_log() -> None:
    """Write buffered evaluation entries to disk in a single I/O operation."""
    if not _eval_log_buffer:
        return
    entries = _eval_log_buffer.copy()
    _eval_log_buffer.clear()
    try:

        def _write():
            with Path(EVAL_LOG_FILE).open("a", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        await asyncio.to_thread(_write)
    except Exception as e:
        logging.error(f"Failed to flush evaluation log: {e}")


async def _execute_processing_loop(
    processor: BookProcessor,
    file_paths: list[Path],
    provider: str,
    model: str | None,
    config: ProcessingConfig,
) -> list[ProcessingResult]:
    """Execute the file processing loop with real-time progress visualization."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    )
    main_task = progress.add_task("[cyan]Processing...", total=len(file_paths))
    status_display = StatusDisplay(progress)
    semaphore = asyncio.Semaphore(config.workers)
    results = []

    try:
        with Live(
            status_display,
            console=console,
            refresh_per_second=4,
            screen=True,  # Isolate from scrollback to prevent stacking
        ) as live:
            if config.benchmark:
                await _run_benchmark_mode(processor, file_paths, status_display, live)
                return []

            # Standard Mode (Rename, Organize, Evaluate)
            async def _process_task(f_path: Path) -> ProcessingResult:
                async with semaphore:
                    status_display.update(f"Scanning {f_path.name}")

                    res = await processor.process_file(
                        f_path,
                        mode=config.mode,
                        fallback_mode=config.fallback_mode,
                        rename_images=config.rename_images,
                        update_metadata=config.update_metadata,
                        add_exif_date=config.add_exif_date,
                    )
                    if config.mode == "evaluate" and res.success:
                        await log_evaluation_result(
                            f_path, provider, model or "unknown", res.metadata
                        )

                    status_display.update(res)
                    progress.advance(main_task)

                    return res

            pending_tasks = [asyncio.ensure_future(_process_task(f)) for f in file_paths]
            results = await asyncio.gather(*pending_tasks)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user. Exiting...[/yellow]")
        return []

    return results


def _print_final_summary(results: list[ProcessingResult]) -> None:
    """Print the final static summary card to the terminal."""
    success_count = sum(1 for r in results if r.success)
    error_count = len(results) - success_count

    summary_table = Table.grid(padding=(0, 1))
    summary_table.add_column(justify="left", style="white")
    summary_table.add_column(justify="right", style="cyan")
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
            title="[bold blue]Processing Result Summary[/]",
            border_style="green" if error_count == 0 else "yellow",
            padding=(1, 2),
            expand=False,
        )
    )


async def run_pipeline(
    root_dir: Path,
    provider: str,
    model: str | None = None,
    config: ProcessingConfig | None = None,
) -> None:
    """Core processing pipeline orchestration."""
    if config is None:
        config = ProcessingConfig()

    # 1. Setup
    _ocr_engine, _llm_engine, processor = await setup_engines(
        provider,
        model,
        config,
    )

    await processor.setup_move_folders(
        root_dir,
        skip_ocr_checks=config.skip_ocr_checks,
        move_scan_failures=config.move_scan_failures,
    )

    # 2. Collection
    try:
        file_paths = collect_files(root_dir, processor, config.recursive, config.rename_images)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user. Exiting...[/yellow]")
        raise typer.Exit(code=0) from None

    if not file_paths:
        console.print("[yellow]No supported files found to process.[/yellow]")
        return

    if config.benchmark:
        file_paths = file_paths[:3]
        console.print(
            f"[yellow]Benchmark mode: only processing first {len(file_paths)} files.[/yellow]"
        )

    # Initial Info Summary
    summary_cols = [
        f"[bold blue]🚀 Files:[/bold blue] {len(file_paths)}",
        f"[bold cyan]⚙ Workers:[/bold cyan] {config.workers}",
        f"[bold yellow]⏱ RPM Limit:[/bold yellow] {app_settings.llm_rpm_limit}",
    ]

    console.print(Columns(summary_cols, expand=True))
    console.print()

    # 3. Execution Loop
    results = await _execute_processing_loop(processor, file_paths, provider, model, config)

    await _flush_eval_log()

    # 4. Final Summary Card (Static, will stay in terminal history)
    if not config.benchmark and results:
        _print_final_summary(results)


async def _run_benchmark_mode(
    processor: BookProcessor,
    file_paths: list[Path],
    status_display: StatusDisplay,
    live: Live,
) -> None:
    """Run PDF extraction benchmark across multiple libraries."""
    extractors_to_test = ["pymupdf", "pdfplumber", "pypdf", "pypdfium2"]
    benchmark_data = []

    for ext_name in extractors_to_test:
        msg = f"[bold yellow]BENCHMARK[/] Testing [cyan]{ext_name}[/]..."
        status_display.update(msg)

        processor.extractor_name = ext_name
        # Re-initialize TextExtractor with new library
        processor.text_extractor = TextExtractor(
            ocr_engine=processor.ocr_engine,
            extractor_name=ext_name,
            target_length=processor.target_length,
        )

        start_t = time.perf_counter()
        success_count = 0

        for f_path in file_paths:
            res = await processor.process_file(f_path, mode="evaluate")
            if res.success:
                success_count += 1
            status_display.update(res)

        duration = time.perf_counter() - start_t
        benchmark_data.append(
            {
                "extractor": ext_name,
                "time": f"{duration:.2f}s",
                "success": f"{success_count}/{len(file_paths)}",
            }
        )
        status_display.update(f"[green]Completed {ext_name} in {duration:.2f}s[/]")

    # Show Benchmark Result Table outside live
    live.stop()
    bench_table = Table(title="PDF Extractor Benchmark Results")
    bench_table.add_column("Extractor", style="cyan")
    bench_table.add_column("Time", style="magenta")
    bench_table.add_column("Success Rate", style="green")

    for data in benchmark_data:
        bench_table.add_row(data["extractor"], data["time"], data["success"])

    console.print("\n")
    console.print(bench_table)
    console.print("\n")


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
            config=ProcessingConfig(
                mode="evaluate",
                workers=workers,
                recursive=False,
                fallback_mode=FallbackMode.OCR,
                extractor=app_settings.extractor,
                text_length=app_settings.text_length,
                text_extraction=app_settings.text_extraction,
                extraction_pages=app_settings.extraction_pages,
            ),
        )
        console.print(f"[dim]Results logged to: {EVAL_LOG_FILE}[/dim]\n")
