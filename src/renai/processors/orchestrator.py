import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import typer
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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
    """Manages the Live UI display for renAI processing."""

    def __init__(self, total_files: int):
        self.total = total_files
        self.completed = 0
        self.success = 0
        self.failed = 0
        self.last_results = []
        self.current_activity = "Initializing..."

    def update(self, result=None, activity=None):
        if result:
            self.completed += 1
            if result.success:
                self.success += 1
                status = "[green]SUCCESS[/green]"
            else:
                self.failed += 1
                status = "[red]FAILED[/red]"

            # Keep last 5 results
            display_name = result.name or (result.file_path.name if result.file_path else "Unknown")
            res_text = f"{status} {display_name}"
            self.last_results.append(res_text)
            if len(self.last_results) > 5:
                self.last_results.pop(0)

        if activity:
            self.current_activity = activity

    def generate_table(self) -> Table:
        table = Table(expand=True, box=None)
        table.add_column("Status", width=12)
        table.add_column("File")

        for res in reversed(self.last_results):
            parts = res.split(" ", 1)
            table.add_row(parts[0], parts[1] if len(parts) > 1 else "")
        return table

    # Status Bar logic
    def __rich__(self) -> Panel:
        pct = (self.completed / self.total) * 100 if self.total > 0 else 0
        filled = int(pct / 5)
        bar = "[bold green]━[/bold green]" * filled + "[dim]━[/dim]" * (20 - filled)

        # Build footer with activity
        footer = Text.assemble(
            (f"{bar} ", "cyan"),
            (f"{pct:>3.0f}%", "bold white"),
            (f" | {self.completed}/{self.total} | ", "dim"),
            (self.current_activity, "italic yellow"),
        )

        content = Layout()
        content.split_column(
            Layout(self.generate_table(), name="results", ratio=3),
            Layout(footer, name="footer", size=1),
        )

        return Panel(
            content,
            title="[bold blue]Processing Live[/bold blue]",
            subtitle=f"[bold green]S: {self.success}[/bold green] • [bold red]F: {self.failed}[/bold red]",
            border_style="cyan",
            padding=(1, 2),
        )


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
    model: str | None,
    mode: str,
    workers: int,
    recursive: bool,
    ocr_enhanced: bool,
    debug_ocr: bool,
    use_mutool: bool,
    mutool_path: str,
    extractor: str,
    text_length: int,
    text_extraction: str,
    extraction_pages: int,
    skip_ocr_checks: bool = False,
    move_scan_failures: bool = False,
    rename_images: bool = False,
    update_metadata: bool = False,
    add_exif_date: bool = False,
    fallback_mode: FallbackMode = FallbackMode.OCR,
    benchmark: bool = False,
) -> None:
    """Run the processing pipeline asynchronously."""
    # 1. Setup
    _, _, processor = await setup_engines(
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
    semaphore = asyncio.Semaphore(workers)
    status_display = StatusDisplay(len(file_paths))

    async def _process_task(f_path):
        async with semaphore:
            status_display.update(activity=f"Processing {f_path.name}...")
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
            return res

    results = []
    with Live(status_display, console=console, refresh_per_second=4, vertical_overflow="visible"):
        pending_tasks = [asyncio.ensure_future(_process_task(f)) for f in file_paths]
        for coro in asyncio.as_completed(pending_tasks):
            result = await coro
            results.append(result)

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
