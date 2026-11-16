import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

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
from renai.engines.llm import LLMCallPolicy, LLMEngine
from renai.engines.ocr import OCREngine
from renai.exceptions import ConfigurationError
from renai.extractors import EXTRACTOR_NAMES
from renai.messages import orchestrator as orch_msg
from renai.models import ProcessingResult
from renai.models.schemas import FallbackMode, ProcessingConfig
from renai.processors.book import BookProcessor
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
    # Initialize OCR engine; OCR is a CPU-bound process pool with its own knob
    # (ocr_workers) so it stays independent of the async `workers` LLM limit.
    ocr_engine = OCREngine(
        enhanced=config.ocr_enhanced,
        debug=config.debug_ocr,
        use_mutool=config.use_mutool,
        mutool_path=config.mutool_path,
        silent=False,
        max_workers=app_settings.ocr_workers,
        dpi=app_settings.ocr_dpi,
        enhanced_dpi=app_settings.ocr_enhanced_dpi,
        head_pages=app_settings.ocr_head_pages,
        tail_pages=app_settings.ocr_tail_pages,
        psm=app_settings.ocr_psm,
    )

    # Get provider configuration
    if provider not in PROVIDERS:
        raise ConfigurationError(
            orch_msg.Error.UNKNOWN_PROVIDER.format(provider, ", ".join(sorted(PROVIDERS)))
        )

    provider_config = PROVIDERS[provider]
    api_key = provider_config.api_key or get_api_key(provider)
    if not api_key:
        raise ConfigurationError(orch_msg.Error.API_KEY_MISSING.format(provider))

    # Determine model to use
    model_name = model or get_effective_model(provider)

    # Initialize Rate Limiter
    rate_limiter = AsyncRateLimiter(rpm=app_settings.llm_rpm_limit)

    # Config-to-engine translation happens here: engines read behavioral knobs
    # from an injected LLMCallPolicy, never from the settings singleton at
    # call time.
    call_policy = LLMCallPolicy(
        max_tokens=app_settings.llm_max_tokens,
        temperature=app_settings.llm_temperature,
        temperature_retry=app_settings.llm_temperature_retry,
        max_retries=app_settings.llm_max_retries,
        refine_mode=app_settings.refine_metadata,
        timeout=app_settings.llm_timeout,
        transport_max_attempts=app_settings.llm_transport_max_attempts,
        backoff_initial=app_settings.llm_backoff_initial,
        backoff_max=app_settings.llm_backoff_max,
    )

    # Initialize Text LLM engine
    llm_engine = LLMEngine(
        provider=provider,
        api_key=api_key,
        base_url=provider_config.base_url,
        model_name=model_name,
        rate_limiter=rate_limiter,
        call_policy=call_policy,
    )

    # Determine vision provider & model
    from renai.config import get_effective_vision_model, get_effective_vision_provider

    vision_provider = config.vision_provider or get_effective_vision_provider()
    vision_model_name = config.vision_model or get_effective_vision_model(vision_provider)

    # Mirror the CLI's vision guard: only demand a working vision provider when
    # vision output is actually requested. Stale vision config must not break
    # text-only runs, so misconfiguration degrades silently in that case.
    vision_required = config.fallback_mode == FallbackMode.VISION or config.rename_images
    if vision_required and not (vision_provider and vision_model_name):
        raise ConfigurationError(orch_msg.Error.VISION_NOT_CONFIGURED)

    vision_llm_engine: LLMEngine | None = None
    if vision_provider and vision_model_name:
        if vision_provider == provider and vision_model_name == model_name:
            vision_llm_engine = llm_engine
        elif vision_provider in PROVIDERS:
            v_prov_config = PROVIDERS[vision_provider]
            v_api_key = v_prov_config.api_key or get_api_key(vision_provider)
            if v_api_key:
                vision_llm_engine = LLMEngine(
                    provider=vision_provider,
                    api_key=v_api_key,
                    base_url=v_prov_config.base_url,
                    model_name=vision_model_name,
                    rate_limiter=rate_limiter,
                    call_policy=call_policy,
                )
            elif vision_required:
                raise ConfigurationError(orch_msg.Error.API_KEY_MISSING.format(vision_provider))
        elif vision_required:
            raise ConfigurationError(
                orch_msg.Error.UNKNOWN_PROVIDER.format(
                    vision_provider, ", ".join(sorted(PROVIDERS))
                )
            )

    # Initialize processor
    processor = BookProcessor(
        ocr_engine=ocr_engine,
        config=config,
        llm_engine=llm_engine,
        vision_llm_engine=vision_llm_engine,
    )

    return ocr_engine, llm_engine, processor


def collect_files(
    root_dir: Path, processor: BookProcessor, recursive: bool, rename_images: bool = False
) -> list[Path]:
    """Collect supported files from directory, excluding quarantine folders.

    Quarantine exclusion uses resolved-path comparison (see
    :class:`~renai.processors.text_extractor.Quarantine`), so user directories
    whose names merely contain a quarantine folder name (e.g.
    ``Needs_Scan_archive``) are still processed. The ``getattr`` fallback keeps
    test doubles that only stub ``supported_extensions`` usable.
    """
    quarantine = getattr(processor.text_extractor, "quarantine", None)
    file_paths = []
    iterator = root_dir.rglob("*") if recursive else root_dir.iterdir()
    for entry in iterator:
        if (
            entry.is_file()
            and (
                entry.suffix.lower() in processor.supported_extensions
                or (rename_images and entry.suffix.lower() in get_image_extensions())
            )
            and not (quarantine is not None and quarantine.contains(entry))
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
        activity_table = Table.grid(expand=True)
        activity_table.add_column(no_wrap=True, overflow="ellipsis")
        for act in self.activities:
            activity_table.add_row(act)

        layout_table = Table.grid(expand=True)

        stats_line = orch_msg.Progress.STATS_LINE.format(
            self.processed, self.successful, self.failed
        )
        layout_table.add_row(
            Panel(
                Group(stats_line, "\n", self.progress),
                title=orch_msg.Progress.TITLE_TOTAL,
                border_style="green",
            )
        )
        layout_table.add_row(
            Panel(
                activity_table,
                title=orch_msg.Progress.TITLE_ACTIVITY,
                border_style="blue",
                padding=(1, 2),
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
    main_task = progress.add_task(orch_msg.Status.PROCESSING, total=len(file_paths))
    status_display = StatusDisplay(progress)
    semaphore = asyncio.Semaphore(config.workers)

    # KeyboardInterrupt is deliberately NOT caught here: it propagates to the
    # CLI layer, which owns the cancel message and exit code. Swallowing it
    # would make `evaluate` continue with the next model after Ctrl+C.
    with Live(
        status_display,
        console=console,
        refresh_per_second=4,
        screen=True,
    ) as live:
        if config.benchmark:
            await _run_benchmark_mode(processor, file_paths, status_display, live)
            return []

        # Standard Mode (Rename, Organize, Evaluate)
        async def _process_task(f_path: Path) -> ProcessingResult:
            async with semaphore:
                status_display.update(orch_msg.Status.SCANNING.format(f_path.name))

                # BookProcessor reads every behavioral knob from self.config
                # (populated once at setup); only file_path varies per call.
                res = await processor.process_file(f_path)
                if config.mode == "evaluate" and res.success:
                    await log_evaluation_result(f_path, provider, model or "unknown", res.metadata)

                status_display.update(res)
                progress.advance(main_task)

                return res

        pending_tasks = [asyncio.ensure_future(_process_task(f)) for f in file_paths]
        return await asyncio.gather(*pending_tasks)


def _print_final_summary(results: list[ProcessingResult], is_dry_run: bool = False) -> None:
    """Print the final static summary card to the terminal."""
    if is_dry_run:
        table = Table(title="[bold yellow]renAI Dry-Run Preview Table[/bold yellow]")
        table.add_column("Original File", style="cyan")
        table.add_column("Proposed Target / Name", style="green")
        table.add_column("Category", style="magenta")
        table.add_column("Status", style="bold")

        for r in results:
            orig_name = r.original_name or r.name or "Unknown"
            dest_name = r.file_path.name if r.file_path else (r.name or "Skipped")
            cat = r.category or "-"
            status = "[green]Ready[/green]" if r.success else f"[red]{r.error}[/red]"
            table.add_row(orig_name, dest_name, cat, status)

        console.print()
        console.print(table)
        console.print()

    success_count = sum(1 for r in results if r.success)
    error_count = len(results) - success_count

    summary_table = Table.grid(padding=(0, 1))
    summary_table.add_column(justify="left", style="white")
    summary_table.add_column(justify="right", style="cyan")
    title_suffix = " (DRY-RUN)" if is_dry_run else ""
    summary_table.add_row(
        orch_msg.Summary.SUCCESS_BADGE, orch_msg.Summary.SUCCESS_COUNT.format(success_count)
    )
    if error_count > 0:
        summary_table.add_row(
            orch_msg.Summary.FAILED_BADGE, orch_msg.Summary.FAILED_COUNT.format(error_count)
        )

    console.print()
    console.print(
        Panel(
            summary_table,
            title=f"{orch_msg.Summary.TITLE}{title_suffix}",
            border_style="yellow" if is_dry_run else ("green" if error_count == 0 else "yellow"),
            padding=(1, 2),
            expand=False,
        )
    )


async def _close_llm_engines(processor: BookProcessor) -> None:
    """Release the HTTP clients of every engine the pipeline built.

    Best-effort: a failing close must not mask the pipeline's own outcome.
    Identity-deduplicated: when vision shares the text engine's provider and
    model, ``setup_engines`` reuses the same ``LLMEngine`` instance, so the
    set here collapses both references into a single close.
    """
    vision_engine = getattr(processor.visual_processor, "llm_engine", None)
    engines = {processor.llm_engine, vision_engine} - {None}
    for engine in engines:
        aclose = getattr(engine, "aclose", None)
        if aclose is None:
            continue
        try:
            await aclose()
        except Exception as e:  # teardown must not raise
            logging.debug("Failed to close LLM engine client: %s", e)


async def run_pipeline(
    root_dir: Path,
    provider: str,
    model: str | None = None,
    config: ProcessingConfig | None = None,
) -> list[ProcessingResult]:
    """Core processing pipeline orchestration.

    Headless-friendly: ``config`` may be a plain ``ProcessingConfig`` — there is
    no app_settings dependency here. Settings-derived values (including
    ``max_filename_length``) are populated exclusively by
    ``ProcessingConfig.from_app_settings``; callers wanting settings defaults
    build their config through that classmethod.
    """
    if config is None:
        config = ProcessingConfig()

    # 1. Setup
    _ocr_engine, _llm_engine, processor = await setup_engines(
        provider,
        model,
        config,
    )

    try:
        await processor.setup_move_folders(root_dir)

        # 2. Collection (KeyboardInterrupt propagates to the CLI, which owns exit UX)
        file_paths = collect_files(root_dir, processor, config.recursive, config.rename_images)

        console.print()

        # 3. Execution Loop
        results = await _execute_processing_loop(processor, file_paths, provider, model, config)

        await _flush_eval_log()

        # 4. Final Summary Card (Static, will stay in terminal history)
        if not config.benchmark and results:
            _print_final_summary(results, is_dry_run=config.dry_run)

        return results
    finally:
        await _close_llm_engines(processor)


async def _run_benchmark_mode(
    processor: BookProcessor,
    file_paths: list[Path],
    status_display: StatusDisplay,
    live: Live,
) -> None:
    """Run PDF extraction benchmark across multiple libraries."""
    # Derived from the extractor registry (SSoT) so a new backend is
    # benchmarked automatically instead of being silently skipped.
    extractors_to_test = EXTRACTOR_NAMES
    benchmark_data = []
    sample_count = app_settings.benchmark_sample_count
    if sample_count < len(file_paths):
        logging.info(
            "Benchmarking on the first %d of %d files (benchmark_sample_count)",
            sample_count,
            len(file_paths),
        )
        file_paths = file_paths[:sample_count]

    for ext_name in extractors_to_test:
        bench_msg = orch_msg.Benchmark.TESTING.format(ext_name)
        status_display.update(bench_msg)

        processor.swap_extractor(ext_name)

        start_t = time.perf_counter()
        success_count = 0

        for f_path in file_paths:
            extracted = await processor.benchmark_extract(f_path)
            if extracted:
                success_count += 1
            status_display.update(
                ProcessingResult(
                    success=extracted,
                    name=f_path.name,
                    original_name=f_path.name,
                )
            )

        duration = time.perf_counter() - start_t
        benchmark_data.append(
            {
                "extractor": ext_name,
                "time": f"{duration:.2f}s",
                "success": f"{success_count}/{len(file_paths)}",
            }
        )
        status_display.update(orch_msg.Benchmark.COMPLETED.format(ext_name, duration))

    # Show Benchmark Result Table outside live
    live.stop()
    bench_table = Table(title=orch_msg.Benchmark.TABLE_TITLE)
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
    target_models = _resolve_target_models(provider_filter)

    if not target_models:
        console.print(orch_msg.Eval.NO_MODELS)
        return

    console.print(orch_msg.Eval.HEADER)
    console.print(orch_msg.Eval.TARGET.format(root_dir))
    console.print(orch_msg.Eval.MODEL_COUNT.format(len(target_models)))

    all_eval_results: dict[str, list[ProcessingResult]] = {}

    for provider, model_name in target_models:
        console.print(orch_msg.Eval.PER_PROVIDER.format(provider, model_name))

        results = await run_pipeline(
            root_dir=root_dir,
            provider=provider,
            model=model_name,
            config=ProcessingConfig.from_app_settings(
                mode="evaluate",
                workers=workers,
                recursive=False,
                fallback_mode=FallbackMode.OCR,
                # Evaluate must never quarantine files, regardless of settings:
                # from_app_settings derives skip_ocr_checks from skip_needs_ocr.
                skip_ocr_checks=False,
            ),
        )
        all_eval_results[f"{provider}/{model_name}"] = results
        console.print(orch_msg.Eval.RESULTS_LOGGED.format(EVAL_LOG_FILE))

    _print_evaluation_summary(all_eval_results)


def _resolve_target_models(provider_filter: str | None) -> list[tuple[str, str]]:
    """Resolve configured models to a list of (provider, model_name) pairs."""
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
    return target_models


def _print_evaluation_summary(all_eval_results: dict[str, list[ProcessingResult]]) -> None:
    """Group and print evaluation comparison tables for each file."""
    from collections import defaultdict

    file_map = defaultdict(list)
    for model_key, results in all_eval_results.items():
        for res in results:
            if res.file_path:
                file_map[res.file_path.name].append((model_key, res))

    if not file_map:
        return

    console.print(orch_msg.Eval.COMPARISON_TITLE)
    for file_name, runs in file_map.items():
        table = Table(
            title=f"File: [cyan]{file_name}[/cyan]",
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
        )
        table.add_column("Model/Provider", style="yellow")
        table.add_column("Status", justify="center")
        table.add_column("Extracted Title")
        table.add_column("Author")
        table.add_column("Category")
        table.add_column("Year", justify="center")

        for model_key, res in runs:
            status = "[green]SUCCESS[/green]" if res.success else "[red]FAILED[/red]"
            title = ""
            author = ""
            category = ""
            year = ""

            if res.success and res.metadata:
                title = res.metadata.get("title") or ""
                author = res.metadata.get("author") or ""
                category = res.metadata.get("category") or ""
                year = str(res.metadata.get("year") or "")
            elif res.error:
                title = f"[dim red]Error: {res.error[:30]}[/dim red]"

            table.add_row(
                model_key,
                status,
                title[:40],
                author[:30],
                category[:20],
                year,
            )
        console.print(table)
        console.print()
