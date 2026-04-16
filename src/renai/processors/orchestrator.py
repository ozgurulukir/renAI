"""
Orchestrator for renAI processing pipelines.
Extracts heavy workflow logic from CLI parameters.
"""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

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
        console.print(f"[red]Invalid provider: {provider}[/red]")
        raise typer.Exit(code=1)

    provider_config = PROVIDERS[provider]
    api_key = provider_config.api_key or get_api_key(provider)
    if not api_key:
        console.print(
            f"[red]API Key for {provider} not found. Set {provider.upper()}_API_KEY environment variable.[/red]"
        )
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
    ocr_engine, llm_engine, processor = await setup_engines(
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
    file_paths = collect_files(root_dir, processor, recursive)
    if not file_paths:
        console.print("[yellow]No supported files found to process.[/yellow]")
        return

    if benchmark:
        file_paths = file_paths[:3]
        console.print(
            f"[yellow]Benchmark mode: only processing first {len(file_paths)} files.[/yellow]"
        )

    console.print(
        f"🚀 [bold blue]renAI[/bold blue] - Found [green]{len(file_paths)}[/green] files to process "
        f"using [cyan]{workers}[/cyan] parallel tasks (RPM Limit: {app_settings.llm_rpm_limit}).\n"
    )

    # 3. Execution
    semaphore = asyncio.Semaphore(workers)

    async def _process_with_semaphore(f_path):
        async with semaphore:
            return await processor.process_file(
                f_path,
                mode=mode,
                fallback_mode=fallback_mode,
                rename_images=rename_images,
                update_metadata=update_metadata,
                add_exif_date=add_exif_date,
            )

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("[cyan]Processing...", total=len(file_paths))

        pending_tasks = [asyncio.ensure_future(_process_with_semaphore(f)) for f in file_paths]

        for coro in asyncio.as_completed(pending_tasks):
            result = await coro
            results.append(result)
            progress.update(
                task_id,
                advance=1,
                description=f"Processing: [bold]{len(results)}/{len(file_paths)}[/bold]",
            )

    # 4. Summary
    success_count = sum(1 for r in results if r.success)
    error_count = len(results) - success_count

    console.print("\n[bold green]Processing complete![/bold green]")
    console.print(f"✅ Success: [green]{success_count}[/green]")
    if error_count > 0:
        console.print(f"❌ Failed: [red]{error_count}[/red]")


async def run_evaluation_pipeline(
    root_dir: Path,
    provider_filter: str | None = None,
    workers: int = 4,
) -> None:
    """Run evaluation for multiple models asynchronously."""
    # Load evaluation models from eval_models.toml
    if provider_filter:
        eval_models = load_eval_models(provider_filter)
    else:
        eval_models = load_all_eval_models()

    if not eval_models:
        console.print("[red]No evaluation models found in eval_models.toml.[/red]")
        return

    console.print("🔍 [bold blue]Evaluation Mode[/bold blue]")
    console.print(f"Target: [cyan]{root_dir}[/cyan]")
    console.print(f"Models: [green]{', '.join(eval_models.keys())}[/green]\n")

    for model_key, (provider, model_name) in eval_models.items():
        console.print(
            f"--- [bold yellow]Evaluating: {model_key} ({provider}/{model_name})[/bold yellow] ---"
        )

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
