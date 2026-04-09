"""
Orchestrator for renAI processing pipelines.
Extracts heavy workflow logic from CLI parameters.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from renai.engines.llm import LLMEngine
from renai.engines.ocr import OCREngine
from renai.models import ProcessingResult
from renai.processors.book import BookProcessor
from renai.processors.text_extractor import FallbackMode
from renai.settings import (
    IMAGE_EXTENSIONS,
    PROVIDERS,
    get_api_key,
    get_effective_model,
)

console = Console()


def setup_engines(
    provider: str,
    model: str | None,
    ocr_enhanced: bool,
    debug_ocr: bool,
    use_mutool: bool,
    mutool_path: str,
    extractor: str,
    text_length: int,
) -> Tuple[OCREngine, LLMEngine, BookProcessor]:
    """Initialize OCR, LLM, and book processor engines."""
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

    # Initialize LLM engine
    llm_engine = LLMEngine(
        provider=provider,
        api_key=api_key,
        base_url=provider_config.base_url,
        model_name=model_name,
    )

    # Initialize processor
    processor = BookProcessor(
        ocr_engine,
        llm_engine,
        extractor_name=extractor,
        provider=provider,
        model=model_name,
        target_length=text_length,
    )

    return ocr_engine, llm_engine, processor


def collect_files(
    root_dir: Path, processor: BookProcessor, recursive: bool
) -> list[Path]:
    """Collect supported files from directory."""
    file_paths = []
    iterator = root_dir.rglob("*") if recursive else root_dir.iterdir()
    for entry in iterator:
        if (
            entry.is_file()
            and not entry.is_symlink()
            and entry.suffix.lower() in processor.supported_extensions
        ):
            file_paths.append(entry)
    return file_paths


def collect_image_files(
    root_dir: Path, processor: BookProcessor, recursive: bool
) -> list[Path]:
    """Collect image files for --rename-images mode."""
    file_paths = []
    iterator = root_dir.rglob("*") if recursive else root_dir.iterdir()
    for entry in iterator:
        if (
            entry.is_file()
            and not entry.is_symlink()
            and entry.suffix.lower() in IMAGE_EXTENSIONS
        ):
            file_paths.append(entry)
    return file_paths


def run_benchmark(processor: BookProcessor, file_paths: list[Path]) -> None:
    """Run extraction benchmark on first 3 files."""
    console.print("[cyan]Starting Benchmark Mode...[/cyan]")
    for p in file_paths[:3]:
        processor.benchmark_extractors(p)
    console.print("[green]Benchmark complete![/green]")


def run_model_eval(
    processor: BookProcessor,
    file_paths: list[Path],
    provider: str,
) -> None:
    """Run model evaluation on first 2 files."""
    from renai.settings import EVAL_MODELS

    eval_models = EVAL_MODELS.get(provider, [])
    if not eval_models:
        console.print(
            f"[red]No evaluation models defined for provider: {provider}[/red]"
        )
        raise typer.Exit(code=1)

    console.print("[cyan]Starting Model Evaluation Mode...[/cyan]")
    for p in file_paths[:2]:
        processor.evaluate_models(p, provider, eval_models)
    console.print("[green]Evaluation complete![/green]")


def process_files(
    processor: BookProcessor,
    file_paths: list[Path],
    root_dir: Path,
    mode: str,
    workers: int,
    fallback_mode: FallbackMode,
    rename_images: bool,
    update_metadata: bool,
    exif: bool,
    move_scan_failures: bool,
) -> None:
    """Process files based on mode."""
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Processing {len(file_paths)} files...", total=len(file_paths)
        )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for p in file_paths:
                future = executor.submit(
                    processor.process_file,
                    p,
                    mode=mode,
                    fallback_mode=fallback_mode,
                    rename_images=rename_images,
                    update_metadata=update_metadata,
                    add_exif_date=exif,
                )
                futures[future] = p

            try:
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        console.print(
                            f"[red]Error processing {futures[future].name}: {e}[/red]"
                        )
                        results.append(ProcessingResult(success=False, error=str(e)))
                    progress.advance(task)
            except KeyboardInterrupt:
                console.print("\n[red]Stopping... Cancelling pending tasks...[/red]")
                executor.shutdown(wait=False, cancel_futures=True)
                # Re-raise to let the CLI handle common exit
                raise

    # Log results
    success_count = sum(1 for r in results if r.success)
    console.print(
        f"[green]Processed {success_count}/{len(file_paths)} files successfully[/green]"
    )

    # Organize files if mode is organize
    if mode == "organize":
        processor.organize_files(results, root_dir)

    # Handle Scan folders
    if fallback_mode == FallbackMode.MOVE_TO_FOLDER:
        needs_scan_files = processor.get_needs_scan_files()
        if needs_scan_files:
            console.print(
                f"[yellow]Files moved to Needs_Scan folder: {len(needs_scan_files)}[/yellow]"
            )

    if move_scan_failures:
        if processor.scan_failed_files:
            console.print(
                f"[red]Total files failed processing: {len(processor.scan_failed_files)}[/red]"
            )
            console.print("[red]Files moved to Scan_Failures directory.[/red]")


def _configure_fallback_folders(
    processor: BookProcessor,
    root_dir: Path,
    fallback_mode: FallbackMode,
    move_scan_failures: bool,
) -> None:
    """Configure special folders for scan fallbacks and failures."""
    if fallback_mode == FallbackMode.MOVE_TO_FOLDER:
        processor.set_needs_scan_folder(root_dir, skip=True)
    if move_scan_failures:
        processor.set_scan_failures_folder(root_dir, enabled=True)


def _collect_all_processing_files(
    root_dir: Path, processor: BookProcessor, recursive: bool, rename_images: bool
) -> List[Path]:
    """Collect all files to be processed, including images if requested."""
    file_paths = collect_files(root_dir, processor, recursive)
    if rename_images:
        image_files = collect_image_files(root_dir, processor, recursive)
        seen = set(file_paths)
        for img in image_files:
            if img not in seen:
                file_paths.append(img)
                seen.add(img)
    return file_paths


def run_pipeline(
    root_dir: Path,
    mode: str,
    provider: str,
    model: str | None,
    ocr_enhanced: bool,
    debug_ocr: bool,
    use_mutool: bool,
    mutool_path: str,
    extractor: str,
    workers: int,
    text_length: int,
    recursive: bool,
    update_metadata: bool,
    fallback_mode: FallbackMode,
    move_scan_failures: bool,
    rename_images: bool,
    exif: bool,
    benchmark: bool = False,
    eval_model: bool = False,
) -> None:
    """
    Main entry point for processing.
    Consolidates engine setup, file collection, and execution.
    """
    # 1. Setup engines
    _, _, processor = setup_engines(
        provider=provider,
        model=model,
        ocr_enhanced=ocr_enhanced,
        debug_ocr=debug_ocr,
        use_mutool=use_mutool,
        mutool_path=mutool_path,
        extractor=extractor,
        text_length=text_length,
    )

    # 2. Configure fallback folders
    _configure_fallback_folders(
        processor, root_dir, fallback_mode, move_scan_failures
    )

    # 3. Collect files
    file_paths = _collect_all_processing_files(
        root_dir, processor, recursive, rename_images
    )

    if not file_paths:
        console.print("[yellow]No supported files found.[/yellow]")
        return

    console.print(f"[cyan]Found {len(file_paths)} files to process[/cyan]\n")

    # 4. Handle Special Modes
    if benchmark:
        run_benchmark(processor, file_paths)
        return

    if eval_model:
        run_model_eval(processor, file_paths, provider)
        return

    # 5. Execute core processing
    process_files(
        processor=processor,
        file_paths=file_paths,
        root_dir=root_dir,
        mode=mode,
        workers=workers,
        fallback_mode=fallback_mode,
        rename_images=rename_images,
        update_metadata=update_metadata,
        exif=exif,
        move_scan_failures=move_scan_failures,
    )
