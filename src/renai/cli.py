"""
renAI CLI - Modern Typer-based command line interface.

This is the main CLI entry point for renAI. All functionality from the
original main.py has been migrated here.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from renai.engines.llm import LLMEngine
from renai.engines.ocr import OCREngine
from renai.logging_config import configure_logging
from renai.processors.book import BookProcessor
from renai.settings import (
    IMAGE_EXTENSIONS,
    PROVIDERS,
    display_warnings,
    get_api_key,
    get_effective_model,
    initialize_directories,
)
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


# =============================================================================
# Helper Functions (migrated from main.py)
# =============================================================================

def setup_environment() -> None:
    """Setup cross-platform encoding and terminal settings."""
    if sys.platform == "win32":
        import codecs

        if sys.stdout.encoding != "utf-8":
            sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
            sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

    try:
        from colorama import init
        init()
    except ImportError:
        pass


def check_requirements() -> bool:
    """Check for essential libraries and display installation tips."""
    missing = []
    libraries = {
        "pymupdf": "pip install pymupdf",
        "pytesseract": "pip install pytesseract",
        "PIL": "pip install Pillow",
        "requests": "pip install requests",
        "markitdown": "pip install markitdown",
        "openai": "pip install openai",
        "numpy": "pip install numpy",
        "cv2": "pip install opencv-python",
    }
    for lib, cmd in libraries.items():
        try:
            __import__(lib)
        except ImportError:
            missing.append((lib, cmd))

    if missing:
        console.print("[red]Missing Python libraries:[/red]")
        for lib, cmd in missing:
            console.print(f"  - {lib}: [cyan]{cmd}[/cyan]")
        return False
    return True


def handle_clear_cache() -> None:
    """Clear extracted text cache and metadata."""
    from renai.settings import CACHE_DIR

    ensure_secure_dir(CACHE_DIR)
    cache_files = list(CACHE_DIR.iterdir())
    if cache_files:
        for cache_file in cache_files:
            try:
                if cache_file.is_file():
                    cache_file.unlink()
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to delete {cache_file.name}: {e}[/yellow]")
        console.print(f"[green]Cache cleared: {len(cache_files)} files removed[/green]")
    else:
        console.print("[green]Cache directory is empty. Nothing to clear.[/green]")


def setup_engines(
    provider: str,
    model: str | None,
    ocr_enhanced: bool,
    debug_ocr: bool,
    use_mutool: bool,
    mutool_path: str,
    extractor: str,
    text_length: int,
    vision: bool,
) -> Tuple[OCREngine, LLMEngine, BookProcessor]:
    """Initialize OCR, LLM, and book processor engines."""
    # Initialize OCR engine
    ocr_engine = OCREngine(
        enhanced=ocr_enhanced,
        debug=debug_ocr,
        use_mutool=use_mutool,
        mutool_path=mutool_path,
        silent=vision,
    )

    # Get provider configuration
    if provider not in PROVIDERS:
        console.print(f"[red]Invalid provider: {provider}[/red]")
        raise typer.Exit(code=1)

    provider_config = PROVIDERS[provider]
    api_key = provider_config.api_key or get_api_key(provider)
    if not api_key:
        console.print(f"[red]API Key for {provider} not found. Set {provider.upper()}_API_KEY environment variable.[/red]")
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


def collect_files(root_dir: Path, processor: BookProcessor, recursive: bool) -> list[Path]:
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


def collect_image_files(root_dir: Path, processor: BookProcessor, recursive: bool) -> list[Path]:
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
        console.print(f"[red]No evaluation models defined for provider: {provider}[/red]")
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
    vision: bool,
    rename_images: bool,
    update_metadata: bool,
    exif: bool,
    skip_needs_ocr: bool,
    move_ocr_failures: bool,
) -> None:
    """Process files based on mode."""
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Processing {len(file_paths)} files...",
            total=len(file_paths)
        )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for p in file_paths:
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    if rename_images:
                        future = executor.submit(
                            processor.process_single_image,
                            p,
                            mode,
                            update_metadata,
                            exif,
                        )
                    else:
                        console.print(f"[dim]Skipping image: {p.name}[/dim]")
                        progress.advance(task)
                        continue
                else:
                    if vision:
                        future = executor.submit(
                            processor.process_single_book_vision,
                            p,
                            mode,
                            update_metadata,
                        )
                    else:
                        future = executor.submit(
                            processor.process_single_book,
                            p,
                            mode,
                            update_metadata,
                        )
                futures[future] = p

            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    console.print(f"[red]Error processing {futures[future].name}: {e}[/red]")
                    results.append((False, None, None, None, None))
                progress.advance(task)

    # Log results
    success_count = sum(1 for r in results if r[0])
    console.print(f"[green]Processed {success_count}/{len(file_paths)} files successfully[/green]")

    # Organize files if mode is organize
    if mode == "organize":
        processor.organize_files(results, root_dir)

    # Handle OCR folders
    if skip_needs_ocr:
        needs_ocr_files = processor.get_needs_ocr_files()
        if needs_ocr_files:
            console.print(f"[yellow]Files moved to Needs_OCR folder: {len(needs_ocr_files)}[/yellow]")

    if move_ocr_failures:
        console.print("[green]OCR failure handling enabled[/green]")


# =============================================================================
# CLI Commands
# =============================================================================

@app.command()
def process(
    directory: str = typer.Argument(
        None,
        help="Directory containing book files (required unless using --clear-cache)"
    ),
    # Mode options
    mode: str = typer.Option(
        "rename",
        "--mode",
        help="Action to perform (rename/organize)",
    ),
    # OCR options
    ocr_enhanced: bool = typer.Option(
        False,
        "--ocr-enhanced",
        help="Enable high-quality OCR with enhancements",
    ),
    debug_ocr: bool = typer.Option(
        False,
        "--debug-ocr",
        help="Save OCR debug images",
    ),
    use_mutool: bool = typer.Option(
        False,
        "--use-mutool",
        help="Use mutool for PDF-to-image conversion (recommended for scanned PDFs)",
    ),
    mutool_path: str = typer.Option(
        "mutool",
        "--mutool-path",
        help="Path to mutool executable",
    ),
    # Provider options
    provider: str = typer.Option(
        "deepinfra",
        "--provider",
        help="LLM Provider (deepinfra/openrouter/openai/custom)",
    ),
    model: str = typer.Option(
        None,
        "--model",
        help="LLM Model name (overrides default)",
    ),
    # Processing options
    extractor: str = typer.Option(
        "pymupdf",
        "--extractor",
        help="PDF Text Extractor library (pymupdf/pdfplumber/pypdf/pypdfium2)",
    ),
    workers: int = typer.Option(
        4,
        "--workers",
        help="Number of concurrent workers",
    ),
    text_length: int = typer.Option(
        10000,
        "--text-length",
        help="Target text length to send to LLM",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        help="Process subdirectories recursively",
    ),
    # Output options
    update_metadata: bool = typer.Option(
        False,
        "--update-metadata",
        help="Force update metadata from LLM (ignore cache)",
    ),
    # OCR fallback options
    skip_needs_ocr: bool = typer.Option(
        False,
        "--skip-needs-ocr",
        help="Skip OCR processing, move files to Needs_OCR folder",
    ),
    move_ocr_failures: bool = typer.Option(
        False,
        "--move-ocr-failures",
        help="Move files that cannot be processed to OCR_Failures folder",
    ),
    # Vision options
    vision: bool = typer.Option(
        False,
        "--vision",
        help="Use vision-capable LLM to process PDFs",
    ),
    rename_images: bool = typer.Option(
        False,
        "--rename-images",
        help="Rename image files using vision-capable LLM",
    ),
    exif: bool = typer.Option(
        False,
        "--exif",
        help="Add EXIF date to image filenames (use with --rename-images)",
    ),
    # Special modes
    benchmark: bool = typer.Option(
        False,
        "--benchmark",
        help="Run extraction benchmark on first 3 files",
    ),
    eval_model: bool = typer.Option(
        False,
        "--eval-model",
        help="Evaluate multiple models for the selected provider",
    ),
    # Cache option
    clear_cache: bool = typer.Option(
        False,
        "--clear-cache",
        help="Clear cache and exit",
    ),
) -> None:
    """Process files in a directory."""
    # Handle clear-cache first
    if clear_cache:
        setup_environment()
        initialize_directories()
        handle_clear_cache()
        raise typer.Exit()

    # Validate directory
    if not directory:
        console.print("[red]Error: Directory is required (use --clear-cache to clear cache only)[/red]")
        raise typer.Exit(code=1)

    root_dir = Path(directory).resolve()
    if not root_dir.is_dir():
        console.print(f"[red]Error: Directory not found: {root_dir}[/red]")
        raise typer.Exit(code=1)

    # Setup
    setup_environment()
    initialize_directories()

    # Check requirements
    if not check_requirements():
        raise typer.Exit(code=1)

    # Display header
    console.print(f"\n[bold cyan]renAI[/bold cyan] - AI-Powered Book Renamer")
    console.print(f"[blue]Directory:[/blue] {root_dir}")
    console.print(f"[green]Mode:[/green] {mode}")
    console.print(f"[yellow]Provider:[/yellow] {provider}")
    if model:
        console.print(f"[magenta]Model:[/magenta] {model}")
    console.print()

    # Vision mode warnings
    if vision or rename_images:
        console.print("[yellow]Note: --vision and --rename-images require a vision-capable model. Please ensure your model supports vision.[/yellow]\n")

    # Setup engines
    try:
        ocr_engine, llm_engine, processor = setup_engines(
            provider=provider,
            model=model,
            ocr_enhanced=ocr_enhanced,
            debug_ocr=debug_ocr,
            use_mutool=use_mutool,
            mutool_path=mutool_path,
            extractor=extractor,
            text_length=text_length,
            vision=vision,
        )
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error setting up engines: {e}[/red]")
        raise typer.Exit(code=1)

    # Configure OCR folders
    if skip_needs_ocr:
        processor.set_needs_ocr_folder(root_dir, skip=True)
    if move_ocr_failures:
        processor.set_ocr_failures_folder(root_dir, enabled=True)

    # Display warnings
    display_warnings(mode, recursive)

    # Collect files
    if vision and rename_images:
        pdf_files = collect_files(root_dir, processor, recursive)
        image_files = collect_image_files(root_dir, processor, recursive)
        file_paths = pdf_files + image_files
        console.print(f"[cyan]Found {len(pdf_files)} PDFs and {len(image_files)} images[/cyan]\n")
    elif rename_images:
        pdf_files = collect_files(root_dir, processor, recursive)
        image_files = collect_image_files(root_dir, processor, recursive)
        file_paths = pdf_files + image_files
    else:
        file_paths = collect_files(root_dir, processor, recursive)

    if not file_paths:
        console.print("[yellow]No supported files found.[/yellow]")
        raise typer.Exit()

    console.print(f"[cyan]Found {len(file_paths)} files to process[/cyan]\n")

    # Run special modes
    if benchmark:
        run_benchmark(processor, file_paths)
        raise typer.Exit()

    if eval_model:
        run_model_eval(processor, file_paths, provider)
        raise typer.Exit()

    # Process files
    try:
        process_files(
            processor=processor,
            file_paths=file_paths,
            root_dir=root_dir,
            mode=mode,
            workers=workers,
            vision=vision,
            rename_images=rename_images,
            update_metadata=update_metadata,
            exif=exif,
            skip_needs_ocr=skip_needs_ocr,
            move_ocr_failures=move_ocr_failures,
        )
    except Exception as e:
        console.print(f"[red]Error processing files: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("providers")
def list_providers() -> None:
    """List available LLM providers."""
    table = Table(title="Available LLM Providers")
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Default Model", style="green")

    for name, config in PROVIDERS.items():
        default = config.default_model if hasattr(config, 'default_model') else 'N/A'
        base_url = config.base_url if hasattr(config, 'base_url') else 'N/A'
        table.add_row(name, default, base_url)

    console.print(table)


@app.command("cache")
def clear_cache_cmd(
    confirm: bool = typer.Option(
        False,
        "--yes", "-y",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Clear the cache directory."""
    if not confirm:
        console.print("[yellow]This will delete all cached text and metadata.[/yellow]")
        response = input("Continue? [y/N]: ")
        if response.lower() not in ("y", "yes"):
            console.print("[cyan]Cancelled.[/cyan]")
            raise typer.Exit()

    setup_environment()
    initialize_directories()
    handle_clear_cache()


@app.command("version")
def show_version() -> None:
    """Show renAI version."""
    try:
        from renai import __version__
        console.print(f"[bold cyan]renAI[/bold cyan] version {__version__}")
    except ImportError:
        console.print("[bold cyan]renAI[/bold cyan] version unknown (not installed)")


# Register process as default command
app.command()(process)


# Backward compatibility: main function for tests
def main():
    """Main entry point for backward compatibility."""
    app()


if __name__ == "__main__":
    app()
