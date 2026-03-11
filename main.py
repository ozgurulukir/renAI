import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import IMAGE_EXTENSIONS, PROVIDERS, display_warnings, initialize_directories
from file_processor import BookProcessor
from llm_engine import LLMEngine
from ocr_engine import OCREngine
from utils import ensure_secure_dir


def setup_environment() -> None:
    """Setup cross-platform encoding and terminal settings."""
    # Setup encoding properly for Windows
    if sys.platform == "win32":
        import codecs

        if sys.stdout.encoding != "utf-8":
            sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
            sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

    # Enable colorama for cross-platform color support
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
        logging.error("\n" + "=" * 50)
        logging.error("MISSING PYTHON LIBRARIES")
        for lib, cmd in missing:
            logging.error(f"- {lib} is missing. Run: {cmd}")
        logging.error("=" * 50 + "\n")
        return False
    return True


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="renAI: AI-Powered Book Renamer and Organizer"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=str,
        help="Directory containing book files (required unless --clear-cache is used)",
    )
    parser.add_argument(
        "--mode",
        choices=["rename", "organize"],
        default="rename",
        help="Action to perform",
    )
    parser.add_argument(
        "--ocr-enhanced",
        action="store_true",
        help="Enable high-quality OCR with enhancements",
    )
    parser.add_argument(
        "--update-metadata",
        action="store_true",
        help="Force update metadata from LLM (ignore cache)",
    )
    parser.add_argument(
        "--debug-ocr", action="store_true", help="Save OCR debug images"
    )
    parser.add_argument(
        "--use-mutool",
        action="store_true",
        help="Use mutool for high-quality PDF-to-image conversion during OCR fallback (recommended for scanned PDFs)",
    )
    parser.add_argument(
        "--mutool-path", type=str, default="mutool", help="Path to mutool executable"
    )
    parser.add_argument(
        "--provider",
        choices=["deepinfra", "openrouter", "openai", "custom"],
        default="deepinfra",
        help="LLM Provider (custom for local inference like Ollama/LM Studio)",
    )
    parser.add_argument("--model", type=str, help="LLM Model name (overrides default)")
    parser.add_argument(
        "--extractor",
        choices=["pymupdf", "pdfplumber", "pypdf", "pypdfium2"],
        default="pymupdf",
        help="PDF Text Extractor library",
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run extraction benchmark on files"
    )
    parser.add_argument(
        "--eval-model",
        action="store_true",
        help="Evaluate multiple models for the selected provider",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of concurrent workers"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear extracted text cache and metadata, then exit",
    )
    parser.add_argument(
        "--skip-needs-ocr",
        action="store_true",
        help="Skip OCR processing for files that require OCR fallback (text extraction failed) and move them directly to Needs_OCR folder",
    )
    parser.add_argument(
        "--move-ocr-failures",
        action="store_true",
        help="Move files that cannot be processed (no text extracted, even with OCR) to OCR_Failures folder",
    )
    parser.add_argument(
        "--recursive",
        action="store_const",
        const=True,
        default=True,
        dest="recursive",
        help="Recursively process subdirectories (default: True)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_const",
        const=False,
        dest="recursive",
        help="Process only the root directory (no recursion)",
    )
    parser.add_argument(
        "--text-length",
        type=int,
        default=10000,
        help="Target text length to send to LLM (default: 10000). Lower values reduce load on local models.",
    )
    # Vision mode arguments
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Use vision-capable LLM to process PDFs instead of OCR fallback. "
        "Ignores --ocr-enhanced, --use-mutool, --skip-needs-ocr, --move-ocr-failures.",
    )
    parser.add_argument(
        "--rename-images",
        action="store_true",
        help="Rename image files using vision-capable LLM. "
        "Requires vision-capable model. Does not auto-enable --vision.",
    )
    parser.add_argument(
        "--exif",
        action="store_true",
        help="Add EXIF date to image filenames when available. Use with --rename-images.",
    )

    return parser.parse_args()


def handle_clear_cache(args: argparse.Namespace) -> None:
    # Clear cache mode - works without directory (clears all cache)
    if args.clear_cache:
        from config import CACHE_DIR

        ensure_secure_dir(CACHE_DIR)
        cache_files = list(CACHE_DIR.iterdir())
        if cache_files:
            for cache_file in cache_files:
                try:
                    if cache_file.is_file():
                        cache_file.unlink()
                except Exception as e:
                    logging.warning(f"Failed to delete {cache_file.name}: {e}")
            logging.info(f"Cache cleared: {len(cache_files)} files removed")
        else:
            logging.info("Cache directory is empty. Nothing to clear.")
        logging.info("Cache cleared successfully.")
        sys.exit(0)


def validate_vision_mode(args: argparse.Namespace) -> bool:
    """Validate that vision mode has proper provider/model configuration.

    Returns True if valid, False otherwise.
    """
    # Model validation is now user responsibility
    # The warning is displayed earlier in the main() function
    # If user selects a non-vision model, API will return an error or unexpected results

    return True


def setup_engines(
    args: argparse.Namespace,
) -> tuple[OCREngine, LLMEngine, BookProcessor]:
    # Initialize engines
    # In vision mode, OCR is only used as fallback - suppress initialization logging
    ocr_engine = OCREngine(
        enhanced=args.ocr_enhanced,
        debug=args.debug_ocr,
        use_mutool=args.use_mutool,
        mutool_path=args.mutool_path,
        silent=args.vision,
    )
    # LLM initialization with provider selection
    provider_config = PROVIDERS.get(args.provider)
    if provider_config is None:
        logging.error(f"Invalid provider: {args.provider}")
        sys.exit(1)
    api_key = provider_config.get("api_key")
    if not api_key:
        logging.error(
            f"API Key for {args.provider} not found in environment variables."
        )
        sys.exit(1)

    # Determine the actual model to use
    model_name = args.model or provider_config.get("default_model", "")

    llm_engine = LLMEngine(
        provider=args.provider,
        api_key=api_key,
        base_url=provider_config["base_url"],  # type: ignore
        model_name=model_name,
    )

    # Initialize processor with provider-model tracking for cache invalidation
    processor = BookProcessor(
        ocr_engine,
        llm_engine,
        extractor_name=args.extractor,
        provider=args.provider,
        model=model_name,
        target_length=args.text_length,
    )

    return ocr_engine, llm_engine, processor


def collect_files(
    root_dir: Path, processor: BookProcessor, recursive: bool
) -> list[Path]:
    # Collect files
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
    from config import IMAGE_EXTENSIONS

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


def is_image_file_by_extension(file_path: Path) -> bool:
    """Check if file is an image based on extension."""
    from config import IMAGE_EXTENSIONS
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def process_special_modes(
    args: argparse.Namespace, processor: BookProcessor, file_paths: list[Path]
) -> None:
    # Check for conflicting special modes (these are mutually exclusive)
    if args.benchmark and args.eval_model:
        logging.error(
            "Cannot use --benchmark and --eval-model at the same time. "
            "Please use only one of these options."
        )
        sys.exit(1)

    # Warnings for special modes combined with other flags
    if args.benchmark and args.vision:
        logging.warning(
            "Note: --benchmark ignores --vision. "
            "Benchmark tests text extraction extractors, not vision models."
        )

    if args.benchmark and args.rename_images:
        logging.warning(
            "Note: --benchmark ignores --rename-images. "
            "Benchmark tests text extraction extractors only."
        )

    if args.eval_model and args.vision:
        logging.warning(
            "Note: --eval-model will use vision mode for PDF text extraction. "
            "This tests how vision-capable models perform on metadata extraction."
        )

    if args.eval_model and args.rename_images:
        logging.warning(
            "Note: --eval-model ignores --rename-images. "
            "Evaluation tests text extraction, not file renaming."
        )

    # Benchmark mode logic
    if args.benchmark:
        logging.info("Starting Benchmark Mode...")
        for p in file_paths[:3]:  # Benchmark first 3 files
            processor.benchmark_extractors(p)
        sys.exit(0)

    # Model evaluation mode
    if args.eval_model:
        logging.info("Starting Model Evaluation Mode...")
        from config import EVAL_MODELS

        eval_models = EVAL_MODELS.get(args.provider, [])
        if not eval_models:
            logging.error(f"No evaluation models defined for provider: {args.provider}")
            sys.exit(1)

        # Select first 2 files for evaluation to represent the directory
        for p in file_paths[:2]:
            processor.evaluate_models(p, args.provider, eval_models)
        sys.exit(0)


def process_files(
    args: argparse.Namespace,
    processor: BookProcessor,
    file_paths: list[Path],
    root_dir: Path,
) -> None:
    """Process files based on mode (regular, vision, or image rename)."""
    results = []

    # Mixed mode: --rename-images handles both images and non-images
    # When --rename-images is used, images go to process_single_image,
    # non-images go to normal processing (with OCR or vision fallback based on --vision)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for p in file_paths:
            # Determine which processing method to use based on file type
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                # Handle images
                if args.rename_images:
                    # --rename-images (with or without --vision): use vision model
                    future = executor.submit(
                        processor.process_single_image,
                        p,
                        args.mode,
                        args.update_metadata,
                        args.exif,
                    )
                else:
                    # --vision alone: skip images
                    logging.info(f"Skipping image in --vision mode: {p.name}")
                    continue
            else:
                # Handle non-image files (PDF, EPUB, TXT, etc.)
                if args.vision:
                    # --vision (with or without --rename-images): use vision fallback
                    future = executor.submit(
                        processor.process_single_book_vision,
                        p,
                        args.mode,
                        args.update_metadata,
                    )
                else:
                    # Regular mode or --rename-images alone: use normal processing with OCR fallback
                    future = executor.submit(
                        processor.process_single_book,
                        p,
                        args.mode,
                        args.update_metadata,
                    )
            futures[future] = p

        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                logging.error(f"Worker failed for {futures[future].name}: {e}")
                results.append((False, None, None, None, None))

    # Log processing mode info
    if args.vision and args.rename_images:
        # Mixed mode: PDFs with vision, images with vision
        pdf_count = sum(1 for p in file_paths if p.suffix.lower() == ".pdf")
        img_count = sum(1 for p in file_paths if p.suffix.lower() in IMAGE_EXTENSIONS)
        logging.info(
            f"Processed {pdf_count} PDFs and {img_count} images "
            f"(PDFs: vision model, Images: vision model)"
        )
    elif args.vision:
        # Vision only: PDFs with vision, images skipped
        pdf_count = sum(1 for p in file_paths if p.suffix.lower() == ".pdf")
        img_count = sum(1 for p in file_paths if p.suffix.lower() in IMAGE_EXTENSIONS)
        logging.info(
            f"Processed {pdf_count} PDFs with vision model, skipped {img_count} images"
        )
    elif args.rename_images:
        # --rename-images only: PDFs with OCR, images with vision
        pdf_count = sum(1 for p in file_paths if p.suffix.lower() == ".pdf")
        img_count = sum(1 for p in file_paths if p.suffix.lower() in IMAGE_EXTENSIONS)
        logging.info(
            f"Processed {pdf_count} PDFs and {img_count} images "
            f"(PDFs: text extraction + OCR, Images: vision model)"
        )
    else:
        logging.info("Using regular mode - extracting text with OCR/LLM")

    # Organize files if mode is organize
    if args.mode == "organize":
        processor.organize_files(results, root_dir)

    # Report files moved to Needs_OCR folder if enabled
    if args.skip_needs_ocr:
        needs_ocr_files = processor.get_needs_ocr_files()
        if needs_ocr_files:
            logging.info(
                f"Files moved to Needs_OCR folder: {len(needs_ocr_files)}. "
                "These files require OCR and can be processed manually."
            )

    success_count = sum(1 for r in results if r[0])
    logging.info(f"Finished. Success: {success_count}/{len(file_paths)}")


def main() -> None:
    # Setup cross-platform environment (encoding, colors)
    setup_environment()

    # Setup paths and environment
    initialize_directories()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if not check_requirements():
        sys.exit(1)

    args = parse_arguments()

    handle_clear_cache(args)

    # Directory is required for other operations
    if not args.directory:
        logging.error(
            "Directory argument is required. Use --clear-cache to clear cache without specifying a directory."
        )
        sys.exit(1)

    root_dir = Path(args.directory).resolve()

    if not root_dir.is_dir():
        logging.error(f"Directory not found: {root_dir}")
        sys.exit(1)

    # Vision mode warning
    if args.vision or args.rename_images:
        logging.warning(
            "Note: --vision and --rename-images require a vision-capable model. "
            "Your selected model will be used for both text extraction and vision tasks."
        )

    # Setup engines
    ocr_engine, llm_engine, processor = setup_engines(args)

    # Validate vision mode requirements
    if args.rename_images and not args.vision:
        # --rename-images alone: images get vision, PDFs get regular text extraction
        logging.info(
            "Note: --rename-images without --vision will process images with vision model, "
            "but PDFs will use regular text extraction (not vision). "
            "Use --vision if you want PDFs processed with vision too."
        )

    if args.vision or args.rename_images:
        # Both flags together = mixed folder processing (valid use case)
        # No warning needed - this is the intended use case for mixed folders

        if not validate_vision_mode(args):
            sys.exit(1)

        # Vision modes skip OCR-related flags
        if args.skip_needs_ocr or args.move_ocr_failures:
            logging.warning(
                "Note: --skip-needs-ocr and --move-ocr-failures are not applicable "
                "in vision mode. These flags will be ignored."
            )

        if args.ocr_enhanced:
            logging.warning(
                "Note: --ocr-enhanced is not applicable in vision mode. "
                "Vision mode processes PDFs directly without OCR. This flag will be ignored."
            )

        if args.use_mutool:
            logging.warning(
                "Note: --use-mutool is not applicable in vision mode. "
                "Vision mode processes PDFs directly. This flag will be ignored."
            )

        if args.text_length != 10000:
            logging.warning(
                "Note: --text-length is not applicable in vision mode. "
                "Vision mode processes images, not extracted text. This flag will be ignored."
            )

        if args.extractor != "pymupdf":
            logging.warning(
                "Note: --extractor is not applicable in vision mode. "
                "Vision mode uses image rendering, not text extraction. This flag will be ignored."
            )

        # Info: --text-length is still useful with --rename-images (controls extracted text for non-image files if any)
        # No warning needed, this is valid

    # Set up OCR folders if enabled (not for vision/image modes)
    if not args.vision and not args.rename_images:
        if args.skip_needs_ocr and args.move_ocr_failures:
            logging.warning(
                "Both --skip-needs-ocr and --move-ocr-failures are enabled. "
                "Files that require OCR will be moved to Needs_OCR folder, "
                "while complete OCR failures will go to OCR_Failures folder."
            )

        if args.skip_needs_ocr:
            processor.set_needs_ocr_folder(root_dir, skip=True)

        if args.move_ocr_failures:
            processor.set_ocr_failures_folder(root_dir, enabled=True)

    # Display warnings before processing (unless disabled)
    display_warnings(args.mode, args.recursive)

    # Collect files based on mode
    if args.vision and args.rename_images:
        # Mixed folder: collect both PDFs and images (all with vision)
        pdf_files = collect_files(root_dir, processor, args.recursive)
        image_files = collect_image_files(root_dir, processor, args.recursive)
        file_paths = pdf_files + image_files
        logging.info(
            f"Collected {len(pdf_files)} PDFs and {len(image_files)} images for mixed processing"
        )
    elif args.rename_images:
        # --rename-images alone: PDFs get text extraction, images get vision
        pdf_files = collect_files(root_dir, processor, args.recursive)
        image_files = collect_image_files(root_dir, processor, args.recursive)
        file_paths = pdf_files + image_files
        logging.info(
            f"Collected {len(pdf_files)} PDFs and {len(image_files)} images for processing "
            f"(PDFs: text extraction, Images: vision)"
        )
    else:
        file_paths = collect_files(root_dir, processor, args.recursive)

    if not file_paths:
        mode_name = "images" if args.rename_images else "supported files"
        logging.warning(f"No {mode_name} found.")
        sys.exit(0)

    process_special_modes(args, processor, file_paths)

    mode_desc = (
        "VISION"
        if args.vision
        else ("IMAGE RENAME" if args.rename_images else args.mode.upper())
    )
    logging.info(f"Found {len(file_paths)} files to process in {mode_desc} mode.")

    process_files(args, processor, file_paths, root_dir)


if __name__ == "__main__":
    main()
