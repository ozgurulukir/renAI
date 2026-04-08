import re

with open("src/renai/cli.py", "r", encoding="utf-8") as f:
    code = f.read()

# Define the replacements manually or semi-automatically
replacements = [
    (r'directory: str = typer\.Argument\(', r'directory: Annotated[Optional[str], typer.Argument('),
    (r'mode: str = typer\.Option\(', r'mode: Annotated[str, typer.Option('),
    (r'ocr_enhanced: bool = typer\.Option\(', r'ocr_enhanced: Annotated[bool, typer.Option('),
    (r'debug_ocr: bool = typer\.Option\(', r'debug_ocr: Annotated[bool, typer.Option('),
    (r'use_mutool: bool = typer\.Option\(', r'use_mutool: Annotated[bool, typer.Option('),
    (r'mutool_path: str = typer\.Option\(', r'mutool_path: Annotated[str, typer.Option('),
    (r'provider: str = typer\.Option\(', r'provider: Annotated[str, typer.Option('),
    (r'model: str = typer\.Option\(', r'model: Annotated[Optional[str], typer.Option('),
    (r'extractor: str = typer\.Option\(', r'extractor: Annotated[str, typer.Option('),
    (r'workers: int = typer\.Option\(', r'workers: Annotated[int, typer.Option('),
    (r'text_length: int = typer\.Option\(', r'text_length: Annotated[int, typer.Option('),
    (r'recursive: bool = typer\.Option\(', r'recursive: Annotated[bool, typer.Option('),
    (r'update_metadata: bool = typer\.Option\(', r'update_metadata: Annotated[bool, typer.Option('),
    (r'fallback_mode: FallbackMode = typer\.Option\(', r'fallback_mode: Annotated[FallbackMode, typer.Option('),
    (r'move_scan_failures: bool = typer\.Option\(', r'move_scan_failures: Annotated[bool, typer.Option('),
    (r'rename_images: bool = typer\.Option\(', r'rename_images: Annotated[bool, typer.Option('),
    (r'exif: bool = typer\.Option\(', r'exif: Annotated[bool, typer.Option('),
    (r'benchmark: bool = typer\.Option\(', r'benchmark: Annotated[bool, typer.Option('),
    (r'eval_model: bool = typer\.Option\(', r'eval_model: Annotated[bool, typer.Option('),
    (r'clear_cache: bool = typer\.Option\(', r'clear_cache: Annotated[bool, typer.Option('),
]

# We must close the Annotated[...] around the default value.
# Wait, this regex is too simplistic. Let me just use replace_file_content for the whole block of process args.
