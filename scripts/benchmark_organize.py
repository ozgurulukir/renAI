import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path if running from root or scripts/
if (Path.cwd() / "src").exists():
    sys.path.insert(0, str(Path.cwd() / "src"))
elif (Path.cwd().parent / "src").exists():
    sys.path.insert(0, str(Path.cwd().parent / "src"))

from renai.models import ProcessingResult
from renai.processors.book import BookProcessor


def setup_files(tmp_path, num_files, file_size_mb):
    """
    Creates a source directory and destination directory with large files
    to test hash collision speed.
    """
    source_dir = tmp_path / "source"
    dest_dir = tmp_path / "dest"
    source_dir.mkdir()
    dest_dir.mkdir()

    # Create random binary data
    data = os.urandom(file_size_mb * 1024 * 1024)

    files = []
    # Create large identical files in both source and dest so that a collision occurs
    for i in range(num_files):
        source_file = source_dir / f"book_{i}.pdf"
        target_dir = dest_dir / "Fiction"
        target_dir.mkdir(exist_ok=True, parents=True)
        dest_file = target_dir / f"book_{i}.pdf"

        # Write same data
        source_file.write_bytes(data)
        dest_file.write_bytes(data)

        # Create ProcessingResult object
        files.append(
            ProcessingResult(
                success=True,
                file_path=source_file,
                name=f"book_{i}.pdf",
                category="Fiction",
            )
        )

    return source_dir, dest_dir, files


async def run_benchmark():
    """Run the benchmark asynchronously."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        print("Setting up files...")
        # 10 files of 20MB each
        source_dir, _dest_dir, _files = setup_files(tmp_path, 10, 20)

        # Mock engines since we only care about organization speed (hash + move)
        processor = BookProcessor(Mock(), Mock())
        # Mock extract_document_metadata to avoid LLM/OCR calls
        processor._extract_document_metadata = Mock(return_value=None)
        # We'll patch write_metadata_cache to avoid disk-bound cache writes during move benchmark
        with patch("renai.processors.book.write_metadata_cache"), \
             patch("renai.processors.book.read_metadata_cache") as mock_read:

            # Mock return value for all files to be organized
            from renai.models.schemas import BookMetadata
            mock_read.return_value = BookMetadata(
                title="Mock Book",
                author="Mock Author",
                category="Fiction",
                year=2024,
                language="en"
            )

            print("Starting benchmark...")
            start_time = time.time()

            # Collect source files
            source_files = list(source_dir.glob("*.pdf"))

            # Process them in organize mode
            tasks = [processor.process_file(f, mode="organize") for f in source_files]
            await asyncio.gather(*tasks)

            elapsed = time.time() - start_time
            print(f"Elapsed time: {elapsed:.3f} seconds")
            return elapsed


def benchmark():
    import asyncio
    return asyncio.run(run_benchmark())


if __name__ == "__main__":
    benchmark()
