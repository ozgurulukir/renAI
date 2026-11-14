import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

# Add src to path if running from root or scripts/
if (Path.cwd() / "src").exists():
    sys.path.insert(0, str(Path.cwd() / "src"))
elif (Path.cwd().parent / "src").exists():
    sys.path.insert(0, str(Path.cwd().parent / "src"))

from renai.models import ProcessingResult
from renai.models.schemas import ProcessingConfig
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
        processor = BookProcessor(Mock(), ProcessingConfig(mode="organize"), Mock())
        # Mock extract_document_metadata to avoid LLM/OCR calls. AsyncMock because
        # the middleware awaits it whenever the mocked cache read misses.
        processor._extract_document_metadata = AsyncMock(return_value=None)
        # Patch the cache layer at its real home (renai.cache.middleware -- the
        # module the with_metadata_cache decorator actually imports) so the
        # benchmark measures hash + move without disk-bound cache I/O. The stale
        # renai.processors.book.read_metadata_cache/write_metadata_cache patch
        # targets started raising AttributeError when the middleware seam landed.
        with (
            patch("renai.cache.middleware.read_metadata_cache") as mock_read,
            patch("renai.cache.middleware.write_metadata_cache"),
        ):
            # Mock return value for all files to be organized: a cache hit is
            # (metadata, CacheSource-or-None).
            from renai.models.schemas import BookMetadata

            mock_read.return_value = (
                BookMetadata(
                    title="Mock Book",
                    author="Mock Author",
                    category="Fiction",
                    year=2024,
                    language="en",
                ),
                None,
            )

            print("Starting benchmark...")
            start_time = time.time()

            # Collect source files
            source_files = list(source_dir.glob("*.pdf"))

            # Process them in organize mode
            tasks = [processor.process_file(f) for f in source_files]
            await asyncio.gather(*tasks)

            elapsed = time.time() - start_time
            print(f"Elapsed time: {elapsed:.3f} seconds")
            return elapsed


def benchmark():
    import asyncio

    return asyncio.run(run_benchmark())


if __name__ == "__main__":
    benchmark()
