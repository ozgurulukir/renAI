import sys
import time
from pathlib import Path
from unittest.mock import Mock

# Add src to path if running from root or scripts/
if (Path.cwd() / "src").exists():
    sys.path.insert(0, str(Path.cwd() / "src"))
elif (Path.cwd().parent / "src").exists():
    sys.path.insert(0, str(Path.cwd().parent / "src"))

from renai.models import ProcessingResult
from renai.processors.book import BookProcessor


def setup_files(tmp_path, num_files, file_size_mb):
    """Creates a source directory and destination directory with large files to test hash collision speed."""
    source_dir = tmp_path / "source"
    dest_dir = tmp_path / "dest"
    source_dir.mkdir()
    dest_dir.mkdir()

    # Create random binary data
    import os
    data = os.urandom(file_size_mb * 1024 * 1024)

    files = []
    # Create large identical files in both source and dest so that a collision occurs
    for i in range(num_files):
        source_file = source_dir / f"book_{i}.pdf"
        dest_file = dest_dir / "Fiction" / f"book_{i}.pdf"
        dest_file.parent.mkdir(exist_ok=True, parents=True)

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

def benchmark():
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        print("Setting up files...")
        # 10 files of 20MB each
        source_dir, dest_dir, files = setup_files(tmp_path, 10, 20)

        processor = BookProcessor(Mock(), Mock())

        print("Starting benchmark...")
        start_time = time.time()

        # ProcessingResult handles its own validation, so we just pass the objects
        processor.organize_files(files, dest_dir)

        elapsed = time.time() - start_time
        print(f"Elapsed time: {elapsed:.3f} seconds")
        return elapsed

if __name__ == "__main__":
    benchmark()
