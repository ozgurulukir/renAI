import time
import shutil
import uuid
from pathlib import Path
from unittest.mock import Mock
from file_processor import BookProcessor

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

        # Add to list (success, current_path, name, category)
        files.append((True, source_file, f"book_{i}.pdf", "Fiction"))

    return source_dir, dest_dir, files

def benchmark():
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        print("Setting up files...")
        # 10 files of 20MB each
        source_dir, dest_dir, files = setup_files(tmp_path, 10, 20)

        processor = BookProcessor(Mock(), Mock())

        print("Starting benchmark (simulate old behavior)...")
        from utils import get_file_hash

        # In the old code, we didn't pass source_hash to safe_move_file,
        # so safe_move_file had to call get_file_hash(source_path) itself.
        # We can simulate this by mocking organize_files to NOT pass source_hash
        # but the current codebase forces us to pass the 5-tuple.
        # So we can just call safe_move_file directly 10 times without source_hash.
        from file_processor import safe_move_file

        start_time = time.time()
        for _, source_file, name, category in files:
            dest_file = dest_dir / category / name
            safe_move_file(source_file, dest_file, "organize", ensure_dirs=False)

        elapsed = time.time() - start_time
        print(f"Elapsed time: {elapsed:.3f} seconds")
        return elapsed

if __name__ == "__main__":
    benchmark()
