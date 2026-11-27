import sys
import tempfile
import time
from pathlib import Path

# Add src to path if running from root or scripts/
if (Path.cwd() / "src").exists():
    sys.path.insert(0, str(Path.cwd() / "src"))
elif (Path.cwd().parent / "src").exists():
    sys.path.insert(0, str(Path.cwd().parent / "src"))

from renai.processors.orchestrator import collect_files


class MockProcessor:
    @property
    def supported_extensions(self):
        return {".pdf", ".epub", ".mobi", ".docx", ".txt", ".md"}


def setup_dummy_files(root_dir: Path, num_dirs: int, num_files_per_dir: int, num_root_files: int):
    # Files in root
    for j in range(num_root_files):
        (root_dir / f"root_file_{j}.txt").touch()

    # Files in subdirs
    for i in range(num_dirs):
        sub_dir = root_dir / f"sub_{i}"
        sub_dir.mkdir()
        for j in range(num_files_per_dir):
            (sub_dir / f"file_{j}.pdf").touch()


def run_benchmark():
    with tempfile.TemporaryDirectory(prefix="renai_bench_") as tmp_dir:
        root_dir = Path(tmp_dir)
        setup_dummy_files(root_dir, 50, 1000, 5000)  # 55000 total files

        processor = MockProcessor()

        # Warmup
        collect_files(root_dir, processor, recursive=True)

        iterations = 5
        total_time = 0.0
        files = []
        for _ in range(iterations):
            start_time = time.time()
            files = collect_files(root_dir, processor, recursive=True)
            end_time = time.time()
            total_time += end_time - start_time

        avg_time = total_time / iterations
        print(f"Collected {len(files)} files. Average time: {avg_time:.4f} seconds.")


if __name__ == "__main__":
    run_benchmark()
