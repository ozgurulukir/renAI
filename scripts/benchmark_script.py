import shutil
import sys
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

def setup_dummy_files(root_dir, num_dirs, num_files_per_dir, num_root_files):
    if root_dir.exists():
        shutil.rmtree(root_dir)
    root_dir.mkdir()

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
    root_dir = Path("dummy_dir_bench")
    setup_dummy_files(root_dir, 50, 1000, 5000) # 55000 total files

    processor = MockProcessor()

    # Warmup
    collect_files(root_dir, processor, recursive=True)

    iterations = 5
    total_time = 0
    for _ in range(iterations):
        start_time = time.time()
        files = collect_files(root_dir, processor, recursive=True)
        end_time = time.time()
        total_time += (end_time - start_time)

    avg_time = total_time / iterations
    print(f"Collected {len(files)} files. Average time: {avg_time:.4f} seconds.")

    shutil.rmtree(root_dir)

if __name__ == "__main__":
    run_benchmark()
