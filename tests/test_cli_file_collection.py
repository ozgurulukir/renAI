
import pytest
from unittest.mock import MagicMock
from renai.cli import collect_all_files

@pytest.fixture
def mock_processor():
    processor = MagicMock()
    processor.supported_extensions = {".pdf", ".epub"}
    return processor

def test_collect_all_files_non_recursive(tmp_path, mock_processor):
    # Create test files
    (tmp_path / "book1.pdf").touch()
    (tmp_path / "book2.epub").touch()
    (tmp_path / "image1.jpg").touch()
    (tmp_path / "other.txt").touch()

    # Create a subdirectory with more files
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    (sub_dir / "book3.pdf").touch()
    (sub_dir / "image2.png").touch()

    doc_paths, image_paths = collect_all_files(tmp_path, mock_processor, recursive=False)

    doc_names = {p.name for p in doc_paths}
    image_names = {p.name for p in image_paths}

    assert doc_names == {"book1.pdf", "book2.epub"}
    assert image_names == {"image1.jpg"}

def test_collect_all_files_recursive(tmp_path, mock_processor):
    # Create test files
    (tmp_path / "book1.pdf").touch()
    (tmp_path / "image1.jpg").touch()

    # Create a subdirectory with more files
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    (sub_dir / "book2.pdf").touch()
    (sub_dir / "image2.png").touch()

    doc_paths, image_paths = collect_all_files(tmp_path, mock_processor, recursive=True)

    doc_names = {p.name for p in doc_paths}
    image_names = {p.name for p in image_paths}

    assert doc_names == {"book1.pdf", "book2.pdf"}
    assert image_names == {"image1.jpg", "image2.png"}

def test_collect_all_files_excludes_symlinks(tmp_path, mock_processor):
    # Create a real file
    real_file = tmp_path / "real.pdf"
    real_file.touch()

    # Create a symlink (if possible on the platform)
    symlink_file = tmp_path / "link.pdf"
    try:
        symlink_file.symlink_to(real_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported on this platform/environment")

    doc_paths, image_paths = collect_all_files(tmp_path, mock_processor, recursive=False)

    assert len(doc_paths) == 1
    assert doc_paths[0].name == "real.pdf"

def test_collect_all_files_case_insensitive(tmp_path, mock_processor):
    (tmp_path / "BOOK.PDF").touch()
    (tmp_path / "IMAGE.JPG").touch()

    doc_paths, image_paths = collect_all_files(tmp_path, mock_processor, recursive=False)

    assert len(doc_paths) == 1
    assert doc_paths[0].name == "BOOK.PDF"
    assert len(image_paths) == 1
    assert image_paths[0].name == "IMAGE.JPG"
