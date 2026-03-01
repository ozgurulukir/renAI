import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from utils import (
    get_file_hash,
    generate_cache_key,
    get_provider_model_key,
    get_metadata_cache_path,
    get_text_cache_path,
    sanitize_filename,
    proportional_chunk_trim,
    ensure_secure_dir,
)
import sys
import os
import subprocess

def test_get_file_hash_small(tmp_path):
    """Test MD5 calculation for a small file."""
    test_file = tmp_path / "test.txt"
    content = b"test content"
    test_file.write_bytes(content)

    expected_hash = hashlib.md5(content).hexdigest()
    assert get_file_hash(test_file) == expected_hash

def test_get_file_hash_empty(tmp_path):
    """Test MD5 calculation for an empty file."""
    test_file = tmp_path / "empty.txt"
    test_file.write_bytes(b"")

    expected_hash = hashlib.md5(b"").hexdigest()
    assert get_file_hash(test_file) == expected_hash

def test_get_file_hash_large(tmp_path):
    """Test MD5 calculation for a file larger than the chunk size (4096)."""
    test_file = tmp_path / "large.txt"
    # Create a 10KB file
    content = b"a" * 10240
    test_file.write_bytes(content)

    expected_hash = hashlib.md5(content).hexdigest()
    assert get_file_hash(test_file) == expected_hash

def test_get_file_hash_nonexistent():
    """Test that get_file_hash raises FileNotFoundError for non-existent files."""
    with pytest.raises(FileNotFoundError):
        get_file_hash(Path("non_existent_file.txt"))

def test_get_file_hash_mocked():
    """Test get_file_hash with mocking to verify chunked reading."""
    mock_file_content = b"first chunk" + b"a" * 4085 + b"second chunk"
    # total length = 11 + 4085 + 12 = 4108 bytes
    # It should read in 2 chunks (4096 and 12)

    mock_path = Path("mocked_file.txt")

    with patch("builtins.open") as mock_open:
        mock_file = MagicMock()
        # Mocking the iterator used in iter(lambda: f.read(4096), b"")
        # f.read(4096) will be called until it returns b""
        mock_file.read.side_effect = [
            mock_file_content[:4096],
            mock_file_content[4096:],
            b""
        ]
        mock_open.return_value.__enter__.return_value = mock_file

        result_hash = get_file_hash(mock_path)

        expected_hash = hashlib.md5(mock_file_content).hexdigest()
        assert result_hash == expected_hash

        # Verify read was called with 4096
        assert mock_file.read.call_count == 3
        mock_file.read.assert_any_call(4096)

def test_generate_cache_key():
    """Test cache key generation from multiple strings."""
    # Basic combination
    assert generate_cache_key("a", "b", "c") == hashlib.md5(b"a|b|c").hexdigest()

    # Handle empty/None values
    assert generate_cache_key("a", "", None, "b") == hashlib.md5(b"a|b").hexdigest()

    # Single value
    assert generate_cache_key("only") == hashlib.md5(b"only").hexdigest()

def test_get_provider_model_key():
    """Test provider-model key generation."""
    provider = "openai"
    model = "gpt-4o"
    expected = generate_cache_key(provider, model)
    assert get_provider_model_key(provider, model) == expected

def test_get_metadata_cache_path(tmp_path):
    """Test metadata cache path construction."""
    file_hash = "abc123hash"
    provider = "deepinfra"
    model = "llama-3"

    pm_key = get_provider_model_key(provider, model)
    expected_name = f"{file_hash}_{pm_key}_metadata.json"

    path = get_metadata_cache_path(file_hash, provider, model, tmp_path)
    assert path == tmp_path / expected_name

def test_get_text_cache_path(tmp_path):
    """Test text cache path construction with various options."""
    file_hash = "texthash"

    # Basic extraction
    path = get_text_cache_path(file_hash, False, False, tmp_path)
    assert path == tmp_path / f"{file_hash}_basic.txt"

    # Enhanced OCR
    path = get_text_cache_path(file_hash, True, False, tmp_path)
    assert path == tmp_path / f"{file_hash}_enhanced.txt"

    # Enhanced OCR + mutool
    path = get_text_cache_path(file_hash, True, True, tmp_path)
    assert path == tmp_path / f"{file_hash}_enhanced_mutool.txt"

    # Basic + mutool
    path = get_text_cache_path(file_hash, False, True, tmp_path)
    assert path == tmp_path / f"{file_hash}_basic_mutool.txt"

@pytest.mark.parametrize(
    "filename, expected",
    [
        # Normal filenames
        ("standard_file.txt", "standard_file.txt"),
        ("file with spaces.txt", "file with spaces.txt"),
        # Windows reserved characters: < > : " / \ | ? *
        (
            'test<one>:two"three/four\\five|six?seven*eight',
            "testonetwothreefourfivesixseveneight",
        ),
        # Control and non-printable characters
        ("file\nname\r\t.txt", "filename.txt"),
        ("file\x00name\x1f.txt", "filename.txt"),
        # Multiple spaces reduction
        ("multiple   spaces.txt", "multiple spaces.txt"),
        ("  leading and trailing  ", "leading and trailing"),
        # Consecutive dots reduction
        ("file....txt", "file.txt"),
        ("file...name..txt", "file.name.txt"),
        # Leading/trailing spaces and dots
        (" . file.txt . ", "file.txt"),
        ("...dots...", "dots"),
        # Unicode support (printable)
        ("unicode_✨_rock.txt", "unicode_✨_rock.txt"),
        ("dokument_📄.pdf", "dokument_📄.pdf"),
        # Edge cases: strings that become empty
        ("   .   ", ""),
        ('<>:"/\\|?*', ""),
        ("", ""),
        # Multiple spaces and dots mixed
        ("  my   file...name..txt  ", "my file.name.txt"),
    ],
)
def test_sanitize_filename_comprehensive(filename, expected):
    """Test filename sanitization for various scenarios."""
    assert sanitize_filename(filename) == expected


def test_sanitize_filename_length():
    """Test length limiting of sanitized filenames."""
    # Length limit
    long_name = "a" * 200
    sanitized = sanitize_filename(long_name, max_length=50)
    assert len(sanitized) == 50
    assert sanitized == "a" * 50

    # Boundary case: exactly max_length
    assert sanitize_filename("a" * 50, max_length=50) == "a" * 50

    # Boundary case: max_length - 1
    assert sanitize_filename("a" * 49, max_length=50) == "a" * 49

    # Truncation with trailing space/dot stripping
    # 'a' * 150 + ' ' -> 'a' * 150 (strip) -> len is 150, not > 150
    assert sanitize_filename("a" * 50 + " .", max_length=50) == "a" * 50

def test_proportional_chunk_trim_basic():
    """Test basic functionality of proportional_chunk_trim."""
    # Empty text
    assert proportional_chunk_trim("", 10, 2) == ""

    # No trim needed
    text = "Short text"
    assert proportional_chunk_trim(text, 50, 2) == text

    # Basic trim (50%)
    # 12345678, target 4, 2 chunks
    # Chunks: "1234", "5678"
    # total_length = 8, reduction_factor = 4/8 = 0.5
    # allowed_length = int(4 * 0.5) = 2
    # result: "12" + "56" = "1256"
    assert proportional_chunk_trim("12345678", 4, 2) == "1256"

def test_proportional_chunk_trim_edge_cases():
    """Test edge cases for proportional_chunk_trim."""
    # num_chunks = 0 (defaults to 1 chunk)
    assert proportional_chunk_trim("abcdef", 3, 0) == "abc"

    # num_chunks = 1
    assert proportional_chunk_trim("abcdef", 3, 1) == "abc"

    # target_length = 0
    assert proportional_chunk_trim("abcdef", 0, 2) == ""

    # chunk_size = 0 (should not happen if num_chunks > 0 and len(text) > 0,
    # but let's see how it handles it via code path)
    # Actually if len(text) > 0 and num_chunks > len(text), chunk_size = 0
    # The code handles chunk_size == 0 by returning text[:target_length]
    assert proportional_chunk_trim("abc", 2, 10) == "ab"

def test_proportional_chunk_trim_rounding_behavior():
    """
    Demonstrate current rounding behavior where many small chunks
    may lead to excessive trimming (even returning empty string).
    """
    # 10 chars, target 5, 10 chunks
    # Each chunk is 1 char. reduction_factor = 0.5
    # allowed_length = int(1 * 0.5) = 0
    # Result will be "" instead of 5 chars
    text = "1234567890"
    assert proportional_chunk_trim(text, 5, 10) == ""

    # Another example: 10 chars, target 9, 10 chunks
    # reduction_factor = 0.9. allowed_length = int(1 * 0.9) = 0
    assert proportional_chunk_trim(text, 9, 10) == ""


def test_ensure_secure_dir_creates_directory(tmp_path):
    """Test that ensure_secure_dir creates the directory if it doesn't exist."""
    new_dir = tmp_path / "new_secure_dir"
    assert not new_dir.exists()

    ensure_secure_dir(new_dir)

    assert new_dir.exists()
    assert new_dir.is_dir()


@patch("utils.sys.platform", "linux")
def test_ensure_secure_dir_unix(tmp_path):
    """Test ensure_secure_dir on Unix systems."""
    test_dir = tmp_path / "unix_dir"

    with patch.object(Path, "chmod") as mock_chmod:
        ensure_secure_dir(test_dir)

        mock_chmod.assert_called_once_with(0o700)
        assert test_dir.exists()


@patch("utils.sys.platform", "linux")
def test_ensure_secure_dir_unix_exception(tmp_path):
    """Test ensure_secure_dir on Unix systems when chmod fails."""
    test_dir = tmp_path / "unix_dir_exc"

    with patch.object(Path, "chmod", side_effect=Exception("chmod failed")) as mock_chmod:
        # Should not raise an exception
        ensure_secure_dir(test_dir)

        mock_chmod.assert_called_once_with(0o700)
        assert test_dir.exists()


@patch("utils.sys.platform", "win32")
@patch.dict(os.environ, {"USERNAME": "testuser"}, clear=True)
def test_ensure_secure_dir_windows_with_username(tmp_path):
    """Test ensure_secure_dir on Windows when USERNAME is set."""
    test_dir = tmp_path / "win_dir"

    with patch("utils.subprocess.run") as mock_run:
        # Mock STARTUPINFO to avoid actual instantiation errors if it's somehow tested on non-Windows
        # But we also mocked sys.platform in utils, let's just patch subprocess.STARTUPINFO if needed,
        # but utils uses subprocess.STARTUPINFO which exists in the mock or real subprocess.
        # It's safer to mock subprocess.STARTUPINFO if we're running tests on linux, as it might not exist.
        if not hasattr(subprocess, "STARTUPINFO"):
            with patch("utils.subprocess.STARTUPINFO", create=True) as mock_startupinfo:
                with patch("utils.subprocess.STARTF_USESHOWWINDOW", create=True):
                    ensure_secure_dir(test_dir)
        else:
            ensure_secure_dir(test_dir)

        assert test_dir.exists()

        # Verify subprocess.run was called with correct arguments
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args

        expected_cmd = [
            "icacls",
            str(test_dir),
            "/inheritance:r",
            "/grant:r",
            "testuser:(OI)(CI)F",
        ]
        assert args[0] == expected_cmd
        assert kwargs.get("capture_output") is True
        assert kwargs.get("check") is False


@patch("utils.sys.platform", "win32")
@patch.dict(os.environ, clear=True) # Ensure USERNAME is not set
def test_ensure_secure_dir_windows_without_username(tmp_path):
    """Test ensure_secure_dir on Windows when USERNAME is not set."""
    test_dir = tmp_path / "win_dir_no_user"

    with patch("utils.subprocess.run") as mock_run:
        ensure_secure_dir(test_dir)

        assert test_dir.exists()
        # subprocess.run should not be called if username is missing
        mock_run.assert_not_called()


@patch("utils.sys.platform", "win32")
@patch.dict(os.environ, {"USERNAME": "testuser"}, clear=True)
def test_ensure_secure_dir_windows_exception(tmp_path):
    """Test ensure_secure_dir on Windows when subprocess.run fails."""
    test_dir = tmp_path / "win_dir_exc"

    with patch("utils.subprocess.run", side_effect=Exception("icacls failed")) as mock_run:
        if not hasattr(subprocess, "STARTUPINFO"):
            with patch("utils.subprocess.STARTUPINFO", create=True):
                with patch("utils.subprocess.STARTF_USESHOWWINDOW", create=True):
                    # Should not raise an exception
                    ensure_secure_dir(test_dir)
        else:
            ensure_secure_dir(test_dir)

        assert test_dir.exists()
        mock_run.assert_called_once()
