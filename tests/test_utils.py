import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from renai.utils import (
    ensure_secure_dir,
    generate_cache_key,
    get_file_hash,
    head_tail_trim,
    read_text_limited,
    sanitize_filename,
)


def test_get_file_hash_small(tmp_path):
    """Test SHA256 calculation for a small file."""
    test_file = tmp_path / "test.txt"
    content = b"test content"
    test_file.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest()
    assert get_file_hash(test_file) == expected_hash


def test_get_file_hash_empty(tmp_path):
    """Test SHA256 calculation for an empty file."""
    test_file = tmp_path / "empty.txt"
    test_file.write_bytes(b"")

    expected_hash = hashlib.sha256(b"").hexdigest()
    assert get_file_hash(test_file) == expected_hash


def test_get_file_hash_large(tmp_path):
    """Test SHA256 calculation for a file larger than the chunk size (4096)."""
    test_file = tmp_path / "large.txt"
    # Create a 10KB file
    content = b"a" * 10240
    test_file.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest()
    assert get_file_hash(test_file) == expected_hash


def test_get_file_hash_nonexistent():
    """Test that get_file_hash raises FileNotFoundError for non-existent files."""
    with pytest.raises(FileNotFoundError) as exc_info:
        get_file_hash(Path("non_existent_file.txt"))
    assert (
        "non_existent_file.txt" in str(exc_info.value)
        or exc_info.value.filename == "non_existent_file.txt"
    )


def test_get_file_hash_mocked():
    """Test get_file_hash with mocked file_digest to avoid heavy I/O."""
    from renai.utils import get_file_hash

    mock_file = MagicMock()
    # Create a mock path object that behaves like a Path
    mock_path = MagicMock(spec=Path)
    mock_path.open.return_value.__enter__.return_value = mock_file

    with patch("hashlib.file_digest") as mock_file_digest:
        # Mock file_digest to return an object with a hexdigest method
        mock_digest_obj = MagicMock()
        mock_digest_obj.hexdigest.return_value = "mocked_hash_value"
        mock_file_digest.return_value = mock_digest_obj

        result = get_file_hash(mock_path)

        assert result == "mocked_hash_value"
        mock_path.open.assert_called_once_with("rb")
        mock_file_digest.assert_called_once_with(mock_file, "sha256")
        mock_digest_obj.hexdigest.assert_called_once_with()


def test_generate_cache_key():
    """Test cache key generation from multiple strings."""
    # Basic combination
    assert generate_cache_key("a", "b", "c") == hashlib.sha256(b"a|b|c").hexdigest()

    # Handle empty/None values
    assert generate_cache_key("a", "", None, "b") == hashlib.sha256(b"a|b").hexdigest()

    # Single value
    assert generate_cache_key("only") == hashlib.sha256(b"only").hexdigest()


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
        ("   .   ", "unnamed"),
        ('<>:"/\\|?*', "unnamed"),
        ("", "unnamed"),
        (".", "unnamed"),
        ("..", "unnamed"),
        ("...", "unnamed"),
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


def test_head_tail_trim_basic():
    """Test basic functionality of head_tail_trim."""
    assert head_tail_trim("", 10) == ""
    text = "Short text"
    assert head_tail_trim(text, 50) == text

    result = head_tail_trim("A" * 100, 40, head_ratio=0.7)
    assert len(result) <= 40
    assert "\n...\n" in result
    head_part, tail_part = result.split("\n...\n")
    assert head_part.startswith("A")
    assert tail_part.endswith("A")


def test_head_tail_trim_edge_cases():
    """Test edge cases for head_tail_trim."""
    assert head_tail_trim("abcdef", 3) == "abc"
    assert head_tail_trim("abcdef", 0) == ""
    assert head_tail_trim("abcdef", 100) == "abcdef"
    assert head_tail_trim("abc", 2, head_ratio=0.7) == "ab"

    with pytest.raises(ValueError, match="head_ratio must be between 0 and 1"):
        head_tail_trim("abc", 10, head_ratio=0.0)
    with pytest.raises(ValueError, match="head_ratio must be between 0 and 1"):
        head_tail_trim("abc", 10, head_ratio=1.0)


def test_head_tail_trim_head_tail_preservation():
    """Test that head and tail content are preserved."""
    text = "HEADER_CONTENT_" + "X" * 200 + "_TAIL_CONTENT"
    result = head_tail_trim(text, 50, head_ratio=0.7)
    assert "HEADER_CONTENT" in result
    assert "TAIL_CONTENT" in result
    assert len(result) <= 50


def test_read_text_limited_small(tmp_path):
    """Test reading a file smaller than max_chars."""
    test_file = tmp_path / "small.txt"
    content = "Hello, world!"
    test_file.write_text(content, encoding="utf-8")

    result = read_text_limited(test_file, max_chars=100)
    assert result == content


def test_read_text_limited_large(tmp_path):
    """Test reading a file larger than max_chars."""
    test_file = tmp_path / "large.txt"
    content = "A" * 1000
    test_file.write_text(content, encoding="utf-8")

    result = read_text_limited(test_file, max_chars=100)
    assert len(result) == 100
    assert result == "A" * 100


def test_read_text_limited_empty(tmp_path):
    """Test reading an empty file."""
    test_file = tmp_path / "empty.txt"
    test_file.write_text("", encoding="utf-8")

    result = read_text_limited(test_file, max_chars=100)
    assert result == ""


@patch("renai.utils.logger.error")
@patch.object(Path, "open")
def test_read_text_limited_error_logging(mock_open, mock_logging_error, tmp_path):
    """Test that an exception triggers logging.error and returns an empty string."""
    test_file = tmp_path / "error.txt"

    # Make open() raise an exception
    mock_open.side_effect = PermissionError("Permission denied")

    result = read_text_limited(test_file, max_chars=100)

    # Should return empty string on error
    assert result == ""

    # Verify logging.error was called
    assert mock_logging_error.call_count == 1
    assert "Error reading" in mock_logging_error.call_args[0][0]
    assert "Permission denied" in mock_logging_error.call_args[0][0]


def test_read_text_limited_zero_chars(tmp_path):
    """Test reading with max_chars set to 0."""
    test_file = tmp_path / "zero.txt"
    test_file.write_text("Some text", encoding="utf-8")

    result = read_text_limited(test_file, max_chars=0)
    assert result == ""


def test_read_text_limited_nonexistent(tmp_path):
    """Test reading a file that does not exist."""
    test_file = tmp_path / "does_not_exist.txt"

    result = read_text_limited(test_file, max_chars=100)
    assert result == ""


def test_read_text_limited_directory(caplog, tmp_path):
    """Test reading from a directory to trigger an actual OS error without mocking open."""
    import logging

    # tmp_path is a directory
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    with caplog.at_level(logging.ERROR):
        result = read_text_limited(test_dir, max_chars=100)

    # Should return empty string on error
    assert result == ""

    # Verify the error was logged
    assert any("Error reading" in record.message for record in caplog.records)
    assert any(
        "Is a directory" in record.message or "Permission denied" in record.message
        for record in caplog.records
    )


@patch("renai.utils.sys")
@patch("renai.utils.os")
@patch("renai.utils.subprocess")
@patch("renai.utils.logger.debug")
def test_ensure_secure_dir_windows_exception_extended(
    mock_debug, mock_subprocess, mock_os, mock_sys, tmp_path
):
    """Test ensure_secure_dir on Windows when subprocess.run raises an exception (extended coverage)."""
    mock_sys.platform = "win32"
    mock_os.environ.get.return_value = "valid_user"

    # Mock Windows-specific subprocess attributes to prevent AttributeError on Linux
    mock_startupinfo = MagicMock()
    mock_startupinfo.dwFlags = 0
    mock_subprocess.STARTUPINFO.return_value = mock_startupinfo
    mock_subprocess.STARTF_USESHOWWINDOW = 1
    mock_subprocess.CREATE_NO_WINDOW = 0x08000000

    mock_subprocess.run.side_effect = Exception("Mock icacls failure")

    test_dir = tmp_path / "secure_test_dir"

    result = ensure_secure_dir(test_dir)

    # Assert directory was created and returned
    assert result == test_dir
    assert test_dir.exists()

    # Assert subprocess.run was called
    assert mock_subprocess.run.call_count == 1

    # Assert the exception was logged
    assert mock_debug.call_count == 1
    log_msg = mock_debug.call_args[0][0]
    assert "Failed to set secure permissions" in log_msg
    assert "Mock icacls failure" in log_msg


@patch("renai.utils.sys")
def test_ensure_secure_dir_unix(mock_sys, tmp_path):
    """Test directory creation with secure permissions on Unix."""
    mock_sys.platform = "linux"
    target_dir = tmp_path / "secure_unix_dir"

    with patch.object(Path, "chmod") as mock_chmod:
        ensure_secure_dir(target_dir)

        # Directory should be created
        assert target_dir.exists()

        # chmod should be called with 0o700
        mock_chmod.assert_called_once_with(0o700)


@patch("renai.utils.sys")
@patch("renai.utils.logger.debug")
def test_ensure_secure_dir_unix_exception(mock_debug, mock_sys, tmp_path):
    """Test exception handling during chmod on Unix."""
    mock_sys.platform = "linux"
    target_dir = tmp_path / "secure_unix_dir_fail"

    with patch.object(Path, "chmod", side_effect=Exception("chmod failed")):
        ensure_secure_dir(target_dir)

        # Exception should be caught and logged
        assert mock_debug.call_count == 1
        # Adjusted to match the log message in src/renai/utils.py if needed,
        # but I'll stick to what was in main for now.
        assert "Failed to set secure permissions" in mock_debug.call_args[0][0]


@patch("renai.utils.sys")
@patch("renai.utils.os")
@patch("renai.utils.subprocess")
def test_ensure_secure_dir_windows_valid_user(mock_subprocess, mock_os, mock_sys, tmp_path):
    """Test icacls permission setting on Windows with a valid username."""
    mock_sys.platform = "win32"
    mock_os.environ.get.return_value = "TestUser"

    # Mock startupinfo attributes
    mock_startupinfo_instance = MagicMock()
    mock_startupinfo_instance.dwFlags = 0
    mock_subprocess.STARTUPINFO.return_value = mock_startupinfo_instance
    mock_subprocess.STARTF_USESHOWWINDOW = 0
    mock_subprocess.run = MagicMock()
    mock_subprocess.CREATE_NO_WINDOW = 0x08000000

    target_dir = tmp_path / "secure_win_dir"
    ensure_secure_dir(target_dir)

    # Directory should be created
    assert target_dir.exists()

    # icacls should be called
    assert mock_subprocess.run.call_count == 1
    args = mock_subprocess.run.call_args[0][0]
    assert args[0] == "icacls"
    assert args[1] == str(target_dir)
    assert "/inheritance:r" in args
    assert "/grant:r" in args
    assert "TestUser:(OI)(CI)F" in args


@patch("renai.utils.sys")
@patch("renai.utils.os")
@patch("renai.utils.logger.warning")
@patch("renai.utils.subprocess")
def test_ensure_secure_dir_windows_invalid_user(
    mock_subprocess, mock_warning, mock_os, mock_sys, tmp_path
):
    """Test Windows permission setting skips and logs warning on invalid username."""
    mock_sys.platform = "win32"
    # Invalid username with command injection attempt
    mock_os.environ.get.return_value = "User&rmdir /s /q C:\\"
    mock_subprocess.run = MagicMock()

    target_dir = tmp_path / "secure_win_dir_invalid"
    ensure_secure_dir(target_dir)

    # Warning should be logged
    assert mock_warning.call_count == 1
    assert "Invalid username" in mock_warning.call_args[0][0]

    # icacls should NOT be called
    mock_subprocess.run.assert_not_called()


@patch("renai.utils.sys")
@patch("renai.utils.os")
@patch("renai.utils.subprocess")
def test_ensure_secure_dir_windows_no_user(mock_subprocess, mock_os, mock_sys, tmp_path):
    """Test Windows permission setting skips if USERNAME is not set."""
    mock_sys.platform = "win32"
    mock_os.environ.get.return_value = None
    mock_subprocess.run = MagicMock()

    target_dir = tmp_path / "secure_win_dir_no_user"
    ensure_secure_dir(target_dir)

    # icacls should NOT be called
    mock_subprocess.run.assert_not_called()


@patch("renai.utils.sys")
@patch("renai.utils.os")
@patch("renai.utils.subprocess")
@patch("renai.utils.logger.debug")
def test_ensure_secure_dir_windows_exception(
    mock_debug, mock_subprocess, mock_os, mock_sys, tmp_path
):
    """Test exception handling during icacls on Windows."""
    mock_sys.platform = "win32"
    mock_os.environ.get.return_value = "TestUser"

    # Mock startupinfo
    mock_startupinfo_instance = MagicMock()
    mock_startupinfo_instance.dwFlags = 0
    mock_subprocess.STARTUPINFO.return_value = mock_startupinfo_instance
    mock_subprocess.STARTF_USESHOWWINDOW = 0
    mock_subprocess.run = MagicMock(side_effect=Exception("icacls failed"))
    mock_subprocess.CREATE_NO_WINDOW = 0x08000000

    target_dir = tmp_path / "secure_win_dir_fail"
    ensure_secure_dir(target_dir)

    # Exception should be caught and logged as debug
    assert mock_debug.call_count == 1
    assert "Failed to set secure permissions" in mock_debug.call_args[0][0]
