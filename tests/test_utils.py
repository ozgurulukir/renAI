import sys

sys.path.insert(0, "src")

import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from renai.utils import (
    get_file_hash,
    generate_cache_key,
    get_provider_model_key,
    get_metadata_cache_path,
    get_text_cache_path,
    sanitize_filename,
    proportional_chunk_trim,
    read_text_limited,
)


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
    """Test get_file_hash with mocking to verify file_digest usage."""
    mock_file_content = b"mocked content for digest"

    mock_path = Path("mocked_file.txt")

    with (
        patch("builtins.open") as mock_open,
        patch("hashlib.file_digest") as mock_file_digest,
    ):
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Mock file_digest to return an object with a hexdigest method
        mock_digest_obj = MagicMock()
        expected_hash = hashlib.md5(mock_file_content).hexdigest()
        mock_digest_obj.hexdigest.return_value = expected_hash
        mock_file_digest.return_value = mock_digest_obj

        result_hash = get_file_hash(mock_path)

        assert result_hash == expected_hash

        # Verify file_digest was called correctly
        mock_file_digest.assert_called_once_with(mock_file, "md5")
        mock_digest_obj.hexdigest.assert_called_once()


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


@patch("renai.utils.logging.error")
@patch("builtins.open")
def test_read_text_limited_error_logging(mock_open, mock_logging_error, tmp_path):
    """Test that an exception triggers logging.error and returns an empty string."""
    test_file = tmp_path / "error.txt"

    # Make open() raise an exception
    mock_open.side_effect = PermissionError("Permission denied")

    result = read_text_limited(test_file, max_chars=100)

    # Should return empty string on error
    assert result == ""

    # Verify logging.error was called
    mock_logging_error.assert_called_once()
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
