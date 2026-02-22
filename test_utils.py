import hashlib
from pathlib import Path
import pytest
from utils import (
    get_file_hash,
    generate_cache_key,
    get_provider_model_key,
    get_metadata_cache_path,
    get_text_cache_path,
    sanitize_filename,
    proportional_chunk_trim,
)

def test_get_file_hash(tmp_path):
    """Test MD5 calculation for a file."""
    test_file = tmp_path / "test.txt"
    content = b"test content"
    test_file.write_bytes(content)

    expected_hash = hashlib.md5(content).hexdigest()
    assert get_file_hash(test_file) == expected_hash

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

def test_sanitize_filename():
    """Test filename sanitization for various edge cases."""
    # Windows invalid characters
    assert sanitize_filename('test<one>:two"three/four\\five|six?seven*eight') == 'testonetwothreefourfivesixseveneight'

    # Control characters
    assert sanitize_filename('file\nname\r\t.txt') == 'filename.txt'

    # Non-printable characters
    assert sanitize_filename('file\x00name.txt') == 'filename.txt'

    # Multiple spaces and dots
    assert sanitize_filename('  my   file...name..txt  ') == 'my file.name.txt'

    # Consecutive dots
    assert sanitize_filename('file....txt') == 'file.txt'

    # Leading/trailing spaces and dots
    assert sanitize_filename(' . file.txt . ') == 'file.txt'

    # Length limit
    long_name = 'a' * 200
    sanitized = sanitize_filename(long_name, max_length=50)
    assert len(sanitized) == 50
    assert sanitized == 'a' * 50

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
