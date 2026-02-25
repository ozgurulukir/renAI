import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


def get_file_hash(file_path: Path) -> str:
    """Calculate the MD5 hash of a file to uniquely identify its content."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def generate_cache_key(*args: str) -> str:
    """Generate a unique cache key from multiple string components.

    Args:
        *args: Variable number of string components to combine

    Returns:
        A hex string hash of the combined components
    """
    combined = "|".join(str(arg) for arg in args if arg)
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


def get_provider_model_key(provider: str, model: str) -> str:
    """Generate a cache key for provider-model pairing.

    Args:
        provider: LLM provider name (e.g., 'deepinfra', 'openai', 'custom')
        model: Model name (e.g., 'gpt-4o-mini', 'llama3')

    Returns:
        A hex string hash combining provider and model
    """
    return generate_cache_key(provider, model)


def get_metadata_cache_path(
    file_hash: str, provider: str, model: str, cache_dir: Path
) -> Path:
    """Get the cache path for metadata with provider-model awareness.

    The cache key includes the file hash and provider-model pairing.
    This ensures that metadata cached with one provider/model is not
    incorrectly reused with a different provider/model.

    Args:
        file_hash: MD5 hash of the file content
        provider: LLM provider name
        model: Model name
        cache_dir: Directory for cache files

    Returns:
        Path to the metadata cache file
    """
    provider_model_key = get_provider_model_key(provider, model)
    cache_filename = f"{file_hash}_{provider_model_key}_metadata.json"
    return cache_dir / cache_filename


def get_text_cache_path(
    file_hash: str, ocr_enhanced: bool, use_mutool: bool, cache_dir: Path
) -> Path:
    """Get the cache path for extracted text.

    Args:
        file_hash: MD5 hash of the file content
        ocr_enhanced: Whether enhanced OCR was used
        use_mutool: Whether mutool was used for extraction
        cache_dir: Directory for cache files

    Returns:
        Path to the text cache file
    """
    quality_suffix = "enhanced" if ocr_enhanced else "basic"
    method_suffix = "_mutool" if use_mutool else ""
    cache_filename = f"{file_hash}_{quality_suffix}{method_suffix}.txt"
    return cache_dir / cache_filename


def sanitize_filename(filename: str, max_length: int = 150) -> str:
    """
    Replace invalid characters and remove all control characters.
    Limits length to stay within Windows path limits.
    """
    # Remove invalid characters for Windows paths: < > : " / \ | ? *
    sanitized = re.sub(r'[<>:"/\\|?*]', "", filename)

    # Remove ALL control characters (including \r, \n, \t) and non-printable chars
    sanitized = "".join(char for char in sanitized if char.isprintable())

    # Replace any multiple spaces with single space
    sanitized = re.sub(r"\s+", " ", sanitized)

    # Remove consecutive dots
    sanitized = re.sub(r"\.\.+", ".", sanitized)

    # Trim leading/trailing spaces and dots
    sanitized = sanitized.strip(" .")

    # Limit length (preserving extension if possible)
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].strip()

    return sanitized


def ensure_secure_dir(path: Path) -> None:
    """
    Ensure a directory exists with secure permissions (owner only).
    Works on both Unix (chmod 700) and Windows (icacls restricted).
    """
    path.mkdir(parents=True, exist_ok=True)

    if sys.platform != "win32":
        try:
            path.chmod(0o700)
        except Exception:
            pass
    else:
        # Windows: Best effort to restrict permissions to current user using icacls
        try:
            username = os.environ.get("USERNAME")
            if username:
                # /inheritance:r - remove inherited ACEs
                # /grant:r %USERNAME%:F - grant full control to current user
                # (OI)(CI) - Object Inherit, Container Inherit
                subprocess.run(
                    [
                        "icacls",
                        str(path),
                        "/inheritance:r",
                        "/grant:r",
                        f"{username}:(OI)(CI)F",
                    ],
                    capture_output=True,
                    check=False,
                    timeout=10,
                )
        except Exception:
            pass


def proportional_chunk_trim(text: str, target_length: int, num_chunks: int) -> str:
    """
    Trim text to a target length by dividing it into equal chunks and applying a reduction factor.
    """
    if not text:
        return ""

    # Calculate the size of each chunk
    chunk_size = len(text) // num_chunks if num_chunks > 0 else len(text)
    if chunk_size == 0:
        return text[:target_length]

    # Divide the text into equal chunks
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Calculate the reduction factor
    total_length = sum(len(chunk) for chunk in chunks)
    reduction_factor = min(1.0, target_length / total_length) if total_length > 0 else 0

    # Apply the reduction factor to each chunk
    trimmed_text = ""
    current_length = 0
    for chunk in chunks:
        allowed_length = int(len(chunk) * reduction_factor)
        trimmed_chunk = chunk[:allowed_length]
        trimmed_text += trimmed_chunk
        current_length += len(trimmed_chunk)

        # Stop if we've reached the target length
        if current_length >= target_length:
            break

    return trimmed_text
