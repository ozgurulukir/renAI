import hashlib
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

# Pre-compiled regular expressions for filename sanitization
RE_INVALID_CHARS = re.compile(r'[<>:"/\\|?*]')
RE_WHITESPACE = re.compile(r"\s+")
RE_CONSECUTIVE_DOTS = re.compile(r"\.\.+")


def get_file_hash(file_path: Path) -> str:
    """Calculate the SHA256 hash of a file to uniquely identify its content."""
    with file_path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def generate_cache_key(*args: str | None) -> str:
    """Generate a unique cache key from multiple string components.

    Args:
        *args: Variable number of string components to combine

    Returns:
        A hex string hash of the combined components
    """
    combined = "|".join(str(arg) for arg in args if arg)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def get_provider_model_key(provider: str, model: str) -> str:
    """Generate a cache key for provider-model pairing.

    Args:
        provider: LLM provider name (e.g., 'deepinfra', 'openai', 'custom')
        model: Model name (e.g., 'gpt-4o-mini', 'llama3')

    Returns:
        A hex string hash combining provider and model
    """
    return generate_cache_key(provider, model)


def sanitize_filename(filename: str, max_length: int | None = None) -> str:
    """
    Replace invalid characters and remove all control characters.
    Limits length to stay within Windows path limits.
    """
    if max_length is None:
        try:
            from renai.config import app_settings

            max_length = app_settings.max_filename_length or 200
        except (ImportError, AttributeError, NameError):
            max_length = 200  # Fallback

    # Ensure max_length is an int for type safety
    max_val: int = int(max_length) if max_length is not None else 200
    # Remove invalid characters for Windows paths: < > : " / \ | ? *
    sanitized = RE_INVALID_CHARS.sub("", filename)

    # Remove ALL control characters (including \r, \n, \t) and non-printable chars
    sanitized = "".join(char for char in sanitized if char.isprintable())

    # Replace any multiple spaces with single space
    sanitized = RE_WHITESPACE.sub(" ", sanitized)

    # Remove consecutive dots
    sanitized = RE_CONSECUTIVE_DOTS.sub(".", sanitized)

    # Trim leading/trailing spaces and dots
    sanitized = sanitized.strip(" .")

    # Prevent empty or traversal paths
    if not sanitized or sanitized in {".", ".."}:
        sanitized = "unnamed"

    # Limit length (preserving extension if possible)
    if len(sanitized) > max_val:
        sanitized = sanitized[:max_val].strip()

    return sanitized


def ensure_secure_dir(path: Path) -> Path:
    """
    Ensure a directory exists with secure permissions (owner only).
    Works on both Unix (chmod 700) and Windows (icacls restricted).
    """
    path.mkdir(parents=True, exist_ok=True)

    if sys.platform != "win32":
        try:
            path.chmod(0o700)
        except Exception as e:
            logging.debug(f"Failed to set secure permissions on {path}: {e}")
        return path

    # Windows: Best effort to restrict permissions to current user using icacls
    username = os.environ.get("USERNAME")
    if not username:
        return path

    # Security: Validate username to prevent command injection
    if not re.match(r"^\w[\w\s.@\\-]*$", username):
        logging.warning(f"Invalid username for secure permissions: {username}")
        return path

    try:
        # /inheritance:r - remove inherited ACEs
        # /grant:r %USERNAME%:F - grant full control to current user
        # (OI)(CI) - Object Inherit, Container Inherit
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

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
            startupinfo=startupinfo,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as e:
        logging.debug(f"Failed to set secure permissions on {path}: {e}")

    return path


def read_text_limited(file_path: Path, max_chars: int) -> str:
    """
    Safely read up to max_chars characters from a text file.
    Uses a stream to avoid loading the entire file into memory.
    """
    try:
        with file_path.open(encoding="utf-8", errors="ignore", newline=None) as f:
            return f.read(max_chars)
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return ""


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


def handle_clear_cache() -> None:
    """Clear extracted text cache, metadata, and OCR debug files."""
    from renai.cache.manager import clear_all_cache
    from renai.config import CACHE_DIR

    ensure_secure_dir(CACHE_DIR)
    results = clear_all_cache(CACHE_DIR)

    total = sum(results.values())
    if total > 0:
        details = ", ".join([f"{count} {ctype}" for ctype, count in results.items() if count > 0])
        logging.info(f"Cache cleared: {total} items removed ({details})")
    else:
        logging.info("Cache is already empty. Nothing to clear.")
