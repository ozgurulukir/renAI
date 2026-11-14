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


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """
    Replace invalid characters and remove all control characters.
    Limits length to stay within filesystem path limits.
    """
    # Ensure max_length is an int for type safety
    max_val: int = int(max_length)
    # Remove invalid characters for Windows paths: < > : " / \ | ? *
    sanitized = RE_INVALID_CHARS.sub("", filename)

    # Remove ALL control characters (including \r, \n, \t) and non-printable chars
    sanitized = "".join(char for char in sanitized if char.isprintable())

    # Replace any multiple spaces with single space
    sanitized = RE_WHITESPACE.sub(" ", sanitized)

    # Remove consecutive dots
    sanitized = RE_CONSECUTIVE_DOTS.sub(".", sanitized)

    # Trim leading/trailing spaces, dots, and quotes
    sanitized = sanitized.strip(" .\"'")

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


def head_tail_trim(text: str, target_length: int, head_ratio: float = 0.7) -> str:
    """
    Trim text to target_length by preserving the head and tail segments.

    Head captures title, author, publisher (beginning of document).
    Tail captures references, ISBN, colophon (end of document).
    The middle (body text, least useful for metadata) is discarded.

    Args:
        text: Input text to trim.
        target_length: Maximum character count for the result.
        head_ratio: Fraction of target_length allocated to the head (0.3-0.9).

    Returns:
        Trimmed text with head + separator + tail, or original if short enough.
    """
    if not text or target_length <= 0:
        return ""
    if not 0.0 < head_ratio < 1.0:
        raise ValueError(f"head_ratio must be between 0 and 1 (exclusive), got {head_ratio}")
    if len(text) <= target_length:
        return text

    head_len = int(target_length * head_ratio)
    tail_len = target_length - head_len

    MIN_TARGET_WITH_SEPARATOR = 3

    if target_length < MIN_TARGET_WITH_SEPARATOR:
        return text[:target_length]

    separator = "\n...\n"
    sep_len = len(separator)
    adjusted_head = head_len - (sep_len // 2)
    adjusted_tail = tail_len - (sep_len - sep_len // 2)

    if adjusted_head <= 0:
        return text[:target_length]
    if adjusted_tail <= 0:
        return text[:target_length]

    return text[:adjusted_head] + separator + text[-adjusted_tail:]
