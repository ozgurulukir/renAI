import hashlib
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Pre-compiled regular expressions for filename sanitization
RE_INVALID_CHARS = re.compile(r'[<>:"/\\|?*]')
RE_WHITESPACE = re.compile(r"\s+")
RE_CONSECUTIVE_DOTS = re.compile(r"\.\.+")


def get_file_hash(file_path: Path) -> str:
    """Calculate the SHA256 hash of a file to uniquely identify its content."""
    with file_path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def _truncate_to_utf8_budget(text: str, max_bytes: int) -> str:
    """Truncate ``text`` to a UTF-8 BYTE budget, stepping back to a valid boundary.

    Filesystems cap a filename component at 255 bytes (not characters), so a
    CJK/emoji title truncated in characters can still exceed the limit. The
    cut is made in bytes and then walked back up to 3 bytes to land on a
    complete UTF-8 character boundary.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    clipped = encoded[:max_bytes]
    while clipped:
        try:
            return clipped.decode("utf-8")
        except UnicodeDecodeError:
            clipped = clipped[:-1]
    return ""


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """
    Replace invalid characters with spaces and remove all control characters.
    Blocks Windows-reserved device names and limits length to stay within
    filesystem path limits.

    ``max_length`` is interpreted as a UTF-8 BYTE budget (matching the 255-byte
    filename component limit on Linux/macOS), and truncation steps back to a
    valid character boundary so multi-byte titles never end mid-character.
    """
    # Ensure max_length is an int for type safety
    max_val: int = int(max_length)
    # Replace invalid characters for Windows paths: < > : " / \ | ? *
    sanitized = RE_INVALID_CHARS.sub(" ", filename)

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

    # Block Windows reserved device names (e.g. CON, PRN, AUX, NUL, COM1-9, LPT1-9)
    stem = sanitized.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED_NAMES:
        sanitized = f"unnamed_{sanitized}"

    # Limit length in UTF-8 BYTES (filesystem component limit), stepping back
    # to a character boundary so CJK/emoji names never end mid-character.
    if len(sanitized.encode("utf-8")) > max_val:
        sanitized = _truncate_to_utf8_budget(sanitized, max_val).strip(" .\"'")

    # Byte truncation followed by stripping can theoretically yield "" (or a
    # lone dot); fall back rather than returning an unusable name.
    if not sanitized or sanitized in {".", ".."}:
        sanitized = "unnamed"

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
            logger.debug(f"Failed to set secure permissions on {path}: {e}")
        return path

    # Windows: Best effort to restrict permissions to current user using icacls
    username = os.environ.get("USERNAME")
    if not username:
        return path

    # Security: Validate username to prevent command injection
    if not re.match(r"^\w[\w\s.@\\-]*$", username):
        logger.warning(f"Invalid username for secure permissions: {username}")
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
        logger.debug(f"Failed to set secure permissions on {path}: {e}")

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
        logger.error(f"Error reading {file_path}: {e}")
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
