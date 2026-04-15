"""
Directory path management for renAI.

This module provides functions for managing application directories
with secure permissions.
"""

from renai.config.constants import CACHE_DIR, DEBUG_DIR
from renai.utils import ensure_secure_dir


def initialize_directories() -> None:
    """Ensure necessary directories exist with secure permissions."""
    ensure_secure_dir(CACHE_DIR)
    ensure_secure_dir(DEBUG_DIR)
