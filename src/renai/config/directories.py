"""
Directory path management for renAI.

This module provides functions for managing application directories
with secure permissions. All directory creation is deferred here
to avoid side effects at import time.
"""

from renai.config.constants import APP_DATA_DIR, CACHE_DIR, DEBUG_DIR
from renai.utils import ensure_secure_dir


def initialize_directories() -> None:
    """Ensure necessary directories exist with secure permissions.

    Creates:
        - APP_DATA_DIR  (base data directory, e.g. LocalAppData/renAI)
        - CACHE_DIR     (APP_DATA_DIR/Cache)
        - DEBUG_DIR     (APP_DATA_DIR/ocr_debug)
    """
    ensure_secure_dir(APP_DATA_DIR)
    ensure_secure_dir(CACHE_DIR)
    ensure_secure_dir(DEBUG_DIR)
