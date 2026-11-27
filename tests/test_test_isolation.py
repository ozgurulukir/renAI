"""Guard tests verifying pytest isolates app data, cache, and undo journal."""

from pathlib import Path

from platformdirs import user_data_dir

import renai.config
from renai.config.constants import APP_AUTHOR, APP_NAME
from renai.file_ops.operations import UNDO_JOURNAL_NAME, record_undo_entry


def test_undo_journal_is_isolated_from_real_user_data_dir(tmp_path: Path):
    """Writing to the undo journal during tests never touches the real user data dir."""
    real_journal = Path(user_data_dir(APP_NAME, APP_AUTHOR)) / "Cache" / UNDO_JOURNAL_NAME
    before_bytes = real_journal.read_bytes() if real_journal.exists() else None

    assert renai.config.CACHE_DIR.is_relative_to(tmp_path)

    src = tmp_path / "source.pdf"
    dst = tmp_path / "dest.pdf"
    src.write_text("dummy", encoding="utf-8")
    record_undo_entry(src, dst, operation_type="move")

    isolated_journal = renai.config.CACHE_DIR / UNDO_JOURNAL_NAME
    assert isolated_journal.exists()

    after_bytes = real_journal.read_bytes() if real_journal.exists() else None
    assert after_bytes == before_bytes
