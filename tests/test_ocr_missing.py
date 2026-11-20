from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from renai.engines.render import create_page_renderer


@pytest.mark.asyncio
async def test_mutool_seam_fallback_process_error():
    """Test the mutool seam backend properly catches and handles errors when mutool fails."""
    process_mock = AsyncMock()
    process_mock.communicate.return_value = (b"", b"Error message")
    process_mock.returncode = 1

    with (
        patch(
            "renai.engines.render.asyncio.create_subprocess_exec",
            AsyncMock(return_value=process_mock),
        ),
        patch("renai.engines.render.logger.error") as mock_log_error,
    ):
        renderer = create_page_renderer(use_mutool=True, mutool_path="/usr/bin/mutool")
        results = await renderer.render_pages(Path("dummy.pdf"), [0, 1], 300)
        assert results == {}
        assert mock_log_error.call_count == 1
        assert "mutool bulk error" in mock_log_error.call_args[0][0]
