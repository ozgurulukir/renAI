from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from renai.engines.ocr import OCREngine


@pytest.mark.asyncio
async def test_ocr_extract_text_mutool_fallback_process_error():
    """Test OCR extraction fallback properly catches and handles errors when mutool fails."""
    engine = OCREngine(use_mutool=False)
    engine.use_mutool = True
    engine.mutool_path = "/usr/bin/mutool"

    with patch("renai.engines.ocr.asyncio.create_subprocess_exec") as mock_exec:
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"Error message")
        mock_process.returncode = 1
        mock_exec.return_value = mock_process

        results = await engine._bulk_extract_images(Path("dummy.pdf"), [0, 1], 300)
        assert results == {}
