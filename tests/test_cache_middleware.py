from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from renai.cache.middleware import with_metadata_cache


class DummyMetadata(BaseModel):
    title: str

class DummyProcessor:
    def __init__(self):
        self.call_count = 0

    @with_metadata_cache()
    async def extract(
        self,
        file_path: Path,
        cache_path: Path,
        model_class: type[BaseModel],
        use_vision_cache: bool,
        fallback_mode: str,
        rename_images: bool,
        update_metadata: bool,
        add_exif_date: bool,
        file_hash: str | None = None,
    ):
        self.call_count += 1
        return DummyMetadata(title="Extracted Title"), False

@pytest.mark.asyncio
@patch("renai.cache.middleware.read_metadata_cache", new_callable=AsyncMock)
@patch("renai.cache.middleware.write_metadata_cache", new_callable=AsyncMock)
async def test_middleware_cache_hit(mock_write, mock_read):
    # Arrange
    mock_read.return_value = DummyMetadata(title="Cached Title")
    processor = DummyProcessor()

    # Act
    result, is_vision = await processor.extract(
        file_path=Path("test.pdf"),
        cache_path=Path("/tmp/cache.json"),
        model_class=DummyMetadata,
        use_vision_cache=False,
        fallback_mode="none",
        rename_images=False,
        update_metadata=False,
        add_exif_date=False,
        file_hash="hash123",
    )

    # Assert
    assert result.title == "Cached Title"
    assert is_vision is False
    assert processor.call_count == 0  # Domain logic bypassed
    mock_read.assert_called_once()
    mock_write.assert_not_called()

@pytest.mark.asyncio
@patch("renai.cache.middleware.read_metadata_cache", new_callable=AsyncMock)
@patch("renai.cache.middleware.write_metadata_cache", new_callable=AsyncMock)
async def test_middleware_cache_miss_and_write(mock_write, mock_read):
    # Arrange
    mock_read.return_value = None  # Cache miss
    processor = DummyProcessor()

    # Act
    result, is_vision = await processor.extract(
        file_path=Path("test.pdf"),
        cache_path=Path("/tmp/cache.json"),
        model_class=DummyMetadata,
        use_vision_cache=False,
        fallback_mode="none",
        rename_images=False,
        update_metadata=False,
        add_exif_date=False,
        file_hash="hash123",
    )

    # Assert
    assert result.title == "Extracted Title"
    assert is_vision is False
    assert processor.call_count == 1  # Domain logic executed
    mock_read.assert_called_once()
    mock_write.assert_called_once_with(
        cache_path=Path("/tmp/cache.json"),
        metadata=result,
        schema_class=DummyMetadata,
        use_vision=False,
        fallback_mode="none",
    )
