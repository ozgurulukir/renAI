from pathlib import Path

import pytest
from pydantic import BaseModel

from renai.cache.middleware import with_metadata_cache


class DummyMetadata(BaseModel):
    title: str

class DummyConfig:
    def __init__(self):
        self.provider = "test_provider"
        self.model = "test_model"

class DummyProcessor:
    def __init__(self):
        self.config = DummyConfig()
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
async def test_middleware_cache_miss_and_hit(tmp_path: Path):
    processor = DummyProcessor()
    test_cache_path = tmp_path / "test.json"

    # First call - cache miss
    result, is_vision = await processor.extract(
        file_path=Path("test.pdf"),
        cache_path=test_cache_path,
        model_class=DummyMetadata,
        use_vision_cache=False,
        fallback_mode="none",
        rename_images=False,
        update_metadata=False,
        add_exif_date=False,
        file_hash="hash123",
    )

    assert result.title == "Extracted Title"
    assert is_vision is False
    assert processor.call_count == 1

    # Second call - cache hit
    result2, is_vision2 = await processor.extract(
        file_path=Path("test.pdf"),
        cache_path=test_cache_path,
        model_class=DummyMetadata,
        use_vision_cache=False,
        fallback_mode="none",
        rename_images=False,
        update_metadata=False,
        add_exif_date=False,
        file_hash="hash123",
    )

    assert result2.title == "Extracted Title"
    assert is_vision2 is False
    assert processor.call_count == 1  # Should not increase

@pytest.mark.asyncio
async def test_middleware_bypass_no_config(tmp_path: Path):
    class NoConfigProcessor:
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

    processor = NoConfigProcessor()
    test_cache_path = tmp_path / "test2.json"

    # Call multiple times, should always miss
    for _ in range(3):
        _result, _ = await processor.extract(
            file_path=Path("test.pdf"),
            cache_path=test_cache_path,
            model_class=DummyMetadata,
            use_vision_cache=False,
            fallback_mode="none",
            rename_images=False,
            update_metadata=False,
            add_exif_date=False,
            file_hash="hash123",
        )

    assert processor.call_count == 3
