from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from renai.cache.middleware import with_metadata_cache


class DummyMetadata(BaseModel):
    title: str


class DummyProcessor:
    """Processor exposing provider/model attributes like BookProcessor."""

    def __init__(self):
        self.provider = "test_provider"
        self.model = "test_model"
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
    ) -> tuple[DummyMetadata, bool]:
        self.call_count += 1
        return DummyMetadata(title="Extracted Title"), False


async def _call(
    processor: DummyProcessor,
    cache_path: Path,
    update_metadata: bool = False,
    use_vision_cache: bool = False,
):
    return await processor.extract(
        file_path=Path("test.pdf"),
        cache_path=cache_path,
        model_class=DummyMetadata,
        use_vision_cache=use_vision_cache,
        fallback_mode="none",
        rename_images=False,
        update_metadata=update_metadata,
        add_exif_date=False,
        file_hash="hash123",
    )


@pytest.mark.asyncio
async def test_middleware_cache_miss_and_hit(tmp_path: Path):
    processor = DummyProcessor()
    test_cache_path = tmp_path / "metadata" / "test.json"

    # First call - cache miss, result written to cache
    result, is_vision = await _call(processor, test_cache_path)
    assert result.title == "Extracted Title"
    assert is_vision is False
    assert processor.call_count == 1

    # Second call - cache hit, engine not invoked again
    result2, is_vision2 = await _call(processor, test_cache_path)
    assert result2.title == "Extracted Title"
    assert is_vision2 is False
    assert processor.call_count == 1


@pytest.mark.asyncio
async def test_middleware_update_metadata_skips_read(tmp_path: Path):
    processor = DummyProcessor()
    test_cache_path = tmp_path / "metadata" / "refresh.json"

    await _call(processor, test_cache_path)
    assert processor.call_count == 1

    # update_metadata=True forces re-extraction (cache read skipped)
    await _call(processor, test_cache_path, update_metadata=True)
    assert processor.call_count == 2

    # Without update_metadata the refreshed entry is served from cache
    await _call(processor, test_cache_path)
    assert processor.call_count == 2


@pytest.mark.asyncio
async def test_middleware_missing_provider_model_uses_unknown(tmp_path: Path):
    class NoProviderProcessor:
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
        ) -> tuple[DummyMetadata, bool]:
            self.call_count += 1
            return DummyMetadata(title="Extracted Title"), False

    processor = NoProviderProcessor()
    test_cache_path = tmp_path / "metadata" / "unknown.json"

    # No provider/model attributes: still caches under the "unknown" key
    for _ in range(3):
        await _call(processor, test_cache_path)

    assert processor.call_count == 1


class VisionDummyProcessor:
    """Processor with separate text and vision provider/model attributes."""

    def __init__(self):
        self.provider = "text_provider"
        self.model = "text_model"
        self.vision_provider = "vision_provider"
        self.vision_model = "vision_model"
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
    ) -> tuple[DummyMetadata, bool]:
        self.call_count += 1
        return DummyMetadata(title="Extracted Title"), use_vision_cache


@pytest.mark.asyncio
async def test_middleware_vision_and_text_caches_are_separate(tmp_path: Path):
    processor = VisionDummyProcessor()
    text_cache = tmp_path / "metadata" / "text.json"
    vision_cache = tmp_path / "metadata" / "vision.json"

    # Vision cache miss + hit
    _result, is_vision = await _call(processor, vision_cache, use_vision_cache=True)
    assert is_vision is True
    assert processor.call_count == 1

    _result, is_vision = await _call(processor, vision_cache, use_vision_cache=True)
    assert is_vision is True
    assert processor.call_count == 1

    # Text cache is a different namespace -> miss
    _result, is_vision = await _call(processor, text_cache)
    assert is_vision is False
    assert processor.call_count == 2


@pytest.mark.asyncio
async def test_middleware_custom_resolvers(tmp_path: Path):
    """Resolvers can be overridden to key caches by custom attributes."""

    class CustomProcessor:
        def __init__(self):
            self.api_config = {"provider": "custom_provider", "model": "custom_model"}
            self.call_count = 0

        @with_metadata_cache(
            provider_resolver=lambda self, use_vision: self.api_config["provider"],
            model_resolver=lambda self, use_vision: self.api_config["model"],
        )
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
        ) -> tuple[DummyMetadata, bool]:
            self.call_count += 1
            return DummyMetadata(title="Extracted Title"), False

    processor: Any = CustomProcessor()
    test_cache_path = tmp_path / "metadata" / "custom.json"

    await _call(processor, test_cache_path)
    await _call(processor, test_cache_path)

    assert processor.call_count == 1
