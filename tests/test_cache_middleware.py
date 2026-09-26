from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from renai.cache.middleware import MetadataRequest, with_metadata_cache


class DummyMetadata(BaseModel):
    title: str


def _make_request(
    cache_path: Path,
    model_class: type[BaseModel],
    update_metadata: bool = False,
    use_vision_cache: bool = False,
    **extra: Any,
) -> MetadataRequest:
    base: dict[str, Any] = {
        "file_path": Path("test.pdf"),
        "cache_path": cache_path,
        "model_class": model_class,
        "use_vision_cache": use_vision_cache,
        "fallback_mode": "none",
        "rename_images": False,
        "update_metadata": update_metadata,
        "add_exif_date": False,
        "file_hash": "hash123",
    }
    base.update(extra)
    return MetadataRequest(**base)


class DummyProcessor:
    """Processor exposing provider/model attributes like BookProcessor."""

    def __init__(self):
        self.provider = "test_provider"
        self.model = "test_model"
        self.call_count = 0

    @with_metadata_cache()
    async def extract(self, request: MetadataRequest) -> tuple[DummyMetadata, bool]:
        self.call_count += 1
        return DummyMetadata(title="Extracted Title"), False


@pytest.mark.asyncio
async def test_middleware_cache_miss_and_hit(tmp_path: Path):
    processor = DummyProcessor()
    test_cache_path = tmp_path / "metadata" / "test.json"

    # First call - cache miss, result written to cache
    result, is_vision = await processor.extract(_make_request(test_cache_path, DummyMetadata))
    assert result.title == "Extracted Title"
    assert is_vision is False
    assert processor.call_count == 1

    # Second call - cache hit, engine not invoked again
    result2, is_vision2 = await processor.extract(_make_request(test_cache_path, DummyMetadata))
    assert result2.title == "Extracted Title"
    assert is_vision2 is False
    assert processor.call_count == 1


@pytest.mark.asyncio
async def test_middleware_update_metadata_skips_read(tmp_path: Path):
    processor = DummyProcessor()
    test_cache_path = tmp_path / "metadata" / "refresh.json"

    await processor.extract(_make_request(test_cache_path, DummyMetadata))
    assert processor.call_count == 1

    # update_metadata=True forces re-extraction (cache read skipped)
    await processor.extract(_make_request(test_cache_path, DummyMetadata, update_metadata=True))
    assert processor.call_count == 2

    # Without update_metadata the refreshed entry is served from cache
    await processor.extract(_make_request(test_cache_path, DummyMetadata))
    assert processor.call_count == 2


@pytest.mark.asyncio
async def test_middleware_missing_provider_model_uses_unknown(tmp_path: Path):
    class NoProviderProcessor:
        def __init__(self):
            self.call_count = 0

        @with_metadata_cache()
        async def extract(self, request: MetadataRequest) -> tuple[DummyMetadata, bool]:
            self.call_count += 1
            return DummyMetadata(title="Extracted Title"), False

    processor = NoProviderProcessor()
    test_cache_path = tmp_path / "metadata" / "unknown.json"

    # No provider/model attributes: still caches under the "unknown" key
    for _ in range(3):
        await processor.extract(_make_request(test_cache_path, DummyMetadata))

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
    async def extract(self, request: MetadataRequest) -> tuple[DummyMetadata, bool]:
        self.call_count += 1
        return DummyMetadata(title="Extracted Title"), request.use_vision_cache


@pytest.mark.asyncio
async def test_middleware_vision_and_text_caches_are_separate(tmp_path: Path):
    processor = VisionDummyProcessor()
    text_cache = tmp_path / "metadata" / "text.json"
    vision_cache = tmp_path / "metadata" / "vision.json"

    # Vision cache miss + hit
    _result, is_vision = await processor.extract(
        _make_request(vision_cache, DummyMetadata, use_vision_cache=True)
    )
    assert is_vision is True
    assert processor.call_count == 1

    _result, is_vision = await processor.extract(
        _make_request(vision_cache, DummyMetadata, use_vision_cache=True)
    )
    assert is_vision is True
    assert processor.call_count == 1

    # Text cache is a different namespace -> miss
    _result, is_vision = await processor.extract(_make_request(text_cache, DummyMetadata))
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
        async def extract(self, request: MetadataRequest) -> tuple[DummyMetadata, bool]:
            self.call_count += 1
            return DummyMetadata(title="Extracted Title"), False

    processor: Any = CustomProcessor()
    test_cache_path = tmp_path / "metadata" / "custom.json"

    await processor.extract(_make_request(test_cache_path, DummyMetadata))
    await processor.extract(_make_request(test_cache_path, DummyMetadata))

    assert processor.call_count == 1


@pytest.mark.asyncio
async def test_middleware_wrong_contract_raises_at_decoration_time():
    """Regression: a method that does not accept ``(self, request: MetadataRequest)``
    must fail loudly at decoration time, never silently at call time."""

    with pytest.raises(ValueError):

        class BadContractProcessor:
            # Missing the explicit MetadataRequest parameter entirely.
            @with_metadata_cache()
            async def extract(self, file_path: Path) -> tuple[DummyMetadata, bool]:
                return DummyMetadata(title="x"), False

        BadContractProcessor()


@pytest.mark.asyncio
async def test_middleware_wrong_request_annotation_raises_at_decoration_time():
    with pytest.raises(ValueError):

        class WrongAnnotationProcessor:
            @with_metadata_cache()
            async def extract(self, not_request: Any) -> tuple[DummyMetadata, bool]:
                return DummyMetadata(title="x"), False

        WrongAnnotationProcessor()


@pytest.mark.asyncio
async def test_middleware_decorated_method_can_be_invoked_with_request():
    """A correctly-typed method accepts a MetadataRequest and behaves normally."""
    processor = DummyProcessor()
    result, is_vision = await processor.extract(_make_request(Path("out.json"), DummyMetadata))
    assert result.title == "Extracted Title"
    assert is_vision is False
