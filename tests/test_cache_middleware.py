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
    provider: str = "test_provider",
    model: str = "test_model",
    vision_provider: str = "vision_provider",
    vision_model: str = "vision_model",
    **extra: Any,
) -> MetadataRequest:
    base: dict[str, Any] = {
        "file_path": Path("test.pdf"),
        "cache_path": cache_path,
        "model_class": model_class,
        "provider": provider,
        "model": model,
        "vision_provider": vision_provider,
        "vision_model": vision_model,
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
    """Processor with provider/model attributes like BookProcessor (unused for keying)."""

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
async def test_middleware_identities_come_from_request_not_instance(tmp_path: Path):
    class NoProviderProcessor:
        """No provider/model attributes at all — the request carries the identities."""

        def __init__(self):
            self.call_count = 0

        @with_metadata_cache()
        async def extract(self, request: MetadataRequest) -> tuple[DummyMetadata, bool]:
            self.call_count += 1
            return DummyMetadata(title="Extracted Title"), False

    processor = NoProviderProcessor()
    test_cache_path = tmp_path / "metadata" / "unknown.json"

    # No provider/model instance attributes: identities ride the request and
    # the cache still round-trips.
    for _ in range(3):
        await processor.extract(
            _make_request(test_cache_path, DummyMetadata, provider="unknown", model="unknown")
        )

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
async def test_middleware_identity_mismatch_reextracts(tmp_path: Path):
    """A row produced by a different engine identity than the request carries is a miss.

    Instance attributes are deliberately misleading: the middleware must key and
    validate purely on the request's identity fields.
    """

    class MisleadingProcessor:
        def __init__(self):
            self.provider = "totally_wrong"
            self.model = "also_wrong"
            self.call_count = 0

        @with_metadata_cache()
        async def extract(self, request: MetadataRequest) -> tuple[DummyMetadata, bool]:
            self.call_count += 1
            return DummyMetadata(title="Extracted Title"), False

    processor: Any = MisleadingProcessor()
    test_cache_path = tmp_path / "metadata" / "custom.json"

    await processor.extract(_make_request(test_cache_path, DummyMetadata))
    await processor.extract(_make_request(test_cache_path, DummyMetadata))
    assert processor.call_count == 1

    # A different producing identity in the request invalidates the row.
    await processor.extract(_make_request(test_cache_path, DummyMetadata, model="other_model"))
    assert processor.call_count == 2


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
async def test_middleware_persists_source_stable_across_cache_hits(tmp_path: Path):
    """Regression: a text-path extraction under a vision-cache request must keep
    reporting is_vision_doc=False on cache hits.

    Previously the hit path reconstructed the flag from use_vision_cache, so a
    text-extractable PDF run with --fallback vision got is_vision_doc=False on
    the cold run but True once cached -- same file, same flags, different
    filename template depending on cache warmth.
    """

    class TextPathProcessor(VisionDummyProcessor):
        """Vision-cache request, but the text path succeeds (returns False)."""

        @with_metadata_cache()
        async def extract(self, request: MetadataRequest) -> tuple[DummyMetadata, bool]:
            self.call_count += 1
            return DummyMetadata(title="From Text Path"), False

    processor = TextPathProcessor()
    vision_cache = tmp_path / "metadata" / "text_success_under_vision.json"
    request = _make_request(vision_cache, DummyMetadata, use_vision_cache=True)

    _result, cold_is_vision = await processor.extract(request)
    assert cold_is_vision is False
    assert processor.call_count == 1

    # Warm run: the persisted marker (False) must win over the flag-based
    # reconstruction (which would say True).
    _result, warm_is_vision = await processor.extract(request)
    assert warm_is_vision is False
    assert processor.call_count == 1


@pytest.mark.asyncio
async def test_middleware_legacy_record_without_marker_reconstructs(tmp_path: Path):
    """Records written before the marker existed keep the old reconstruction."""
    processor = VisionDummyProcessor()
    vision_cache = tmp_path / "metadata" / "legacy.json"

    # Write a record the old way: no is_vision_doc marker.
    from renai.cache.manager import write_metadata_cache

    write_metadata_cache(
        vision_cache,
        DummyMetadata(title="Legacy"),
        "vision_provider",
        "vision_model",
    )

    request = _make_request(vision_cache, DummyMetadata, use_vision_cache=True)
    result, is_vision = await processor.extract(request)
    assert result is not None and result.title == "Legacy"
    # Legacy reconstruction: use_vision_cache=True and not an image request.
    assert is_vision is True
    assert processor.call_count == 0


@pytest.mark.asyncio
async def test_middleware_rule_signature_mismatch_reextracts(tmp_path: Path):
    """A cached record from a different category universe is a miss, not a hit.

    The mismatched row is not deleted immediately; the re-extraction overwrite
    replaces it atomically.
    """
    processor = DummyProcessor()
    test_cache_path = tmp_path / "metadata" / "rule_scoped.json"

    # Seed the cache the old way (global categories -> empty signature).
    from renai.cache.manager import write_metadata_cache

    write_metadata_cache(
        test_cache_path,
        DummyMetadata(title="Global Universe"),
        "test_provider",
        "test_model",
        is_vision_doc=False,
    )

    request = _make_request(test_cache_path, DummyMetadata, rule_signature="rule-sig-1")
    result, _ = await processor.extract(request)
    assert processor.call_count == 1  # mismatch -> re-extraction ran
    assert result is not None and result.title == "Extracted Title"

    # The rewrite carries the rule signature, so the next rule run is a hit.
    _result, _ = await processor.extract(request)
    assert processor.call_count == 1

    # A global run (no rule) now mismatches the rule-scoped row -> miss.
    global_request = _make_request(test_cache_path, DummyMetadata)
    await processor.extract(global_request)
    assert processor.call_count == 2


@pytest.mark.asyncio
async def test_middleware_rule_signature_match_serves_cache(tmp_path: Path):
    from renai.cache.manager import read_metadata_cache_with_source, write_metadata_cache

    processor = DummyProcessor()
    test_cache_path = tmp_path / "metadata" / "rule_match.json"

    write_metadata_cache(
        test_cache_path,
        DummyMetadata(title="Rule Universe"),
        "test_provider",
        "test_model",
        is_vision_doc=False,
        rule_signature="rule-sig-1",
    )

    request = _make_request(test_cache_path, DummyMetadata, rule_signature="rule-sig-1")
    result, _ = await processor.extract(request)
    assert processor.call_count == 0
    assert result is not None and result.title == "Rule Universe"

    # Sanity: the stored source really carries the signature.
    _meta, source = read_metadata_cache_with_source(
        test_cache_path, "test_provider", "test_model", model_class=DummyMetadata
    )
    assert source is not None and source.rule_signature == "rule-sig-1"


@pytest.mark.asyncio
async def test_middleware_decorated_method_can_be_invoked_with_request():
    """A correctly-typed method accepts a MetadataRequest and behaves normally."""
    processor = DummyProcessor()
    result, is_vision = await processor.extract(_make_request(Path("out.json"), DummyMetadata))
    assert result.title == "Extracted Title"
    assert is_vision is False


@pytest.mark.asyncio
async def test_text_produced_record_survives_vision_model_change(tmp_path: Path):
    """Acceptance (#193): under a vision-cache request, a record produced by the
    TEXT LLM stays a hit when only the vision model changes.

    The key is identity-free and the row validates against the producing
    (text) engine, so the vision-model swap must not touch it.
    """
    vision_cache = tmp_path / "metadata" / "text_success_under_vision.json"

    class TextPathProcessor(VisionDummyProcessor):
        @with_metadata_cache()
        async def extract(self, request: MetadataRequest) -> tuple[DummyMetadata, bool]:
            self.call_count += 1
            return DummyMetadata(title="From Text Path"), False

    processor = TextPathProcessor()
    # Cold run: text path succeeds (is_vision_fallback=False) under a
    # vision-cache request keyed for vision_model=v1.
    request_v1 = _make_request(
        vision_cache, DummyMetadata, use_vision_cache=True, vision_model="v1"
    )
    _result, cold_is_vision = await processor.extract(request_v1)
    assert cold_is_vision is False
    assert processor.call_count == 1

    # Vision model changes v1 -> v2: same key (identity-free), text-produced
    # row still validates against the unchanged text engine.
    request_v2 = _make_request(
        vision_cache,
        DummyMetadata,
        use_vision_cache=True,
        vision_model="v2",
    )
    _result, warm_is_vision = await processor.extract(request_v2)
    assert warm_is_vision is False
    assert processor.call_count == 1


@pytest.mark.asyncio
async def test_text_model_change_invalidates_text_produced_record(tmp_path: Path):
    """Acceptance (#193): changing the text model re-extracts a text-produced row."""
    processor = VisionDummyProcessor()
    vision_cache = tmp_path / "metadata" / "text_model_change.json"

    class TextPathProcessor(VisionDummyProcessor):
        @with_metadata_cache()
        async def extract(self, request: MetadataRequest) -> tuple[DummyMetadata, bool]:
            self.call_count += 1
            return DummyMetadata(title="From Text Path"), False

    processor = TextPathProcessor()
    request_old = _make_request(vision_cache, DummyMetadata, use_vision_cache=True, model="text_m1")
    await processor.extract(request_old)
    assert processor.call_count == 1

    # Same vision model, new text model: the text-produced row is stale.
    request_new = _make_request(vision_cache, DummyMetadata, use_vision_cache=True, model="text_m2")
    await processor.extract(request_new)
    assert processor.call_count == 2

    # And the rewritten row is keyed to the new text engine.
    await processor.extract(request_new)
    assert processor.call_count == 2


@pytest.mark.asyncio
async def test_vision_produced_record_invalidated_by_vision_model_change(tmp_path: Path):
    """A vision-engine row validates against the vision engine: a vision model
    change is a miss, while a text model change does not affect it."""
    processor = VisionDummyProcessor()
    vision_cache = tmp_path / "metadata" / "vision_produced.json"

    request_v1 = _make_request(
        vision_cache, DummyMetadata, use_vision_cache=True, vision_model="v1"
    )
    await processor.extract(request_v1)  # VisionDummyProcessor returns use_vision_cache -> True
    assert processor.call_count == 1

    # Text model change does not invalidate a vision-produced row.
    request_text_change = _make_request(
        vision_cache, DummyMetadata, use_vision_cache=True, vision_model="v1", model="other_text"
    )
    await processor.extract(request_text_change)
    assert processor.call_count == 1

    # Vision model change does.
    request_v2 = _make_request(
        vision_cache, DummyMetadata, use_vision_cache=True, vision_model="v2"
    )
    await processor.extract(request_v2)
    assert processor.call_count == 2


@pytest.mark.asyncio
async def test_middleware_legacy_row_without_producing_identity_still_reads(tmp_path: Path):
    """A true legacy row (no _prod_* markers, pre-provenance key format) is still
    served when its stored identity matches the requested engine identity."""
    import json as json_lib

    from renai.cache.manager import _get_db_connection

    processor = VisionDummyProcessor()
    # Build the exact pre-provenance key: doc_<hash>_<pmk>_vision.json
    from renai.utils import get_provider_model_key

    pmk = get_provider_model_key("vision_provider", "vision_model")
    legacy_key = f"doc_hash123_{pmk}_vision.json"

    test_cache_path = tmp_path / "metadata" / "doc_hash123_vision.json"
    db_dir = tmp_path  # _get_db_dir maps metadata/ -> its parent
    payload = {"title": "Legacy Row"}
    with _get_db_connection(db_dir) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO metadata_cache (cache_key, provider, model, metadata_json)"
            " VALUES (?, ?, ?, ?)",
            (legacy_key, "vision_provider", "vision_model", json_lib.dumps(payload)),
        )
    # Also make sure no current-format row shadows it.
    assert test_cache_path.name != legacy_key

    request = _make_request(test_cache_path, DummyMetadata, use_vision_cache=True)
    result, is_vision = await processor.extract(request)
    assert processor.call_count == 0
    assert result is not None and result.title == "Legacy Row"
    # Legacy vision-request reconstruction: vision template.
    assert is_vision is True
