import asyncio
import functools
import inspect
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from renai.cache.manager import (
    MetadataCacheKey,
    metadata_version_fingerprint,
    read_metadata_cache,
    write_metadata_cache,
)
from renai.config import get_image_extensions
from renai.config.folder_rules import global_categories_signature
from renai.models.schemas import FallbackMode

logger = logging.getLogger(__name__)

# Per-key in-flight bookkeeping (#249): concurrent identical MetadataRequests
# share one underlying extraction call instead of each missing the cache and
# paying for an LLM round-trip. Keyed by a hashable identity of the request.
_INFLIGHT_CALLS: dict[tuple[Any, ...], asyncio.Task[tuple[Any, bool]]] = {}


@dataclass(frozen=True)
class MetadataRequest:
    """Explicit, type-safe contract shared by the cache decorator and its callers.

    Replaces the previous implicit *named-kwargs* convention (``file_path``,
    ``cache_path``, ``model_class``, ...) which let the decorator read arguments
    by name and broke silently with ``KeyError``/``TypeError`` at call time if a
    parameter name drifted. Wrapping every call in one discoverable, editable
    dataclass collapses the implicit schema into a single source of truth.

    The four identity fields carry BOTH engine identities from the single
    decision point that builds the request (``BookProcessor.
    _build_metadata_request``): ``provider``/``model`` for the text engine,
    ``vision_provider``/``vision_model`` for the vision engine. The middleware
    picks the producing identity from the observed extraction outcome instead
    of duplicating the request-flag-vs-instance-attribute decision.
    """

    file_path: Path
    cache_key: MetadataCacheKey
    model_class: type[BaseModel]
    provider: str
    model: str
    vision_provider: str
    vision_model: str
    use_vision_cache: bool
    fallback_mode: FallbackMode
    rename_images: bool
    update_metadata: bool
    add_exif_date: bool
    file_hash: str | None = None
    doc_categories: dict[str, list] | None = None
    image_categories: dict[str, list] | None = None
    rule_signature: str | None = None


_NUM_REQUIRED_PARAMS = 2  # self + request


def _validate_request_contract(func: Callable[..., Any]) -> None:
    """Fail loudly at decoration time when a method does not expose ``(self, request)``.

    Moves the failure from "runtime on a random call" to "import/decoration time",
    where it is immediate, loud, and easy to locate instead of a ``KeyError`` deep
    inside an unrelated code path.

    The wrapped method must expose ``request`` (annotated ``MetadataRequest``) as its
    second parameter -- the parameter immediately after ``self`` -- so the decorator
    can always find and forward the contract. Optional keyword-only parameters (e.g.
    per-folder category overrides) are permitted after it, but the contract parameter
    itself is required and may not be renamed.
    """
    params = list(inspect.signature(func).parameters.values())
    if len(params) < _NUM_REQUIRED_PARAMS:
        raise ValueError(
            "with_metadata_cache requires the decorated method to accept "
            f"(self, request: MetadataRequest); found {len(params) - 1} non-self parameter(s)"
        )

    # ``request`` must be the parameter immediately after ``self`` (position 1).
    request_param = params[1]
    if request_param.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.VAR_POSITIONAL,
    ):
        raise ValueError(
            "with_metadata_cache requires the metadata parameter to be named 'request' "
            "and accepted as a regular keyword/positional parameter after 'self'; "
            f"got {request_param.kind}"
        )
    if request_param.name != "request":
        raise ValueError(
            "with_metadata_cache requires the metadata parameter to be named 'request'; "
            f"got {request_param.name!r}"
        )
    if request_param.annotation is not MetadataRequest:
        raise ValueError(
            "with_metadata_cache requires the metadata parameter to be annotated as "
            f"MetadataRequest; got {request_param.annotation!r}"
        )

    # No additional positional-or-keyword parameters are allowed: everything beyond the
    # contract must be keyword-only, otherwise they could shadow the decorator's call
    # convention. Keyword-only parameters (e.g. category overrides) are fine.
    for extra in params[2:]:
        if extra.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            raise ValueError(
                "with_metadata_cache allows only keyword-only parameters after "
                f"(self, request); got positional parameter {extra.name!r}"
            )


def _producing_identity(request: MetadataRequest, is_vision_doc: bool | None) -> tuple[str, str]:
    """Identity of the engine expected to have produced a record for this request.

    Image requests are always served by the vision engine. Document requests
    under a vision cache namespace were produced by the vision engine only when
    the persisted ``is_vision_doc`` marker says so (``None`` = legacy record,
    reconstructed as vision — the historical behavior for vision-request rows);
    otherwise the text LLM succeeded first.
    """
    is_image = request.file_path.suffix.lower() in get_image_extensions()
    if is_image and request.rename_images:
        return request.vision_provider, request.vision_model
    if request.use_vision_cache and is_vision_doc is not False:
        return request.vision_provider, request.vision_model
    return request.provider, request.model


def _produced_by_vision(request: MetadataRequest, is_vision_fallback: bool) -> bool:
    """Whether the engine that produced ``is_vision_fallback`` output was the vision engine."""
    is_image = request.file_path.suffix.lower() in get_image_extensions()
    return (is_image and request.rename_images) or bool(is_vision_fallback)


def _effective_rule_signature(request: MetadataRequest) -> str:
    """Category-universe signature the row must carry (#247).

    A folder rule's signature takes precedence; a request without category
    restrictions signs the effective GLOBAL category list
    (``global_categories_signature``), so editing the global list invalidates
    rows written under the previous list — the same contract rule-scoped rows
    already had.
    """
    if request.rule_signature:
        return request.rule_signature
    is_image = request.file_path.suffix.lower() in get_image_extensions()
    return global_categories_signature(
        mode="img" if (is_image and request.rename_images) else "doc"
    )


def _cache_lookup(request: MetadataRequest) -> tuple[Any, bool] | None:
    """Resolve a valid cache hit for ``request``, or ``None`` on any miss.

    The single place cache-validity judgments are applied (#247). A stored row
    is a hit only when its category-universe signature, version fingerprint and
    producing identity all match; ``source is None`` (legacy file fallback /
    pre-provenance data) skips signature/version gating and falls through to
    the legacy reconstruction.

    Synchronous so callers can run it via ``asyncio.to_thread``.
    """
    if request.update_metadata:
        return None

    cached_data, source = read_metadata_cache(request.cache_key, request.model_class)
    if not cached_data:
        return None

    legacy_provider = request.vision_provider if request.use_vision_cache else request.provider
    legacy_model = request.vision_model if request.use_vision_cache else request.model

    if source is not None:
        expected_sig = _effective_rule_signature(request)
        stored_sig = source.rule_signature
        # Legacy records (no stored signature) are treated as global:
        # rule-scoped runs re-extract under their own category universe.
        if (stored_sig or "") != expected_sig:
            stored_repr: str = (
                stored_sig if stored_sig else ("legacy" if stored_sig is None else "global")
            )
            # Info, not debug: a mismatch means a real paid LLM re-extraction
            # the user did not ask for (typically a changed folder rule),
            # which deserves visibility.
            logger.info(
                "Metadata cache rule-signature mismatch; re-extracting "
                "(file=%s, expected=%s, stored=%s)",
                request.file_path.name,
                expected_sig or "global",
                stored_repr,
            )
            return None

        current_version = metadata_version_fingerprint()
        stored_version = source.version_fingerprint
        if (stored_version or "") != current_version:
            # Prompt/schema/app-version change (#247): the stored row predates
            # the current extraction contract.
            logger.info(
                "Metadata cache version mismatch; re-extracting "
                "(file=%s, stored_version=%s, current_version=%s)",
                request.file_path.name,
                stored_version or "legacy (none)",
                current_version,
            )
            return None

    if (
        source is None
        or source.produced_by_provider is None
        or source.produced_by_model is None
        or (source.produced_by_provider, source.produced_by_model)
        == _producing_identity(request, source.is_vision_doc)
    ):
        is_vision_doc = source.is_vision_doc if source else None
        logger.debug(
            "Metadata cache hit (file=%s, produced_by=%s)",
            request.file_path.name,
            (
                f"{source.produced_by_provider}/{source.produced_by_model}"
                if source and source.produced_by_provider
                else f"{legacy_provider}/{legacy_model} (legacy)"
            ),
        )
        if is_vision_doc is None:
            # Legacy record predating the persisted source marker:
            # reconstruct from request flags (previous behavior).
            is_image = request.file_path.suffix.lower() in get_image_extensions()
            is_vision_doc = request.use_vision_cache and not (is_image and request.rename_images)
        return cached_data, is_vision_doc

    # Info, not debug: a mismatch means a real paid LLM re-extraction the user
    # did not ask for (the engine that produced the row changed), which
    # deserves visibility.
    logger.info(
        "Metadata cache producing-engine mismatch; re-extracting "
        "(file=%s, produced_by=%s, expected=%s)",
        request.file_path.name,
        f"{source.produced_by_provider}/{source.produced_by_model}",
        "/".join(_producing_identity(request, source.is_vision_doc)),
    )
    return None


def with_metadata_cache() -> Callable[
    [Callable[..., Coroutine[Any, Any, tuple[Any, bool]]]],
    Callable[..., Coroutine[Any, Any, tuple[Any, bool]]],
]:
    """Decorator that wraps an async metadata-resolving method with cache logic.

    The decorated method must accept ``(self, request: MetadataRequest, ...)`` and
    return ``tuple[BaseModel | None, bool]`` (metadata plus whether a vision fallback
    was used). The contract is validated at decoration time by
    :func:`_validate_request_contract`.

    Cache identity is provenance-based: rows are keyed by file + mode only, and
    each row records which engine produced it (``CacheSource.produced_by_*``).
    On read, a row is a hit only when ALL of the following hold (the validity
    judgments live in :func:`_cache_lookup`, the single place they are applied):

    - its stored category-universe signature matches the effective one for
      this request (folder-rule override, or the signed global category list);
    - its stored version fingerprint (prompts + Pydantic schemas + app
      version, ``metadata_version_fingerprint``) matches the current one;
    - its producing identity still matches the current identity of that same
      engine, taken from the request contract — no instance attributes are
      consulted.

    Concurrent identical requests are de-duplicated: while one extraction for
    a cache key is in flight, other requests with the same key await its
    result instead of issuing their own underlying call (#249).
    """

    def decorator(
        func: Callable[..., Coroutine[Any, Any, tuple[Any, bool]]],
    ) -> Callable[..., Coroutine[Any, Any, tuple[Any, bool]]]:
        _validate_request_contract(func)

        async def _execute(
            self: Any,
            request: MetadataRequest,
            **extra: Any,
        ) -> tuple[BaseModel | None, bool]:
            hit = await asyncio.to_thread(_cache_lookup, request)
            if hit is not None:
                return hit

            metadata, is_vision_fallback = await func(self, request, **extra)

            if metadata:
                # Key the row by the identity of the engine that actually
                # produced it, so a text-LLM record under a vision-request
                # namespace validates against the text engine, and vice versa.
                if _produced_by_vision(request, is_vision_fallback):
                    provider = request.vision_provider
                    model = request.vision_model
                else:
                    provider = request.provider
                    model = request.model
                await asyncio.to_thread(
                    write_metadata_cache,
                    request.cache_key,
                    metadata,
                    is_vision_doc=is_vision_fallback,
                    rule_signature=_effective_rule_signature(request),
                    produced_provider=provider,
                    produced_model=model,
                )

            return metadata, is_vision_fallback

        @functools.wraps(func)
        async def wrapper(
            self: Any,
            request: MetadataRequest,
            **extra: Any,
        ) -> tuple[BaseModel | None, bool]:
            # Per-key in-flight de-duplication (#249): workers processing
            # byte-identical files concurrently share a single extraction
            # call instead of each missing the (not yet written) cache row.
            dedup_key = (
                request.cache_key,
                request.rule_signature or "",
                bool(request.update_metadata),
                tuple(sorted((name, repr(value)) for name, value in extra.items())),
            )
            in_flight = _INFLIGHT_CALLS.get(dedup_key)
            if in_flight is not None:
                # Shielded so a cancelled awaiter cannot cancel the shared
                # extraction for everyone else; the owner populates the cache
                # on success.
                return await asyncio.shield(in_flight)

            task: asyncio.Task[tuple[Any, bool]] = asyncio.create_task(
                _execute(self, request, **extra)
            )
            _INFLIGHT_CALLS[dedup_key] = task
            try:
                return await task
            finally:
                if _INFLIGHT_CALLS.get(dedup_key) is task:
                    _INFLIGHT_CALLS.pop(dedup_key, None)

        return wrapper

    return decorator
