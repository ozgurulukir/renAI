import asyncio
import functools
import inspect
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from renai.cache.manager import read_metadata_cache_with_source, write_metadata_cache
from renai.config import get_image_extensions
from renai.logging_config import get_logger

logger = get_logger("renai.cache.middleware")


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
    cache_path: Path
    model_class: type[BaseModel]
    provider: str
    model: str
    vision_provider: str
    vision_model: str
    use_vision_cache: bool
    fallback_mode: Any
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
    On read, a row is a hit only when its producing identity still matches the
    current identity of that same engine, taken from the request contract — no
    instance attributes are consulted, so the provenance decision lives in
    exactly one place (the code that builds the ``MetadataRequest``).
    """

    def decorator(
        func: Callable[..., Coroutine[Any, Any, tuple[Any, bool]]],
    ) -> Callable[..., Coroutine[Any, Any, tuple[Any, bool]]]:
        _validate_request_contract(func)

        @functools.wraps(func)
        async def wrapper(
            self: Any,
            request: MetadataRequest,
            **extra: Any,
        ) -> tuple[BaseModel | None, bool]:
            # Legacy rows were keyed (and identity-stamped) by the *requested*
            # engine: the vision engine for vision-cache requests, else text.
            legacy_provider = (
                request.vision_provider if request.use_vision_cache else request.provider
            )
            legacy_model = request.vision_model if request.use_vision_cache else request.model

            if not request.update_metadata:
                cached_data, source = await asyncio.to_thread(
                    read_metadata_cache_with_source,
                    request.cache_path,
                    legacy_provider,
                    legacy_model,
                    model_class=request.model_class,
                )
                if cached_data:
                    expected_sig = request.rule_signature or ""
                    stored_sig = source.rule_signature if source else None
                    # Legacy records (no stored signature) are treated as
                    # global: rule-scoped runs re-extract under their own
                    # category universe instead of being served global data.
                    if (stored_sig or "") != expected_sig:
                        # Info, not debug: a mismatch means a real paid LLM
                        # re-extraction the user did not ask for (typically a
                        # changed folder rule), which deserves visibility.
                        logger.info(
                            "Metadata cache rule-signature mismatch; re-extracting",
                            file=request.file_path.name,
                            expected=expected_sig or "global",
                            stored=(
                                stored_sig
                                if stored_sig
                                else ("legacy" if stored_sig is None else "global")
                            ),
                        )
                    elif (
                        source is None
                        or source.produced_by_provider is None
                        or source.produced_by_model is None
                        or (source.produced_by_provider, source.produced_by_model)
                        == _producing_identity(request, source.is_vision_doc)
                    ):
                        is_vision_doc = source.is_vision_doc if source else None
                        logger.debug(
                            "Metadata cache hit",
                            file=request.file_path.name,
                            produced_by=(
                                f"{source.produced_by_provider}/{source.produced_by_model}"
                                if source and source.produced_by_provider
                                else f"{legacy_provider}/{legacy_model} (legacy)"
                            ),
                        )
                        if is_vision_doc is None:
                            # Legacy record predating the persisted source marker:
                            # reconstruct from request flags (previous behavior).
                            is_image = request.file_path.suffix.lower() in get_image_extensions()
                            is_vision_doc = request.use_vision_cache and not (
                                is_image and request.rename_images
                            )
                        return cached_data, is_vision_doc
                    else:
                        # Info, not debug: a mismatch means a real paid LLM
                        # re-extraction the user did not ask for (the engine
                        # that produced the row changed), which deserves
                        # visibility.
                        logger.info(
                            "Metadata cache producing-engine mismatch; re-extracting",
                            file=request.file_path.name,
                            produced_by=(
                                f"{source.produced_by_provider}/{source.produced_by_model}"
                            ),
                            expected="/".join(_producing_identity(request, source.is_vision_doc)),
                        )

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
                    request.cache_path,
                    metadata,
                    provider,
                    model,
                    is_vision_doc=is_vision_fallback,
                    rule_signature=request.rule_signature,
                )

            return metadata, is_vision_fallback

        return wrapper

    return decorator
