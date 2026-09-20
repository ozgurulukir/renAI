import asyncio
import functools
import inspect
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from renai.cache.manager import read_metadata_cache, write_metadata_cache
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
    """

    file_path: Path
    cache_path: Path
    model_class: type[BaseModel]
    use_vision_cache: bool
    fallback_mode: Any
    rename_images: bool
    update_metadata: bool
    add_exif_date: bool
    file_hash: str | None = None
    doc_categories: dict[str, list] | None = None
    image_categories: dict[str, list] | None = None


def _fallback_provider(self: Any, use_vision_cache: bool = False) -> str:
    """Resolve the cache provider from the wrapped instance."""
    attr = "vision_provider" if use_vision_cache else "provider"
    return getattr(self, attr, "unknown")


def _fallback_model(self: Any, use_vision_cache: bool = False) -> str:
    """Resolve the cache model from the wrapped instance."""
    attr = "vision_model" if use_vision_cache else "model"
    return getattr(self, attr, "unknown")


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


def with_metadata_cache(
    provider_resolver: Callable[[Any, bool], str] = _fallback_provider,
    model_resolver: Callable[[Any, bool], str] = _fallback_model,
) -> Callable[
    [Callable[..., Coroutine[Any, Any, tuple[Any, bool]]]],
    Callable[..., Coroutine[Any, Any, tuple[Any, bool]]],
]:
    """Decorator that wraps an async metadata-resolving method with cache logic.

    The decorated method must accept ``(self, request: MetadataRequest, ...)`` and
    return ``tuple[BaseModel | None, bool]`` (metadata plus whether a vision fallback
    was used). The contract is validated at decoration time by
    :func:`_validate_request_contract`.

    The provider and model used for cache keying are resolved lazily from the
    instance at call time via the ``provider_resolver`` and ``model_resolver``
    callables. By default they read the instance's ``provider``/``model``
    attributes (or ``vision_provider``/``vision_model`` when the vision cache
    is used), so the wrapper does not depend on a specific ``config`` object.
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
            provider = provider_resolver(self, request.use_vision_cache)
            model = model_resolver(self, request.use_vision_cache)

            if not request.update_metadata:
                cached_data = await asyncio.to_thread(
                    read_metadata_cache,
                    request.cache_path,
                    provider,
                    model,
                    model_class=request.model_class,
                )
                if cached_data:
                    logger.debug(
                        "Metadata cache hit",
                        file=request.file_path.name,
                        provider=provider,
                        model=model,
                    )
                    is_image = request.file_path.suffix.lower() in get_image_extensions()
                    is_vision_doc = request.use_vision_cache and not (
                        is_image and request.rename_images
                    )
                    return cached_data, is_vision_doc

            metadata, is_vision_fallback = await func(self, request, **extra)

            if metadata:
                await asyncio.to_thread(
                    write_metadata_cache,
                    request.cache_path,
                    metadata,
                    provider,
                    model,
                )

            return metadata, is_vision_fallback

        return wrapper

    return decorator
