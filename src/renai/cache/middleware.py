import asyncio
import functools
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from renai.cache.manager import read_metadata_cache, write_metadata_cache
from renai.config import get_image_extensions
from renai.logging_config import get_logger

logger = get_logger("renai.cache.middleware")


def _fallback_provider(self: Any, use_vision_cache: bool = False) -> str:
    """Resolve the cache provider from the wrapped instance."""
    attr = "vision_provider" if use_vision_cache else "provider"
    return getattr(self, attr, "unknown")


def _fallback_model(self: Any, use_vision_cache: bool = False) -> str:
    """Resolve the cache model from the wrapped instance."""
    attr = "vision_model" if use_vision_cache else "model"
    return getattr(self, attr, "unknown")


def with_metadata_cache(
    provider_resolver: Callable[[Any, bool], str] = _fallback_provider,
    model_resolver: Callable[[Any, bool], str] = _fallback_model,
) -> Callable[
    [Callable[..., Coroutine[Any, Any, tuple[BaseModel | None, bool]]]],
    Callable[..., Coroutine[Any, Any, tuple[BaseModel | None, bool]]],
]:
    """Decorator that wraps an async metadata-resolving method with cache logic.

    The wrapped method must accept the keyword arguments ``file_path``,
    ``cache_path``, ``model_class``, ``use_vision_cache``, ``fallback_mode``,
    ``rename_images``, ``update_metadata``, ``add_exif_date`` and the optional
    ``file_hash``, and must return ``tuple[BaseModel | None, bool]``
    (metadata plus whether a vision fallback was used).

    The provider and model used for cache keying are resolved lazily from the
    instance at call time via the ``provider_resolver`` and ``model_resolver``
    callables. By default they read the instance's ``provider``/``model``
    attributes (or ``vision_provider``/``vision_model`` when the vision cache
    is used), so the wrapper does not depend on a specific ``config`` object.
    """

    def decorator(
        func: Callable[..., Coroutine[Any, Any, tuple[BaseModel | None, bool]]],
    ) -> Callable[..., Coroutine[Any, Any, tuple[BaseModel | None, bool]]]:
        @functools.wraps(func)
        async def wrapper(
            self: Any,
            *args: Any,
            file_path: Path,
            cache_path: Path,
            model_class: type[BaseModel],
            use_vision_cache: bool,
            fallback_mode: Any,
            rename_images: bool,
            update_metadata: bool,
            add_exif_date: bool,
            file_hash: str | None = None,
            **kwargs: Any,
        ) -> tuple[BaseModel | None, bool]:
            provider = provider_resolver(self, use_vision_cache)
            model = model_resolver(self, use_vision_cache)

            if not update_metadata:
                cached_data = await asyncio.to_thread(
                    read_metadata_cache,
                    cache_path,
                    provider,
                    model,
                    model_class=model_class,
                )
                if cached_data:
                    logger.debug(
                        "Metadata cache hit",
                        file=file_path.name,
                        provider=provider,
                        model=model,
                    )
                    is_image = file_path.suffix.lower() in get_image_extensions()
                    is_vision_doc = use_vision_cache and not (is_image and rename_images)
                    return cached_data, is_vision_doc

            metadata, is_vision_fallback = await func(
                self,
                *args,
                file_path=file_path,
                cache_path=cache_path,
                model_class=model_class,
                use_vision_cache=use_vision_cache,
                fallback_mode=fallback_mode,
                rename_images=rename_images,
                update_metadata=update_metadata,
                add_exif_date=add_exif_date,
                file_hash=file_hash,
                **kwargs,
            )

            if metadata:
                await asyncio.to_thread(
                    write_metadata_cache,
                    cache_path,
                    metadata,
                    provider,
                    model,
                )

            return metadata, is_vision_fallback

        return wrapper

    return decorator
