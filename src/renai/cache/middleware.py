import asyncio
import functools
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from renai.cache.manager import read_metadata_cache, write_metadata_cache
from renai.logging_config import get_logger

logger = get_logger("renai.cache.middleware")

def with_metadata_cache(
) -> Callable[
    [Callable[..., Coroutine[Any, Any, tuple[BaseModel | None, bool]]]],
    Callable[..., Coroutine[Any, Any, tuple[BaseModel | None, bool]]]
]:
    def decorator(
        func: Callable[..., Coroutine[Any, Any, tuple[BaseModel | None, bool]]]
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
            **kwargs: Any
        ) -> tuple[BaseModel | None, bool]:

            config = getattr(self, "config", None)
            if not config:
                logger.warning(
                    "Cache middleware bypassed: 'config' attribute missing on instance.",
                    file=file_path.name
                )
                return await func(
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
                    **kwargs
                )

            provider = getattr(config, "provider", "unknown")
            model = getattr(config, "model", "unknown")

            if not update_metadata:
                cached_data = await asyncio.to_thread(
                    read_metadata_cache,
                    cache_path=cache_path,
                    provider=provider,
                    model=model,
                    model_class=model_class
                )
                if cached_data:
                    logger.debug("Metadata cache hit", file=file_path.name, provider=provider, model=model)
                    return cached_data, use_vision_cache

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
                **kwargs
            )

            if metadata:
                await asyncio.to_thread(
                    write_metadata_cache,
                    cache_path=cache_path,
                    metadata=metadata,
                    provider=provider,
                    model=model
                )

            return metadata, is_vision_fallback
        return wrapper
    return decorator
