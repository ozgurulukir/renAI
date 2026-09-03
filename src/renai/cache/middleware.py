import functools
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from renai.cache.manager import read_metadata_cache, write_metadata_cache
from renai.logging_config import get_logger

logger = get_logger("renai.cache.middleware")

T = TypeVar("T", bound=tuple[BaseModel | None, bool])

def with_metadata_cache() -> Callable[[Callable[..., Coroutine[Any, Any, T]]],\
    Callable[..., Coroutine[Any, Any, T]]]:
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) ->\
        Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(
            self: Any,
            file_path: Path,
            cache_path: Path,
            model_class: type[BaseModel],
            use_vision_cache: bool,
            fallback_mode: Any,
            rename_images: bool,
            update_metadata: bool,
            add_exif_date: bool,
            file_hash: str | None = None,
            *args: Any,
            **kwargs: Any
        ) -> T:

            if not update_metadata:
                cached_data = await read_metadata_cache(
                    cache_path=cache_path,
                    schema_class=model_class,
                    use_vision=use_vision_cache,
                    fallback_mode=fallback_mode,
                )
                if cached_data:
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
                await write_metadata_cache(
                    cache_path=cache_path,
                    metadata=metadata,
                    schema_class=model_class,
                    use_vision=is_vision_fallback,
                    fallback_mode=fallback_mode,
                )

            return metadata, is_vision_fallback
        return wrapper
    return decorator
