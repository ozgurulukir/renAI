import re

with open("src/renai/cache/middleware.py", "r") as f:
    content = f.read()

# Make the middleware signature match what _resolve_metadata actually expects
old_wrapper = """        async def wrapper(self: Any, file_path: Path, file_hash: str, *args: Any, **kwargs: Any) -> T:
            config = getattr(self, "config", None)
            if not config:
                logger.warning("Cache middleware bypassed: 'config' attribute missing on instance.")
                return await func(self, file_path, file_hash, *args, **kwargs)

            use_vision = kwargs.get("is_vision_fallback", False) or getattr(config, "rename_images", False)
            schema_class = kwargs.get("schema_class")

            if not config.update_metadata and schema_class:
                cached_data = await read_metadata_cache(
                    file_hash=file_hash,
                    provider=config.provider,
                    model=config.model,
                    schema_class=schema_class,
                    use_vision=use_vision
                )
                if cached_data:
                    return cached_data

            result = await func(self, file_path, file_hash, *args, **kwargs)

            if result and schema_class:
                await write_metadata_cache(
                    file_hash=file_hash,
                    provider=config.provider,
                    model=config.model,
                    metadata=result,
                    schema_class=schema_class,
                    use_vision=use_vision
                )
            return result"""

new_wrapper = """        async def wrapper(
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
        ) -> tuple[BaseModel | None, bool]:

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
                file_path=file_path,
                cache_path=cache_path,
                model_class=model_class,
                use_vision_cache=use_vision_cache,
                fallback_mode=fallback_mode,
                rename_images=rename_images,
                update_metadata=update_metadata,
                add_exif_date=add_exif_date,
                file_hash=file_hash,
                *args,
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

            return metadata, is_vision_fallback"""

content = content.replace(old_wrapper, new_wrapper)
content = content.replace("T = TypeVar(\"T\", bound=BaseModel | None)", "T = TypeVar(\"T\", bound=tuple[BaseModel | None, bool])")

with open("src/renai/cache/middleware.py", "w") as f:
    f.write(content)
