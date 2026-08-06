import asyncio
import logging
from pathlib import Path
from typing import cast

from renai.file_ops.operations import resolve_target_destination, safe_move_file
from renai.filename.builder import (
    build_book_filename,
    build_image_filename,
    build_vision_filename,
    extract_category_from_metadata,
)
from renai.models import ProcessingResult
from renai.models.schemas import BookMetadata, ImageMetadata


class FileOrganizer:
    def __init__(self, root_dir: Path | None = None):
        self.root_dir = root_dir
        self.file_operation_lock = asyncio.Lock()

    async def organize_file(
        self,
        file_path: Path,
        metadata: BookMetadata | ImageMetadata,
        mode: str = "rename",
        add_exif_date: bool = False,
        file_hash: str | None = None,
        is_vision_doc: bool = False,
        dry_run: bool = False,
    ) -> ProcessingResult:
        try:
            category = extract_category_from_metadata(metadata)
            is_image = isinstance(metadata, ImageMetadata)

            if is_image:
                new_name = build_image_filename(
                    cast("ImageMetadata", metadata), file_path.suffix, add_exif_date=add_exif_date
                )
                category = "Images"
            elif is_vision_doc:
                new_name = build_vision_filename(cast("BookMetadata", metadata), file_path.suffix)
            else:
                new_name = build_book_filename(cast("BookMetadata", metadata), file_path.suffix)

            target_dest = resolve_target_destination(
                file_path, new_name, category=category, mode=mode, root_dir=self.root_dir
            )

            if self.root_dir:
                resolved_root = self.root_dir.resolve()
                resolved_dest = target_dest.resolve()
                if not resolved_dest.is_relative_to(resolved_root):
                    logging.error("Target path %s escapes root %s", target_dest, self.root_dir)
                    return ProcessingResult(
                        success=False,
                        name=file_path.name,
                        original_name=file_path.name,
                        error="Security: Path traversal blocked",
                    )

            if file_path.resolve() == target_dest.resolve():
                logging.debug(f"Skipping (Already organized/renamed): {file_path.name}")
                m_data = (
                    metadata.model_dump()
                    if hasattr(metadata, "model_dump")
                    else cast("dict", metadata)
                )
                return ProcessingResult(
                    success=True,
                    metadata=m_data,
                    file_path=file_path,
                    name=file_path.name,
                    original_name=file_path.name,
                    category=category,
                )

            async with self.file_operation_lock:

                def _do_move():
                    return safe_move_file(
                        file_path,
                        target_dest,
                        operation_type=mode,
                        source_hash=file_hash,
                        dry_run=dry_run,
                    )

                final_path = await asyncio.to_thread(_do_move)

            logging.debug(f"SUCCESS ({mode}): {file_path.name} -> {final_path.name}")
            m_data = (
                metadata.model_dump() if hasattr(metadata, "model_dump") else cast("dict", metadata)
            )
            return ProcessingResult(
                success=True,
                metadata=m_data,
                file_path=final_path,
                name=final_path.name,
                original_name=file_path.name,
                category=category,
            )

        except Exception as e:
            logging.error(f"Renaming logic failed for {file_path.name}: {e}")
            return ProcessingResult(
                success=False,
                name=file_path.name,
                original_name=file_path.name,
                error=f"Renaming logic failed: {e!s}",
            )
