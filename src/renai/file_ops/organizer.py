import asyncio
import logging
from pathlib import Path
from typing import cast

from renai.config.folder_rules import FolderRule
from renai.exceptions import FileOperationError
from renai.file_ops.operations import (
    resolve_target_destination,
    safe_copy_file,
    safe_move_file,
)
from renai.filename.builder import (
    build_book_filename,
    build_image_filename,
    build_vision_filename,
    extract_category_from_metadata,
)
from renai.models import ProcessingResult
from renai.models.schemas import BookMetadata, ImageMetadata

logger = logging.getLogger(__name__)


class FileOrganizer:
    def __init__(self, root_dir: Path | None = None, *, max_filename_length: int):
        self.root_dir = root_dir
        self.max_filename_length = max_filename_length
        # Per-destination-directory locks (issue #244): file operations that
        # target different directories (and their SHA-256 uniqueness checks)
        # can proceed in parallel instead of serialising behind one global
        # lock. Dict access happens on the event loop thread only.
        self._dir_locks: dict[Path, asyncio.Lock] = {}
        self._dir_locks_guard = asyncio.Lock()

    async def _lock_for(self, dest_dir: Path) -> asyncio.Lock:
        """Return the per-directory lock guarding file ops into ``dest_dir``."""
        async with self._dir_locks_guard:
            return self._dir_locks.setdefault(dest_dir, asyncio.Lock())

    async def organize_file(
        self,
        file_path: Path,
        metadata: BookMetadata | ImageMetadata,
        mode: str = "rename",
        add_exif_date: bool = False,
        file_hash: str | None = None,
        is_vision_doc: bool = False,
        dry_run: bool = False,
        folder_rule: FolderRule | None = None,
    ) -> ProcessingResult:
        try:
            category = extract_category_from_metadata(metadata, self.max_filename_length)
            is_image = isinstance(metadata, ImageMetadata)

            if is_image:
                new_name = build_image_filename(
                    cast("ImageMetadata", metadata),
                    file_path.suffix,
                    self.max_filename_length,
                    add_exif_date=add_exif_date,
                    template=folder_rule.image_template if folder_rule else None,
                )
                category = "Images"
            elif is_vision_doc:
                new_name = build_vision_filename(
                    cast("BookMetadata", metadata),
                    file_path.suffix,
                    self.max_filename_length,
                    template=folder_rule.vision_template if folder_rule else None,
                )
            else:
                new_name = build_book_filename(
                    cast("BookMetadata", metadata),
                    file_path.suffix,
                    self.max_filename_length,
                    template=folder_rule.doc_template if folder_rule else None,
                )

            if folder_rule:
                logger.debug("Applied folder rule %s to %s", folder_rule.path, file_path.name)

            target_dest = resolve_target_destination(
                file_path, new_name, category=category, mode=mode, root_dir=self.root_dir
            )

            if file_path.resolve() == target_dest.resolve():
                logger.debug(f"Skipping (Already organized/renamed): {file_path.name}")
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

            # Copy vs move distinction:
            # - "move" transfers the source to the destination and (on success) the
            #   source is gone, because the move removes the original.
            # - "copy" duplicates the source bytes and MUST leave the original intact.
            # Branch on the mode here so a copy is never routed through safe_move_file,
            # which would silently delete the source (the original data-loss bug).
            # Note: the default rename/organize paths still move the file. Only the
            # explicit "copy" mode duplicates the source without deleting it.
            transfer = safe_copy_file if mode == "copy" else safe_move_file

            def _do_transfer():
                return transfer(
                    file_path,
                    target_dest,
                    source_hash=file_hash,
                    dry_run=dry_run,
                    root_dir=self.root_dir,
                )

            if dry_run:
                # Dry runs never touch the filesystem, so no lock is needed.
                final_path = _do_transfer()
            else:
                dir_lock = await self._lock_for(target_dest.parent)
                async with dir_lock:
                    final_path = await asyncio.to_thread(_do_transfer)

            logger.debug(f"SUCCESS ({mode}): {file_path.name} -> {final_path.name}")
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

        except FileOperationError as e:
            # Path-traversal and other boundary violations from the file_ops
            # security controls are reported verbatim so the security reason
            # stays visible in the processing result.
            logger.error("Blocked file operation for %s: %s", file_path.name, e)
            return ProcessingResult(
                success=False,
                name=file_path.name,
                original_name=file_path.name,
                error=str(e),
            )

        except Exception as e:
            logger.error(f"Renaming logic failed for {file_path.name}: {e}")
            return ProcessingResult(
                success=False,
                name=file_path.name,
                original_name=file_path.name,
                error=f"Renaming logic failed: {e!s}",
            )
