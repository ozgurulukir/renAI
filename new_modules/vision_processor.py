"""
Vision processing module - handles PDF processing with vision-capable LLMs.

This module was extracted from file_processor.py as part of the refactoring
to improve code organization and testability.

Overview:
=========
The VisionProcessor class handles document processing using vision-capable
Large Language Models (LLMs). Unlike text extraction which reads document text
directly, vision mode "sees" the document as an image.

When to Use Vision Mode:
========================
Vision mode is ideal for:
- Text-based PDFs where extraction fails
- Documents with complex formatting
- When you don't have OCR (Tesseract) installed
- Faster processing of clean digital documents
- Better accuracy on formatted documents

When NOT to Use Vision Mode:
===========================
- Scanned documents (use OCR instead)
- Multi-language documents (OCR better for Turkish)
- When cost is a concern (vision uses image tokens, more expensive)
- When you need the highest accuracy on scanned content

Processing Flow:
===============
1. Try text extraction WITHOUT OCR fallback
   - If successful → use LLM metadata (same as regular mode)
   
2. If text extraction fails + file is PDF → use vision model
   - Convert PDF pages to images
   - Send images to vision-capable LLM
   - Extract metadata from vision response
   
3. If text extraction fails + file is NOT PDF → return failure
   - Non-PDF files can't use vision fallback

Recommended Vision Models:
========================
| Provider   | Model                              | Best For        |
|------------|------------------------------------|-----------------|
| OpenAI     | gpt-4o                            | Best accuracy   |
| OpenRouter | openai/gpt-4o                     | Flexible pricing|
| OpenRouter | google/gemini-1.5-flash           | Free tier       |
| DeepInfra  | meta-llama/Llama-3.2-90B-Vision  | Open source     |
| DeepInfra  | Qwen/Qwen2-VL-72B-Instruct        | Excellent value |

Image Optimization:
==================
To balance quality vs API token cost:
- Max size: 1024x1024 pixels
- Format: JPEG (85% quality)
- DPI: 150 (good balance of detail vs size)

Page Selection Strategy:
=======================
- Documents ≤ 5 pages: Process all pages
- Documents > 5 pages: Sample strategically
  - First 3 pages (front matter, intro)
  - Middle page (content sample)
  - Last page (conclusion, references)

This strategy balances thoroughness with API cost.

Usage:
======
    from new_modules.vision_processor import VisionProcessor
    from new_modules.text_extractor import TextExtractor
    from llm_engine import LLMEngine
    
    # Create processor
    llm = LLMEngine(provider="openai", model="gpt-4o")
    text_extractor = TextExtractor(ocr_engine, "pymupdf")
    
    vision = VisionProcessor(
        llm_engine=llm,
        text_extractor=text_extractor,
        provider="openai",
        model="gpt-4o"
    )
    
    # Check if model supports vision
    if vision.is_vision_capable():
        success, metadata = vision.process(Path("document.pdf"))
        if success:
            print(f"Title: {metadata.get('title')}")

Class API:
=========
VisionProcessor(llm_engine, text_extractor, provider, model)
    - is_vision_capable(): Check if model supports vision
    - process(): Process document with vision
    - _extract_with_vision(): Extract via vision model
    - _convert_pdf_page_to_image(): Convert PDF page to JPEG

Module Functions:
================
- create_vision_processor(): Factory function for easy creation
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf
from PIL import Image

from llm_engine import LLMEngine

from .cache_manager import (
    get_metadata_cache_path,
    read_metadata_cache,
    write_metadata_cache,
)
from .text_extractor import TextExtractor


# Vision mode image optimization settings
# These settings balance quality vs token cost
MULTIMODAL_IMAGE_MAX_SIZE = 1024  # Max width/height in pixels
MULTIMODAL_IMAGE_QUALITY = 85  # JPEG quality (1-100)


class VisionProcessor:
    """
    Handles document processing using vision-capable LLMs.

    This processor can:
    - Convert PDF pages to images
    - Extract metadata using vision models
    - Fall back to text extraction when possible
    """

    def __init__(
        self,
        llm_engine: LLMEngine,
        text_extractor: TextExtractor,
        provider: str = "deepinfra",
        model: str = "",
    ):
        """
        Initialize the vision processor.

        Args:
            llm_engine: LLM engine with vision capabilities
            text_extractor: Text extractor instance
            provider: LLM provider name
            model: LLM model name
        """
        self.llm_engine = llm_engine
        self.text_extractor = text_extractor
        self.provider = provider
        self.model = model or getattr(llm_engine, "model_name", "")
        self.image_max_size = MULTIMODAL_IMAGE_MAX_SIZE
        self.image_quality = MULTIMODAL_IMAGE_QUALITY

    def is_vision_capable(self) -> bool:
        """
        Check if the configured model supports vision.

        Delegates to the LLM engine to determine if the current model
        has multimodal (vision) capabilities.

        Returns:
            True if model is multimodal-capable, False otherwise.

        Example:
            >>> processor = VisionProcessor(llm_engine, text_extractor, "openai", "gpt-4o")
            >>> processor.is_vision_capable()
            True
            
            >>> processor = VisionProcessor(llm_engine, text_extractor, "openai", "gpt-3.5-turbo")
            >>> processor.is_vision_capable()
            False
        """
        return self.llm_engine.is_multimodal_capable()

    def process(
        self,
        file_path: Path,
        update_metadata: bool = False,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Process a document using vision mode.

        This is the main entry point for vision-based document processing.
        It implements a smart fallback strategy:

        1. Try text extraction WITHOUT OCR fallback
           - If successful → use regular LLM metadata extraction
           - This is faster and cheaper than vision

        2. If text extraction fails + file is PDF → use vision model
           - Convert PDF pages to images
           - Send to vision-capable LLM
           - Extract metadata from response

        3. If text extraction fails + file is NOT PDF → return failure
           - Vision fallback only works for PDFs

        Args:
            file_path: Path to the document file (preferably PDF).
            update_metadata: If True, ignore cached metadata and regenerate.
                           Default False (use cache if available).

        Returns:
            Tuple of (success: bool, metadata: Optional[Dict])
            
            On success, metadata contains:
            - title: Document title
            - subtitle: Document subtitle (if any)
            - author: Author name (if any)
            - year: Publication year (if any)
            - category: Document category

        Example:
            >>> success, metadata = processor.process(Path("document.pdf"))
            >>> if success:
            ...     print(f"Title: {metadata['title']}")
        """
        # Check vision capability first
        if not self.is_vision_capable():
            logging.error(
                f"Model {self.llm_engine.model_name} does not support vision. "
                "Please use a vision-capable model."
            )
            return False, None

        # Skip symlinks
        if file_path.is_symlink():
            logging.warning(f"Skipping symlink: {file_path.name}")
            return False, None

        # Check metadata cache
        metadata = self._get_cached_metadata(file_path, update_metadata)
        if metadata is not None:
            return True, metadata

        # Step 1: Try text extraction WITHOUT OCR
        text = self.text_extractor.extract_without_ocr(file_path)

        if text:
            # Step 2: Text extraction succeeded → use LLM
            metadata = self.llm_engine.get_book_metadata(text)
            if metadata:
                self._save_metadata_cache(file_path, metadata)
                return True, metadata

        # Text extraction failed
        if file_path.suffix.lower() == ".pdf":
            # Step 3: PDF → use vision model
            logging.info(
                f"Text extraction failed for {file_path.name}, using vision model"
            )
            metadata = self._extract_with_vision(file_path)
            if metadata:
                self._save_metadata_cache(file_path, metadata)
                return True, metadata
            else:
                return False, None
        else:
            # Step 4: Non-PDF → cannot process with vision
            logging.warning(
                f"Text extraction failed for {file_path.name} and vision "
                "fallback is only for PDFs."
            )
            return False, None

    def _get_cached_metadata(
        self,
        file_path: Path,
        update_metadata: bool,
    ) -> Optional[Dict[str, Any]]:
        """Get cached metadata if available and valid."""
        from utils import get_file_hash

        file_hash = get_file_hash(file_path)
        cache_path = get_metadata_cache_path(
            file_hash, self.provider, self.model, use_vision=True
        )

        if update_metadata:
            return None

        return read_metadata_cache(cache_path, self.provider, self.model)

    def _save_metadata_cache(
        self,
        file_path: Path,
        metadata: Dict[str, Any],
    ) -> None:
        """Save metadata to cache."""
        from utils import get_file_hash

        file_hash = get_file_hash(file_path)
        cache_path = get_metadata_cache_path(
            file_hash, self.provider, self.model, use_vision=True
        )
        write_metadata_cache(cache_path, metadata, self.provider, self.model)

    def _extract_with_vision(
        self,
        file_path: Path,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from PDF using vision model.

        Processes a sample of pages and returns metadata from the first successful result.
        """
        try:
            pdf_doc = pymupdf.open(str(file_path))
            total_pages = len(pdf_doc)
            pdf_doc.close()

            # Select pages to process
            pages_to_process = self._select_pages(total_pages)

            # Process each page until we get valid metadata
            for page_num in pages_to_process:
                img_bytes = self._convert_pdf_page_to_image(
                    file_path, page_number=page_num, dpi=150
                )
                if not img_bytes:
                    continue

                metadata = self.llm_engine.get_book_metadata_from_image(
                    file_path, img_bytes
                )
                if metadata:
                    return metadata

            logging.error(
                f"Vision model failed to extract metadata from any page of {file_path.name}"
            )
            return None

        except Exception as e:
            logging.error(f"Vision extraction failed for {file_path.name}: {e}")
            return None

    def _select_pages(self, total_pages: int) -> List[int]:
        """
        Select which pages to process based on document length.

        For short documents, process all pages.
        For longer documents, sample strategically.
        """
        if total_pages <= 5:
            return list(range(total_pages))
        else:
            # Sample: first 3, middle, and last page
            return [0, 1, 2, total_pages // 2, total_pages - 1]

    def _convert_pdf_page_to_image(
        self,
        file_path: Path,
        page_number: int = 0,
        dpi: int = 150,
    ) -> Optional[bytes]:
        """
        Convert a PDF page to optimized image bytes for vision model processing.

        Uses JPEG compression and resizing to reduce token cost while maintaining
        readability for text extraction.

        Args:
            file_path: Path to the PDF file
            page_number: Page number to convert (0-indexed)
            dpi: Resolution for the rendered image

        Returns:
            JPEG image bytes or None on failure
        """
        try:
            from io import BytesIO

            with pymupdf.open(str(file_path)) as doc:
                if page_number >= len(doc):
                    logging.error(
                        f"Page {page_number + 1} does not exist in {file_path.name}"
                    )
                    return None

                page = doc.load_page(page_number)
                zoom = dpi / 72
                pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))

                # Convert PyMuPDF pixmap to PIL Image
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                # Resize if larger than max size (maintain aspect ratio)
                if img.width > self.image_max_size or img.height > self.image_max_size:
                    img.thumbnail(
                        (self.image_max_size, self.image_max_size),
                        Image.Resampling.LANCZOS
                    )

                # Compress to JPEG to reduce token cost
                buffer = BytesIO()
                img.save(
                    buffer,
                    format="JPEG",
                    quality=self.image_quality,
                    optimize=True,
                )
                return buffer.getvalue()

        except Exception as e:
            logging.error(
                f"Failed to convert PDF page to image for {file_path.name}: {e}"
            )
            return None


def create_vision_processor(
    llm_engine: LLMEngine,
    text_extractor: TextExtractor,
    provider: str,
    model: str,
) -> VisionProcessor:
    """
    Factory function to create a VisionProcessor.

    Args:
        llm_engine: LLM engine instance
        text_extractor: Text extractor instance
        provider: Provider name
        model: Model name

    Returns:
        Configured VisionProcessor instance
    """
    return VisionProcessor(
        llm_engine=llm_engine,
        text_extractor=text_extractor,
        provider=provider,
        model=model,
    )
