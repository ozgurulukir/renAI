import asyncio
import logging
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
import pytesseract  # type: ignore[import-untyped]
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from ..settings import (
    CACHE_DIR,
    DEBUG_DIR,
    TESSERACT_PATH,
    get_ocr_lang,
    get_subprocess_timeout,
)
from ..utils import ensure_secure_dir, sanitize_filename

# Try to detect OpenCV at module level for test collection support
try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Configure Tesseract
if TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)  # type: ignore[assignment]


class OCREngine:
    def __init__(
        self,
        enhanced: bool = False,
        debug: bool = False,
        use_mutool: bool = False,
        mutool_path: str = "mutool",
        silent: bool = False,
    ):
        self.enhanced = enhanced
        self.debug = debug
        self.use_mutool = use_mutool
        self.mutool_path = mutool_path
        self.silent = silent

        # Validate mutool if requested
        if self.use_mutool:
            resolved_path = self._validate_mutool(self.mutool_path)
            if resolved_path:
                self.mutool_path = resolved_path
            else:
                logging.warning(
                    f"mutool not found or invalid at {self.mutool_path}. "
                    "Falling back to standard extraction."
                )
                self.use_mutool = False

        # Simple two-tier configuration
        if self.enhanced:
            self.cfg = {
                "dpi": 300,
                "resize": 1.0,
                "enhance": True,
                "sharpen": True,
                "deskew": True,
                "threshold": True,
            }
            if not self.silent:
                logging.info(
                    "OCR Engine initialized in ENHANCED mode (300 DPI, deskew, threshold, sharpen)"
                )
        else:
            self.cfg = {
                "dpi": 150,
                "resize": 1.0,
                "enhance": False,
                "sharpen": False,
                "deskew": False,
                "threshold": False,
            }
            if not self.silent:
                logging.info("OCR Engine initialized in MINIMAL mode (150 DPI)")

        self.MAX_DIMENSION = 2000

    def _get_subprocess_kwargs(self) -> dict[str, Any]:
        """Consolidate Windows-specific subprocess flags."""
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            kwargs["startupinfo"] = startupinfo
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return kwargs

    def _validate_mutool(self, path: str) -> str | None:
        """
        Verify that the provided path points to a valid mutool executable.
        Returns the resolved absolute path if valid, otherwise None.
        """
        found = shutil.which(path)
        if not found:
            return None

        # Resolve to absolute path to prevent relative path bypasses
        resolved_path = Path(found).resolve().absolute()
        resolved = str(resolved_path)

        # 1. Strict character whitelist check for the path
        # Allows alphanumeric, dots, underscores, hyphens, spaces, parentheses, and path separators
        if not re.match(r"^[a-zA-Z0-9._/\\:\- ()]+$", resolved):
            logging.warning(f"Rejected mutool path with suspicious characters: {resolved}")
            return None

        # 2. Basic filename check (case-insensitive)
        name = resolved_path.name.lower()
        # On Windows it might be mutool.exe
        if not name.startswith("mutool"):
            logging.warning(f"Rejected non-mutool executable: {resolved}")
            return None

        # 3. Identity check via version command
        try:
            kwargs = self._get_subprocess_kwargs()
            result = subprocess.run(
                [resolved, "-v"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
                **kwargs,
            )
            # mutool -v output typically contains "mutool" and "MuPDF"
            output = (result.stdout + result.stderr).lower()
            if "mutool" in output or "mupdf" in output:
                return resolved
            logging.warning(f"Executable at {resolved} did not respond like mutool")
            return None
        except Exception as e:
            logging.warning(f"Failed to verify mutool at {resolved}: {e}")
            return None

    @staticmethod
    def get_page_source_dpi(page) -> int:
        """Estimate the maximum DPI of images on a PDF page."""
        try:
            images = page.get_image_info(xrefs=True)
            if not images:
                return 72
            max_dpi = 72
            for img in images:
                display_width = img["bbox"][2] - img["bbox"][0]
                display_height = img["bbox"][3] - img["bbox"][1]
                if display_width > 0 and display_height > 0:
                    dpi_x = (img["width"] / display_width) * 72
                    dpi_y = (img["height"] / display_height) * 72
                    max_dpi = max(max_dpi, dpi_x, dpi_y)
            return max_dpi
        except Exception as e:
            logging.debug(f"Failed to calculate source DPI: {e}")
            return 72

    def _deskew(self, image: Image.Image) -> Image.Image:
        """Estimate skew angle and rotate image to straighten it."""
        if not OPENCV_AVAILABLE:
            return image

        try:
            # Convert PIL image to numpy array
            image_array = np.asarray(image)

            # Handle grayscale vs color (ensure we have a 1-channel image for thresholding)
            THREE_CHANNELS = 3
            FOUR_CHANNELS = 4
            if len(image_array.shape) == THREE_CHANNELS:
                if image_array.shape[2] == FOUR_CHANNELS:
                    gray = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
                else:
                    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array

            # Threshold to get binary image (text white, background black)
            # Use Otsu's thresholding for better results on varying contrast
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Morphological operations to join text into lines/blocks
            # This helps in detecting the overall skew of text lines rather than individual chars
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
            dilated = cv2.dilate(binary, kernel, iterations=1)

            # Find contours of the dilated blocks
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return image

            angles = []
            for contour in contours:
                # Filter small contours that might be noise
                MIN_CONTOUR_AREA = 100
                if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                    continue

                rect = cv2.minAreaRect(contour)
                (w, h) = rect[1]
                angle = rect[2]

                # Adjust angle to be relative to the horizontal axis
                # In OpenCV 4.x, minAreaRect angle is usually in [-90, 0)
                if w < h:
                    angle += 90

                # After adjustment, angle is in [-45, 45] for horizontal-ish boxes.
                angles.append(angle)

            if not angles:
                return image

            # Calculate median angle to avoid outliers
            median_angle = np.median(angles)

            # If the angle is very small, don't bother rotating
            MIN_ROTATION_ANGLE = 0.1
            if abs(median_angle) < MIN_ROTATION_ANGLE:
                return image

            # Rotate image to correct skew
            height, width = image_array.shape[:2]
            center = (width // 2, height // 2)

            # Get rotation matrix (positive values mean counter-clockwise rotation)
            rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

            # Calculate new bounding dimensions
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])
            new_width = int((height * sin) + (width * cos))
            new_height = int((height * cos) + (width * sin))

            # Adjust rotation matrix to account for translation
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]

            # Perform rotation - Use white background for documents
            IS_GRAYSCALE = 2
            rotated_array = cv2.warpAffine(
                image_array,
                rotation_matrix,
                (new_width, new_height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255 if len(image_array.shape) == IS_GRAYSCALE else (255, 255, 255),
            )
            return Image.fromarray(rotated_array)
        except Exception as e:
            logging.debug(f"Skew correction failed: {e}")
            return image

    def _get_page_sampling(self, total_pages: int) -> list[int]:
        """Page sampling strategy."""
        SAMPLE_THRESHOLD = 30
        pages_to_extract = list(range(min(15, total_pages)))
        if total_pages > SAMPLE_THRESHOLD:
            mid = total_pages // 2
            pages_to_extract += [mid, mid + 1]
        if total_pages > SAMPLE_THRESHOLD:
            pages_to_extract += [total_pages - 1, total_pages]
        return sorted({p for p in pages_to_extract if 0 <= p < total_pages})

    async def _bulk_extract_images(
        self, file_path: Path, pages: list[int], target_dpi: int
    ) -> dict[int, Path]:
        """Extract multiple pages from a PDF in a single mutool call asynchronously."""
        if not pages:
            return {}

        batch_id = uuid.uuid4().hex
        output_pattern = CACHE_DIR / f"temp_batch_{batch_id}_%d.png"

        # mutool uses 1-based page numbers
        page_range = ",".join(str(p + 1) for p in pages)

        try:
            cmd = [
                self.mutool_path,
                "draw",
                "-q",
                "-o",
                str(output_pattern),
                "-c",
                "gray",
                "-r",
                str(int(target_dpi)),
                str(file_path),
                page_range,
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **self._get_subprocess_kwargs(),
            )

            try:
                _stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=get_subprocess_timeout()
                )
                if process.returncode != 0:
                    logging.error(f"mutool bulk error: {stderr.decode()}")
                    return {}
            except TimeoutError:
                process.kill()
                logging.error("mutool bulk extraction timed out")
                return {}

            # Map back to 0-based pno
            results = {}
            for pno in pages:
                img_path = CACHE_DIR / f"temp_batch_{batch_id}_{pno + 1}.png"
                if img_path.exists():
                    results[pno] = img_path
            return results

        except Exception as e:
            logging.error(f"Bulk mutool extraction failed: {e}")
            return {}

    async def _extract_image_from_page(
        self,
        doc: pymupdf.Document,
        pno: int,
        file_path: Path,
        target_dpi: int,
        pre_extracted_path: Path | None = None,
    ) -> tuple[Image.Image | None, Path | None]:
        """Handles mutool or pymupdf fallback to extract an image from a page asynchronously."""
        image: Image.Image | None = None
        if pre_extracted_path and pre_extracted_path.exists():
            try:
                # CPU intensive image loading
                image = await asyncio.to_thread(Image.open, pre_extracted_path)
                await asyncio.to_thread(image.load)
                return image, pre_extracted_path
            except Exception as e:
                logging.error(f"Failed to load pre-extracted image for page {pno + 1}: {e}")

        use_mutool_for_page = self.use_mutool and file_path.suffix.lower() == ".pdf"
        temp_img = None

        if use_mutool_for_page:
            temp_img = CACHE_DIR / f"temp_page_{pno}_{uuid.uuid4().hex}.png"
            try:
                cmd = [
                    self.mutool_path,
                    "draw",
                    "-q",
                    "-o",
                    str(temp_img),
                    "-c",
                    "gray",
                    "-r",
                    str(int(target_dpi)),
                    str(file_path),
                    str(pno + 1),
                ]

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **self._get_subprocess_kwargs(),
                )

                try:
                    _stdout, stderr = await asyncio.wait_for(
                        process.communicate(), timeout=get_subprocess_timeout()
                    )
                    if process.returncode == 0:
                        image = await asyncio.to_thread(Image.open, temp_img)
                        await asyncio.to_thread(image.load)
                    else:
                        logging.error(f"mutool page error: {stderr.decode()}")
                except TimeoutError:
                    process.kill()
                    logging.error(f"mutool timed out for page {pno + 1}")
            except Exception as e:
                logging.error(f"mutool failed for page {pno + 1}: {e}")

        if not image:

            def _get_pixmap_thread():
                page = doc.load_page(pno)
                zoom = target_dpi / 72
                pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
                return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            image = await asyncio.to_thread(_get_pixmap_thread)

        return image, temp_img

    async def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Handles resizing and enhancements in a thread."""
        return await asyncio.to_thread(self._sync_preprocess_image, image)

    def _sync_preprocess_image(self, image: Image.Image) -> Image.Image:
        """Sync version of preprocessing for thread execution."""
        scale = self.cfg["resize"]
        if max(image.width, image.height) * scale > self.MAX_DIMENSION:
            scale = self.MAX_DIMENSION / max(image.width, image.height)

        if scale != 1.0:
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        image = image.convert("L")

        if self.cfg["enhance"]:
            if self.cfg["deskew"]:
                image = self._deskew(image)
            image = ImageOps.autocontrast(image)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            if self.cfg["sharpen"]:
                image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            if self.cfg["threshold"]:
                CONTRAST_THRESHOLD = 140
                image = image.point(lambda x: 0 if x < CONTRAST_THRESHOLD else 255)
        return image

    async def _perform_ocr(self, image: Image.Image, pno: int) -> str:
        """Calls Tesseract in a thread."""
        try:

            def _tesseract_thread():
                return pytesseract.image_to_string(
                    image,
                    lang=get_ocr_lang(),
                    config=r"--oem 3 --psm 3",
                    timeout=get_subprocess_timeout(),
                )

            return await asyncio.to_thread(_tesseract_thread)
        except Exception as e:
            logging.error(f"Tesseract failed for page {pno + 1}: {e}")
            return ""

    async def extract_text(self, file_path: Path) -> str:  # noqa: C901, PLR0912
        """Extract text from a document asynchronously using OCR."""
        try:
            file_debug_dir = None
            if self.debug:
                base_name = file_path.stem
                file_debug_dir = DEBUG_DIR / sanitize_filename(base_name)
                ensure_secure_dir(file_debug_dir)

            # PyMuPDF open is fast, but let's be safe
            doc = await asyncio.to_thread(pymupdf.open, str(file_path))
            total_pages = len(doc)

            pages_to_extract = self._get_page_sampling(total_pages)

            bulk_images = {}
            if self.use_mutool and file_path.suffix.lower() == ".pdf":
                bulk_images = await self._bulk_extract_images(
                    file_path, pages_to_extract, int(self.cfg["dpi"])
                )

            extracted_text = []
            total_chars = 0
            for pno in pages_to_extract:
                pre_path = bulk_images.get(pno)
                target_dpi = self.cfg["dpi"]

                if not pre_path:
                    # In thread to avoid blocking loop with PDF info extraction
                    def _get_dpi(p_no):
                        page = doc.load_page(p_no)
                        return min(self.cfg["dpi"], self.get_page_source_dpi(page))

                    target_dpi = await asyncio.to_thread(_get_dpi, pno)

                image, temp_img = await self._extract_image_from_page(
                    doc, pno, file_path, int(target_dpi), pre_extracted_path=pre_path
                )

                if image:
                    image = await self._preprocess_image(image)

                    if self.debug and file_debug_dir:

                        def _save_debug(img, p_idx):
                            quality_tag = "enhanced" if self.enhanced else "minimal"
                            img.save(file_debug_dir / f"page_{p_idx + 1}_{quality_tag}.png")

                        await asyncio.to_thread(_save_debug, image, pno)

                    page_text = await self._perform_ocr(image, pno)
                    extracted_text.append(page_text)
                    total_chars += len(page_text)

                    if temp_img and temp_img.exists():
                        try:
                            image.close()
                            await asyncio.to_thread(temp_img.unlink)
                        except Exception as e:
                            logging.debug(f"Cleanup of temp_img failed: {e}")

                MAX_OCR_CHARS = 10000
                if total_chars > MAX_OCR_CHARS:
                    break

            await asyncio.to_thread(doc.close)

            for img_path in bulk_images.values():
                if img_path.exists():
                    try:
                        await asyncio.to_thread(img_path.unlink)
                    except Exception as e:
                        logging.debug(f"Cleanup of bulk_img failed: {e}")

            logging.info(
                f"OCR extracted text from {len(pages_to_extract)} pages of {file_path.name}"
            )
            return "\n\n".join(extracted_text).strip()

        except Exception as e:
            logging.error(f"OCR Pipeline failed for {file_path.name}: {e}")
            return ""
