"""OCR Engine for document text extraction using Tesseract, MuPDF, and OpenCV.

Note:
    atexit registration (atexit.register(OCREngine.shutdown_process_pool, wait=False))
    handles normal interpreter exit. Abrupt process termination (e.g., os._exit() or SIGKILL)
    will not trigger atexit cleanup hooks.
"""

import asyncio
import atexit
import contextlib
import logging
import re
import shutil
import subprocess
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pymupdf
import pytesseract  # type: ignore[import-untyped]
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from ..config import (
    DEBUG_DIR,
    TESSERACT_PATH,
    get_ocr_lang,
    get_subprocess_timeout,
)
from ..utils import ensure_secure_dir, sanitize_filename
from .render import RenderedPage, create_page_renderer, get_subprocess_kwargs

if TYPE_CHECKING:
    import cv2  # type: ignore[import-untyped]

# Try to detect OpenCV at module level for test collection support
try:
    import cv2 as _cv2

    cv2 = _cv2
    OPENCV_AVAILABLE: bool = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    OPENCV_AVAILABLE = False

# Configure Tesseract
if TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)  # type: ignore[assignment]

# Preprocessing tuning constants. These are heuristics tuned for the default
# 150/300 DPI render tiers (see ocr_dpi / ocr_enhanced_dpi settings); they are
# image-domain fixed rather than operator knobs.
# Longest-edge downscale limit before Tesseract: keeps per-page OCR memory and
# runtime bounded regardless of source scan resolution.
OCR_MAX_DIMENSION = 2000
# Morphological kernel for deskew text-line detection (~150 DPI text lines).
DESKEW_LINE_KERNEL = (30, 5)
# Enhanced-mode contrast boost and unsharp mask; tuned for scanned documents
# where faint print needs reinforcement before binarization.
ENHANCED_CONTRAST_FACTOR = 1.5
ENHANCED_UNSHARP_RADIUS = 2
ENHANCED_UNSHARP_PERCENT = 150
ENHANCED_UNSHARP_THRESHOLD = 3
# Tesseract OCR engine mode: 3 = default, OEM-chosen (LSTM where available).
TESSERACT_OEM = 3


class OCREngine:
    """OCR Engine managing image preprocessing and Tesseract OCR execution.

    The shared ProcessPoolExecutor handles CPU-heavy image operations and is registered for
    clean shutdown via atexit on normal interpreter exit.
    """

    _process_pool: ProcessPoolExecutor | None = None
    _max_workers: int = 4
    _process_pool_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        enhanced: bool = False,
        debug: bool = False,
        use_mutool: bool = False,
        mutool_path: str = "mutool",
        silent: bool = False,
        max_workers: int = 4,
        dpi: int = 150,
        enhanced_dpi: int = 300,
        head_pages: int = 5,
        tail_pages: int = 3,
        psm: int = 3,
    ):
        self.enhanced = enhanced
        self.debug = debug
        self.use_mutool = use_mutool
        self.mutool_path = mutool_path
        self.silent = silent
        self.max_workers = max_workers
        self.head_pages = head_pages
        self.tail_pages = tail_pages
        self.psm = psm
        OCREngine._max_workers = max_workers

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

        # Simple two-tier configuration; DPI tiers are settings-backed knobs
        # (ocr_dpi / ocr_enhanced_dpi) because scan density varies per library.
        if self.enhanced:
            self.cfg = {
                "dpi": enhanced_dpi,
                "resize": 1.0,
                "enhance": True,
                "sharpen": True,
                "deskew": True,
                "threshold": True,
            }
            if not self.silent:
                logging.debug(
                    f"OCR Engine initialized in ENHANCED mode ({enhanced_dpi} DPI, "
                    "deskew, threshold, sharpen)"
                )
        else:
            self.cfg = {
                "dpi": dpi,
                "resize": 1.0,
                "enhance": False,
                "sharpen": False,
                "deskew": False,
                "threshold": False,
            }
            if not self.silent:
                logging.debug(f"OCR Engine initialized in MINIMAL mode ({dpi} DPI)")

        self.MAX_DIMENSION = OCR_MAX_DIMENSION

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
            kwargs = get_subprocess_kwargs()
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

    def _calculate_skew_angle(self, image_array: np.ndarray) -> float | None:
        """Calculate skew angle using OpenCV contours and Otsu thresholding."""
        if cv2 is None:
            return None
        THREE_CHANNELS = 3
        FOUR_CHANNELS = 4
        if len(image_array.shape) == THREE_CHANNELS:
            if image_array.shape[2] == FOUR_CHANNELS:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
            else:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DESKEW_LINE_KERNEL)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        angles = []
        for contour in contours:
            MIN_CONTOUR_AREA = 100
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue
            rect = cv2.minAreaRect(contour)
            (w, h) = rect[1]
            angle = rect[2]
            if w < h:
                angle += 90
            angles.append(angle)

        if not angles:
            return None

        median_angle = float(np.median(angles))
        MIN_ROTATION_ANGLE = 0.1
        if abs(median_angle) < MIN_ROTATION_ANGLE:
            return None
        return median_angle

    def _rotate_image(self, image_array: np.ndarray, angle: float) -> Image.Image:
        """Rotate image by specified angle to correct skew."""
        assert cv2 is not None
        height, width = image_array.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

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

    def _deskew(self, image: Image.Image) -> Image.Image:
        """Estimate skew angle and rotate image to straighten it."""
        if not OPENCV_AVAILABLE:
            return image

        try:
            image_array = np.asarray(image)
            skew_angle = self._calculate_skew_angle(image_array)
            if skew_angle is None:
                return image
            return self._rotate_image(image_array, skew_angle)
        except Exception as e:
            logging.debug(f"Skew correction failed: {e}")
            return image

    def _get_page_sampling(self, total_pages: int) -> list[int]:
        """Head-tail page sampling strategy for OCR.

        Samples the first ``head_pages`` and last ``tail_pages`` of the document,
        which is where metadata (title, author, references, ISBN) is located.
        Page counts are settings-backed (ocr_head_pages / ocr_tail_pages) because
        documents with metadata in appendices/colophons need deeper sampling,
        while OCR is roughly 100x slower than text extraction.
        """
        head = list(range(min(self.head_pages, total_pages)))
        tail_start = max(self.head_pages, total_pages - self.tail_pages)
        tail = list(range(tail_start, total_pages))
        return sorted(set(head + tail))

    @classmethod
    def shutdown_process_pool(cls, wait: bool = False) -> None:
        """Shutdown the shared ProcessPoolExecutor if running.

        Guarded by a module-level lock so concurrent callers can't race on
        shutdown before another call has created the pool.
        """
        with cls._process_pool_lock:
            if cls._process_pool is not None:
                cls._process_pool.shutdown(wait=wait)
                cls._process_pool = None

    def close(self, wait: bool = False) -> None:
        """Shutdown the shared ProcessPoolExecutor."""
        self.shutdown_process_pool(wait=wait)

    async def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Handles resizing and enhancements in a ProcessPoolExecutor to bypass GIL.

        Note: On Windows or environments using 'spawn' process context, pickling bound
        methods requires top-level module pickling support. Any pickling failure automatically
        falls back to asyncio.to_thread.
        """
        try:
            loop = asyncio.get_running_loop()
            # Guard the check-then-act on pool creation so concurrent callers
            # don't race on startup (two pools / interleaved None checks).
            with OCREngine._process_pool_lock:
                if OCREngine._process_pool is None:
                    OCREngine._process_pool = ProcessPoolExecutor(
                        max_workers=getattr(self, "max_workers", OCREngine._max_workers)
                    )
            return await loop.run_in_executor(
                OCREngine._process_pool, self._sync_preprocess_image, image
            )
        except Exception as e:
            logging.debug(f"ProcessPool execution fallback to thread: {e}")
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
            image = enhancer.enhance(ENHANCED_CONTRAST_FACTOR)
            if self.cfg["sharpen"]:
                image = image.filter(
                    ImageFilter.UnsharpMask(
                        radius=ENHANCED_UNSHARP_RADIUS,
                        percent=ENHANCED_UNSHARP_PERCENT,
                        threshold=ENHANCED_UNSHARP_THRESHOLD,
                    )
                )
            if self.cfg["threshold"]:
                CONTRAST_THRESHOLD = 140
                image = image.point(lambda x: 0 if x < CONTRAST_THRESHOLD else 255)  # type: ignore[operator]
        return image

    async def _run_tesseract(self, image: Image.Image, pno: int) -> str:
        """Execute Tesseract OCR and clean up extracted text."""
        return await self._perform_ocr(image, pno)

    async def _perform_ocr(self, image: Image.Image, pno: int) -> str:
        """Calls Tesseract in a thread with configured timeout."""
        try:

            def _tesseract_thread():
                return pytesseract.image_to_string(
                    image,
                    lang=get_ocr_lang(),
                    config=f"--oem {TESSERACT_OEM} --psm {self.psm}",
                    timeout=get_subprocess_timeout(),
                )

            return await asyncio.to_thread(_tesseract_thread)
        except Exception as e:
            logging.error(f"Tesseract failed for page {pno + 1}: {e}")
            return ""

    async def extract_text(self, file_path: Path, target_chars: int = 10000) -> str:  # noqa: C901, PLR0912, PLR0915
        """Extract text from a document asynchronously using OCR."""
        if file_path.is_symlink():
            logging.warning(
                "Skipping symlink to prevent arbitrary file read (OCR): %s",
                file_path.name,
            )
            return ""
        doc = None
        bulk_pages: dict[int, RenderedPage] = {}
        per_page_temps: list[Path] = []
        try:
            file_debug_dir = None
            if self.debug:
                base_name = file_path.stem
                file_debug_dir = DEBUG_DIR / sanitize_filename(base_name)
                ensure_secure_dir(file_debug_dir)

            doc = await asyncio.to_thread(pymupdf.open, str(file_path))
            total_pages = len(doc)

            pages_to_extract = self._get_page_sampling(total_pages)

            is_pdf = file_path.suffix.lower() == ".pdf"
            use_mutool = self.use_mutool and is_pdf
            # Factory picks the backend; this flow only orchestrates its own
            # sampling + DPI policy (bulk @ cfg dpi, per-page @ dynamic dpi).
            mutool_renderer = (
                create_page_renderer(use_mutool=True, mutool_path=self.mutool_path)
                if use_mutool
                else None
            )
            pymupdf_renderer = create_page_renderer(doc=doc)

            if mutool_renderer is not None:
                bulk_pages = await mutool_renderer.render_pages(
                    file_path, pages_to_extract, int(self.cfg["dpi"])
                )

            extracted_text = []
            total_chars = 0
            for pno in pages_to_extract:
                page_result = bulk_pages.get(pno)
                page_temp: Path | None = None

                if page_result is None:

                    def _get_dpi(p_no):
                        page = doc.load_page(p_no)
                        return min(self.cfg["dpi"], self.get_page_source_dpi(page))

                    target_dpi = await asyncio.to_thread(_get_dpi, pno)

                    # Fallback chain preserved: per-page mutool retry, then pymupdf.
                    if mutool_renderer is not None:
                        page_result = (
                            await mutool_renderer.render_pages(file_path, [pno], int(target_dpi))
                        ).get(pno)
                    if page_result is None:
                        page_result = (
                            await pymupdf_renderer.render_pages(file_path, [pno], int(target_dpi))
                        ).get(pno)
                    page_temp = page_result.temp_path if page_result else None
                    if page_temp:
                        per_page_temps.append(page_temp)

                if page_result is not None and page_result.image is not None:
                    image = await self._preprocess_image(page_result.image)

                    if self.debug and file_debug_dir:

                        def _save_debug(img, p_idx):
                            quality_tag = "enhanced" if self.enhanced else "minimal"
                            img.save(file_debug_dir / f"page_{p_idx + 1}_{quality_tag}.png")

                        await asyncio.to_thread(_save_debug, image, pno)

                    page_text = await self._run_tesseract(image, pno)
                    extracted_text.append(page_text)
                    total_chars += len(page_text)

                    # Only per-page rendered temps are cleaned here; bulk temps
                    # are caller-owned until the finally block below.
                    if page_temp and page_temp.exists():
                        try:
                            image.close()
                            await asyncio.to_thread(page_temp.unlink)
                        except Exception as e:
                            logging.debug(f"Cleanup of temp_img failed: {e}")

                if total_chars > target_chars:
                    break

            logging.debug(
                f"OCR extracted text from {len(pages_to_extract)} pages of {file_path.name}"
            )
            return "\n\n".join(extracted_text).strip()

        except Exception as e:
            logging.error(f"OCR Pipeline failed for {file_path.name}: {e}")
            return ""
        finally:
            if doc:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(doc.close)
            for rendered in bulk_pages.values():
                if rendered.temp_path and rendered.temp_path.exists():
                    try:
                        await asyncio.to_thread(rendered.temp_path.unlink)
                    except Exception as e:
                        logging.debug(f"Cleanup of bulk temp file failed: {e}")
            # Per-page temps are usually unlinked right after their OCR; this
            # catches any left behind by an exception between hand-back and
            # that cleanup (previously a leak window).
            for temp_path in per_page_temps:
                if temp_path.exists():
                    try:
                        await asyncio.to_thread(temp_path.unlink)
                    except Exception as e:
                        logging.debug(f"Cleanup of per-page temp file failed: {e}")


atexit.register(OCREngine.shutdown_process_pool, wait=False)
