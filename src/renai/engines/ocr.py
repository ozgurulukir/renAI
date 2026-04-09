import logging
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pymupdf
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from ..settings import (
    CACHE_DIR,
    DEBUG_DIR,
    OCR_LANG,
    SUBPROCESS_TIMEOUT,
    TESSERACT_PATH,
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

    def _get_subprocess_kwargs(self) -> dict:
        """Consolidate Windows-specific subprocess flags."""
        kwargs = {}
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            kwargs["startupinfo"] = startupinfo
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return kwargs

    def _validate_mutool(self, path: str) -> Optional[str]:
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
                timeout=5,
                **kwargs,
            )
            # mutool -v output typically contains "mutool" and "MuPDF"
            output = (result.stdout + result.stderr).lower()
            if "mutool" in output or "mupdf" in output:
                return resolved
            else:
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
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 4:
                    gray = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
                else:
                    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array

            # Threshold to get binary image (text white, background black)
            # Use Otsu's thresholding for better results on varying contrast
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # Morphological operations to join text into lines/blocks
            # This helps in detecting the overall skew of text lines rather than individual chars
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
            dilated = cv2.dilate(binary, kernel, iterations=1)

            # Find contours of the dilated blocks
            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return image

            angles = []
            for contour in contours:
                # Filter small contours that might be noise
                if cv2.contourArea(contour) < 100:
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
            if abs(median_angle) < 0.1:
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
            border_value = 255 if len(image_array.shape) == 2 else (255, 255, 255)
            rotated_array = cv2.warpAffine(
                image_array,
                rotation_matrix,
                (new_width, new_height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_value,
            )

            # Convert back to PIL Image
            rotated_image = Image.fromarray(rotated_array)

            return rotated_image
        except Exception as e:
            logging.debug(f"Skew correction failed: {e}")
            return image

    def _get_pages_to_extract(self, total_pages: int) -> list[int]:
        """Page sampling strategy."""
        pages_to_extract = list(range(min(15, total_pages)))
        if total_pages > 30:
            mid = total_pages // 2
            pages_to_extract += [mid, mid + 1]
        if total_pages > 30:
            pages_to_extract += [total_pages - 1, total_pages]
        return sorted(set(p for p in pages_to_extract if 0 <= p < total_pages))

    def _bulk_extract_images(
        self, file_path: Path, pages: list[int], target_dpi: int
    ) -> dict[int, Path]:
        """Extract multiple pages from a PDF in a single mutool call."""
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

            kwargs = self._get_subprocess_kwargs()

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=SUBPROCESS_TIMEOUT,
                **kwargs,
            )

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

    def _extract_image_from_page(
        self,
        doc: pymupdf.Document,
        pno: int,
        file_path: Path,
        target_dpi: int,
        pre_extracted_path: Optional[Path] = None,
    ) -> tuple[Optional[Image.Image], Optional[Path]]:
        """Handles mutool or pymupdf fallback to extract an image from a page."""
        if pre_extracted_path and pre_extracted_path.exists():
            try:
                image = Image.open(pre_extracted_path)
                image.load()
                return image, pre_extracted_path
            except Exception as e:
                logging.error(
                    f"Failed to load pre-extracted image for page {pno + 1}: {e}"
                )
        use_mutool_for_page = self.use_mutool and file_path.suffix.lower() == ".pdf"

        image = None
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
                kwargs = self._get_subprocess_kwargs()

                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    timeout=SUBPROCESS_TIMEOUT,
                    **kwargs,
                )
                image = Image.open(temp_img)
                image.load()
            except subprocess.TimeoutExpired:
                logging.error(f"mutool timed out for page {pno + 1}")
            except Exception as e:
                logging.error(f"mutool failed for page {pno + 1}: {e}")

        if not image:
            page = doc.load_page(pno)
            zoom = target_dpi / 72
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
            image = Image.frombytes(
                "RGB",
                (pix.width, pix.height),
                pix.samples,
            )
        return image, temp_img

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Handles resizing, grayscale conversion, deskewing, and enhancements."""
        # Pre-processing pipeline
        scale = self.cfg["resize"]
        if max(image.width, image.height) * scale > self.MAX_DIMENSION:
            scale = self.MAX_DIMENSION / max(image.width, image.height)

        if scale != 1.0:
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # 1. Grayscale
        image = image.convert("L")

        if self.cfg["enhance"]:
            # 1. Deskew (if enabled)
            if self.cfg["deskew"]:
                image = self._deskew(image)

            # 2. Contrast enhancement (Auto-contrast + Brightness boost)
            image = ImageOps.autocontrast(image)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)

            # 3. Sharpening (Unsharp Mask equivalent)
            if self.cfg["sharpen"]:
                image = image.filter(
                    ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
                )

            # 4. Thresholding (Binarization)
            if self.cfg["threshold"]:
                # Convert to 1-bit black and white
                image = image.point(lambda x: 0 if x < 140 else 255)
        return image

    def _perform_ocr(self, image: Image.Image, pno: int) -> str:
        """Calls Tesseract to extract text from the processed image."""
        # OCR with optimized config
        # --oem 3 (Default, base + LSTM), --psm 3
        try:
            page_text = pytesseract.image_to_string(
                image,
                lang=OCR_LANG,
                config=r"--oem 3 --psm 3",
                timeout=SUBPROCESS_TIMEOUT,
            )
            return page_text
        except RuntimeError as e:
            logging.error(f"Tesseract timed out or failed for page {pno + 1}: {e}")
            return ""

    def extract_text(self, file_path: Path) -> str:
        """Extract text from a document using OCR."""
        try:
            file_debug_dir = None
            if self.debug:
                base_name = file_path.stem
                file_debug_dir = DEBUG_DIR / sanitize_filename(base_name)
                ensure_secure_dir(file_debug_dir)

            doc = pymupdf.open(str(file_path))
            total_pages = len(doc)

            # Page sampling strategy
            pages_to_extract = self._get_pages_to_extract(total_pages)

            # Bulk extract if mutool is available and file is PDF
            bulk_images = {}
            if self.use_mutool and file_path.suffix.lower() == ".pdf":
                bulk_images = self._bulk_extract_images(
                    file_path, pages_to_extract, int(self.cfg["dpi"])
                )

            extracted_text = []
            total_chars = 0
            for pno in pages_to_extract:
                pre_path = bulk_images.get(pno)

                if pre_path:
                    # Use a default target_dpi when using pre-extracted images
                    # as we don't want to load the page just to calculate source_dpi
                    target_dpi = self.cfg["dpi"]
                else:
                    page = doc.load_page(pno)
                    source_dpi = self.get_page_source_dpi(page)
                    target_dpi = min(self.cfg["dpi"], source_dpi)

                image, temp_img = self._extract_image_from_page(
                    doc, pno, file_path, int(target_dpi), pre_extracted_path=pre_path
                )

                if image:
                    image = self._preprocess_image(image)

                    if self.debug and file_debug_dir:
                        quality_tag = "enhanced" if self.enhanced else "minimal"
                        image.save(file_debug_dir / f"page_{pno + 1}_{quality_tag}.png")

                    page_text = self._perform_ocr(image, pno)

                    extracted_text.append(page_text)
                    total_chars += len(page_text)

                    if temp_img and temp_img.exists():
                        try:
                            image.close()
                            temp_img.unlink()
                        except Exception as e:
                            logging.debug(f"Cleanup of temp_img failed: {e}")

                if total_chars > 10000:
                    break

            doc.close()

            # Cleanup any remaining bulk images that weren't processed/unlinked
            for img_path in bulk_images.values():
                if img_path.exists():
                    try:
                        img_path.unlink()
                    except Exception as e:
                        logging.debug(f"Cleanup of bulk_img failed: {e}")

            logging.info(
                f"OCR extracted text from {len(pages_to_extract)} pages of {file_path.name}"
            )
            return "\n\n".join(extracted_text).strip()

        except Exception as e:
            logging.error(f"OCR Pipeline failed for {file_path.name}: {e}")
            return ""
