import logging
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np
import pymupdf
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from config import CACHE_DIR, DEBUG_DIR, OCR_LANG, SUBPROCESS_TIMEOUT, TESSERACT_PATH
from utils import ensure_secure_dir, sanitize_filename

# Try to import OpenCV for skew correction
try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logging.warning(
        "OpenCV not installed. Skew correction will be disabled. Install with: pip install opencv-python"
    )

# Configure Tesseract
if TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)


class OCREngine:
    def __init__(
        self,
        enhanced: bool = False,
        debug: bool = False,
        use_mutool: bool = False,
        mutool_path: str = "mutool",
    ):
        self.enhanced = enhanced
        self.debug = debug
        self.use_mutool = use_mutool
        self.mutool_path = mutool_path

        # Validate mutool if requested
        if self.use_mutool:
            resolved_path = shutil.which(self.mutool_path)
            if resolved_path:
                self.mutool_path = resolved_path
            else:
                logging.warning(
                    f"mutool not found at {self.mutool_path}. "
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
            logging.info("OCR Engine initialized in MINIMAL mode (150 DPI)")

        self.MAX_DIMENSION = 2000

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
        except Exception:
            return 72

    def _deskew(self, image: Image.Image) -> Image.Image:
        """Estimate skew angle and rotate image to straighten it."""
        if not OPENCV_AVAILABLE:
            return image

        try:
            # Convert PIL image to numpy array
            data = np.asarray(image.convert("L"))

            # Threshold to get binary image
            _, binary = cv2.threshold(data, 128, 255, cv2.THRESH_BINARY_INV)  # type: ignore

            # Find contours
            contours, _ = cv2.findContours(  # type: ignore
                binary,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,  # type: ignore
            )

            if not contours:
                return image

            # Fit lines to contours and calculate skew angle
            angles = []
            for contour in contours:
                if len(contour) < 5:
                    continue
                # Fit ellipse to contour
                ellipse = cv2.fitEllipse(contour)  # type: ignore
                angle = ellipse[2]
                # Store the angle (minimum area rectangle gives angle relative to horizontal)
                angles.append(angle)

            if not angles:
                return image

            # Calculate median angle
            median_angle = np.median(angles)

            # Convert to degrees and adjust
            angle_degrees = median_angle

            # If angle is close to 90, it's actually 0 (vertical text)
            if abs(angle_degrees) > 45:
                angle_degrees = 90 - angle_degrees

            # Rotate image to correct skew
            image_array = np.asarray(image)
            height, width = image_array.shape[:2]
            center = (width // 2, height // 2)

            # Get rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)  # type: ignore

            # Calculate new bounding dimensions
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])
            new_width = int((height * sin) + (width * cos))
            new_height = int((height * cos) + (width * sin))

            # Adjust rotation matrix to account for translation
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]

            # Perform rotation
            rotated_array = cv2.warpAffine(  # type: ignore
                image_array,
                rotation_matrix,
                (new_width, new_height),
                flags=cv2.INTER_CUBIC,  # type: ignore
                borderMode=cv2.BORDER_REPLICATE,  # type: ignore
            )

            # Convert back to PIL Image
            rotated_image = Image.fromarray(rotated_array)

            return rotated_image
        except Exception as e:
            logging.debug(f"Skew correction failed: {e}")
            return image

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
            pages_to_extract = list(range(min(15, total_pages)))
            if total_pages > 30:
                mid = total_pages // 2
                pages_to_extract += [mid, mid + 1]
            if total_pages > 30:
                pages_to_extract += [total_pages - 1, total_pages]
            pages_to_extract = sorted(
                set(p for p in pages_to_extract if 0 <= p < total_pages)
            )

            extracted_text = []
            total_chars = 0
            for pno in pages_to_extract:
                page = doc.load_page(pno)
                source_dpi = self.get_page_source_dpi(page)
                target_dpi = min(self.cfg["dpi"], source_dpi)

                use_mutool_for_page = (
                    self.use_mutool and file_path.suffix.lower() == ".pdf"
                )

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
                        subprocess.run(
                            cmd, check=True, capture_output=True, timeout=SUBPROCESS_TIMEOUT
                        )
                        image = Image.open(temp_img)
                        image.load()
                    except subprocess.TimeoutExpired:
                        logging.error(f"mutool timed out for page {pno + 1}")
                    except Exception as e:
                        logging.error(f"mutool failed for page {pno + 1}: {e}")

                if not image:
                    zoom = target_dpi / 72
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
                    image = Image.frombytes(  # type: ignore
                        "RGB",
                        (pix.width, pix.height),
                        pix.samples,
                    )

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
                        image = image.point(lambda x: 0 if x < 140 else 255)  # type: ignore

                if self.debug and file_debug_dir:
                    quality_tag = "enhanced" if self.enhanced else "minimal"
                    image.save(file_debug_dir / f"page_{pno + 1}_{quality_tag}.png")

                # OCR with optimized config
                # --oem 3 (Default, base + LSTM), --psm 3
                try:
                    page_text = pytesseract.image_to_string(
                        image,
                        lang=OCR_LANG,
                        config=r"--oem 3 --psm 3",
                        timeout=SUBPROCESS_TIMEOUT,
                    )
                except RuntimeError as e:
                    logging.error(
                        f"Tesseract timed out or failed for page {pno + 1}: {e}"
                    )
                    page_text = ""

                extracted_text.append(page_text)
                total_chars += len(page_text)

                if temp_img and temp_img.exists():
                    try:
                        image.close()
                        temp_img.unlink()
                    except Exception:
                        pass

                if total_chars > 10000:
                    break

            doc.close()
            logging.info(
                f"OCR extracted text from {len(pages_to_extract)} pages of {file_path.name}"
            )
            return "\n\n".join(extracted_text).strip()

        except Exception as e:
            logging.error(f"OCR Pipeline failed for {file_path.name}: {e}")
            return ""
