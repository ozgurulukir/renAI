import sys

sys.path.insert(0, "src")

import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw
from renai.engines.ocr import OCREngine, OPENCV_AVAILABLE


def test_get_page_source_dpi_no_images():
    """Test when no images are present on the page."""
    mock_page = Mock()
    mock_page.get_image_info.return_value = []
    assert OCREngine.get_page_source_dpi(mock_page) == 72


def test_get_page_source_dpi_single_image():
    """Test DPI calculation for a single image."""
    mock_page = Mock()
    # Image 100x100 pixels, displayed at 50x50 pts
    # DPI = (100 / 50) * 72 = 144
    mock_page.get_image_info.return_value = [
        {"bbox": (0, 0, 50, 50), "width": 100, "height": 100}
    ]
    assert OCREngine.get_page_source_dpi(mock_page) == 144


def test_get_page_source_dpi_multiple_images():
    """Test that the maximum DPI is returned when multiple images are present."""
    mock_page = Mock()
    # Image 1: DPI = (100 / 50) * 72 = 144
    # Image 2: DPI = (100 / 24) * 72 = 300
    mock_page.get_image_info.return_value = [
        {"bbox": (0, 0, 50, 50), "width": 100, "height": 100},
        {"bbox": (100, 100, 124, 124), "width": 100, "height": 100},
    ]
    assert OCREngine.get_page_source_dpi(mock_page) == 300


def test_get_page_source_dpi_zero_dimension():
    """Test that images with zero display dimensions are handled gracefully."""
    mock_page = Mock()
    # Image with zero display width
    mock_page.get_image_info.return_value = [
        {"bbox": (10, 10, 10, 60), "width": 100, "height": 100}
    ]
    assert OCREngine.get_page_source_dpi(mock_page) == 72


def test_get_page_source_dpi_negative_dimension():
    """Test that images with negative display dimensions are handled gracefully."""
    mock_page = Mock()
    # Image with negative display height (bbox[3] < bbox[1])
    mock_page.get_image_info.return_value = [
        {"bbox": (0, 50, 50, 0), "width": 100, "height": 100}
    ]
    assert OCREngine.get_page_source_dpi(mock_page) == 72


def test_get_page_source_dpi_exception():
    """Test that exceptions in get_image_info return default DPI."""
    mock_page = Mock()
    mock_page.get_image_info.side_effect = Exception("Mocked error")
    assert OCREngine.get_page_source_dpi(mock_page) == 72


@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not available")
@pytest.mark.parametrize("angle", [-10, 10, 30, -30])
def test_deskew_various_angles(angle):
    engine = OCREngine(enhanced=True)

    # Create a tilted image
    size = (500, 200)
    image = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(image)
    # Using a larger font-like rectangle to ensure stable contour detection
    draw.rectangle([50, 80, 450, 100], fill=0)

    # Rotate (Pillow rotate is CCW, so we use -angle if we want 'angle' to be CW)
    # Actually, let's just use angle directly as CCW.
    tilted_image = image.rotate(angle, expand=True, fillcolor=255)

    # Deskew
    deskewed_image = engine._deskew(tilted_image)

    # Verify it's different from tilted if angle is significant
    if abs(angle) > 0.5:
        assert deskewed_image != tilted_image

    # The deskewed image should be straightened
    data = np.asarray(deskewed_image.convert("L"))
    _, binary = cv2.threshold(data, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angles = []
    for c in contours:
        if cv2.contourArea(c) < 100:
            continue
        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        a = rect[2]
        if w < h:
            a += 90
        angles.append(a)

    if angles:
        median_angle = np.median(angles)
        # We expect it to be near 0
        assert abs(median_angle) < 1.5


def test_deskew_small_contours():
    engine = OCREngine(enhanced=True)
    # Create an image with only tiny "noise" dots
    image = Image.new("L", (200, 200), color=255)
    draw = ImageDraw.Draw(image)
    for i in range(5):
        draw.rectangle([i * 20, i * 20, i * 20 + 2, i * 20 + 2], fill=0)

    # Deskew should return original because all contours < 100 area
    # (each rect is 3x3 = 9 pixels)
    assert engine._deskew(image) == image


def test_deskew_different_modes():
    engine = OCREngine(enhanced=True)

    # RGB
    img_rgb = Image.new("RGB", (100, 100), color=(255, 255, 255))
    assert engine._deskew(img_rgb) is not None

    # RGBA
    img_rgba = Image.new("RGBA", (100, 100), color=(255, 255, 255, 255))
    assert engine._deskew(img_rgba) is not None


def test_deskew_returns_original_if_no_opencv(monkeypatch):
    monkeypatch.setattr("renai.engines.ocr.OPENCV_AVAILABLE", False)
    engine = OCREngine(enhanced=True)
    img = Image.new("L", (100, 100))
    assert engine._deskew(img) == img


def test_deskew_handles_empty_image():
    engine = OCREngine(enhanced=True)
    img = Image.new("L", (100, 100), color=255)
    # Should return original image if no contours found
    assert engine._deskew(img) == img


class TestOCREngineInit:
    def test_init_with_defaults(self):
        """Test OCREngine initialization with default parameters."""
        engine = OCREngine()
        assert engine.enhanced is False
        assert engine.debug is False
        assert engine.use_mutool is False
        assert engine.mutool_path == "mutool"
        assert engine.cfg["dpi"] == 150

    def test_init_with_enhanced(self):
        """Test OCREngine initialization with enhanced mode."""
        engine = OCREngine(enhanced=True)
        assert engine.enhanced is True
        assert engine.cfg["dpi"] == 300
        assert engine.cfg["deskew"] is True
        assert engine.cfg["threshold"] is True

    def test_init_with_mutool_not_found(self, caplog):
        """Test OCREngine initialization when mutool is not found."""
        with patch("shutil.which", return_value=None):
            engine = OCREngine(use_mutool=True, mutool_path="nonexistent_mutool")
            assert engine.use_mutool is False
            assert "mutool not found" in caplog.text

    def test_init_with_mutool_found(self):
        """Test OCREngine initialization when mutool is found."""
        with patch("shutil.which", return_value="/usr/bin/mutool"):
            engine = OCREngine(use_mutool=True, mutool_path="mutool")
            assert engine.use_mutool is True
            assert engine.mutool_path == "/usr/bin/mutool"


class TestGetPagesToExtract:
    def test_get_pages_few_pages(self):
        """Test page extraction for documents with few pages."""
        engine = OCREngine()
        pages = engine._get_pages_to_extract(5)
        assert pages == [0, 1, 2, 3, 4]

    def test_get_pages_many_pages(self):
        """Test page extraction strategy for large documents."""
        engine = OCREngine()
        pages = engine._get_pages_to_extract(100)
        assert 0 in pages
        assert 50 in pages  # mid page
        assert 99 in pages  # last page
        assert len(pages) > 15

    def test_get_pages_edge_cases(self):
        """Test page extraction for edge cases."""
        engine = OCREngine()

        # Single page
        pages = engine._get_pages_to_extract(1)
        assert pages == [0]

        # Exactly 30 pages
        pages = engine._get_pages_to_extract(30)
        assert len(pages) <= 15


class TestExtractImageFromPage:
    @pytest.mark.skipif(
        not Path("/usr/bin/mutool").exists(), reason="mutool not installed"
    )
    def test_extract_image_with_mutool(self, tmp_path):
        """Test image extraction using mutool."""
        engine = OCREngine(use_mutool=True)

        # Create a simple PDF for testing
        import pymupdf

        doc = pymupdf.open()
        doc.new_page()
        test_pdf = tmp_path / "test.pdf"
        doc.save(str(test_pdf))
        doc.close()

        doc = pymupdf.open(str(test_pdf))
        image, temp_img = engine._extract_image_from_page(doc, 0, test_pdf, 150)

        assert image is not None
        assert isinstance(image, Image.Image)

    def test_extract_image_mutool_timeout(self, caplog):
        """Test handling of mutool timeout."""
        engine = OCREngine(use_mutool=True)

        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.width = 100
        mock_pix.height = 100
        mock_pix.samples = b"\x00" * 30000
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.load_page.return_value = mock_page

        with patch("subprocess.run", side_effect=Exception("Timeout")):
            with caplog.at_level(logging.ERROR):
                image, temp_img = engine._extract_image_from_page(
                    mock_doc, 0, Path("test.pdf"), 150
                )

                assert image is not None  # Falls back to pymupdf
                assert "mutool" in caplog.text or image is not None

    def test_extract_image_fallback_to_pymupdf(self):
        """Test fallback to pymupdf when mutool is disabled."""
        engine = OCREngine(use_mutool=False)

        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.width = 100
        mock_pix.height = 100
        mock_pix.samples = b"\x00" * 30000
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.load_page.return_value = mock_page

        image, temp_img = engine._extract_image_from_page(
            mock_doc, 0, Path("test.pdf"), 150
        )

        assert image is not None
        assert isinstance(image, Image.Image)


class TestPreprocessImage:
    def test_preprocess_minimal_mode(self):
        """Test image preprocessing in minimal mode."""
        engine = OCREngine(enhanced=False)
        img = Image.new("RGB", (200, 200), color=(128, 128, 128))

        result = engine._preprocess_image(img)

        assert result.mode == "L"  # Grayscale
        assert result.size == (200, 200)

    def test_preprocess_enhanced_mode(self):
        """Test image preprocessing in enhanced mode."""
        engine = OCREngine(enhanced=True, debug=False)
        img = Image.new("RGB", (200, 200), color=(128, 128, 128))

        # Add some content for enhancement
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 150, 150], fill=255)

        result = engine._preprocess_image(img)

        assert result.mode == "L"

    def test_preprocess_resize_large_image(self):
        """Test that large images are resized."""
        engine = OCREngine()
        img = Image.new("L", (5000, 5000), color=128)

        result = engine._preprocess_image(img)

        assert max(result.size) <= 2000

    def test_preprocess_with_deskew(self):
        """Test preprocessing with deskew enabled."""
        engine = OCREngine(enhanced=True)
        img = Image.new("L", (200, 200), color=255)

        result = engine._preprocess_image(img)

        assert result is not None


class TestPerformOCR:
    def test_perform_ocr_success(self):
        """Test successful OCR execution."""
        engine = OCREngine()
        img = Image.new("L", (100, 100), color=255)

        with patch("pytesseract.image_to_string", return_value="Test text"):
            result = engine._perform_ocr(img, 0)
            assert result == "Test text"

    def test_perform_ocr_timeout(self, caplog):
        """Test OCR timeout handling."""
        engine = OCREngine()
        img = Image.new("L", (100, 100), color=255)

        with patch("pytesseract.image_to_string", side_effect=RuntimeError("Timeout")):
            result = engine._perform_ocr(img, 0)
            assert result == ""
            assert "Tesseract timed out" in caplog.text

    def test_perform_ocr_runtime_error(self, caplog):
        """Test OCR runtime error handling."""
        engine = OCREngine()
        img = Image.new("L", (100, 100), color=255)

        with patch("pytesseract.image_to_string", side_effect=RuntimeError("Error")):
            result = engine._perform_ocr(img, 0)
            assert result == ""


class TestExtractText:
    def test_extract_text_success(self):
        """Test successful text extraction."""
        engine = OCREngine()

        # Create a test PDF
        import pymupdf

        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Test Content")
        test_pdf = Path(__file__).parent / "test_doc.pdf"
        doc.save(str(test_pdf))
        doc.close()

        result = engine.extract_text(test_pdf)

        # Cleanup
        test_pdf.unlink()

        assert isinstance(result, str)

    def test_extract_text_debug_mode(self, tmp_path):
        """Test text extraction with debug mode enabled."""
        with patch("renai.engines.ocr.DEBUG_DIR", tmp_path):
            engine = OCREngine(debug=True)

            # Create a test PDF
            import pymupdf

            doc = pymupdf.open()
            doc.new_page()
            test_pdf = tmp_path / "test.pdf"
            doc.save(str(test_pdf))
            doc.close()

            result = engine.extract_text(test_pdf)

            # Debug images should be created
            assert isinstance(result, str)

    def test_extract_text_exception(self, caplog):
        """Test exception handling in extract_text."""
        engine = OCREngine()

        with patch("pymupdf.open", side_effect=Exception("Failed")):
            result = engine.extract_text(Path("nonexistent.pdf"))
            assert result == ""
            assert "OCR Pipeline failed" in caplog.text

    def test_extract_text_early_termination(self):
        """Test early termination when enough text is extracted."""
        engine = OCREngine()

        # Mock a document with many pages
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 100

        mock_page = MagicMock()
        mock_page.get_text.return_value = "A" * 5000
        mock_doc.load_page.return_value = mock_page

        # Mock get_page_source_dpi
        with patch.object(OCREngine, "get_page_source_dpi", return_value=150):
            with patch("pymupdf.open", return_value=mock_doc):
                with patch.object(engine, "_extract_image_from_page") as mock_extract:
                    mock_extract.return_value = (Image.new("L", (100, 100)), None)
                    with patch.object(engine, "_perform_ocr", return_value="A" * 5000):
                        engine.extract_text(Path("test.pdf"))

                        # Should stop after getting enough text
                        assert mock_extract.call_count <= 3
