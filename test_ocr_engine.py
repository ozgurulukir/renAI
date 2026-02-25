import pytest
import numpy as np
import cv2
from unittest.mock import Mock
from PIL import Image, ImageDraw
from ocr_engine import OCREngine, OPENCV_AVAILABLE

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
        {"bbox": (100, 100, 124, 124), "width": 100, "height": 100}
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
def test_deskew_straightens_image():
    engine = OCREngine(enhanced=True)

    # Create a tilted image
    size = (500, 200)
    image = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(image)
    draw.text((50, 80), "This is a test line for deskewing verification.", fill=0)

    # Rotate 10 degrees CCW (Pillow rotate is CCW)
    tilted_image = image.rotate(10, expand=True, fillcolor=255)

    # Deskew
    deskewed_image = engine._deskew(tilted_image)

    # Verify it's different from tilted
    assert deskewed_image != tilted_image

    # The deskewed image should be straightened
    data = np.asarray(deskewed_image.convert("L"))
    _, binary = cv2.threshold(data, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angles = []
    for c in contours:
        if cv2.contourArea(c) < 100: continue
        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        angle = rect[2]
        if w < h: angle += 90
        angles.append(angle)

    if angles:
        median_angle = np.median(angles)
        assert abs(median_angle) < 1.0

def test_deskew_returns_original_if_no_opencv(monkeypatch):
    monkeypatch.setattr("ocr_engine.OPENCV_AVAILABLE", False)
    engine = OCREngine(enhanced=True)
    img = Image.new("L", (100, 100))
    assert engine._deskew(img) == img

def test_deskew_handles_empty_image():
    engine = OCREngine(enhanced=True)
    img = Image.new("L", (100, 100), color=255)
    # Should return original image if no contours found
    assert engine._deskew(img) == img
