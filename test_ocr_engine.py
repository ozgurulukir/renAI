from unittest.mock import Mock, patch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from ocr_engine import OCREngine

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

def test_deskew_opencv_not_available():
    """Test deskew when OpenCV is not available."""
    with patch("ocr_engine.OPENCV_AVAILABLE", False):
        engine = OCREngine()
        img = Image.new("L", (100, 100), 255)
        result = engine._deskew(img)
        assert result is img

def test_deskew_no_contours():
    """Test deskew when no contours are found."""
    # All white image should have no contours after THRESH_BINARY_INV
    img = Image.new("L", (100, 100), 255)
    engine = OCREngine()
    result = engine._deskew(img)
    assert result is img

def test_deskew_no_valid_angles():
    """Test deskew when contours are too small to fit an ellipse."""
    # Create an image with a single pixel dot (1 point contour)
    img = Image.new("L", (100, 100), 255)
    img.putpixel((50, 50), 0)
    engine = OCREngine()
    result = engine._deskew(img)
    assert result is img

def test_deskew_happy_path():
    """Test deskew with a clearly skewed image."""
    # Create a white image with a black rectangle rotated by 10 degrees
    img = Image.new("L", (200, 200), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 100], fill=0)
    skewed_img = img.rotate(10, expand=True, fillcolor=255)

    engine = OCREngine()
    result = engine._deskew(skewed_img)

    # The result should be different from the skewed image
    assert result is not skewed_img
    # Check that it's different in size or content
    if result.size != skewed_img.size:
        assert True
    else:
        assert np.any(np.asarray(result) != np.asarray(skewed_img))

def test_deskew_angle_adjustment():
    """Test the logic for angles > 45 degrees."""
    engine = OCREngine()
    img = Image.new("L", (100, 100), 255)

    # Mocking cv2 functions to return a specific angle (60 degrees)
    # Using a 5-point contour to pass the len(contour) < 5 check
    mock_contour = np.array([[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]], [[5, 5]]])
    with patch("cv2.threshold", return_value=(None, np.zeros((100, 100), dtype=np.uint8))), \
         patch("cv2.findContours", return_value=([mock_contour], None)), \
         patch("cv2.fitEllipse", return_value=((50, 50), (10, 20), 60.0)), \
         patch("cv2.getRotationMatrix2D", return_value=np.eye(2, 3)) as mock_rotate, \
         patch("cv2.warpAffine", return_value=np.zeros((100, 100), dtype=np.uint8)):

        engine._deskew(img)

        # Check that the angle passed to getRotationMatrix2D was 90 - 60 = 30
        args, _ = mock_rotate.call_args
        assert args[1] == 30.0

def test_deskew_exception():
    """Test that deskew handles exceptions gracefully."""
    engine = OCREngine()
    img = Image.new("L", (100, 100), 255)

    with patch("cv2.threshold", side_effect=Exception("OpenCV Error")):
        result = engine._deskew(img)
        assert result is img
