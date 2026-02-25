from unittest.mock import Mock
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
