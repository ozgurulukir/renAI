import pytest
from unittest.mock import patch, mock_open
import config

def test_load_categories_from_file_not_exists():
    """Test when CUSTOM_CATEGORIES_FILE does not exist."""
    with patch("config.CUSTOM_CATEGORIES_FILE") as mock_file:
        mock_file.exists.return_value = False
        result = config.load_categories_from_file()
        assert result == {}

def test_load_categories_from_file_empty():
    """Test when CUSTOM_CATEGORIES_FILE exists but is empty."""
    with patch("config.CUSTOM_CATEGORIES_FILE") as mock_file:
        mock_file.exists.return_value = True
        with patch("builtins.open", mock_open(read_data="")):
            result = config.load_categories_from_file()
            # categories["en"] = [] is initialized before the loop
            assert result == {"en": []}

def test_load_categories_from_file_comments_only():
    """Test when CUSTOM_CATEGORIES_FILE has only comments and empty lines."""
    content = "# This is a comment\n\n   \n# Another comment"
    with patch("config.CUSTOM_CATEGORIES_FILE") as mock_file:
        mock_file.exists.return_value = True
        with patch("builtins.open", mock_open(read_data=content)):
            result = config.load_categories_from_file()
            assert result == {"en": []}

def test_load_categories_from_file_default_lang():
    """Test when CUSTOM_CATEGORIES_FILE has categories for default language (en)."""
    content = "Fiction\nNon-Fiction\nArticle"
    with patch("config.CUSTOM_CATEGORIES_FILE") as mock_file:
        mock_file.exists.return_value = True
        with patch("builtins.open", mock_open(read_data=content)):
            result = config.load_categories_from_file()
            assert result == {"en": ["Fiction", "Non-Fiction", "Article"]}

def test_load_categories_from_file_multiple_langs():
    """Test when CUSTOM_CATEGORIES_FILE has multiple languages."""
    content = "LANG: tr\nMakale\nKurgu\nLANG: en\nArticle\nFiction"
    with patch("config.CUSTOM_CATEGORIES_FILE") as mock_file:
        mock_file.exists.return_value = True
        with patch("builtins.open", mock_open(read_data=content)):
            result = config.load_categories_from_file()
            assert result == {
                "en": ["Article", "Fiction"],
                "tr": ["Makale", "Kurgu"]
            }

def test_load_categories_from_file_mixed_order():
    """Test categories before any LANG: spec and multiple LANG: specs."""
    content = "DefaultCat\nLANG: tr\nTrCat\nLANG: en\nEnCat"
    with patch("config.CUSTOM_CATEGORIES_FILE") as mock_file:
        mock_file.exists.return_value = True
        with patch("builtins.open", mock_open(read_data=content)):
            result = config.load_categories_from_file()
            assert result == {
                "en": ["DefaultCat", "EnCat"],
                "tr": ["TrCat"]
            }

def test_load_categories_from_file_lang_parsing():
    """Test LANG: parsing with different casing and whitespace."""
    content = "  lAnG:  TR  \nKurgu\n  Lang:en  \nFiction"
    with patch("config.CUSTOM_CATEGORIES_FILE") as mock_file:
        mock_file.exists.return_value = True
        with patch("builtins.open", mock_open(read_data=content)):
            result = config.load_categories_from_file()
            assert result == {
                "en": ["Fiction"],
                "tr": ["Kurgu"]
            }

def test_load_categories_from_file_exception():
    """Test exception handling during file operations."""
    with patch("config.CUSTOM_CATEGORIES_FILE") as mock_file:
        mock_file.exists.return_value = True
        with patch("builtins.open", side_effect=OSError("Disk failure")):
            result = config.load_categories_from_file()
            assert result == {}
