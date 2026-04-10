from renai.filename.builder import (
    build_book_filename,
    build_image_filename,
    build_vision_filename,
    extract_category_from_metadata,
    is_filename_changed,
)
from renai.models.schemas import BookMetadata, ImageMetadata


class TestBuildBookFilename:
    """Test build_book_filename function with SSoT models."""

    def test_full_metadata_model(self):
        """Test filename with BookMetadata model."""
        data = BookMetadata(
            title="Machine Learning",
            subtitle="A Probabilistic Perspective",
            edition="2nd",
            author="Kevin Murphy",
            year=2012,
            category="Technology",
            language="en"
        )
        result = build_book_filename(data, ".pdf")
        assert "Machine Learning" in result
        assert "Kevin Murphy" in result
        assert "(2012)" in result
        assert result.endswith(".pdf")

    def test_full_metadata_dict(self):
        """Test filename with dictionary (backward compatibility)."""
        data = {
            "title": "Machine Learning",
            "author": "Kevin Murphy",
            "year": 2012,
        }
        result = build_book_filename(data, ".pdf")
        assert "Machine Learning" in result
        assert "(2012)" in result

    def test_minimal_metadata(self):
        """Test filename with only title."""
        data = BookMetadata(title="My Book", category="Other", language="en")
        result = build_book_filename(data, ".pdf")
        assert result == "My Book.pdf"


class TestBuildImageFilename:
    """Test build_image_filename function with SSoT models."""

    def test_full_metadata_model(self):
        """Test image filename with ImageMetadata model."""
        data = ImageMetadata(
            description="Sunset", 
            category="Photo", 
            language="en", 
            visual_context="Beach",
            exif_date="2024-07-15"
        )
        result = build_image_filename(
            data, ".jpg", add_exif_date=True
        )
        assert "Sunset" in result
        assert "[2024-07-15]" in result

    def test_turkish_language(self):
        """Test image filename with Turkish language."""
        data = ImageMetadata(description="İstanbul", language="tr", category="Photo")
        result = build_image_filename(data, ".jpg")
        assert "(tr)" in result


class TestBuildVisionFilename:
    """Test build_vision_filename function with SSoT models."""

    def test_model_conversion(self):
        """Test vision filename with BookMetadata model."""
        data = BookMetadata(title="Research Paper", author="Author", year=2023, category="Paper", language="en")
        result = build_vision_filename(data, ".pdf")
        assert "Research Paper" in result
        assert "(2023)" in result
        assert "(en)" in result


class TestExtractCategoryFromMetadata:
    """Test extract_category_from_metadata function."""

    def test_model_safe_category(self):
        """Test category extraction from Model using safe_category property."""
        data = BookMetadata(title="T", category="Science/Fiction", language="en")
        result = extract_category_from_metadata(data)
        assert result == "Science&Fiction"

    def test_dict_fallback_sanitization(self):
        """Test that dicts still get sanitized for backward compatibility."""
        data = {"category": "Science/Fiction"}
        result = extract_category_from_metadata(data)
        assert result == "Science&Fiction"


class TestIsFilenameChanged:
    """Test is_filename_changed function."""

    def test_identical_names(self):
        """Test identical filenames return False."""
        from pathlib import Path

        result = is_filename_changed(Path("book.pdf"), "book.pdf")
        assert result is False

    def test_different_names(self):
        """Test different filenames return True."""
        from pathlib import Path

        result = is_filename_changed(Path("old.pdf"), "new.pdf")
        assert result is True
