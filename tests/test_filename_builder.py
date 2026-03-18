import sys

sys.path.insert(0, "src")


from renai.filename.builder import (
    build_book_filename,
    build_image_filename,
    build_vision_filename,
    extract_category_from_metadata,
    is_filename_changed,
)


class TestBuildBookFilename:
    """Test build_book_filename function."""

    def test_full_metadata(self):
        """Test filename with all fields."""
        data = {
            "title": "Machine Learning",
            "subtitle": "A Probabilistic Perspective",
            "edition": "2nd",
            "author": "Kevin Murphy",
            "year": 2012,
        }
        result = build_book_filename(data, ".pdf")
        assert "Machine Learning" in result
        assert "Kevin Murphy" in result
        assert "(2012)" in result
        assert result.endswith(".pdf")

    def test_minimal_metadata(self):
        """Test filename with only title."""
        data = {"title": "My Book"}
        result = build_book_filename(data, ".pdf")
        assert result == "My Book.pdf"

    def test_with_year(self):
        """Test filename with title and year."""
        data = {"title": "Python Guide", "year": 2024}
        result = build_book_filename(data, ".epub")
        assert "Python Guide" in result
        assert "(2024)" in result

    def test_with_author(self):
        """Test filename with author."""
        data = {"title": "Test Book", "author": "John Doe"}
        result = build_book_filename(data, ".pdf")
        assert "John Doe" in result


class TestBuildImageFilename:
    """Test build_image_filename function."""

    def test_full_metadata(self):
        """Test image filename with all fields."""
        data = {"title": "Sunset", "category": "Photo", "language": "en"}
        result = build_image_filename(
            data, ".jpg", exif_date="2024-07-15", add_exif_date=True
        )
        assert "Sunset" in result
        assert "Photo" in result
        assert "[2024-07-15]" in result

    def test_minimal(self):
        """Test image filename with minimal data."""
        data = {"title": "Beach"}
        result = build_image_filename(data, ".png")
        assert "Beach" in result

    def test_turkish_language(self):
        """Test image filename with Turkish language."""
        data = {"title": "İstanbul", "language": "tr"}
        result = build_image_filename(data, ".jpg")
        assert "(tr)" in result


class TestBuildVisionFilename:
    """Test build_vision_filename function."""

    def test_with_category(self):
        """Test vision filename with category."""
        data = {"title": "Research Paper", "category": "Academic", "language": "en"}
        result = build_vision_filename(data, ".pdf")
        assert "Research Paper" in result
        assert "Academic" in result

    def test_without_category(self):
        """Test vision filename without category."""
        data = {"title": "Document"}
        result = build_vision_filename(data, ".pdf")
        assert "Document" in result


class TestExtractCategoryFromMetadata:
    """Test extract_category_from_metadata function."""

    def test_basic_category(self):
        """Test basic category extraction."""
        data = {"category": "Fiction"}
        result = extract_category_from_metadata(data)
        assert result == "Fiction"

    def test_slash_replacement(self):
        """Test slash replacement in category."""
        data = {"category": "Science/Fiction"}
        result = extract_category_from_metadata(data)
        assert "/" not in result
        assert "&" in result

    def test_default_category(self):
        """Test default category."""
        data = {}
        result = extract_category_from_metadata(data)
        assert result == "Uncategorized"


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
