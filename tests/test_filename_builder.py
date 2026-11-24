from renai.filename.builder import (
    build_book_filename,
    build_image_filename,
    build_vision_filename,
    extract_category_from_metadata,
    render_filename_template,
)
from renai.models.schemas import BookMetadata, ImageMetadata


class TestBookMetadataSchemaValidation:
    """Test new BookMetadata field validators."""

    def test_sanitize_year_none_returns_none(self):
        data = BookMetadata(title="T", year=None, language="en", category="Other")
        assert data.year is None

    def test_sanitize_year_valid_int_and_float(self):
        data_int = BookMetadata(title="T", year=2024, language="en", category="Other")
        assert data_int.year == 2024

        data_float = BookMetadata(title="T", year=2020.0, language="en", category="Other")
        assert data_float.year == 2020

    def test_sanitize_year_clean_string(self):
        data = BookMetadata(title="T", year="2024", language="en", category="Other")
        assert data.year == 2024

    def test_sanitize_year_extracts_year_from_messy_string(self):
        data1 = BookMetadata(title="T", year="2024 (E-book)", language="en", category="Other")
        assert data1.year == 2024

        data2 = BookMetadata(
            title="T", year="Published in 1999 (2nd ed)", language="en", category="Other"
        )
        assert data2.year == 1999

    def test_sanitize_year_placeholders_return_none(self):
        for placeholder in ("", "   ", "unknown", "N/A", "-", "null", "none"):
            data = BookMetadata(title="T", year=placeholder, language="en", category="Other")
            assert data.year is None

    def test_sanitize_year_invalid_string_or_type_returns_none(self):
        data_no_year = BookMetadata(title="T", year="no year here", language="en", category="Other")
        assert data_no_year.year is None

        data_invalid_range = BookMetadata(
            title="T", year="In the year 3000", language="en", category="Other"
        )
        assert data_invalid_range.year is None

        data_list = BookMetadata(title="T", year=["2024"], language="en", category="Other")
        assert data_list.year is None

    def test_isbn_valid(self):
        data = BookMetadata(title="T", isbn="978-0-13-235088-4", language="en", category="Other")
        assert data.isbn == "978-0-13-235088-4"

    def test_isbn_none_for_unknown(self):
        data = BookMetadata(title="T", isbn="unknown", language="en", category="Other")
        assert data.isbn is None

    def test_isbn_none_for_empty(self):
        data = BookMetadata(title="T", isbn="  ", language="en", category="Other")
        assert data.isbn is None

    def test_page_count_int(self):
        data = BookMetadata(title="T", page_count=464, language="en", category="Other")
        assert data.page_count == 464

    def test_page_count_from_string(self):
        data = BookMetadata(title="T", page_count="350 pages", language="en", category="Other")
        assert data.page_count == 350

    def test_page_count_none_for_unknown(self):
        data = BookMetadata(title="T", page_count="N/A", language="en", category="Other")
        assert data.page_count is None

    def test_page_count_none_for_zero(self):
        data = BookMetadata(title="T", page_count=0, language="en", category="Other")
        assert data.page_count is None

    def test_publisher_none_for_empty(self):
        data = BookMetadata(title="T", publisher="", language="en", category="Other")
        assert data.publisher is None

    def test_series_none_for_na(self):
        data = BookMetadata(title="T", series="N/A", language="en", category="Other")
        assert data.series is None

    def test_translator_stripped(self):
        data = BookMetadata(title="T", translator="  Jane Doe  ", language="en", category="Other")
        assert data.translator == "Jane Doe"

    def test_doi_preserved(self):
        data = BookMetadata(title="T", doi="10.1000/xyz123", language="en", category="Other")
        assert data.doi == "10.1000/xyz123"

    def test_volume_none_for_dash(self):
        data = BookMetadata(title="T", volume="-", language="en", category="Other")
        assert data.volume is None


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
            language="en",
        )
        result = build_book_filename(data, ".pdf", 200)
        assert "Machine Learning" in result
        assert "Kevin Murphy" in result
        assert "(2012)" in result
        assert result.endswith(".pdf")

    def test_minimal_metadata(self):
        """Test filename with only title."""
        data = BookMetadata(title="My Book", category="Other", language="en")
        result = build_book_filename(data, ".pdf", 200)
        assert result == "My Book.pdf"


class TestQuotedFieldStripping:
    """Test that LLM-returned quoted values are stripped in filenames."""

    def test_single_quoted_title_and_author(self):
        data = BookMetadata(
            title="'Clean Code'",
            author="'Robert C. Martin'",
            year=2008,
            category="Other",
            language="en",
        )
        result = build_book_filename(data, ".pdf", 200)
        assert result.startswith("Clean Code")
        assert "'Clean Code'" not in result
        assert "'Robert C. Martin'" not in result

    def test_double_quoted_title_and_author(self):
        data = BookMetadata(
            title='"Clean Code"',
            author='"Robert C. Martin"',
            year=2008,
            category="Other",
            language="en",
        )
        result = build_book_filename(data, ".pdf", 200)
        assert result.startswith("Clean Code")
        assert '"Clean Code"' not in result

    def test_quoted_edition_no_double_quotes(self):
        data = BookMetadata(
            title="Clean Code",
            edition="'2nd'",
            author="Bob",
            year=2008,
            category="Other",
            language="en",
        )
        result = build_book_filename(data, ".pdf", 200)
        assert "'2nd'" in result
        assert "''2nd''" not in result

    def test_quoted_subtitle_stripped(self):
        data = BookMetadata(
            title="Clean Code",
            subtitle="'A Handbook of Agile Software'",
            author="Bob",
            year=2008,
            category="Other",
            language="en",
        )
        result = build_book_filename(data, ".pdf", 200)
        assert "'A Handbook" not in result
        assert "A Handbook" in result

    def test_image_quoted_description_stripped(self):
        data = ImageMetadata(description="'A beautiful sunset'", language="en", category="Photo")
        result = build_image_filename(data, ".jpg", 200)
        assert "'A beautiful" not in result
        assert "A beautiful sunset" in result

    def test_vision_quoted_fields_stripped(self):
        data = BookMetadata(
            title="'Clean Code'", author="'Bob'", year=2008, category="Other", language="en"
        )
        result = build_vision_filename(data, ".pdf", 200)
        assert "'Clean Code'" not in result
        assert "Clean Code" in result


class TestBuildImageFilename:
    """Test build_image_filename function with SSoT models."""

    def test_full_metadata_model(self):
        """Test image filename with ImageMetadata model."""
        data = ImageMetadata(
            description="Sunset",
            category="Photo",
            language="en",
            visual_context="Beach",
            exif_date="2024-07-15",
        )
        result = build_image_filename(data, ".jpg", 200, add_exif_date=True)
        assert "Sunset" in result
        assert "[2024-07-15]" in result

    def test_turkish_language(self):
        """Test image filename with Turkish language."""
        data = ImageMetadata(description="İstanbul", language="tr", category="Photo")
        result = build_image_filename(data, ".jpg", 200)
        assert "(tr)" in result


class TestBuildVisionFilename:
    """Test build_vision_filename function with SSoT models."""

    def test_model_conversion(self):
        """Test vision filename with BookMetadata model."""
        data = BookMetadata(
            title="Research Paper", author="Author", year=2023, category="Paper", language="en"
        )
        result = build_vision_filename(data, ".pdf", 200)
        assert "Research Paper" in result
        assert "(2023)" in result
        assert "(en)" in result


class TestExtractCategoryFromMetadata:
    """Test extract_category_from_metadata function."""

    def test_model_safe_category(self):
        """Test category extraction from Model using safe_category property."""
        data = BookMetadata(title="T", category="Science/Fiction", language="en")
        result = extract_category_from_metadata(data, 200)
        assert result == "Science&Fiction"

    def test_model_empty_category_safe_category_returns_uncategorized(self):
        """Test model with empty string category uses safe_category fallback to 'Uncategorized'."""
        data = BookMetadata(title="T", category="", language="en")
        result = extract_category_from_metadata(data, 200)
        assert result == "Uncategorized"

    def test_model_empty_string_category_returns_default(self):
        """Test model with empty string category returns default."""
        data = BookMetadata(title="T", category="", language="en")
        result = extract_category_from_metadata(data, 200)
        assert result == "Uncategorized"

    def test_model_backslash_sanitization(self):
        """Test backslash is replaced with ampersand in model category."""
        metadata = BookMetadata(category="Fiction\\Sci-Fi", language="en")
        result = extract_category_from_metadata(metadata, 200)
        assert result == "Fiction&Sci-Fi"

    def test_model_slash_and_backslash_sanitization(self):
        """Test both slash types are replaced in model category."""
        metadata = BookMetadata(category="History/Military\\Biography", language="en")
        result = extract_category_from_metadata(metadata, 200)
        assert result == "History&Military&Biography"


class TestSeriesAndVolumeInFilename:
    """Test series and volume fields in book and vision filenames."""

    def test_series_with_volume(self):
        data = BookMetadata(
            title="The Two Towers",
            author="J.R.R. Tolkien",
            year=1954,
            series="Lord of the Rings",
            volume="2",
            language="en",
            category="Other",
        )
        result = build_book_filename(data, ".pdf", 200)
        assert "Lord of the Rings #2" in result

    def test_series_without_volume(self):
        data = BookMetadata(
            title="Dune",
            author="Frank Herbert",
            year=1965,
            series="Dune Chronicles",
            language="en",
            category="Other",
        )
        result = build_book_filename(data, ".pdf", 200)
        assert "Dune Chronicles" in result
        assert "#" not in result

    def test_volume_without_series(self):
        data = BookMetadata(
            title="Collected Works",
            author="Author",
            year=2000,
            volume="3",
            language="en",
            category="Other",
        )
        result = build_book_filename(data, ".pdf", 200)
        assert "Vol. 3" in result

    def test_no_series_no_volume(self):
        data = BookMetadata(
            title="Clean Code",
            author="Robert C. Martin",
            year=2008,
            language="en",
            category="Other",
        )
        result = build_book_filename(data, ".pdf", 200)
        assert "Vol." not in result
        assert "#" not in result

    def test_vision_series_with_volume(self):
        data = BookMetadata(
            title="The Two Towers",
            author="J.R.R. Tolkien",
            year=1954,
            series="Lord of the Rings",
            volume="2",
            language="en",
            category="Other",
        )
        result = build_vision_filename(data, ".pdf", 200)
        assert "Lord of the Rings #2" in result

    def test_metadata_only_fields_not_in_filename(self):
        data = BookMetadata(
            title="Clean Code",
            author="Robert C. Martin",
            year=2008,
            isbn="978-0-13-235088-4",
            publisher="Prentice Hall",
            doi="10.1234/test",
            translator="Jane Doe",
            page_count=464,
            language="en",
            category="Other",
        )
        result = build_book_filename(data, ".pdf", 200)
        assert "978-0-13-235088-4" not in result
        assert "Prentice Hall" not in result
        assert "10.1234" not in result
        assert "Jane Doe" not in result
        assert "464" not in result


class TestFilenameLengthAndConsistencyFixes:
    """Tests for #232 and #233 (length cap with extension, validation, year bounds, lang)."""

    def test_total_filename_never_exceeds_max_length(self):
        long_title = "A" * 100
        book = BookMetadata(
            title=long_title, author="Author", year=2024, language="en", category="Other"
        )
        img = ImageMetadata(description=long_title, language="de", category="Photo")

        book_name = build_book_filename(book, ".pdf", max_filename_length=50)
        assert len(book_name) <= 50
        assert book_name.endswith(".pdf")

        vision_name = build_vision_filename(book, ".pdf", max_filename_length=50)
        assert len(vision_name) <= 50
        assert vision_name.endswith(" -- (en).pdf")

        img_name = build_image_filename(img, ".jpg", max_filename_length=50)
        assert len(img_name) <= 50
        assert img_name.endswith(" -- (de).jpg")

        tmpl_name = render_filename_template("{title}", book, ".epub", max_filename_length=50)
        assert tmpl_name is not None
        assert len(tmpl_name) <= 50
        assert tmpl_name.endswith(".epub")

    def test_unknown_document_fallback_respects_max_length_and_sanitizes_ext(self):
        empty_book = BookMetadata(language="en", category="Other")
        result = build_book_filename(empty_book, ".p<d>f", max_filename_length=15)
        assert "<" not in result and ">" not in result
        assert len(result) <= 15

    def test_render_filename_template_strips_duplicate_extension(self):
        book = BookMetadata(title="My Book", language="en", category="Other")
        result = render_filename_template("{title}.pdf", book, ".pdf", max_filename_length=100)
        assert result == "My Book.pdf"

    def test_build_vision_filename_applies_year_bounds(self):
        out_of_range = BookMetadata.model_construct(
            title="Future", year=3000, language="en", category="Other"
        )
        result = build_vision_filename(out_of_range, ".pdf", max_filename_length=100)
        assert "(3000)" not in result

    def test_lang_suffix_preserves_non_turkish_iso_codes(self):
        img = ImageMetadata(description="Berlin", language="de", category="Photo")
        assert build_image_filename(img, ".png", 100) == "Berlin -- (de).png"

    def test_validate_filename_template_rejects_empty_static_output(self):
        from renai.filename.builder import validate_filename_template

        assert validate_filename_template(" -- () [] ", BookMetadata) is None
        assert validate_filename_template("{title} -- {author}", BookMetadata) is not None
