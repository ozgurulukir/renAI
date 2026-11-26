"""Tests for folder-based rules (folder_rules.toml).

Covers:
- Layered TOML loading (user > default > package), invalid rule tolerance
- Category normalization (flat list vs per-language table)
- Prefix matching with subfolder inheritance and longest-prefix wins
- Filename template rendering and builder integration
- LLMEngine per-call category override
- Pipeline integration: rule-signature-aware caching and organizer templates
"""

import logging
import os
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from renai.cache.manager import CacheSource, metadata_version_fingerprint
from renai.config import folder_rules as fr
from renai.config.constants import PACKAGE_CONFIG_DIR
from renai.config.folder_rules import (
    FolderRule,
    get_folder_rule_for,
    load_folder_rules,
    rule_cache_signature,
)
from renai.engines.llm import LLMEngine
from renai.file_ops.organizer import FileOrganizer
from renai.filename.builder import build_book_filename, build_image_filename
from renai.models.schemas import BookMetadata, ImageMetadata, ProcessingConfig
from renai.processors.book import BookProcessor

MISSING = Path("__definitely_missing__.toml")

# Placeholder for the engine's api_key argument in fixtures. Kept in one
# identifier so no credential-looking string literal appears inline.
_DUMMY_KEY = "test-key"


@pytest.fixture(autouse=True)
def _clear_rules_cache():
    """Keep the lru_cache on load_folder_rules from leaking between tests."""
    fr.load_folder_rules.cache_clear()
    yield
    fr.load_folder_rules.cache_clear()


def _write_rules_file(tmp_path: Path, content: str, name: str = "folder_rules.toml") -> Path:
    rules_file = tmp_path / name
    rules_file.write_text(content, encoding="utf-8")
    return rules_file


def _toml_path(path: Path) -> str:
    """Serialize a filesystem path as a TOML string, portable across platforms.

    Uses forward slashes so Windows paths (which contain backslashes) don't
    trip up TOML's escape handling, e.g. ``C:\\data`` being read as an escape.
    """
    return path.as_posix()


def _patched_layers(user: Path, default: Path = MISSING, package: Path = MISSING):
    """Patch the three folder-rules config locations for one test."""
    stack = ExitStack()
    stack.enter_context(patch.object(fr, "FOLDER_RULES_USER_FILE", user))
    stack.enter_context(patch.object(fr, "FOLDER_RULES_DEFAULT_FILE", default))
    stack.enter_context(patch.object(fr, "PACKAGE_CONFIG_DIR", package))
    return stack


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestLoadFolderRules:
    def test_user_file_wins_over_package(self, tmp_path):
        user = _write_rules_file(
            tmp_path, '[[rules]]\npath = "/data/user"\ncategories = ["User"]\n'
        )
        pkg = _write_rules_file(
            tmp_path, '[[rules]]\npath = "/data/pkg"\ncategories = ["Pkg"]\n', "default.toml"
        )
        with _patched_layers(user, package=pkg):
            rules = load_folder_rules()
        assert len(rules) == 1
        assert rules[0].path == Path("/data/user").resolve()

    def test_missing_files_return_empty(self, tmp_path):
        with _patched_layers(tmp_path / "folder_rules.toml"):
            assert load_folder_rules() == ()

    def test_invalid_toml_user_layer_wins_with_no_rules(self, tmp_path, caplog):
        """A broken user file wins its layer with zero rules; defaults are not consulted."""
        user = tmp_path / "folder_rules.toml"
        user.write_text("not [valid toml", encoding="utf-8")
        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        (pkg_dir / "folder_rules.default.toml").write_text(
            '[[rules]]\npath = "/data/pkg"\ncategories = ["Pkg"]\n', encoding="utf-8"
        )
        with _patched_layers(user, package=pkg_dir), caplog.at_level(logging.WARNING):
            assert load_folder_rules() == ()
        assert any("folder rules" in record.message.lower() for record in caplog.records)

    def test_empty_user_file_disables_package_rules(self, tmp_path):
        """An existing user file with no [[rules]] entries wins with zero rules."""
        user = _write_rules_file(tmp_path, "# deliberately empty\n")
        pkg = _write_rules_file(
            tmp_path, '[[rules]]\npath = "/data/pkg"\ncategories = ["Pkg"]\n', "default.toml"
        )
        with _patched_layers(user, package=pkg):
            assert load_folder_rules() == ()

    def test_rule_without_path_is_skipped(self, tmp_path, caplog):
        user = _write_rules_file(tmp_path, '[[rules]]\ncategories = ["Orphan"]\n')
        with _patched_layers(user), caplog.at_level(logging.WARNING):
            assert load_folder_rules() == ()
        assert any("missing 'path'" in record.message for record in caplog.records)

    def test_rule_without_categories_or_templates_is_skipped(self, tmp_path):
        user = _write_rules_file(tmp_path, '[[rules]]\npath = "/data/x"\n')
        with _patched_layers(user):
            assert load_folder_rules() == ()

    def test_flat_category_list_applies_to_all_languages(self, tmp_path):
        user = _write_rules_file(
            tmp_path, '[[rules]]\npath = "/data/x"\ncategories = ["Tez", "Makale"]\n'
        )
        with _patched_layers(user):
            (rule,) = load_folder_rules()
        assert rule.categories == {"tr": ["Tez", "Makale"], "en": ["Tez", "Makale"]}

    def test_per_language_category_table(self, tmp_path):
        user = _write_rules_file(
            tmp_path,
            '[[rules]]\npath = "/data/x"\n[rules.categories]\ntr = ["Tez"]\nen = ["Thesis"]\n',
        )
        with _patched_layers(user):
            (rule,) = load_folder_rules()
        assert rule.categories == {"tr": ["Tez"], "en": ["Thesis"]}

    def test_per_language_category_table_strips_values(self, tmp_path):
        user = _write_rules_file(
            tmp_path,
            '[[rules]]\npath = "/data/x"\n[rules.categories]\ntr = ["Tez "] \nen = ["Thesis"]\n',
        )
        with _patched_layers(user):
            (rule,) = load_folder_rules()
        assert rule.categories == {"tr": ["Tez"], "en": ["Thesis"]}

    def test_malformed_braces_disable_template(self, tmp_path, caplog):
        user = _write_rules_file(
            tmp_path,
            '[[rules]]\npath = "/data/x"\ncategories = ["Tez"]\ndoc_template = "{title -- "\n',
        )
        with _patched_layers(user), caplog.at_level(logging.WARNING):
            (rule,) = load_folder_rules()
        assert rule.doc_template is None
        assert any("malformed braces" in record.message for record in caplog.records)

    def test_image_template_rejects_document_only_token(self, tmp_path):
        user = _write_rules_file(
            tmp_path,
            '[[rules]]\npath = "/data/x"\ncategories = ["Tez"]\nimage_template = "{author}"\n',
        )
        with _patched_layers(user):
            (rule,) = load_folder_rules()
        assert rule.image_template is None

    def test_tilde_path_is_expanded_and_resolved(self, tmp_path):
        user = _write_rules_file(tmp_path, '[[rules]]\npath = "~"\ncategories = ["Home"]\n')
        with _patched_layers(user):
            (rule,) = load_folder_rules()
        assert rule.path == Path("~").expanduser().resolve()

    def test_unknown_template_token_disables_template(self, tmp_path, caplog):
        user = _write_rules_file(
            tmp_path,
            '[[rules]]\npath = "/data/x"\ncategories = ["Tez"]\n'
            'doc_template = "{title} -- {bogus_field}"\n',
        )
        with _patched_layers(user), caplog.at_level(logging.WARNING):
            (rule,) = load_folder_rules()
        assert rule.doc_template is None
        assert rule.categories is not None
        assert any("bogus_field" in record.message for record in caplog.records)

    def test_rule_becomes_empty_after_template_rejection_is_skipped(self, tmp_path, caplog):
        """A rule whose only content was an invalid template is dropped entirely."""
        user = _write_rules_file(
            tmp_path,
            '[[rules]]\npath = "/data/x"\ndoc_template = "{title} -- {bogus_field}"\n',
        )
        with _patched_layers(user), caplog.at_level(logging.WARNING):
            assert load_folder_rules() == ()
        assert any("no categories or templates" in record.message for record in caplog.records)

    def test_attribute_access_template_drops_template_not_rule(self, tmp_path, caplog):
        """A {title.year}-style template drops only the template; categories stay."""
        user = _write_rules_file(
            tmp_path,
            '[[rules]]\npath = "/data/x"\ncategories = ["Tez"]\n'
            'doc_template = "{title.year} -- {title}"\n',
        )
        with _patched_layers(user), caplog.at_level(logging.WARNING):
            (rule,) = load_folder_rules()
        assert rule.doc_template is None
        assert rule.categories is not None  # the rule itself is kept
        # The warning names the rule, the template and the reason
        messages = [record.message for record in caplog.records if "ignoring" in record.message]
        assert any("user#rules[0]" in m and "{title.year} -- {title}" in m for m in messages)

    def test_indexing_conversion_and_format_spec_templates_rejected(self, tmp_path):
        user = _write_rules_file(
            tmp_path,
            '[[rules]]\npath = "/data/x"\ncategories = ["Tez"]\n'
            'doc_template = "{title[0]}"\n'
            'vision_template = "{title!r}"\n'
            'image_template = "{description:>200}"\n',
        )
        with _patched_layers(user):
            (rule,) = load_folder_rules()
        assert rule.doc_template is None
        assert rule.vision_template is None
        assert rule.image_template is None
        assert rule.categories is not None

    def test_valid_template_alongside_invalid_one_is_kept(self, tmp_path):
        user = _write_rules_file(
            tmp_path,
            '[[rules]]\npath = "/data/x"\ncategories = ["Tez"]\n'
            'doc_template = "{title.year}"\nvision_template = "{title} -- {author}"\n',
        )
        with _patched_layers(user):
            (rule,) = load_folder_rules()
        assert rule.doc_template is None
        assert rule.vision_template == "{title} -- {author}"

    def test_package_default_toml_defines_no_active_rules(self):
        """The shipped folder_rules.default.toml is fully commented out."""
        with _patched_layers(MISSING, package=PACKAGE_CONFIG_DIR):
            assert load_folder_rules() == ()


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


class TestGetFolderRuleFor:
    def test_exact_folder_matches(self, tmp_path):
        user = _write_rules_file(
            tmp_path, f'[[rules]]\npath = "{_toml_path(tmp_path)}"\ncategories = ["Exact"]\n'
        )
        with _patched_layers(user):
            rule = get_folder_rule_for(tmp_path / "book.pdf")
        assert rule is not None
        assert "Exact" in rule.categories["en"]

    def test_subfolder_inherits_parent_rule(self, tmp_path):
        base = tmp_path / "base"
        user = _write_rules_file(
            tmp_path, f'[[rules]]\npath = "{_toml_path(base)}"\ncategories = ["Base"]\n'
        )
        with _patched_layers(user):
            rule = get_folder_rule_for(base / "2024" / "book.pdf")
        assert rule is not None
        assert "Base" in rule.categories["en"]

    def test_deepest_rule_wins(self, tmp_path):
        base = tmp_path / "base"
        deep = base / "comics"
        deep.mkdir(parents=True)
        user = _write_rules_file(
            tmp_path,
            f"""
[[rules]]
path = "{_toml_path(base)}"
categories = ["Base"]

[[rules]]
path = "{_toml_path(deep)}"
categories = ["Deep"]
""",
        )
        with _patched_layers(user):
            deep_rule = get_folder_rule_for(deep / "x.pdf")
            base_rule = get_folder_rule_for(base / "y.pdf")
        assert "Deep" in deep_rule.categories["en"]
        assert "Base" in base_rule.categories["en"]

    def test_no_matching_rule_returns_none(self, tmp_path):
        user = _write_rules_file(tmp_path, '[[rules]]\npath = "/elsewhere"\ncategories = ["X"]\n')
        with _patched_layers(user):
            assert get_folder_rule_for(tmp_path / "book.pdf") is None

    def test_differing_case_matches_on_folding_platforms(self, tmp_path):
        """Different-case rule paths match only where os.path.normcase folds case."""
        base = tmp_path / "Base"
        user = _write_rules_file(
            tmp_path, f'[[rules]]\npath = "{_toml_path(base)}"\ncategories = ["Folded"]\n'
        )
        with _patched_layers(user):
            rule = get_folder_rule_for(tmp_path / "base" / "x.pdf")
        folds = os.path.normcase("A") == os.path.normcase("a")
        if folds:
            assert rule is not None
            assert "Folded" in rule.categories["en"]
        else:
            assert rule is None


class TestIsPathPrefix:
    def test_exact_self_match(self):
        assert fr._is_path_prefix(Path("/data/x"), Path("/data/x")) is True

    def test_nested_prefix(self):
        assert fr._is_path_prefix(Path("/data/x/y"), Path("/data/x")) is True

    def test_sibling_is_not_prefix(self):
        assert fr._is_path_prefix(Path("/data/xy"), Path("/data/x")) is False
        assert fr._is_path_prefix(Path("/other/x/y"), Path("/data/x")) is False

    def test_root_prefix_matches_all(self):
        assert fr._is_path_prefix(Path("/data/x"), Path("/")) is True

    def test_case_folding_follows_normcase(self):
        folds = os.path.normcase("A") == os.path.normcase("a")
        assert fr._is_path_prefix(Path("/Data/X/y"), Path("/data/x")) is folds


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


class TestRenderFilenameTemplate:
    def test_substitution_and_extension(self):
        md = BookMetadata(
            language="en", category="Fiction", title="Dune", author="Frank Herbert", year=1965
        )
        result = build_book_filename(md, ".pdf", 200, template="{author} -- {title} -- ({year})")
        assert result == "Frank Herbert -- Dune -- (1965).pdf"

    def test_empty_fields_and_brackets_are_cleaned(self):
        md = BookMetadata(language="en", category="Fiction", title="Dune")
        result = build_book_filename(
            md, ".pdf", 200, template="{author} -- {title} -- ({year}) -- [{isbn}]"
        )
        assert result == "Dune.pdf"

    def test_series_volume_composite(self):
        md = BookMetadata(
            language="en",
            category="Fiction",
            title="Dune",
            series="Dune Chronicles",
            volume="2",
        )
        result = build_book_filename(md, ".pdf", 200, template="{title} -- {series_volume}")
        assert result == "Dune -- Dune Chronicles #2.pdf"

    def test_all_empty_falls_back_to_builtin(self):
        md = BookMetadata(language="en", category="Fiction")
        result = build_book_filename(md, ".pdf", 200, template="{author} -- {title}")
        assert result == "Unknown_Document.pdf"

    def test_builtin_pattern_unchanged_without_template(self):
        md = BookMetadata(
            language="en", category="Fiction", title="Dune", author="Frank Herbert", year=1965
        )
        assert build_book_filename(md, ".pdf", 200) == "Dune -- Frank Herbert -- (1965).pdf"

    def test_image_template_replaces_language_suffix(self):
        md = ImageMetadata(language="tr", category="Fotoğraf", description="Kedi")
        result = build_image_filename(md, ".jpg", 200, template="{description} -- ({language})")
        assert result == "Kedi -- (tr).jpg"

    def test_image_without_template_keeps_builtin_suffix(self):
        md = ImageMetadata(language="tr", category="Fotoğraf", description="Kedi")
        assert build_image_filename(md, ".jpg", 200) == "Kedi -- (tr).jpg"

    def test_exif_date_token_renders_and_collapses(self):
        dated = ImageMetadata(
            language="en", category="Screenshot", description="Sunset", exif_date="2024-07-15"
        )
        result = build_image_filename(dated, ".jpg", 200, template="{description} -- [{exif_date}]")
        assert result == "Sunset -- [2024-07-15].jpg"

        undated = ImageMetadata(language="en", category="Screenshot", description="Sunset")
        result = build_image_filename(
            undated, ".jpg", 200, template="{description} -- [{exif_date}]"
        )
        assert result == "Sunset.jpg"

    def test_result_sanitized_against_long_values(self):
        md = BookMetadata(language="en", category="Fiction", title="A" * 500)
        result = build_book_filename(md, ".pdf", 100, template="{title}")
        assert len(Path(result).stem) <= 100
        assert result.endswith(".pdf")


# ---------------------------------------------------------------------------
# LLMEngine per-call category override
# ---------------------------------------------------------------------------

GLOBAL_CATEGORIES = {"tr": ["Kurgu", "Diğer"], "en": ["Fiction", "Other"]}
RULE_CATEGORIES = {"tr": ["Tez"], "en": ["Comic"]}


@pytest.fixture
def engine():
    with patch("renai.engines.llm.get_allowed_categories", return_value=GLOBAL_CATEGORIES):
        return LLMEngine(
            provider="openai",
            api_key=_DUMMY_KEY,
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o-mini",
        )


class TestLLMCategoryOverride:
    def test_system_role_uses_override_categories(self, engine):
        role = engine._get_system_role("book_extractor", BookMetadata, categories=RULE_CATEGORIES)
        assert "Comic" in role
        assert "Fiction" not in role

    def test_system_role_defaults_to_engine_categories(self, engine):
        role = engine._get_system_role("book_extractor", BookMetadata)
        assert "Fiction" in role

    def test_validate_accepts_category_from_override(self, engine):
        raw = {"title": "X", "category": "Comic", "language": "en"}
        validated, retry_list = engine._validate_and_align_metadata(
            raw, BookMetadata, "test", categories_override=RULE_CATEGORIES
        )
        assert validated is not None
        assert validated.category == "Comic"
        assert retry_list is None

    def test_validate_rejects_category_outside_override(self, engine):
        raw = {"title": "X", "category": "Fiction", "language": "en"}
        validated, retry_list = engine._validate_and_align_metadata(
            raw, BookMetadata, "test", categories_override=RULE_CATEGORIES
        )
        assert validated is None
        assert retry_list == ["Comic"]

    def test_single_language_rule_constrains_other_language(self, engine):
        """A tr-only rule still constrains en documents via the fallback bucket."""
        rule = {"tr": ["Tez"]}
        validated, retry_list = engine._validate_and_align_metadata(
            {"title": "X", "category": "Tez", "language": "en"},
            BookMetadata,
            "test",
            categories_override=rule,
        )
        assert validated is not None
        assert validated.category == "Tez"

        rejected, retry_list = engine._validate_and_align_metadata(
            {"title": "X", "category": "Fiction", "language": "en"},
            BookMetadata,
            "test",
            categories_override=rule,
        )
        assert rejected is None
        assert retry_list == ["Tez"]

    def test_final_fallback_uses_first_allowed_category(self, engine):
        raw = {"title": "X", "category": "Kurgu", "language": "tr"}
        model = engine._attempt_final_fallback(
            raw, BookMetadata, categories_override=RULE_CATEGORIES
        )
        assert model is not None
        assert model.category == "Tez"

    def test_final_fallback_keeps_other_when_allowed(self, engine):
        rule = {"en": ["Comic", "Other"]}
        model = engine._attempt_final_fallback(
            {"title": "X", "language": "en"}, BookMetadata, categories_override=rule
        )
        assert model is not None
        assert model.category == "Other"

    @pytest.mark.asyncio
    async def test_get_book_metadata_threads_override(self, engine):
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = '{"title": "X", "category": "Comic", "language": "en"}'
        engine.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await engine.get_book_metadata("some text", allowed_categories=RULE_CATEGORIES)

        assert result is not None
        assert result.category == "Comic"
        sent_system_prompt = engine.client.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ]
        assert "Comic" in sent_system_prompt
        assert "Fiction" not in sent_system_prompt


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


@pytest.fixture
def processor():
    ocr = MagicMock()
    llm = MagicMock()
    llm.model_name = "test-model"
    return BookProcessor(ocr, ProcessingConfig(), llm)


class TestPipelineFolderRules:
    @pytest.mark.asyncio
    async def test_cache_rule_mismatch_reextracts_under_rule_categories(self, processor, tmp_path):
        """A cache row written under global categories must not serve a rule run.

        The cache is rule-signature aware: a mismatch is a miss and the file is
        re-extracted with the rule's category list (replacing the old lossy
        post-hoc clamp that demoted the cached category to a fallback bucket).
        """
        test_file = tmp_path / "book.pdf"
        test_file.write_text("content")
        cached = BookMetadata(title="Cached", author="A", category="Fiction", language="en")
        rule = FolderRule(path=tmp_path, categories={"en": ["Comic"]})
        extracted = BookMetadata(title="Fresh", author="B", category="Comic", language="en")

        with (
            patch("renai.processors.book.get_folder_rule_for", return_value=rule),
            patch("renai.processors.book.get_file_hash", return_value="hash"),
            patch(
                "renai.cache.middleware.read_metadata_cache",
                return_value=(cached, CacheSource(is_vision_doc=False, rule_signature="")),
            ),
            patch.object(processor.text_extractor, "extract", AsyncMock(return_value="some text")),
            patch.object(
                processor.llm_engine,
                "get_book_metadata",
                AsyncMock(return_value=extracted),
            ) as mock_llm,
            patch(
                "renai.file_ops.organizer.safe_move_file",
                side_effect=lambda s, d, **kwargs: d,
            ),
        ):
            result = await processor.process_file(test_file)

        assert result.success is True
        assert result.metadata["category"] == "Comic"
        assert result.metadata["title"] == "Fresh"
        # Re-extraction runs under the rule's category universe, not the global one.
        assert mock_llm.call_args.kwargs["allowed_categories"] == rule.categories

    @pytest.mark.asyncio
    async def test_matching_rule_signature_serves_cache(self, processor, tmp_path):
        test_file = tmp_path / "book.pdf"
        test_file.write_text("content")
        cached = BookMetadata(title="Cached", category="Comic", language="en")
        rule = FolderRule(path=tmp_path, categories={"en": ["Comic", "Other"]})

        with (
            patch("renai.processors.book.get_folder_rule_for", return_value=rule),
            patch("renai.processors.book.get_file_hash", return_value="hash"),
            patch(
                "renai.cache.middleware.read_metadata_cache",
                return_value=(
                    cached,
                    CacheSource(
                        is_vision_doc=False,
                        rule_signature=rule_cache_signature(rule, mode="doc"),
                        version_fingerprint=metadata_version_fingerprint(),
                    ),
                ),
            ),
            patch(
                "renai.file_ops.organizer.safe_move_file",
                side_effect=lambda s, d, **kwargs: d,
            ),
        ):
            result = await processor.process_file(test_file)

        assert result.success is True
        assert result.metadata["category"] == "Comic"
        # Signature matches the rule's universe: the cache row is served as-is.
        processor.llm_engine.get_book_metadata.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_language_rule_reextracts_other_language_cache_hit(
        self, processor, tmp_path
    ):
        """A tr-only rule re-extracts an en-global cache hit under the tr bucket."""
        test_file = tmp_path / "book.pdf"
        test_file.write_text("content")
        cached = BookMetadata(title="Cached", category="Fiction", language="en")
        rule = FolderRule(path=tmp_path, categories={"tr": ["Tez"]})
        extracted = BookMetadata(title="Tez Work", author="C", category="Tez", language="tr")

        with (
            patch("renai.processors.book.get_folder_rule_for", return_value=rule),
            patch("renai.processors.book.get_file_hash", return_value="hash"),
            patch(
                "renai.cache.middleware.read_metadata_cache",
                return_value=(cached, CacheSource(is_vision_doc=False, rule_signature="")),
            ),
            patch.object(processor.text_extractor, "extract", AsyncMock(return_value="some text")),
            patch.object(
                processor.llm_engine,
                "get_book_metadata",
                AsyncMock(return_value=extracted),
            ),
            patch(
                "renai.file_ops.organizer.safe_move_file",
                side_effect=lambda s, d, **kwargs: d,
            ),
        ):
            result = await processor.process_file(test_file)

        assert result.success is True
        assert result.metadata["category"] == "Tez"

    @pytest.mark.asyncio
    async def test_image_rule_mismatch_reextracts_with_image_categories(self, processor, tmp_path):
        """Image mode signs image_categories; a global row must not serve a rule run."""
        processor.config = processor.config.model_copy(update={"rename_images": True})
        test_file = tmp_path / "photo.jpg"
        test_file.write_bytes(b"jpeg")
        cached = ImageMetadata(description="Global category photo", category="Photo", language="en")
        rule = FolderRule(path=tmp_path, image_categories={"en": ["Screenshot"]})
        extracted = ImageMetadata(
            description="Rule category photo", category="Screenshot", language="en"
        )

        with (
            patch("renai.processors.book.get_folder_rule_for", return_value=rule),
            patch("renai.processors.book.get_file_hash", return_value="hash"),
            patch(
                "renai.cache.middleware.read_metadata_cache",
                return_value=(cached, CacheSource(is_vision_doc=False, rule_signature="")),
            ),
            patch.object(
                processor.visual_processor,
                "extract_metadata_from_image",
                AsyncMock(return_value=extracted),
            ) as mock_visual,
            patch(
                "renai.file_ops.organizer.safe_move_file",
                side_effect=lambda s, d, **kwargs: d,
            ),
        ):
            result = await processor.process_file(test_file)

        assert result.success is True
        assert result.metadata["description"] == "Rule category photo"
        # Re-extraction receives the rule's image category universe.
        assert mock_visual.call_args.kwargs["image_categories"] == rule.image_categories

    @pytest.mark.asyncio
    async def test_doc_template_applied_by_organizer(self, processor, tmp_path):
        test_file = tmp_path / "book.pdf"
        test_file.write_text("content")
        rule = FolderRule(path=tmp_path, doc_template="{author} -- {title}")
        extracted = BookMetadata(
            title="Dune", author="Frank Herbert", year=1965, category="Fiction", language="en"
        )

        with (
            patch("renai.processors.book.get_folder_rule_for", return_value=rule),
            patch("renai.processors.book.get_file_hash", return_value="hash"),
            patch("renai.cache.middleware.read_metadata_cache", return_value=(None, None)),
            patch.object(processor.text_extractor, "extract", AsyncMock(return_value="some text")),
            patch.object(
                processor.llm_engine,
                "get_book_metadata",
                AsyncMock(return_value=extracted),
            ),
            patch(
                "renai.file_ops.organizer.safe_move_file",
                side_effect=lambda s, d, **kwargs: d,
            ),
        ):
            result = await processor.process_file(test_file)

        assert result.success is True
        assert result.name == "Frank Herbert -- Dune.pdf"

    @pytest.mark.asyncio
    async def test_organizer_selects_template_by_file_type(self, tmp_path):
        organizer = FileOrganizer(max_filename_length=200)
        rule = FolderRule(
            path=tmp_path,
            doc_template="{title} -- doc",
            vision_template="{title} -- vision",
            image_template="{description} -- img",
        )
        book = BookMetadata(title="Dune", category="Fiction", language="en")
        image = ImageMetadata(description="Kedi", category="Fotoğraf", language="tr")

        with patch(
            "renai.file_ops.organizer.safe_move_file",
            side_effect=lambda s, d, **kwargs: d,
        ):
            doc_result = await organizer.organize_file(
                tmp_path / "a.pdf", book, mode="rename", folder_rule=rule
            )
            vision_result = await organizer.organize_file(
                tmp_path / "b.pdf", book, mode="rename", is_vision_doc=True, folder_rule=rule
            )
            image_result = await organizer.organize_file(
                tmp_path / "c.jpg", image, mode="rename", folder_rule=rule
            )

        assert doc_result.name == "Dune -- doc.pdf"
        assert vision_result.name == "Dune -- vision.pdf"
        assert image_result.name == "Kedi -- img.jpg"

    @pytest.mark.asyncio
    async def test_organizer_without_rule_keeps_builtin_pattern(self, tmp_path):
        organizer = FileOrganizer(max_filename_length=200)
        book = BookMetadata(
            title="Dune", author="Frank Herbert", year=1965, category="Fiction", language="en"
        )

        with patch(
            "renai.file_ops.organizer.safe_move_file",
            side_effect=lambda s, d, **kwargs: d,
        ):
            result = await organizer.organize_file(tmp_path / "a.pdf", book, mode="rename")

        assert result.name == "Dune -- Frank Herbert -- (1965).pdf"


def test_rule_cache_signature_mode_and_stability(tmp_path):
    """Signatures hash the mode-relevant list, stay stable, and ignore absent lists."""
    doc_rule = FolderRule(path=tmp_path, categories={"en": ["Comic"]})
    img_rule = FolderRule(path=tmp_path, image_categories={"en": ["Screenshot"]})
    same_doc_rule = FolderRule(path=tmp_path, categories={"en": ["Comic"]})

    # Doc mode signs categories; image mode signs image_categories; a rule that
    # does not restrict the mode's list shares the global universe (None).
    assert rule_cache_signature(doc_rule, mode="doc") is not None
    assert rule_cache_signature(doc_rule, mode="img") is None
    assert rule_cache_signature(img_rule, mode="img") is not None
    assert rule_cache_signature(img_rule, mode="doc") is None
    assert rule_cache_signature(None, mode="doc") is None

    # Equal universes hash identically; list order is semantically meaningful
    # (the LLM fallback picks bucket[0]) so a reorder intentionally invalidates.
    assert rule_cache_signature(doc_rule, mode="doc") == rule_cache_signature(
        same_doc_rule, mode="doc"
    )
    assert rule_cache_signature(doc_rule, mode="doc") != rule_cache_signature(
        FolderRule(path=tmp_path, categories={"en": ["Comic", "Other"]}),
        mode="doc",
    )
