"""Folder-based processing rules for renAI.

Loads layered ``folder_rules`` TOML configs and resolves which rule (if any)
applies to a given source path. A rule can restrict the category list offered
to the LLM and override the filename template per file type.

Layered loading (first one found wins, no merging):
1. User config (``FOLDER_RULES_USER_FILE``)
2. Physical default (``FOLDER_RULES_DEFAULT_FILE``)
3. Package-internal default (``folder_rules.default.toml``)
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from renai.config.constants import (
    FOLDER_RULES_DEFAULT_FILE,
    FOLDER_RULES_USER_FILE,
    PACKAGE_CONFIG_DIR,
)
from renai.filename.builder import validate_filename_template
from renai.models.schemas import BookMetadata, ImageMetadata


class FolderRule(BaseModel):
    """A per-folder rule restricting categories and/or overriding templates."""

    path: Path = Field(description="Folder (prefix) this rule applies to")
    categories: dict[str, list[str]] | None = Field(
        default=None, description="Document category list per language, replacing the global one"
    )
    image_categories: dict[str, list[str]] | None = Field(
        default=None, description="Image category list per language, replacing the global one"
    )
    doc_template: str | None = Field(
        default=None, description="Filename template for text-extracted documents"
    )
    vision_template: str | None = Field(
        default=None, description="Filename template for vision-fallback documents"
    )
    image_template: str | None = Field(
        default=None, description="Filename template for image files"
    )

    @field_validator("path")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @property
    def specificity(self) -> int:
        """Depth of the rule path; deeper paths win when several rules match."""
        return len(self.path.parts)


def resolve_allowed_bucket(allowed_dict: dict[str, list[str]], language: str) -> list[str]:
    """Resolve the category bucket for a metadata language.

    Falls back to the ``en`` bucket and then to the first defined bucket, so
    single-language rules also constrain files detected in another language.
    """
    lang = (language or "en").lower()
    return allowed_dict.get(lang) or allowed_dict.get("en") or next(iter(allowed_dict.values()), [])


def resolve_fallback_category(language: str, allowed_dict: dict[str, list[str]]) -> str:
    """Pick the forced category for a restricted rule list.

    Prefers ``Diğer``/``Other`` when the matching bucket contains it, otherwise
    the first category of the bucket. Shared by the LLM fallback path and the
    cache-hit clamp so both paths agree.
    """
    lang = (language or "en").lower()
    bucket = resolve_allowed_bucket(allowed_dict, language)
    if not bucket:
        return "Diğer" if lang == "tr" else "Other"
    fallback = "Diğer" if lang == "tr" else "Other"
    return fallback if fallback in bucket else bucket[0]


def _normalize_category_map(value: Any) -> dict[str, list[str]] | None:
    """Accept a flat list (same list for every language) or a per-language table."""
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return {"tr": items, "en": items} if items else None
    if isinstance(value, dict):
        normalized = {
            str(lang).lower(): [str(cat).strip() for cat in cats if str(cat).strip()]
            for lang, cats in value.items()
            if isinstance(cats, list)
        }
        return {lang: cats for lang, cats in normalized.items() if cats} or None
    return None


def _validated_template(
    value: Any, rule_label: str, key: str, model_class: type[BookMetadata] | type[ImageMetadata]
) -> str | None:
    """Validate a template against the metadata model it will be rendered with."""
    if value is None:
        return None
    template = validate_filename_template(str(value), model_class)
    if template is None:
        logging.warning("Folder rule %s: ignoring %s", rule_label, key)
        return None
    return template


def _parse_rule(raw: Any, source: str, index: int) -> FolderRule | None:
    """Parse a single [[rules]] entry; invalid rules are skipped with a warning."""
    label = f"{source}#rules[{index}]"
    if not isinstance(raw, dict) or not raw.get("path"):
        logging.warning("Folder rule %s skipped: missing 'path'", label)
        return None
    try:
        rule = FolderRule(
            path=raw["path"],
            categories=_normalize_category_map(raw.get("categories")),
            image_categories=_normalize_category_map(raw.get("image_categories")),
            doc_template=_validated_template(
                raw.get("doc_template"), label, "doc_template", BookMetadata
            ),
            vision_template=_validated_template(
                raw.get("vision_template"), label, "vision_template", BookMetadata
            ),
            image_template=_validated_template(
                raw.get("image_template"), label, "image_template", ImageMetadata
            ),
        )
    except Exception as e:
        logging.warning("Folder rule %s skipped: %s", label, e)
        return None

    if not (
        rule.categories
        or rule.image_categories
        or rule.doc_template
        or rule.vision_template
        or rule.image_template
    ):
        logging.warning("Folder rule %s skipped: no categories or templates defined", label)
        return None
    return rule


def _try_load_rules(path: Path, label: str) -> tuple[FolderRule, ...] | None:
    """Load rules from one TOML layer.

    ``None`` means the file does not exist (the next layer is consulted). An
    existing file always wins its layer: parse errors, a missing ``rules``
    table, or zero valid rules all yield ``()``, so an empty or broken user
    config disables packaged defaults instead of falling through to them.
    """
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        logging.warning("Could not load %s folder rules from %s: %s", label, path, e)
        return ()

    raw_rules = data.get("rules")
    if not isinstance(raw_rules, list):
        return ()
    return tuple(
        rule
        for index, raw in enumerate(raw_rules)
        if (rule := _parse_rule(raw, label, index)) is not None
    )


@lru_cache(maxsize=1)
def load_folder_rules() -> tuple[FolderRule, ...]:
    """Load folder rules from layered TOML configs.

    Priority (First one found wins, no merging):
    1. User config (FOLDER_RULES_USER_FILE)
    2. Physical default (FOLDER_RULES_DEFAULT_FILE)
    3. Package-internal default

    A file that exists but defines no valid rules wins its layer with no rules;
    only a missing file lets the next layer be consulted.
    """
    for path, label in (
        (FOLDER_RULES_USER_FILE, "user"),
        (FOLDER_RULES_DEFAULT_FILE, "default"),
        (PACKAGE_CONFIG_DIR / "folder_rules.default.toml", "package"),
    ):
        result = _try_load_rules(path, label)
        if result is not None:
            return result
    return ()


def _is_path_prefix(child: Path, parent: Path) -> bool:
    """Prefix containment between paths, folding case where the platform does.

    Uses ``os.path.normcase`` so rule paths written with different letter case
    still match on case-insensitive filesystems (Windows) while POSIX matching
    stays exact.
    """
    child_norm = os.path.normcase(str(child))
    parent_norm = os.path.normcase(str(parent))
    return child_norm == parent_norm or child_norm.startswith(parent_norm.rstrip(os.sep) + os.sep)


def get_folder_rule_for(file_path: Path) -> FolderRule | None:
    """Resolve the most specific rule whose path is a prefix of the file's folder."""
    rules = load_folder_rules()
    if not rules:
        return None

    folder = file_path.parent.resolve()
    best: FolderRule | None = None
    for rule in rules:
        if _is_path_prefix(folder, rule.path) and (
            best is None or rule.specificity > best.specificity
        ):
            best = rule
    return best
