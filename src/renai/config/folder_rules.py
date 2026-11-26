"""Folder-based processing rules for renAI.

Loads layered ``folder_rules`` TOML configs and resolves which rule (if any)
applies to a given source path. A rule can restrict the category list offered
to the LLM and override the filename template per file type.

Layered loading (first one found wins, no merging):
1. User config (``FOLDER_RULES_USER_FILE``)
2. Physical default (``FOLDER_RULES_DEFAULT_FILE``)
3. Package-internal default (``folder_rules.default.toml``)
"""

import hashlib
import json
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
from renai.filename.builder import filename_template_rejection_reason
from renai.messages import config as config_msg
from renai.models.schemas import BookMetadata, ImageMetadata

logger = logging.getLogger(__name__)


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


def rule_cache_signature(rule: FolderRule | None, *, mode: str) -> str | None:
    """Stable signature of the category universe a rule imposes on extraction.

    The metadata cache stores this signature next to each record so a cached
    extraction is only served to runs whose category universe matches (the
    same role provider/model play in the cache row). Templates are excluded:
    they shape filenames, not extracted metadata. Returns ``None`` when the
    rule does not restrict categories for ``mode`` ("doc"/"img").

    When the rule does not restrict categories for ``mode``, callers fall back
    to :func:`global_categories_signature` so the effective global list is
    signed exactly like a rule's list (#247).
    """
    if rule is None:
        return None
    categories = rule.image_categories if mode == "img" else rule.categories
    if not categories:
        return None
    return _categories_signature(categories)


def _categories_signature(categories: dict[str, list[str]]) -> str:
    payload = json.dumps(categories, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=2)
def global_categories_signature(*, mode: str) -> str:
    """Stable signature of the effective global category list (#247).

    Signs whatever ``config.providers.get_allowed_categories`` /
    ``get_image_categories`` resolve for ``mode`` ("doc"/"img") — user TOML
    overrides included — so editing the global category list invalidates
    metadata rows written under the previous list, exactly like a folder-rule
    category change does.
    """
    from renai.config.providers import get_allowed_categories, get_image_categories

    categories = get_image_categories() if mode == "img" else get_allowed_categories()
    return _categories_signature(categories)


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
    the first category of the bucket. Used by the LLM validation fallback so a
    restricted rule's metadata always lands inside the rule's category list.
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
    """Validate a template against the metadata model it will be rendered with.

    An invalid template drops ONLY the template (the rule keeps its
    categories); the warning names the rule, the template and the reason.
    """
    if value is None:
        return None
    template = str(value)
    reason = filename_template_rejection_reason(template, model_class)
    if reason is not None:
        logger.warning(
            config_msg.Warn.folder_rule_template_ignored(rule_label, key, template, reason)
        )
        return None
    return template


def _parse_rule(raw: Any, source: str, index: int) -> FolderRule | None:
    """Parse a single [[rules]] entry; invalid rules are skipped with a warning."""
    label = f"{source}#rules[{index}]"
    if not isinstance(raw, dict) or not raw.get("path"):
        logger.warning("Folder rule %s skipped: missing 'path'", label)
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
        logger.warning("Folder rule %s skipped: %s", label, e)
        return None

    if not (
        rule.categories
        or rule.image_categories
        or rule.doc_template
        or rule.vision_template
        or rule.image_template
    ):
        logger.warning("Folder rule %s skipped: no categories or templates defined", label)
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
        logger.warning("Could not load %s folder rules from %s: %s", label, path, e)
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
