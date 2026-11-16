import asyncio
import base64
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessageParam

from ..config import get_allowed_categories, get_image_categories
from ..config.folder_rules import resolve_allowed_bucket, resolve_fallback_category
from ..exceptions import LLMError
from ..models.schemas import BaseMetadata, BookMetadata, ImageMetadata
from ..utils.rate_limiter import AsyncRateLimiter

T = TypeVar("T", bound=BaseMetadata)

# Values that indicate missing/placeholder metadata (P0-1 refinement gate).
_METADATA_PLACEHOLDERS = frozenset(
    {
        "",
        "unknown",
        "unknown title",
        "unknown author",
        "untitled",
        "n/a",
        "na",
        "none",
        "null",
        "-",
        "--",
    }
)

# HTTP status threshold separating retryable server errors (>= threshold, i.e.
# 5xx) from fail-fast client errors (< threshold, i.e. 4xx other than 429).
_RETRYABLE_STATUS_THRESHOLD = 500

# Fallback category map used by ``resolve_fallback_category`` when no per-folder
# restriction applies. Wrapped in the same ``dict[str, list[str]]`` shape so a
# single owner (folder_rules) decides fallback-category semantics.
_GLOBAL_FALLBACK_CATEGORIES: dict[str, list[str]] = {"tr": ["Diğer"], "en": ["Other"]}


@dataclass(frozen=True)
class LLMCallPolicy:
    """Behavioral knobs for LLM calls, injected at construction time.

    Translated from the global settings singleton by ``orchestrator.setup_engines`` so the
    engine never reads the global settings singleton at call time. Frozen:
    configuration is fixed per engine; two engines may safely differ.

    Field defaults mirror the settings defaults, so constructing
    ``LLMEngine`` without an explicit policy (tests, scripts) preserves the
    documented default behavior. Production always passes a policy built by
    ``setup_engines``.
    """

    max_tokens: int = 1024
    temperature: float = 0.3
    temperature_retry: float = 0.5
    max_retries: int = 2
    refine_mode: str = "auto"
    # Transport-level knobs (HTTP timeout + retryable-error backoff). Defaults
    # mirror AppSettings; see settings.py for the domain rationale.
    timeout: float = 120.0
    transport_max_attempts: int = 5
    backoff_initial: float = 2.0
    backoff_max: float = 60.0


# Standard exponential-backoff growth factor; fixed by convention, not a knob.
_BACKOFF_MULTIPLIER = 2.0


def _filter_model_fields(
    raw_data: dict[str, Any], model_class: type[BaseMetadata]
) -> dict[str, Any]:
    """Drop keys not present on the Pydantic model (SSoT field filtering)."""
    model_fields = model_class.model_fields.keys()
    return {k: v for k, v in raw_data.items() if k in model_fields}


def _classify_llm_error(err: Exception) -> tuple[bool, str]:
    """Classify an LLM API error as (retryable, label).

    ``label`` feeds both the retry warning and the "retries exhausted" error
    message. Retryable: RateLimitError (HTTP 429), timeouts/connection errors,
    and server-side HTTP 5xx. Non-retryable: other client errors (HTTP 4xx) and
    unexpected exceptions.
    """
    if isinstance(err, RateLimitError):
        return True, "Rate limit"
    if isinstance(err, (APITimeoutError, APIConnectionError)):
        return True, "Connection/timeout"
    if isinstance(err, APIStatusError):
        if err.status_code >= _RETRYABLE_STATUS_THRESHOLD:
            return True, "Server error"
        return False, f"Non-retryable API error (HTTP {err.status_code})"
    return False, "LLM call failure"


class LLMEngine:
    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model_name: str,
        rate_limiter: AsyncRateLimiter | None = None,
        call_policy: LLMCallPolicy | None = None,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        # Resolve the effective policy first: the HTTP client timeout is a
        # policy knob, so the client cannot be built before it is normalized.
        self.call_policy = call_policy or LLMCallPolicy()
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            # Fail a hung endpoint in bounded time instead of the SDK's implicit
            # 600s; every other timeout in the app is settings-backed too.
            timeout=self.call_policy.timeout,
        )
        self.rate_limiter = rate_limiter
        self.allowed_categories = get_allowed_categories()
        # Path to the prompts directory within the package
        self.prompts_dir = Path(__file__).parent.parent / "prompts"

        # Load templates once
        self._templates = self._load_all_templates()

    async def aclose(self) -> None:
        """Release the underlying HTTP client (sockets, connection pool).

        The engine holds an ``AsyncOpenAI`` client backed by ``httpx``; without
        an explicit close the pool lingers until garbage collection, which
        matters for long-lived sessions and repeated pipeline runs.
        """
        await self.client.close()

    def _load_all_templates(self) -> dict[str, str]:
        """Load all prompt templates from the prompts directory."""
        templates = {}
        template_names = [
            "system_base",
            "book_extractor",
            "refinement",
            "image_analyzer",
        ]

        for name in template_names:
            path = self.prompts_dir / f"{name}.txt"
            try:
                if path.exists():
                    with path.open(encoding="utf-8") as f:
                        templates[name] = f.read()
                else:
                    logging.error(f"Prompt template not found: {path}")
                    templates[name] = ""
            except Exception as e:
                logging.error(f"Failed to load prompt template {name}: {e}")
                templates[name] = ""
        return templates

    def _format_category_table(self, categories: dict[str, list]) -> str:
        """Format a category dictionary into a markdown table for prompts."""
        tr_cats = ", ".join(categories.get("tr", []))
        en_cats = ", ".join(categories.get("en", []))
        return f"| **Turkish (tr)** | {tr_cats} |\n| **English/Other (en)** | {en_cats} |"

    def _get_system_role(
        self,
        template_name: str,
        model_class: type[BaseMetadata],
        is_image_mode: bool = False,
        categories: dict[str, list] | None = None,
    ) -> str:
        """Compose a full system role prompt using templates and schema injection.

        ``categories`` replaces the engine-level category table (per-folder rules).
        """
        base = self._templates.get("system_base", "")
        template = self._templates.get(template_name, "")

        # Determine which categories to inject
        cats = categories or (get_image_categories() if is_image_mode else self.allowed_categories)
        category_table = self._format_category_table(cats)

        # Generate JSON schema example from Pydantic model
        schema_dict = model_class.model_json_schema()
        # Create a simplified example for the LLM
        example_json = {
            prop: info.get("description", prop)
            for prop, info in schema_dict.get("properties", {}).items()
            if prop != "exif_date"  # Hide internal fields
        }

        formatted_base = base.format(category_table=category_table)

        # Handle cases where templates might have slightly different placeholders or combined needs
        security_warning = formatted_base.split("### **Guidelines:**")[0].strip()

        return template.format(
            system_base=formatted_base,
            security_warning=security_warning,
            category_table=category_table,
            category_rules=f"### **Category & Language Rules:**\n{category_table}",
            json_schema=json.dumps(example_json, indent=4, ensure_ascii=False),
        )

    async def get_book_metadata(
        self, text: str, allowed_categories: dict[str, list] | None = None
    ) -> BookMetadata | None:
        """Generate book metadata from text using SSoT models and modular prompts.

        ``allowed_categories`` (per-folder rule) replaces the engine-wide list
        in both the prompt and category validation.
        """
        if not self.client.api_key:
            logging.error(f"API Key for {self.provider} not found.")
            return None

        prompt = self._build_document_prompt(text)
        system_role = self._get_system_role(
            "book_extractor", BookMetadata, categories=allowed_categories
        )

        # 1. Initial Extraction
        initial_model = await self._call_llm(
            prompt=prompt,
            system_role=system_role,
            model_class=BookMetadata,
            allowed_categories=allowed_categories,
        )

        if not initial_model:
            return None

        # 2. Refinement Pass — gated by the refine_metadata cost setting (P0-1)
        return await self._maybe_refine(initial_model, allowed_categories)

    @staticmethod
    def _build_document_prompt(text: str) -> str:
        """Build LLM prompt with head/tail markers if separator is present."""
        separator = "\n---\n"
        if separator in text:
            head_text, tail_text = text.split(separator, 1)
            return (
                f"<document_content>\n"
                f"<document_head>\n{head_text}\n</document_head>\n"
                f"<document_tail>\n{tail_text}\n</document_tail>\n"
                f"</document_content>"
            )
        return f"<document_content>\n{text}\n</document_content>"

    def _is_metadata_complete(self, model: BookMetadata) -> bool:
        """True iff title and author are present and not placeholder-ish."""
        title = (model.title or "").strip().lower()
        author = (getattr(model, "author", None) or "").strip().lower()
        return (
            bool(title and author)
            and title not in _METADATA_PLACEHOLDERS
            and author not in _METADATA_PLACEHOLDERS
        )

    async def _maybe_refine(
        self, initial_model: BookMetadata, allowed_categories: dict[str, list] | None = None
    ) -> BookMetadata:
        """Gate the refinement pass by the refine_metadata cost setting.

        Shared by the text and vision extraction paths: refines on "always",
        or on "auto" when the initial metadata is incomplete; otherwise returns
        the initial model untouched.
        """
        mode = self.call_policy.refine_mode
        if mode == "always" or (mode == "auto" and not self._is_metadata_complete(initial_model)):
            return await self._refine_metadata(initial_model, allowed_categories)
        return initial_model

    async def _refine_metadata(
        self, initial_model: BookMetadata, allowed_categories: dict[str, list] | None = None
    ) -> BookMetadata:
        """Refine extracted metadata using the editor persona and SSoT schema."""
        logging.debug("Refining metadata using SSoT-based validation pass...")

        system_role = self._get_system_role(
            "refinement", BookMetadata, categories=allowed_categories
        )
        refinement_prompt = (
            f"<initial_metadata>\n"
            f"{initial_model.model_dump_json(ensure_ascii=False)}\n"
            f"</initial_metadata>"
        )

        refined_model = await self._call_llm(
            prompt=refinement_prompt,
            system_role=system_role,
            model_class=BookMetadata,
            allowed_categories=allowed_categories,
        )

        return refined_model if refined_model else initial_model

    async def get_book_metadata_from_image(
        self,
        image_path: Path,
        image_data: bytes | list[bytes],
        allowed_categories: dict[str, list] | None = None,
    ) -> BookMetadata | None:
        """Extract book metadata via vision using modular prompts."""
        if not self.client.api_key:
            logging.error(f"API Key for {self.provider} not found.")
            return None

        try:
            if isinstance(image_data, list):
                image_base64: str | list[str] = [
                    base64.b64encode(img).decode("utf-8") for img in image_data
                ]
            else:
                image_base64 = base64.b64encode(image_data).decode("utf-8")

            system_role = self._get_system_role(
                "book_extractor", BookMetadata, categories=allowed_categories
            )

            initial_model = await self._call_vision_llm(
                image_base64=image_base64,
                system_role=system_role,
                file_name=image_path.name,
                model_class=BookMetadata,
                allowed_categories=allowed_categories,
            )

            if not initial_model:
                return None

            return await self._maybe_refine(initial_model, allowed_categories)
        except Exception as e:
            logging.error(f"Vision book extraction failed for {image_path.name}: {e}")
            return None

    async def get_image_metadata(
        self,
        image_path: Path,
        image_data: bytes,
        image_categories: dict[str, list] | None = None,
    ) -> ImageMetadata | None:
        """Extract image metadata for renaming using image analyzer SSoT."""
        if not self.client.api_key:
            logging.error(f"API Key for {self.provider} not found.")
            return None

        try:
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            system_role = self._get_system_role(
                "image_analyzer",
                ImageMetadata,
                is_image_mode=True,
                categories=image_categories,
            )

            return await self._call_vision_llm(
                image_base64=image_base64,
                system_role=system_role,
                file_name=image_path.name,
                model_class=ImageMetadata,
                allowed_categories=image_categories,
            )
        except Exception as e:
            logging.error(f"Vision image extraction failed for {image_path.name}: {e}")
            return None

    async def _call_vision_llm(
        self,
        image_base64: str | list[str],
        system_role: str,
        file_name: str,
        model_class: type[T],
        allowed_categories: dict[str, list] | None = None,
    ) -> T | None:
        """Helper for vision calls with type-safe model validation."""
        image_format = Path(file_name).suffix.lower().replace(".", "") or "jpeg"
        if image_format not in ("jpeg", "png", "gif", "webp"):
            image_format = "jpeg"

        images = [image_base64] if isinstance(image_base64, str) else image_base64

        content: list[dict[str, Any]] = []
        for img in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{image_format};base64,{img}"},
                }
            )
        content.append(
            {
                "type": "text",
                "text": "Analyze the provided pages/images and extract metadata as JSON.",
            }
        )

        messages = cast(
            "list[ChatCompletionMessageParam]",
            [
                {"role": "system", "content": system_role},
                {
                    "role": "user",
                    "content": content,
                },
            ],
        )
        return await self._generate_json_completion(
            messages, model_class, file_name, allowed_categories=allowed_categories
        )

    async def _call_llm(
        self,
        prompt: str,
        system_role: str,
        model_class: type[T],
        allowed_categories: dict[str, list] | None = None,
    ) -> T | None:
        """Helper for text calls with type-safe model validation."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt},
        ]
        return await self._generate_json_completion(
            messages, model_class, allowed_categories=allowed_categories
        )

    def _prepare_completion_headers(self) -> dict[str, str] | None:
        """Prepare extra headers for the API request if necessary."""
        if self.provider == "openrouter":
            return {"HTTP-Referer": "https://github.com/renAI", "X-Title": "renAI"}
        return None

    @staticmethod
    def _try_repair_json(content: str) -> str | None:
        """Attempt local string repairs for common LLM JSON syntax errors."""
        # 1. Remove trailing commas before closing braces/brackets
        repaired = re.sub(r",\s*([\}\]])", r"\1", content)
        # 2. Extract first '{' to last '}' if surrounded by extraneous text
        start = repaired.find("{")
        end = repaired.rfind("}")
        if start != -1 and end != -1 and end > start:
            return repaired[start : end + 1]
        return None

    def _parse_json_content(self, content: str) -> dict[str, Any] | None:
        """Clean markdown formatting and parse JSON content."""
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?|```$", "", content, flags=re.MULTILINE).strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            repaired = self._try_repair_json(content)
            if repaired:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass
            return None

    def _validate_and_align_metadata(
        self,
        raw_data: dict[str, Any],
        model_class: type[T],
        context_name: str,
        categories_override: dict[str, list] | None = None,
    ) -> tuple[T | None, list[str] | None]:
        """Validate parsed data against Pydantic model and check category constraints.

        On a schema/validation failure returns ``(None, None)``. Otherwise returns the
        validated model; an invalid category is rejected and the allowed list returned
        as ``retry_list`` so the caller can surface the forbidden value.
        """
        try:
            filtered_data = _filter_model_fields(raw_data, model_class)
            validated_model = model_class(**filtered_data)
        except Exception as e:
            logging.warning(f"Schema validation failed for {context_name}: {e}")
            return None, None

        # Category & Language SSoT Alignment
        lang = validated_model.language.lower()
        is_image = isinstance(validated_model, ImageMetadata)
        allowed_dict = categories_override or (
            get_image_categories() if is_image else self.allowed_categories
        )

        allowed = resolve_allowed_bucket(allowed_dict, lang)

        if validated_model.category in allowed:
            return validated_model, None

        # Item 2: invalid category -> reject and return the allowed list so the
        # caller (or an explicit retry pass) can surface the forbidden value.
        return None, allowed

    def _attempt_final_fallback(
        self,
        last_parsed_data: dict[str, Any] | None,
        model_class: type[T],
        categories_override: dict[str, list] | None = None,
    ) -> T | None:
        """Fall back to a forced category if validation initially failed.

        With a per-folder restriction, the fallback category is resolved from
        the restricted list (preferring "Other"/"Diğer", else the first entry).
        """
        if not last_parsed_data:
            return None
        try:
            filtered_data = _filter_model_fields(last_parsed_data, model_class)
            if categories_override:
                filtered_data["category"] = resolve_fallback_category(
                    filtered_data.get("language", "en"), categories_override
                )
            else:
                filtered_data["category"] = resolve_fallback_category(
                    filtered_data.get("language", "en"), _GLOBAL_FALLBACK_CATEGORIES
                )
            return model_class(**filtered_data)
        except Exception as e:
            logging.debug(
                "Could not coerce LLM JSON payload to %s (%s); treating as extraction failure.",
                model_class.__name__,
                e,
            )
            return None

    async def _execute_completion_with_backoff(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
        headers: dict[str, str] | None,
        context_name: str,
    ) -> str | None:
        """Call the LLM API with progressive exponential backoff.

        Retries on RateLimitError (HTTP 429), timeouts, connection errors, and
        server-side HTTP 5xx errors; non-retryable client errors (HTTP 4xx other
        than 429) fail fast with LLMError. Error classification lives in
        :func:`_classify_llm_error` so retry behavior is testable without
        patching globals or sleeping.
        """
        backoff_sec = self.call_policy.backoff_initial
        max_backoff_sec = self.call_policy.backoff_max
        max_attempts = self.call_policy.transport_max_attempts

        for attempt in range(max_attempts):
            try:
                if self.rate_limiter:
                    await self.rate_limiter.acquire()

                completion = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.call_policy.max_tokens,
                    temperature=temperature,
                    extra_headers=headers,
                )
                return completion.choices[0].message.content
            except Exception as err:
                retryable, label = _classify_llm_error(err)
                if not retryable:
                    logging.error("%s for %s: %s", label, context_name, err)
                    raise LLMError(f"{label} for {context_name}: {err}") from err
                if attempt == max_attempts - 1:
                    logging.error(
                        "%s retries exhausted (%d attempts) for %s.",
                        label,
                        max_attempts,
                        context_name,
                    )
                    raise LLMError(
                        f"{label} retries exhausted ({max_attempts} attempts) "
                        f"for {context_name}: {err}"
                    ) from err
                logging.warning(
                    "%s for %s. Retrying in %.1fs... Error: %s",
                    label,
                    context_name,
                    backoff_sec,
                    err,
                )
                await asyncio.sleep(backoff_sec)
                backoff_sec = min(backoff_sec * _BACKOFF_MULTIPLIER, max_backoff_sec)
        raise LLMError(f"LLM call returned no completion for {context_name}")

    async def _generate_json_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        model_class: type[T],
        context_name: str = "document",
        allowed_categories: dict[str, list] | None = None,
    ) -> T | None:
        """Unified LLM caller returning typed Pydantic models with SSoT enforcement."""
        headers = self._prepare_completion_headers()
        last_parsed_data: dict[str, Any] | None = None

        for attempt in range(self.call_policy.max_retries):
            try:
                temperature = (
                    self.call_policy.temperature
                    if attempt == 0
                    else self.call_policy.temperature_retry
                )

                content = await self._execute_completion_with_backoff(
                    messages=messages,
                    temperature=temperature,
                    headers=headers,
                    context_name=context_name,
                )
                if not content:
                    continue

                raw_data = self._parse_json_content(content)
                if raw_data is None:
                    continue
                last_parsed_data = raw_data

                validated_model, allowed = self._validate_and_align_metadata(
                    raw_data, model_class, context_name, categories_override=allowed_categories
                )
                if validated_model:
                    return validated_model

                if allowed:
                    logging.warning(
                        "Invalid category '%s' for %s (Attempt %d)",
                        raw_data.get("category"),
                        context_name,
                        attempt + 1,
                    )
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Category must be ONLY from: {', '.join(allowed)}",
                        }
                    )
            except Exception as e:
                logging.error(f"Failed to generate JSON completion for {context_name}: {e}")

        # Final Fallback to a forced category if JSON exists
        return self._attempt_final_fallback(
            last_parsed_data, model_class, categories_override=allowed_categories
        )
