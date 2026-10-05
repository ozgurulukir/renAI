import asyncio
import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, TypeVar, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from ..config import app_settings, get_allowed_categories, get_image_categories
from ..config.folder_rules import resolve_allowed_bucket, resolve_fallback_category
from ..exceptions import LLMError
from ..models.schemas import BaseMetadata, BookMetadata, ImageMetadata
from ..utils.rate_limiter import AsyncRateLimiter

T = TypeVar("T", bound=BaseMetadata)


class LLMEngine:
    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model_name: str,
        rate_limiter: AsyncRateLimiter | None = None,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.rate_limiter = rate_limiter
        self.allowed_categories = get_allowed_categories()
        # Path to the prompts directory within the package
        self.prompts_dir = Path(__file__).parent.parent / "prompts"

        # Load templates once
        self._templates = self._load_all_templates()

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

        # 2. Refinement Pass
        return await self._refine_metadata(initial_model, allowed_categories)

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

            return await self._refine_metadata(initial_model, allowed_categories)
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
            model_fields = model_class.model_fields.keys()
            filtered_data = {k: v for k, v in raw_data.items() if k in model_fields}
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
            model_fields = model_class.model_fields.keys()
            filtered_data = {k: v for k, v in last_parsed_data.items() if k in model_fields}
            if categories_override:
                filtered_data["category"] = resolve_fallback_category(
                    filtered_data.get("language", "en"), categories_override
                )
            else:
                lang = filtered_data.get("language", "en").lower()
                filtered_data["category"] = "Diğer" if lang == "tr" else "Other"
            return model_class(**filtered_data)
        except Exception:
            return None

    async def _execute_completion_with_backoff(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
        headers: dict[str, str] | None,
        context_name: str,
    ) -> str | None:
        """Call the LLM API with progressive exponential backoff on RateLimitError."""
        from openai import RateLimitError

        backoff_sec = 2.0
        max_backoff_sec = 60.0
        rate_limit_attempts = 5

        for rl_attempt in range(rate_limit_attempts):
            try:
                if self.rate_limiter:
                    await self.rate_limiter.acquire()

                completion = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=app_settings.llm_max_tokens,
                    temperature=temperature,
                    extra_headers=headers,
                )
                return completion.choices[0].message.content
            except RateLimitError as rle:
                if rl_attempt == rate_limit_attempts - 1:
                    logging.error(
                        "Rate limit retries exhausted (%d attempts) for %s.",
                        rate_limit_attempts,
                        context_name,
                    )
                    raise LLMError(
                        f"Rate limit retries exhausted ({rate_limit_attempts} attempts) "
                        f"for {context_name}: {rle}"
                    ) from rle
                logging.warning(
                    "Rate limit exceeded (HTTP 429) for %s. Retrying in %.1fs... Error: %s",
                    context_name,
                    backoff_sec,
                    rle,
                )
                await asyncio.sleep(backoff_sec)
                backoff_sec = min(backoff_sec * 2.0, max_backoff_sec)
            except Exception as e:
                logging.error("LLM call failure for %s: %s", context_name, e)
                raise LLMError(f"LLM call failed for {context_name}: {e}") from e
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

        for attempt in range(app_settings.llm_max_retries):
            try:
                temperature = (
                    app_settings.llm_temperature
                    if attempt == 0
                    else app_settings.llm_temperature_retry
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
