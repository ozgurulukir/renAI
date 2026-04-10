import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from ..settings import get_allowed_categories, get_image_categories, app_settings
from ..models.schemas import BookMetadata, ImageMetadata, BaseMetadata

T = TypeVar("T", bound=BaseMetadata)

class LLMEngine:
    def __init__(
        self, provider: str, api_key: str, base_url: str, model_name: str
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.allowed_categories = get_allowed_categories()
        # Path to the prompts directory within the package
        self.prompts_dir = Path(__file__).parent.parent / "prompts"
        
        # Load templates once
        self._templates = self._load_all_templates()

    def _load_all_templates(self) -> Dict[str, str]:
        """Load all prompt templates from the prompts directory."""
        templates = {}
        template_names = ["system_base", "book_extractor", "refinement", "image_analyzer"]
        
        for name in template_names:
            path = self.prompts_dir / f"{name}.txt"
            try:
                if path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        templates[name] = f.read()
                else:
                    logging.error(f"Prompt template not found: {path}")
                    templates[name] = ""
            except Exception as e:
                logging.error(f"Failed to load prompt template {name}: {e}")
                templates[name] = ""
        return templates

    def _format_category_table(self, categories: Dict[str, list]) -> str:
        """Format a category dictionary into a markdown table for prompts."""
        tr_cats = ", ".join(categories.get("tr", []))
        en_cats = ", ".join(categories.get("en", []))
        return f"| **Turkish (tr)** | {tr_cats} |\n| **English/Other (en)** | {en_cats} |"

    def _get_system_role(self, template_name: str, model_class: Type[BaseMetadata], is_image_mode: bool = False) -> str:
        """Compose a full system role prompt using templates and schema injection."""
        base = self._templates.get("system_base", "")
        template = self._templates.get(template_name, "")
        
        # Determine which categories to inject
        cats = get_image_categories() if is_image_mode else self.allowed_categories
        category_table = self._format_category_table(cats)
        
        # Generate JSON schema example from Pydantic model
        schema_dict = model_class.model_json_schema()
        # Create a simplified example for the LLM
        example_json = {
            prop: info.get("description", prop)
            for prop, info in schema_dict.get("properties", {}).items()
            if prop != "exif_date" # Hide internal fields
        }
        
        formatted_base = base.format(category_table=category_table)
        
        # Handle cases where templates might have slightly different placeholders or combined needs
        security_warning = formatted_base.split("### **Guidelines:**")[0].strip()
        
        return template.format(
            system_base=formatted_base,
            security_warning=security_warning,
            category_table=category_table,
            category_rules=f"### **Category & Language Rules:**\n{category_table}",
            json_schema=json.dumps(example_json, indent=4, ensure_ascii=False)
        )

    def get_book_metadata(self, text: str) -> Optional[BookMetadata]:
        """Generate book metadata from text using SSoT models and modular prompts."""
        if not self.client.api_key:
            logging.error(f"API Key for {self.provider} not found.")
            return None

        truncated_text = text[:app_settings.text_length]
        system_role = self._get_system_role("book_extractor", BookMetadata)

        # 1. Initial Extraction
        initial_model = self._call_llm(
            prompt=f"<document_content>\n{truncated_text}\n</document_content>",
            system_role=system_role,
            model_class=BookMetadata
        )

        if not initial_model:
            return None

        # 2. Refinement Pass
        return self._refine_metadata(initial_model)

    def _refine_metadata(self, initial_model: BookMetadata) -> BookMetadata:
        """Refine extracted metadata using the editor persona and SSoT schema."""
        logging.info("Refining metadata using SSoT-based validation pass...")
        
        system_role = self._get_system_role("refinement", BookMetadata)
        refinement_prompt = f"<initial_metadata>\n{initial_model.model_dump_json(ensure_ascii=False)}\n</initial_metadata>"

        refined_model = self._call_llm(
            prompt=refinement_prompt,
            system_role=system_role,
            model_class=BookMetadata
        )

        return refined_model if refined_model else initial_model

    def get_book_metadata_from_image(self, image_path: Path, image_data: bytes) -> Optional[BookMetadata]:
        """Extract book metadata via vision using modular prompts."""
        if not self.client.api_key:
            logging.error(f"API Key for {self.provider} not found.")
            return None

        try:
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            system_role = self._get_system_role("book_extractor", BookMetadata)

            initial_model = self._call_vision_llm(
                image_base64=image_base64,
                system_role=system_role,
                file_name=image_path.name,
                model_class=BookMetadata
            )

            if not initial_model:
                return None

            return self._refine_metadata(initial_model)
        except Exception as e:
            logging.error(f"Vision book extraction failed for {image_path.name}: {e}")
            return None

    def get_image_metadata(self, image_path: Path, image_data: bytes) -> Optional[ImageMetadata]:
        """Extract image metadata for renaming using image analyzer SSoT."""
        if not self.client.api_key:
            logging.error(f"API Key for {self.provider} not found.")
            return None

        try:
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            system_role = self._get_system_role("image_analyzer", ImageMetadata, is_image_mode=True)

            return self._call_vision_llm(
                image_base64=image_base64,
                system_role=system_role,
                file_name=image_path.name,
                model_class=ImageMetadata
            )
        except Exception as e:
            logging.error(f"Vision image extraction failed for {image_path.name}: {e}")
            return None

    def _call_vision_llm(self, image_base64: str, system_role: str, file_name: str, model_class: Type[T]) -> Optional[T]:
        """Helper for vision calls with type-safe model validation."""
        image_format = Path(file_name).suffix.lower().replace(".", "") or "jpeg"
        if image_format not in ("jpeg", "png", "gif", "webp"):
            image_format = "jpeg"

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_role},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{image_base64}"}},
                    {"type": "text", "text": "Analyze this document/image and extract metadata as JSON."},
                ],
            },
        ]
        return self._generate_json_completion(messages, model_class, file_name)

    def _call_llm(self, prompt: str, system_role: str, model_class: Type[T]) -> Optional[T]:
        """Helper for text calls with type-safe model validation."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt},
        ]
        return self._generate_json_completion(messages, model_class)

    def _generate_json_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        model_class: Type[T],
        context_name: str = "document",
    ) -> Optional[T]:
        """Unified LLM caller returning typed Pydantic models with SSoT enforcement."""
        headers = {"HTTP-Referer": "https://github.com/renAI", "X-Title": "renAI"} if self.provider == "openrouter" else None
        last_parsed_data: dict[str, Any] | None = None

        for attempt in range(app_settings.llm_max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=app_settings.llm_max_tokens,
                    temperature=app_settings.llm_temperature if attempt == 0 else app_settings.llm_temperature_retry,
                    extra_headers=headers,
                )

                content = completion.choices[0].message.content
                if not content:
                    continue
                
                # Cleanup markdown and parse JSON
                content = content.strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\n?|```$", "", content, flags=re.MULTILINE).strip()
                
                try:
                    raw_data = json.loads(content)
                    last_parsed_data = raw_data
                except json.JSONDecodeError:
                    continue

                # Pydantic Validation (The SSoT Core)
                try:
                    # Filter out fields not in model to handle LLM talkativeness
                    model_fields = model_class.model_fields.keys()
                    filtered_data = {k: v for k, v in raw_data.items() if k in model_fields}
                    
                    validated_model = model_class(**filtered_data)
                except Exception as e:
                    logging.warning(f"Schema validation failed for {context_name}: {e}")
                    continue

                # Category & Language SSoT Alignment
                lang = validated_model.language.lower()
                if lang not in self.allowed_categories:
                    # Don't mutate here yet, check allowed list
                    pass
                
                is_image = isinstance(validated_model, ImageMetadata)
                allowed_dict = get_image_categories() if is_image else self.allowed_categories
                
                target_lang = lang if lang in allowed_dict else "en"
                allowed = allowed_dict.get(target_lang, allowed_dict.get("en", []))

                if validated_model.category in allowed:
                    return validated_model
                else:
                    logging.warning(f"Invalid category '{validated_model.category}' for {context_name} (Attempt {attempt + 1})")
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": f"Category must be ONLY from: {', '.join(allowed)}"})

            except Exception as e:
                logging.error(f"LLM call failure for {context_name}: {e}")

        # Final Fallback to "Other" if JSON exists
        if last_parsed_data:
            try:
                model_fields = model_class.model_fields.keys()
                filtered_data = {k: v for k, v in last_parsed_data.items() if k in model_fields}
                # Force a safe category
                lang = filtered_data.get("language", "en").lower()
                filtered_data["category"] = "Diğer" if lang == "tr" else "Other"
                return model_class(**filtered_data)
            except Exception:
                pass

        return None
