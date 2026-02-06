import json
import logging
import re
from typing import Any, Dict, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from config import get_allowed_categories


class LLMEngine:
    def __init__(self, provider: str, api_key: str, base_url: str, model_name: str) -> None:
        self.provider = provider
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.allowed_categories = get_allowed_categories()
        self.prompt_template = """
Imagine you are an experienced librarian specializing in cataloging and categorization.
Your task is to extract or generate metadata for the source document based on the provided text fragment.

### **Guidelines:**
- Detect and respect the source document's language.
- Provide the language code in the "language" field ("tr" for Turkish, "en" for English/Other).
- Make the metadata concise and clean.
- Use proper capitalization for titles, subtitles, and other fields.
- Provide ONLY a valid JSON object. No explanations, no markdown blocks unless necessary.

### **Category & Language Rules:**
You must choose the nearest category from the list below matching the detected language.

| Language | Categories |
| :--- | :--- |
| **Turkish (tr)** | Makale, Kurgu, Kurgu Dışı, Ders Kitabı, Rapor, Devlet, Sunum, Kılavuz, Tez, Araştırma, Diğer, Sözlük |
| **English/Other (en)** | Article, Fiction, Non-Fiction, Textbook, Report, Gov, Presentation, Manual, Thesis, Research, Other, Dictionary |

*Note: Use 'Devlet' or 'Gov' for laws, standards, government specs, and official regulations.*

### **Output Format (JSON):**
{{
    "title": "Main title",
    "subtitle": "Subtitle (if any)",
    "edition": "Edition (e.g., 2nd Edition)",
    "author": "Author or Editor name",
    "year": "Publication year (4 digits)",
    "category": "Selected category label",
    "language": "tr or en"
}}

**Text fragment**:
{text}

JSON:
"""
        self.refinement_prompt_template = """
You are a meticulous senior book editor and librarian.
Your task is to REVIEW and REFINE the initial metadata JSON extracted from a text fragment.

### **Goal:**
Ensure the metadata is accurate, complete, and standardized.

### **Refinement Rules:**
1. **Missing Information & Knowledge:** If fields are missing (like year, author), try to find them in the text. For well-known books, you MAY use your internal knowledge to fill in missing details if the identity of the book is clear.
2. **Corrections:** Fix OCR typos or spelling errors in Title/Author using your knowledge of the actual book (e.g., 'Harry Poter' -> 'Harry Potter').
3. **Category Validation:** Verify the 'category' field. If the content clearly belongs to a more specific valid category from the allowed list, update it.
4. **Standardization:**
    - Ignore minor variations in 'Edition' or 'Publisher' unless they are completely wrong.
    - Ensure Title and Author are properly capitalized.
5. **Language:** Ensure the 'language' field accurately reflects the text source ("tr" or "en").

### **Allowed Categories:**
{allowed_categories}

### **Input Data:**
**Text Fragment:**
{text}

**Initial Metadata:**
{current_json}

### **Output:**
Provide ONLY the refined valid JSON object.
"""

    def get_book_metadata(self, text: str) -> Optional[Dict[str, Any]]:
        """Generate book metadata using the LLM with category validation and retries."""
        if not self.client.api_key:
            logging.error(f"API Key for {self.provider} not found.")
            return None

        truncated_text = text[:8000]

        # 1. Initial Extraction
        initial_metadata = self._call_llm(
            prompt=self.prompt_template.format(text=truncated_text),
            system_role="You are a professional librarian. Extract metadata in JSON format.",
        )

        if not initial_metadata:
            return None

        # 2. Refinement Pass
        logging.info("Refining metadata with secondary pass...")
        refined_metadata = self._call_llm(
            prompt=self.refinement_prompt_template.format(
                text=truncated_text,
                current_json=json.dumps(initial_metadata),
                allowed_categories=json.dumps(
                    self.allowed_categories, ensure_ascii=False
                ),
            ),
            system_role="You are a senior editor. Refine and correct the metadata.",
        )

        # Return refined if successful, otherwise fallback to initial
        return refined_metadata if refined_metadata else initial_metadata

    def _call_llm(self, prompt: str, system_role: str) -> Optional[Dict[str, Any]]:
        """Helper method to handle LLM calls, retries, and JSON parsing."""
        extra_headers = {}
        if self.provider == "openrouter":
            extra_headers = {
                "HTTP-Referer": "https://github.com/renAI",
                "X-Title": "renAI Book Renamer",
            }

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt},
        ]

        book_data: dict[str, Any] | None = None

        for attempt in range(2):
            try:
                chat_completion = self.client.chat.completions.create(  # type: ignore
                    model=self.model_name,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.3 if attempt == 0 else 0.5,
                    extra_headers=extra_headers
                    if self.provider == "openrouter"
                    else None,
                )

                content = chat_completion.choices[0].message.content
                if content is None:
                    return None
                content = content.strip()

                # Clean markdown blocks
                if content.startswith("```"):
                    content = re.sub(
                        r"^```(?:json)?\n?|```$", "", content, flags=re.MULTILINE
                    ).strip()

                book_data = json.loads(content)

                # Basic Validation
                if book_data is None:
                    return None
                lang = book_data.get("language", "en").lower()
                if lang not in self.allowed_categories:
                    lang = "en"

                category = book_data.get("category", "")
                allowed = self.allowed_categories[lang]

                if category in allowed:
                    return book_data
                else:
                    logging.warning(
                        f"Invalid category '{category}' for language '{lang}' (Attempt {attempt + 1}). Retrying..."
                    )
                    # Add feedback for next turn in retry loop, though for simplicity in helper we might just retry or fix
                    # For this helper, let's just retry the generation with a hint if we were essentially doing a conversation loop
                    # But since we simplified to a helper, we'll just continue to next attempt or strict fail.
                    # To mimic previous logic better:
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": f"The category '{category}' is not in the allowed list for '{lang}'. Please choose ONLY from: {', '.join(allowed)}",
                        }
                    )

            except Exception as e:
                logging.error(f"LLM request failed (Attempt {attempt + 1}): {e}")

        # Fallback if we have parsed JSON but category was wrong
        if book_data is not None:
            lang = book_data.get("language", "en").lower()
            if lang not in self.allowed_categories:
                lang = "en"
            book_data["category"] = "Diğer" if lang == "tr" else "Other"
            return book_data

        return None
