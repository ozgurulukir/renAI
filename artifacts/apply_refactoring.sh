#!/bin/bash
set -e

# Update BookProcessor init
cat << 'INNER_EOF' > update_book.py
with open("src/renai/processors/book.py", "r") as f:
    content = f.read()

old_init = """    def __init__(
        self,
        ocr_engine: OCREngine,
        llm_engine: LLMEngine | None = None,
        extractor_name: str = "pymupdf",
        provider: str = "deepinfra",
        model: str = "",
        target_length: int = 10000,
        text_extraction: str = "head-tail",
        extraction_pages: int = 10,
        move_scan_failures: bool = False,
        vision_max_pages: int | None = None,
        vision_llm_engine: LLMEngine | None = None,
        vision_provider: str | None = None,
        vision_model: str | None = None,
    ):"""
new_init = """    def __init__(
        self,
        ocr_engine: OCREngine,
        config: 'ProcessingConfig',
        llm_engine: LLMEngine | None = None,
        vision_llm_engine: LLMEngine | None = None,
    ):"""
content = content.replace(old_init, new_init)

old_body = """        self.ocr_engine = ocr_engine
        self.llm_engine = llm_engine
        self.vision_llm_engine = vision_llm_engine or llm_engine
        self.provider = provider
        self.model = model or (getattr(llm_engine, "model_name", "") if llm_engine else "")
        self.vision_provider = vision_provider or (
            getattr(self.vision_llm_engine, "provider", provider)
            if self.vision_llm_engine
            else provider
        )
        self.vision_model = vision_model or (
            getattr(self.vision_llm_engine, "model_name", self.model)
            if self.vision_llm_engine
            else self.model
        )
        self.target_length = target_length
        self.extractor_name = extractor_name
        self.move_scan_failures = move_scan_failures

        self.text_extractor = TextExtractor(
            ocr_engine=ocr_engine,
            extractor_name=extractor_name,
            target_length=target_length,
            text_extraction=text_extraction,
            extraction_pages=extraction_pages,
        )
        self.visual_processor = VisualProcessor(
            llm_engine=self.vision_llm_engine,
            provider=self.vision_provider,
            model=self.vision_model,
            vision_max_pages=vision_max_pages,
        )"""

new_body = """        self.config = config
        self.ocr_engine = ocr_engine
        self.llm_engine = llm_engine
        self.vision_llm_engine = vision_llm_engine or llm_engine

        # Pull from config
        self.provider = config.provider
        self.model = config.model or ""
        self.vision_provider = config.vision_provider or self.provider
        self.vision_model = config.vision_model or self.model
        self.target_length = config.text_length
        self.extractor_name = config.extractor
        self.move_scan_failures = config.move_scan_failures

        self.text_extractor = TextExtractor(
            ocr_engine=ocr_engine,
            extractor_name=config.extractor,
            target_length=config.text_length,
            text_extraction=config.text_extraction,
            extraction_pages=config.extraction_pages,
        )
        self.visual_processor = VisualProcessor(
            llm_engine=self.vision_llm_engine,
            provider=self.vision_provider,
            model=self.vision_model,
            vision_max_pages=config.vision_max_pages,
        )"""
content = content.replace(old_body, new_body)

import_str = "from ..models.schemas import BookMetadata, ImageMetadata"
new_import_str = "from ..models.schemas import BookMetadata, ImageMetadata, ProcessingConfig"
content = content.replace(import_str, new_import_str)

with open("src/renai/processors/book.py", "w") as f:
    f.write(content)
INNER_EOF
python3 update_book.py

# Update Orchestrator
cat << 'INNER_EOF' > update_orchestrator.py
with open("src/renai/processors/orchestrator.py", "r") as f:
    content = f.read()

old_processor_init = """    # Initialize BookProcessor
    processor = BookProcessor(
        ocr_engine,
        llm_engine,
        extractor_name=config.extractor,
        provider=provider,
        model=model_name,
        target_length=config.text_length,
        text_extraction=config.text_extraction,
        extraction_pages=config.extraction_pages,
        move_scan_failures=config.move_scan_failures,
        vision_max_pages=config.vision_max_pages,
        vision_llm_engine=vision_llm_engine,
        vision_provider=vision_provider,
        vision_model=vision_model_name,
    )"""

new_processor_init = """    # Initialize BookProcessor
    processor = BookProcessor(
        ocr_engine=ocr_engine,
        config=config,
        llm_engine=llm_engine,
        vision_llm_engine=vision_llm_engine,
    )"""
content = content.replace(old_processor_init, new_processor_init)

with open("src/renai/processors/orchestrator.py", "w") as f:
    f.write(content)
INNER_EOF
python3 update_orchestrator.py

# Update CLI
cat << 'INNER_EOF' > update_cli.py
with open("src/renai/cli.py", "r") as f:
    content = f.read()

old_build = """    config = _build_processing_config(
        mode=mode,
        workers=workers,
        recursive=recursive,"""

new_build = """    config = _build_processing_config(
        mode=mode,
        provider=provider,
        model=model,
        workers=workers,
        recursive=recursive,"""
content = content.replace(old_build, new_build)

with open("src/renai/cli.py", "w") as f:
    f.write(content)
INNER_EOF
python3 update_cli.py

# Update Schemas
cat << 'INNER_EOF' > update_schemas.py
with open("src/renai/models/schemas.py", "r") as f:
    content = f.read()

old_config = """class ProcessingConfig(BaseModel):
    \"\"\"Configuration for the processing pipeline.\"\"\"

    mode: Literal["rename", "organize", "evaluate"] = "rename\""""

new_config = """class ProcessingConfig(BaseModel):
    \"\"\"Configuration for the processing pipeline.\"\"\"

    mode: Literal["rename", "organize", "evaluate"] = "rename"
    provider: str = ""
    model: str | None = None\""""
content = content.replace(old_config, new_config)

with open("src/renai/models/schemas.py", "w") as f:
    f.write(content)
INNER_EOF
python3 update_schemas.py

# Update tests
cat << 'INNER_EOF' > update_tests.py
with open("tests/test_file_processor.py", "r") as f:
    content = f.read()

old_processor_fixture = """    @pytest.fixture
    def processor(self, mock_ocr, mock_llm):
        return BookProcessor(mock_ocr, mock_llm)"""

new_processor_fixture = """    @pytest.fixture
    def processor(self, mock_ocr, mock_llm):
        from renai.models.schemas import ProcessingConfig
        config = ProcessingConfig(
            extractor="pymupdf",
            text_length=10000,
            text_extraction="head-tail",
            extraction_pages=10,
            move_scan_failures=False,
            vision_max_pages=3,
            provider="deepinfra",
            model="test-model"
        )
        return BookProcessor(ocr_engine=mock_ocr, config=config, llm_engine=mock_llm)"""
content = content.replace(old_processor_fixture, new_processor_fixture)

with open("tests/test_file_processor.py", "w") as f:
    f.write(content)

with open("tests/test_integration.py", "r") as f:
    content = f.read()

old_fixture = """@pytest.fixture
def processor(mock_engines):
    \"\"\"Fixture to provide a BookProcessor with mocked engines and extractor.\"\"\"
    ocr, llm = mock_engines
    proc = BookProcessor(ocr, llm)"""

new_fixture = """@pytest.fixture
def processor(mock_engines):
    \"\"\"Fixture to provide a BookProcessor with mocked engines and extractor.\"\"\"
    from renai.models.schemas import ProcessingConfig
    ocr, llm = mock_engines
    config = ProcessingConfig(extractor="pymupdf", text_length=10000, provider="deepinfra", model="test-model")
    proc = BookProcessor(ocr_engine=ocr, config=config, llm_engine=llm)"""
content = content.replace(old_fixture, new_fixture)

old_engines = """def mock_engines():
    \"\"\"Fixture to provide mocked OCR and LLM engines.\"\"\"
    ocr = MagicMock()
    llm = MagicMock()
    llm.model_name = "test-model\""""

new_engines = """def mock_engines():
    \"\"\"Fixture to provide mocked OCR and LLM engines.\"\"\"
    ocr = MagicMock()
    llm = MagicMock()
    llm.provider = "deepinfra"
    llm.model_name = "test-model\""""
content = content.replace(old_engines, new_engines)

with open("tests/test_integration.py", "w") as f:
    f.write(content)
INNER_EOF
python3 update_tests.py

# Cleanup temp files
rm update_book.py update_orchestrator.py update_cli.py update_schemas.py update_tests.py

echo "Refactoring applied successfully."
