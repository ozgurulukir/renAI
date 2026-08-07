from unittest.mock import MagicMock
from renai.processors.book import BookProcessor
from renai.models.schemas import ProcessingConfig
from renai.engines.ocr import OCREngine
from renai.engines.llm import LLMEngine

def test_config_injection():
    ocr_engine = MagicMock(spec=OCREngine)
    llm_engine = MagicMock(spec=LLMEngine)

    config = ProcessingConfig(
        provider="openai",
        model="gpt-4o",
        extractor="pdfplumber",
        text_length=15000,
        text_extraction="regular",
        extraction_pages=5,
        move_scan_failures=True,
        vision_max_pages=2,
        vision_provider="openai",
        vision_model="gpt-4o"
    )

    processor = BookProcessor(
        ocr_engine=ocr_engine,
        config=config,
        llm_engine=llm_engine,
    )

    assert processor.extractor_name == "pdfplumber"
    assert processor.target_length == 15000
    assert processor.move_scan_failures == True
    assert processor.vision_provider == "openai"
    assert processor.vision_model == "gpt-4o"
    assert processor.provider == "openai"
    assert processor.model == "gpt-4o"
    print("Validation passed!")

if __name__ == "__main__":
    test_config_injection()
