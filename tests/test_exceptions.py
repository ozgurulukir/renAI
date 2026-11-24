from renai.exceptions import (
    CacheError,
    ConfigurationError,
    ExtractionError,
    FileOperationError,
    LLMError,
    RenAIError,
)


def test_renai_error():
    err = RenAIError("base error")
    assert str(err) == "base error"
    assert isinstance(err, Exception)


def test_extraction_error():
    err = ExtractionError("extraction failed")
    assert str(err) == "extraction failed"
    assert isinstance(err, RenAIError)


def test_cache_error():
    err = CacheError("cache miss")
    assert str(err) == "cache miss"
    assert isinstance(err, RenAIError)


def test_configuration_error():
    err = ConfigurationError("invalid config")
    assert str(err) == "invalid config"
    assert isinstance(err, RenAIError)


def test_file_operation_error():
    err = FileOperationError("file moved")
    assert str(err) == "file moved"
    assert isinstance(err, RenAIError)


def test_llm_error():
    err = LLMError("llm timeout")
    assert str(err) == "llm timeout"
    assert isinstance(err, RenAIError)


def test_exception_structured_context():
    from pathlib import Path

    from renai.exceptions import OCRError

    cause = ValueError("root cause")
    err = ExtractionError(
        "extract failed",
        file_path=Path("/tmp/doc.pdf"),
        provider="openai",
        cause=cause,
        missing_tool="mutool",
    )
    assert err.file_path == Path("/tmp/doc.pdf")
    assert err.provider == "openai"
    assert err.cause is cause
    assert err.missing_tool == "mutool"

    cfg_err = ConfigurationError("bad key", provider="openrouter", missing_api_key=True)
    assert cfg_err.missing_api_key is True
    assert cfg_err.provider == "openrouter"

    ocr_err = OCRError("no binary", missing_tool="tesseract")
    assert ocr_err.missing_tool == "tesseract"
