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


def test_extraction_error():
    err = ExtractionError("extraction failed")
    assert str(err) == "extraction failed"


def test_cache_error():
    err = CacheError("cache miss")
    assert str(err) == "cache miss"


def test_configuration_error():
    err = ConfigurationError("invalid config")
    assert str(err) == "invalid config"


def test_file_operation_error():
    err = FileOperationError("file moved")
    assert str(err) == "file moved"


def test_llm_error():
    err = LLMError("llm timeout")
    assert str(err) == "llm timeout"
