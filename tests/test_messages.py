from renai.messages import errors


def test_error_messages():
    """Test Error message constants structure and formatting."""
    assert "unexpected system error" in errors.DEFAULT_WHY
    assert "API Key not found" in errors.API_KEY_WHY
    assert "A required external tool" in errors.TOOL_MISSING_WHY

    formatted = errors.FILE_NOT_FOUND_WHY.format("test.pdf")
    assert "test.pdf" in formatted

    assert "Missing required file system" in errors.PERMISSION_WHY
    assert "configuration files" in errors.CONFIG_WHY
    assert "AI (LLM) service provider" in errors.LLM_WHY
    assert "OCR processing" in errors.OCR_WHY
    assert "extract text or visual data" in errors.EXTRACTION_WHY
    assert "Failed to read" in errors.CACHE_WHY
    assert "Failed to move" in errors.FILE_OP_WHY

    formatted = errors.REN_AI_WHY + " " + errors.REN_AI_HOW.format("detail")
    assert "detail" in formatted

    assert "ERROR" in errors.ERROR_LABEL
    assert "WHY" in errors.WHY_LABEL
    assert "HOW" in errors.HOW_LABEL
