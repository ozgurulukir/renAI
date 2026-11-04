from renai.messages import errors
from renai.messages.cli import Warn


def test_warn_desc_vision():
    """Test Warn.desc_vision static method formatting across model inputs."""
    # Arrange & Act
    msg_gpt = Warn.desc_vision("gpt-4o")
    msg_empty = Warn.desc_vision("")
    msg_special = Warn.desc_vision("org/model-v1.0:latest")

    # Assert
    assert "[bold yellow]'gpt-4o'[/bold yellow]" in msg_gpt
    assert "might not support vision inputs on models.dev." in msg_gpt
    assert "Running in vision fallback mode or renaming images may fail." in msg_gpt

    assert "[bold yellow]''[/bold yellow]" in msg_empty
    assert "[bold yellow]'org/model-v1.0:latest'[/bold yellow]" in msg_special


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
