import argparse
import logging
import builtins
from unittest.mock import patch, MagicMock
import pytest
import main

def test_check_requirements_success():
    """Test check_requirements when all libraries are present."""
    # Mock __import__ to always succeed
    with patch("builtins.__import__") as mock_import:
        mock_import.return_value = MagicMock()
        assert main.check_requirements() is True

def test_check_requirements_missing_one(caplog):
    """Test check_requirements when one library is missing."""
    def mocked_import(name, *args, **kwargs):
        if name == "pymupdf":
            raise ImportError(f"No module named '{name}'")
        return MagicMock()

    with patch("builtins.__import__", side_effect=mocked_import):
        # We don't want to use with caplog.at_level(logging.ERROR) if we want to catch logs
        # from the function which might have already set up logging or uses global logging.
        # But main.py doesn't call basicConfig until main(), so check_requirements uses default.
        assert main.check_requirements() is False

    assert "MISSING PYTHON LIBRARIES" in caplog.text
    assert "pymupdf is missing. Run: pip install pymupdf" in caplog.text

def test_check_requirements_missing_multiple(caplog):
    """Test check_requirements when multiple libraries are missing."""
    def mocked_import(name, *args, **kwargs):
        if name in ["pymupdf", "openai", "numpy"]:
            raise ImportError(f"No module named '{name}'")
        return MagicMock()

    with patch("builtins.__import__", side_effect=mocked_import):
        assert main.check_requirements() is False

    assert "pymupdf is missing. Run: pip install pymupdf" in caplog.text
    assert "openai is missing. Run: pip install openai" in caplog.text
    assert "numpy is missing. Run: pip install numpy" in caplog.text

@patch("main.OCREngine")
@patch("main.LLMEngine")
@patch("main.BookProcessor")
def test_setup_engines_success(mock_processor, mock_llm_engine, mock_ocr_engine):
    """Test setup_engines successful initialization with explicit model."""
    args = argparse.Namespace(
        ocr_enhanced=True,
        debug_ocr=False,
        use_mutool=True,
        mutool_path="/path/to/mutool",
        provider="test_provider",
        model="test-model",
        extractor="test_extractor",
        text_length=1000,
    )

    mock_providers = {
        "test_provider": {
            "base_url": "https://api.test.com",
            "api_key": "test_api_key",
            "default_model": "default-test-model",
        }
    }

    with patch.dict(main.PROVIDERS, mock_providers, clear=True):
        ocr, llm, processor = main.setup_engines(args)

    mock_ocr_engine.assert_called_once_with(
        enhanced=True,
        debug=False,
        use_mutool=True,
        mutool_path="/path/to/mutool",
    )
    mock_llm_engine.assert_called_once_with(
        provider="test_provider",
        api_key="test_api_key",
        base_url="https://api.test.com",
        model_name="test-model",
    )
    mock_processor.assert_called_once_with(
        mock_ocr_engine.return_value,
        mock_llm_engine.return_value,
        extractor_name="test_extractor",
        provider="test_provider",
        model="test-model",
        target_length=1000,
    )
    assert ocr == mock_ocr_engine.return_value
    assert llm == mock_llm_engine.return_value
    assert processor == mock_processor.return_value

@patch("main.OCREngine")
@patch("main.LLMEngine")
@patch("main.BookProcessor")
def test_setup_engines_default_model(mock_processor, mock_llm_engine, mock_ocr_engine):
    """Test setup_engines uses default model if model is not provided in args."""
    args = argparse.Namespace(
        ocr_enhanced=False,
        debug_ocr=True,
        use_mutool=False,
        mutool_path=None,
        provider="test_provider",
        model=None,
        extractor="default_extractor",
        text_length=500,
    )

    mock_providers = {
        "test_provider": {
            "base_url": "https://api.test.com",
            "api_key": "test_api_key",
            "default_model": "default-test-model",
        }
    }

    with patch.dict(main.PROVIDERS, mock_providers, clear=True):
        ocr, llm, processor = main.setup_engines(args)

    mock_llm_engine.assert_called_once_with(
        provider="test_provider",
        api_key="test_api_key",
        base_url="https://api.test.com",
        model_name="default-test-model",
    )

@patch("sys.exit")
def test_setup_engines_invalid_provider(mock_exit, caplog):
    """Test setup_engines handles invalid provider correctly."""
    # When sys.exit is called, we should raise an exception to stop execution
    mock_exit.side_effect = SystemExit(1)

    args = argparse.Namespace(
        ocr_enhanced=False,
        debug_ocr=False,
        use_mutool=False,
        mutool_path=None,
        provider="invalid_provider",
        model=None,
    )

    with patch.dict(main.PROVIDERS, {}, clear=True):
        with pytest.raises(SystemExit) as exc_info:
            main.setup_engines(args)

    mock_exit.assert_called_once_with(1)
    assert exc_info.value.code == 1
    assert "Invalid provider: invalid_provider" in caplog.text

@patch("sys.exit")
def test_setup_engines_missing_api_key(mock_exit, caplog):
    """Test setup_engines handles missing API key correctly."""
    # When sys.exit is called, we should raise an exception to stop execution
    mock_exit.side_effect = SystemExit(1)

    args = argparse.Namespace(
        ocr_enhanced=False,
        debug_ocr=False,
        use_mutool=False,
        mutool_path=None,
        provider="test_provider",
        model=None,
    )

    mock_providers = {
        "test_provider": {
            "base_url": "https://api.test.com",
            "api_key": None,
            "default_model": "default-test-model",
        }
    }

    with patch.dict(main.PROVIDERS, mock_providers, clear=True):
        with pytest.raises(SystemExit) as exc_info:
            main.setup_engines(args)

    mock_exit.assert_called_once_with(1)
    assert exc_info.value.code == 1
    assert "API Key for test_provider not found in environment variables." in caplog.text
