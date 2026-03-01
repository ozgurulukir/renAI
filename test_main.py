import argparse
import builtins
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

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

def test_process_special_modes_no_modes():
    """Test process_special_modes when neither mode is active."""
    args = argparse.Namespace(benchmark=False, eval_model=False)
    processor = MagicMock()
    file_paths = [Path("test1.pdf"), Path("test2.pdf")]

    # Should not raise anything and not call processor methods
    main.process_special_modes(args, processor, file_paths)
    processor.benchmark_extractors.assert_not_called()
    processor.evaluate_models.assert_not_called()

def test_process_special_modes_benchmark():
    """Test process_special_modes when benchmark mode is active."""
    args = argparse.Namespace(benchmark=True, eval_model=False)
    processor = MagicMock()
    file_paths = [Path("test1.pdf"), Path("test2.pdf"), Path("test3.pdf"), Path("test4.pdf")]

    with pytest.raises(SystemExit) as excinfo:
        main.process_special_modes(args, processor, file_paths)

    assert excinfo.value.code == 0
    assert processor.benchmark_extractors.call_count == 3

    # Check that it was called with the first 3 files
    calls = processor.benchmark_extractors.call_args_list
    assert calls[0][0][0] == file_paths[0]
    assert calls[1][0][0] == file_paths[1]
    assert calls[2][0][0] == file_paths[2]

@patch("main.logging.error")
def test_process_special_modes_eval_model_no_models(mock_logging_error):
    """Test process_special_modes when eval_model mode is active but no models exist."""
    args = argparse.Namespace(benchmark=False, eval_model=True, provider="my_provider")
    processor = MagicMock()
    file_paths = [Path("test1.pdf"), Path("test2.pdf")]

    with patch.dict("config.EVAL_MODELS", clear=True):
        with pytest.raises(SystemExit) as excinfo:
            main.process_special_modes(args, processor, file_paths)

        assert excinfo.value.code == 1
        mock_logging_error.assert_called_with("No evaluation models defined for provider: my_provider")
        processor.evaluate_models.assert_not_called()

def test_process_special_modes_eval_model_success():
    """Test process_special_modes when eval_model mode is active and models exist."""
    args = argparse.Namespace(benchmark=False, eval_model=True, provider="my_provider")
    processor = MagicMock()
    file_paths = [Path("test1.pdf"), Path("test2.pdf"), Path("test3.pdf")]
    eval_models = ["model1", "model2"]

    with patch.dict("config.EVAL_MODELS", {"my_provider": eval_models}, clear=True):
        with pytest.raises(SystemExit) as excinfo:
            main.process_special_modes(args, processor, file_paths)

        assert excinfo.value.code == 0
        assert processor.evaluate_models.call_count == 2

        # Check that it was called with the first 2 files
        calls = processor.evaluate_models.call_args_list
        assert calls[0][0] == (file_paths[0], "my_provider", eval_models)
        assert calls[1][0] == (file_paths[1], "my_provider", eval_models)
