import sys
from pathlib import Path

sys.path.insert(0, "src")

import pytest
import subprocess
from unittest.mock import MagicMock, patch
from renai.engines.ocr import OCREngine

class TestOCREngineSecurity:

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_validate_mutool_valid(self, mock_run, mock_which):
        """Test validation with a legitimate-looking mutool."""
        mock_which.return_value = "/usr/bin/mutool"
        mock_run.return_value = MagicMock(
            stdout="mutool version 1.23.0\n",
            stderr="",
            returncode=0
        )

        engine = OCREngine(use_mutool=False) # Don't trigger validation in init yet
        result = engine._validate_mutool("mutool")

        assert result == "/usr/bin/mutool"
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["/usr/bin/mutool", "-v"]

    @patch("shutil.which")
    def test_validate_mutool_not_found(self, mock_which):
        """Test validation when executable is not found."""
        mock_which.return_value = None

        engine = OCREngine(use_mutool=False)
        result = engine._validate_mutool("nonexistent")

        assert result is None

    @patch("shutil.which")
    def test_validate_mutool_rejected_name(self, mock_which):
        """Test validation when executable name is not allowed (e.g., /usr/bin/ls)."""
        mock_which.return_value = "/usr/bin/ls"

        engine = OCREngine(use_mutool=False)
        result = engine._validate_mutool("/usr/bin/ls")

        assert result is None

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_validate_mutool_wrong_output(self, mock_run, mock_which):
        """Test validation when executable is named mutool but gives wrong output."""
        mock_which.return_value = "/tmp/mutool" # Name starts with mutool
        mock_run.return_value = MagicMock(
            stdout="I am a malicious script\n",
            stderr="",
            returncode=0
        )

        engine = OCREngine(use_mutool=False)
        result = engine._validate_mutool("/tmp/mutool")

        assert result is None

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_validate_mutool_exception(self, mock_run, mock_which):
        """Test validation when subprocess raises an exception."""
        mock_which.return_value = "/usr/bin/mutool"
        mock_run.side_effect = subprocess.SubprocessError("Failed")

        engine = OCREngine(use_mutool=False)
        result = engine._validate_mutool("mutool")

        assert result is None

    @patch("renai.engines.ocr.OCREngine._validate_mutool")
    def test_init_validation_success(self, mock_validate):
        """Test OCREngine init with successful mutool validation."""
        mock_validate.return_value = "/resolved/mutool"

        engine = OCREngine(use_mutool=True, mutool_path="mutool")

        assert engine.use_mutool is True
        assert engine.mutool_path == "/resolved/mutool"
        mock_validate.assert_called_with("mutool")

    @patch("renai.engines.ocr.OCREngine._validate_mutool")
    def test_init_validation_failure(self, mock_validate):
        """Test OCREngine init with failed mutool validation."""
        mock_validate.return_value = None

        engine = OCREngine(use_mutool=True, mutool_path="malicious_path")

        assert engine.use_mutool is False
        mock_validate.assert_called_with("malicious_path")
