from pathlib import Path
from unittest.mock import AsyncMock, patch

from renai.models import ProcessingResult
from renai.processors.orchestrator import run_evaluation_pipeline


def test_run_evaluation_pipeline_no_models(tmp_path):
    with (
        patch("renai.processors.orchestrator.load_eval_models", return_value=[]),
        patch("renai.processors.orchestrator.console.print") as mock_print,
    ):
        import asyncio

        asyncio.run(run_evaluation_pipeline(tmp_path, "openai"))
        mock_print.assert_any_call("[red]No evaluation models found in eval_models.toml.[/red]")


def test_run_evaluation_pipeline_success(tmp_path):
    mock_results = [
        ProcessingResult(
            success=True,
            file_path=Path("book1.pdf"),
            metadata={"title": "ML Intro", "author": "Andrew Ng", "category": "AI", "year": 2018},
        )
    ]

    with (
        patch("renai.processors.orchestrator.load_eval_models", return_value=["gpt-4"]),
        patch(
            "renai.processors.orchestrator.run_pipeline", new_callable=AsyncMock
        ) as mock_run_pipeline,
        patch("renai.processors.orchestrator.console.print") as mock_print,
    ):
        mock_run_pipeline.return_value = mock_results
        import asyncio

        asyncio.run(run_evaluation_pipeline(tmp_path, "openai"))

        # Verify that run_pipeline was called
        assert mock_run_pipeline.call_count == 1
        call_kwargs = mock_run_pipeline.call_args[1]
        assert call_kwargs["root_dir"] == tmp_path
        # Verify provider and model arguments
        assert mock_run_pipeline.call_args[1]["provider"] == "openai"
        assert mock_run_pipeline.call_args[1]["model"] == "gpt-4"

        # Verify that the comparison summary printed the success message
        any_success = any(
            "Evaluation Comparison Summary" in str(args[0]) for args, _ in mock_print.call_args_list
        )
        assert any_success
