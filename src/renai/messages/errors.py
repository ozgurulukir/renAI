"""
Exception analysis messages for the empathetic error handler.
"""

DEFAULT_WHY = "An unexpected system error occurred."
DEFAULT_HOW = "Please review the error details or open an issue on GitHub."
API_KEY_WHY = "API Key not found or invalid."
API_KEY_HOW = "Check the RENAI_PROVIDER_API_KEY environment variable or update your .env file."
TOOL_MISSING_WHY = "A required external tool (OCR or MuTool) was not found."
TOOL_MISSING_HOW = "Ensure Tesseract or MuTool is installed and available in your PATH."
FILE_NOT_FOUND_HOW = "Verify the accuracy of the file path."
PERMISSION_WHY = "Missing required file system permissions."
PERMISSION_HOW = "Make sure you have sufficient read/write privileges for the file/directory."
CONFIG_WHY = "Error in renAI configuration files or settings."
CONFIG_HOW = "Check the formatting and values inside the settings.toml file."
LLM_WHY = "The AI (LLM) service provider returned an error."
LLM_HOW = "Check the API provider status (OpenAI, OpenRouter, etc.) and your internet connection."
OCR_WHY = "OCR processing on the image or PDF failed."
OCR_HOW = "Verify your Tesseract installation and ensure the input file is not corrupted."
EXTRACTION_WHY = "Failed to extract text or visual data from the file."
EXTRACTION_HOW = "Ensure the file format is supported and the file is not corrupted."
CACHE_WHY = "Failed to read from or write to the cache database."
CACHE_HOW = "Check cache directory permissions or run the 'renai clear-cache' command."
FILE_OP_WHY = "Failed to move, copy, or rename the file."
FILE_OP_HOW = (
    "Ensure the destination directory is writable and no locked file with the same name exists."
)
REN_AI_WHY = "An internal renAI-specific error occurred."

PANEL_TITLE_SYSTEM_ERROR = "[bold red]renAI System Error[/bold red]"

ERROR_LABEL = "[bold red]ERROR:[/bold red] {}"
WHY_LABEL = "[bold yellow]WHY:[/bold yellow] {}"
HOW_LABEL = "[bold green]HOW TO FIX:[/bold green] {}"
