# renAI: AI-Powered Document and Image Organizer

`renAI` is a Python tool that automatically renames and organizes your digital files using Large Language Models (LLMs) and Optical Character Recognition (OCR). It extracts meaningful metadata from your files using either text extraction or multimodal AI vision, creating consistent, searchable filenames.

Last updated: 2026-06-06

---

## 📖 Introduction

Managing large collections of digital files can be overwhelming when they arrive with cryptic or inconsistent names. renAI solves this by intelligently analyzing your files and generating descriptive, organized filenames.

**Two Powerful Approaches:**

- **Text Extraction Mode**: Reads document content (PDF, EPUB, DOCX, etc.) to identify titles, authors, and publication years using traditional text-based LLMs
- **Vision Mode**: Uses vision-capable LLMs to "see" and understand documents and images. If text extraction fails, falls back to vision model instead of OCR.

Whether you're dealing with scanned PDFs, digital documents, or photos, renAI adapts to give you the best results.

The tool supports both cloud-based LLMs (DeepInfra, OpenRouter, OpenAI) and local inference (Ollama, LM Studio), giving you flexibility between convenience and privacy.

### Naming Convention

**Documents (Text Extraction Mode):**

```text
Title -- Subtitle -- Edition -- Author -- (Year).ext
```

*Example: `Introduction to Machine Learning -- 3rd Edition -- Andrew Ng -- (2018).pdf`*

**Documents (Vision Mode):**

```text
Title -- Author -- (Year) -- (Language).ext
```

*Example: `Deep Learning Research -- Ian Goodfellow -- (2016) -- (en).pdf`*

**Images (Vision Mode):**

```text
Description -- [Date] -- (Language).ext
```

*Example: `Sunset -- [2024-07-15] -- (en).jpg`*

---

## ✨ Features

| Feature | Description |
| :--- | :--- |
| **Multi-Format Support** | Processes PDF, EPUB, MOBI, DOCX, PPTX, XLSX, TXT, MD, and image files (JPG, PNG, HEIC, etc.) |
| **Multiple PDF Extractors** | Choose from PyMuPDF, pdfplumber, pypdf, or pypdfium2 |
| **AI-Driven Metadata** | Uses LLMs to extract titles, authors, years, and categories |
| **Vision Mode** | Tries text extraction first; falls back to vision-capable LLM for PDFs if text extraction fails (instead of OCR) |
| **Image Renaming** | Rename image files using AI vision analysis |
| **Advanced OCR** | Tesseract integration with Enhanced Mode for low-quality scans |
| **mutool Support** | Extracts high-quality images from fixed-layout PDFs |
| **Smart Caching** | SHA256-based caching system (separate layers for both text and metadata). |
| **AsyncIO-First** | Fully asynchronous architecture using `asyncio` for non-blocking I/O and high concurrency |
| **Async RPM Limiting** | Integrated `AsyncRateLimiter` ensures provider compliance across concurrent tasks |
| **Ruff Powered** | ultra-fast linting and formatting for consistent code style |
| **SSoT Architecture** | Single Source of Truth for metadata using strictly-typed Pydantic models |
| **Modular Prompts** | Centralized, reusable prompt components with dynamic JSON schema injection |
| **Unified Refinement** | Standardized second-pass 'Senior Editor' logic for both text and vision modes |
| **Token Optimization** | Context-free refinement pass reduces secondary API costs by ~90% |
| **Custom Categories** | Define your own categories per language in `categories.toml` |

### Operating Modes

- **`rename`**: Renames files based on extracted metadata
- **`organize`**: Moves renamed files into category folders
- **`evaluate`**: Tests multiple providers and models for accuracy
- **`--fallback-mode none`**: No fallback; returns empty string on extraction failure.
- **`--fallback-mode ocr`**: Tries text extraction first. If it fails, falls back to Tesseract OCR.
- **`--fallback-mode vision`**: Tries text extraction first (no OCR). If it fails, falls back to vision-capable LLM for PDFs.
- **`--fallback-mode move`**: Moves files where text extraction fails to `Needs_Scan/` folder.
- **`--rename-images`**: Renames image files using vision-capable LLM. Also processes non-image files with text extraction + selected fallback.
- **`--benchmark`**: Compares PDF extraction library performance on the first few files.

---

## 🛠️ Installation

### Prerequisites

1. **Python 3.12+**
2. **Tesseract OCR** - [Download for Windows](https://github.com/UB-Mannheim/tesseract/wiki/Tesseract-at-UB-Mannheim) or install via package manager:

   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr

   # macOS
   brew install tesseract
   ```

3. **mutool (Optional)** - Part of [MuPDF](https://mupdf.com/releases/index.html)

### Setup

```bash
# Clone the repository
git clone https://github.com/ozgurulukir/renAI.git
cd renAI

# Install using uv (creates 'renai' command)
uv sync --all-extras

# Initialize renAI (Interactive Wizard)
# This will guide you through provider setup and create necessary config files
renai init
```

### Setup & API Keys

renAI uses an interactive wizard for first-time setup. Run the following command:

```bash
renai init
```

This will create your configuration files in the appropriate OS directory:

- **Windows**: `%LOCALAPPDATA%\renAI`
- **Linux**: `~/.config/renAI`
- **macOS**: `~/Library/Application Support/renAI`

The wizard helps you configure:

- **LLM Provider**: DeepInfra (default), OpenRouter, OpenAI, or Custom
- **API Keys**: Stored securely in `secrets.toml`
- **General Settings**: Concurrent workers, fallback strategies, etc.

### Configuration Hierarchy

renAI uses a robust, layered configuration system:

1. **User Overrides**: `settings.toml` in your user config directory.
2. **Physical Defaults**: `settings.default.toml` (auto-generated reference).
3. **Internal Defaults**: Hardcoded baseline settings for zero-setup operation.

#### Essential Files

- `settings.toml`: Customize application behavior and override provider defaults.
- `secrets.toml`: Sensitive API keys. **Never share this file!**
- `providers.toml`: Register custom LLM providers or override endpoints.
- `categories.toml`: Define your own categories for organization (`--mode organize`).
- `eval_models.toml`: List models to test with the `evaluate` command.

Get keys from: [DeepInfra](https://deepinfra.com) | [OpenRouter](https://openrouter.ai) | [OpenAI](https://platform.openai.com/api-keys)

### Environment Variables

You can also configure renAI using environment variables (prefixed with `RENAI_`):

- `RENAI_PROVIDER`: e.g., `openai`
- `RENAI_MODEL`: e.g., `gpt-4o`
- `RENAI_WORKERS`: e.g., `8`

### Zero-Code Defaults

renAI follows a "Zero-Code Defaults" philosophy. You don't *need* to edit configuration files to start. Baseline factory defaults are included within the package, allowing it to "just work" after simple initialization.

Override options are available via:

1. Environment variables (prefixed with `RENAI_`)
2. `settings.toml` in your user config directory
3. `settings.default.toml` (Global baseline)
4. CLI flags (highest priority)

---

## 📖 Usage Examples

### Basic Renaming

```bash
# Rename all files in directory
uv run renai process "C:/Path/To/Books" --mode rename
```

### Organizing into Folders

```bash
# Rename and organize into category folders
renai process "C:/Path/To/Books" --mode organize
```

### Model Evaluation

Evaluate multiple LLM providers and models against your sample files to find the best performer. Results are saved to `evaluation.log` in your app data directory.

```bash
# Evaluate ALL providers and models defined in eval_models.toml
renai evaluate "C:/Path/To/Samples"

# Evaluate only a specific provider
renai evaluate "C:/Path/To/Samples" --provider deepinfra

# Evaluate with custom worker count
renai evaluate "C:/Path/To/Samples" --workers 8
```

### Vision Mode (PDF without OCR)

Vision mode implements a "smart fallback" strategy: it tries text extraction first (which is faster and cheaper). If extraction fails, it falls back to a vision-capable LLM to process the PDF as an image.

```bash
# Process PDFs with vision model (no OCR) - uses default provider/model
renai process "C:/Path/To/Books" --fallback-mode vision

# Use specific vision model
renai process "C:/Path/To/Books" --fallback-mode vision --provider openrouter --model openai/gpt-4o

# DeepInfra with Llama Vision
renai process "C:/Path/To/Books" --fallback-mode vision --provider deepinfra --model meta-llama/Llama-3.2-90B-Vision-Instruct

# OpenAI GPT-4o
renai process "C:/Path/To/Books" --fallback-mode vision --provider openai --model gpt-4o
```

### Image Renaming

Rename image files using AI vision analysis. Requires a vision-capable model:

```bash
# Rename images with vision model
renai process "C:/Path/To/Photos" --rename-images --provider openai --model gpt-4o

# With EXIF date (if available)
renai process "C:/Path/To/Photos" --fallback-mode vision --rename-images --provider openai --model gpt-4o --exif

# With DeepInfra
renai process "C:/Path/To/Images" --fallback-mode vision --rename-images --provider deepinfra --model Qwen/Qwen2-VL-72B-Instruct
```

### Mixed Folders (PDF + Images)

Process both PDFs and images in the same folder:

| Parameter | PDF/Supported | Image |
| :--- | :--- | :--- |
| `--fallback-mode vision` | Normal → Vision fallback | ⏭️ Skipped |
| `--fallback-mode vision --rename-images` | Normal → Vision fallback | 🔄 Vision rename |
| `--rename-images` | Normal → OCR fallback | 🔄 Vision rename |

```bash
# PDFs with vision, images with vision
renai process "C:/Path/To/Mixed" --fallback-mode vision --rename-images --provider openai --model gpt-4o

# Only rename images (PDFs with OCR)
renai process "C:/Path/To/Mixed" --rename-images --provider openai --model gpt-4o

# Only process PDFs with vision (skip images)
renai process "C:/Path/To/PDFs" --fallback-mode vision --provider openai --model gpt-4o
```

### High-Quality OCR for Scanned Documents

```bash
# Enhanced OCR with image preprocessing
renai process "C:/Path/To/Books" --ocr-enhanced --use-mutool
```

### LLM Providers

#### DeepInfra (default)

uv run renai process "C:/Path/To/Books" --mode rename --provider deepinfra

#### OpenRouter

uv run renai process "C:/Path/To/Books" --mode rename --provider openrouter

#### OpenAI

uv run renai process "C:/Path/To/Books" --mode rename --provider openai

### Local Inference (Ollama / LM Studio)

Use the `custom` provider to connect to local inference servers like [Ollama](https://ollama.com/) or [LM Studio](https://lmstudio.ai/). These tools provide OpenAI-compatible APIs.

> **⚠️ Important: Not all models are capable of structured output, particularly LLMs below 7B parameters. Check the model card README if you are unsure if the model supports structured output.**
>
> renAI requires JSON output from the LLM for reliable metadata extraction. Models with fewer than 7B parameters may struggle with structured output formatting.

```bash
# Set environment variables (Windows PowerShell)
$env:CUSTOM_API_BASE_URL = "http://localhost:11434/v1"  # Ollama default
$env:CUSTOM_MODEL = "llama3"  # Your model name

# Run with custom provider
renai process "C:/Path/To/Books" --mode rename --provider custom

# Or override model directly
renai process "C:/Path/To/Books" --mode rename --provider custom --model mistral
```

## Controlling Extraction Length & Chunking

When running local models on consumer hardware, or to save tokens, you can control how much text is extracted and sent to the LLM:

```bash
# Reduce text length to 5000 characters (default: 10000)
renai process "C:/Path/To/Books" --mode rename --provider custom --text-length 5000

# Increase for better context if your model can handle it
renai process "C:/Path/To/Books" --mode rename --provider custom --text-length 20000

# Regular mode: Stop reading after 5 pages, no skipping
renai process "C:/Path/To/Books" --mode rename --text-extraction regular --extraction-pages 5
```

| Option | Description | Default |
| :--- | :--- | :--- |
| `--text-length` | Target text length sent to LLM | 10000 |
| `--text-extraction` | Extraction chunking logic (`mixed` or `regular`) | `mixed` |
| `--extraction-pages` | Target pages to process from the document | 10 |

### Environment Variables for Custom Provider

| Variable | Description | Default |
| :--- | :--- | :--- |
| `CUSTOM_API_BASE_URL` | Base URL of your local API server | `http://localhost:8000/v1` |
| `CUSTOM_MODEL` | Model name to use | `llama3` |
| `CUSTOM_API_KEY` | API key (usually not needed for local) | `not-required` |

#### Common Local Server URLs

| Tool | URL |
| :--- | :--- |
| Ollama | `http://localhost:11434/v1` |
| LM Studio | `http://localhost:1234/v1` |
| LocalAI | `http://localhost:8080/v1` |
| Text Generation WebUI | `http://localhost:5001/v1` |

### Different Models

```bash
# DeepInfra with Qwen model
renai process "C:/Path/To/Books" --mode rename --model Qwen/Qwen2.5-72B-Instruct

# OpenRouter with Gemini
renai process "C:/Path/To/Books" --mode rename --provider openrouter --model google/gemini-2.0-flash-exp:free
```

### Faster Processing

```bash
# Increase concurrent workers
renai process "C:/Path/To/Books" --mode rename --workers 8
```

### Force Refresh Metadata

```bash
# Bypass cache and re-extract
renai process "C:/Path/To/Books" --mode rename --update-metadata
```

### Benchmarking & Model Evaluation

#### PDF Extractor Benchmark

Compare the performance and accuracy of different PDF extraction libraries on your documents.

```bash
renai process "C:/Path/To/Books" --benchmark
# or
renai benchmark "C:/Path/To/Books"
```

#### Advanced Model Evaluation

The `evaluate` command performs a competitive test:

1. It takes the first few files from your directory.
2. It processes them using **every** model listed in your `eval_models.toml`.
3. It logs the extracted metadata and a confidence score for each attempt.

**Configuration:**
Models are loaded from `eval_models.toml` (User > Default > Internal).

Example `eval_models.toml`:

```toml
[deepinfra]
models = ["meta-llama/Llama-3.2-90B-Vision-Instruct", "meta-llama/Llama-3.1-405B-Instruct"]

[openai]
models = ["gpt-4o", "gpt-4o-mini"]
```

### Cache Management

The cache is automatically managed based on file SHA256 hashes. Metadata cache entries are scoped by provider-model pairs, so switching providers or models automatically uses a different cache entry.

```bash
# Manually clear all cache (text, metadata, and ocr_debug)
renai cache --yes
```

---

## 🤖 Vision Models

renAI supports vision-capable LLMs that can process PDFs and images as visual content. By default, it tries text extraction first to save costs, falling back to vision only if necessary.

### Recommended Vision Models

| Provider | Model | Best For |
| :--- | :--- | :--- |
| **OpenAI** | `gpt-4o` | Best accuracy, reliable |
| **OpenAI** | `gpt-4o-mini` | Fast, cost-effective |
| **OpenRouter** | `openai/gpt-4o` | Flexible pricing |
| **OpenRouter** | `google/gemini-1.5-flash` | Free tier available |
| **DeepInfra** | `meta-llama/Llama-3.2-90B-Vision-Instruct` | Open source, powerful |
| **DeepInfra** | `Qwen/Qwen2-VL-72B-Instruct` | Excellent value |
| **Custom** | `llava`, `qwen-vl` | Local inference |

### Vision Mode vs OCR

| Feature | Vision Fallback (`--fallback-mode vision`) | OCR Fallback (`--fallback-mode ocr`) |
| :--- | :--- | :--- |
| **Speed** | Faster for text PDFs | Slower |
| **Accuracy** | Higher for clear text | Depends on scan quality |
| **Scanned Docs** | May struggle | Optimized for scans |
| **Languages** | Limited to model | Turkish+English optimized |
| **API Cost** | Image tokens | Text tokens |

**When to use Vision Mode:**

- PDFs where text extraction might fail
- When you don't have Tesseract installed
- When OCR is too slow or inaccurate
- When you want better results on formatted documents

**When to use OCR:**

- Scanned documents
- Low-quality images
- Multi-language documents
- When you need Turkish character support

---

## ⚠️ Warnings & Limitations

### Data Loss Prevention

> **⚠️ Always backup your files before running renAI in `rename` or `organize` mode.**

- The tool renames and moves files
- Safety checks are included, but unexpected issues can occur
- Review the naming convention before applying to your entire collection
- Test with a small directory first

### Directory Layout Warning (Organize Mode)

When using `--mode organize`:

- Files from all subdirectories are moved to category folders at the root
- Empty subdirectories may remain after files are moved
- Category folders are created at the root level, not within subdirectories
- Running multiple times will re-process files

### Technical Limitations

| Limitation | Description |
| :--- | :--- |
| **Text Length** | Configurable via `--text-length` (e.g., 10000 chars) |
| **API Costs** | Each file processed sends text to an LLM API |
| **Rate Limits** | Providers may enforce rate limits |
| **OCR Quality** | Scanned documents require good quality scans |
| **Language Support** | Optimized for Turkish and English |
| **Vision Mode** | PDF only (images supported with `--rename-images`) |

### OCR Handling Options

- **`--fallback-mode move`**: Moves files with insufficient text to `Needs_Scan` folder (skips fallback processing)
- **`--move-scan-failures`**: Moves files that fail both standard and fallback extraction to `Scan_Failures` folder
- **`--fallback-mode none`**: No fallback processing; extraction failures are reported but not retried
- **`--benchmark`**: Run a benchmark comparing PDF extractors on the first few files

---

## ✅ Best Practices

1. **Regular Backups**: Always keep a backup of your original files
2. **Start Small**: Test with 5-10 files first to verify quality
3. **Monitor Costs**: Set up usage alerts with your LLM provider
4. **Use Caching**: Avoid `--update-metadata` unless necessary (cache is automatically scoped by provider/model)
5. **Choose OCR Wisely**: Use `--ocr-enhanced --use-mutool` only for scanned documents
6. **Vision for PDFs**: Use `--fallback-mode vision` for text-based PDFs (faster, more accurate)
7. **Compare Providers**: OpenRouter often offers competitive pricing
8. **Protect Privacy**: For sensitive or private documents, use a local LLM (Ollama, LM Studio) with the `custom` provider. This keeps your document content on your machine rather than sending it to cloud services.

### Caching Strategy

renAI uses an intelligent caching system that includes provider-model awareness:

| Cache Type | Key Components | Invalidation |
| :--- | :--- | :--- |
| **Text Cache** | File hash + OCR settings | **File Content**: Cache is invalidated when the SHA256 hash of the file changes. |
| **Metadata Cache** | File hash + provider + model + schema validation | When file content, provider, model, or schema changes |

**How it works:**

- Each file is identified by its SHA256 hash
- Metadata cache keys include a hash of the provider-model pair
- Switching providers or models automatically invalidates old cache
- Use `--update-metadata` to force fresh extraction regardless of cache

**Example:**

```bash
# First run with DeepInfra - caches metadata
renai process "C:/Books" --provider deepinfra

# Same files with OpenRouter - NEW cache entry (old one preserved)
renai process "C:/Books" --provider openrouter

# Same provider/model - uses cached metadata
renai process "C:/Books" --provider deepinfra

# Force refresh - bypasses cache
renai process "C:/Books" --provider deepinfra --update-metadata
```

**Cache Location:** Managed automatically by the OS via `platformdirs` (e.g., `~/.cache/renAI` on Linux, `~/Library/Caches/renAI` on macOS, `%LOCALAPPDATA%\renAI\Cache` on Windows).

---

### Privacy Considerations

| Scenario | Recommended Provider |
| :--- | :--- |
| Public documents | Any cloud provider (DeepInfra, OpenRouter, OpenAI) |
| Private/personal files | `custom` provider with local LLM |
| Sensitive business documents | `custom` provider with local LLM |
| Testing/experimentation | Cloud providers for convenience |

For detailed privacy guidance, data retention policies, and compliance considerations, see [PRIVACY.md](docs/PRIVACY.md).

### Cost Estimation

| Files | Est. Cost (~$0.001/file) |
| :--- | :--- |
| 100 | $0.10 |
| 1,000 | $1.00 |
| 10,000 | $10.00 |

Actual costs vary by provider and model. Vision mode may have different pricing (image tokens vs text tokens).

---

## 🔧 Configuration

### OS-Agnostic File Management

renAI now utilizes modern OS-agnostic path management via the `platformdirs` library.

- **Cache Data**: Managed in `%LOCALAPPDATA%\renAI\Cache` (Windows) or `~/.cache/renAI` (Linux).
- **Log and Debug Data**: Managed in the OS user data directory.
- **Configuration**: Managed via `settings.toml`, `secrets.toml`, and `categories.toml` in the flat OS config directory (no nested folders).

### Defining Custom Categories

Create `categories.toml` in your OS's respective user config directory for `renAI` (e.g., `~/.config/renAI/categories.toml` on Linux, `%APPDATA%\renAI\categories.toml` on Windows):

```toml
[en]
categories = [
    "Article", "Fiction", "Non-Fiction", "Textbook", "Report",
    "Gov", "Presentation", "Manual", "Thesis", "Research",
    "Other", "Dictionary", "Magazine"
]

[tr]
categories = [
    "Makale", "Kurgu", "Kurgu Dışı", "Ders Kitabı", "Rapor",
    "Devlet", "Sunum", "Kılavuz", "Tez", "Araştırma",
    "Diğer", "Sözlük", "Dergi"
]
```

### Command Options

| Option | Description |
| :--- | :--- |
| `--provider` | `deepinfra`, `openrouter`, `openai`, or `custom` |
| `--model` | Model name (for text processing or vision if --fallback-mode vision is used) |
| `--extractor` | `pymupdf`, `pypdf`, `pdfplumber`, `pypdfium2` |
| `--workers` | Concurrent workers (default: 4) |
| `--ocr-enhanced` | Enable enhanced OCR |
| `--use-mutool` | Use mutool for high-quality PDF-to-image conversion during OCR fallback (recommended for scanned PDFs) |
| `--debug-ocr` | Save debug images |
| `--recursive` | Process subdirectories (default: True) |
| `--no-recursive` | Process root only |
| `--exif` | Add EXIF date to image filenames when available |
| `--text-length` | Target text length for LLM (default: 10000) |
| `--text-extraction` | Extraction mode (`mixed` or `regular`) (default: `regular`) |
| `--extraction-pages` | Pages to process per document |
| `--update-metadata` | Force refresh metadata cache |
| `--fallback-mode` | Strategy when extraction fails (`none`, `ocr`, `vision`, `move`) |
| `--move-scan-failures` | Move failed files to `Scan_Failures` folder |
| `--benchmark` | Benchmark PDF extractors on the first few files |
| `cache` | Command to clear cache |
| `init` | Command to start interactive setup wizard |
| `providers` | Command to list configured providers |
| `--version` | Show renAI version |

> **Important:** When using `--fallback-mode vision` or `--rename-images`, you must use a vision-capable model. renAI does not validate vision capability - please ensure your model supports multimodal/vision input.

---

## 🏗️ Architecture

```text
├── src/
│   └── renai/
│       ├── cli.py               # CLI entry point (Typer)
│       ├── config/              # Configuration & settings loading
│       │   ├── settings.py      # Pydantic settings management
│       │   ├── settings.default.toml
│       │   ├── providers.py     # Provider registry
│       │   └── constants.py     # Cache & app directory constants
│       ├── cache/               # [NEW] Cache management with atomic writes
│       │   ├── manager.py       # Text & metadata cache operations
│       │   └── __init__.py
│       ├── prompts/             # Modular .txt prompt templates
│       ├── models/              # SSoT Pydantic Schemas (Book, Image)
│       ├── engines/             # Vision & OCR engines
│       ├── extractors/          # Document parsers
│       ├── processors/          # Core processing logic
│       │   ├── book.py
│       │   ├── orchestrator.py  # Pipeline orchestration
│       │   ├── setup.py         # [NEW] Interactive Init Wizard
│       │   ├── text_extractor.py
│       │   └── visual.py
│       └── ...
├── tests/                       # Async-powered unit tests (pytest-asyncio)
├── pyproject.toml               # Package metadata & dependencies
└── ...
```

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/NewFeature`
3. Commit changes: `git commit -m 'Add NewFeature'`
4. Push to branch: `git push origin feature/NewFeature`
5. Open a Pull Request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 📧 Support

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Email**: Contact maintainers

---

## 🖼️ Advanced Vision Workflows

Vision mode allows renAI to process PDFs and images without text extraction, using multimodal AI models that can "see" document content.

### Direct Image Renaming

Rename image files (photos, screenshots) using AI:

```bash
renai process "C:/Photos" --mode rename --fallback-mode vision --provider deepinfra --model meta-llama/Llama-3.2-90B-Vision-Instruct --rename-images
```

This will analyze each image and rename it based on its visual content.

### Processing Mixed Folders Strategy

Process folders containing both PDFs and images:

```bash
renai process "C:/Documents" --mode rename --fallback-mode vision --provider deepinfra --model meta-llama/Llama-3.2-90B-Vision-Instruct --rename-images
```

This processes:

- PDFs with vision model (no OCR needed)
- Images with vision model

### EXIF Date Support

When processing images, you can include EXIF date information in filenames:

```bash
renai process "C:/Photos" --mode rename --fallback-mode vision --provider deepinfra --model meta-llama/Llama-3.2-90B-Vision-Instruct --rename-images --exif
```

This adds the photo date from EXIF metadata to the filename when available.

### Top-Tier Vision Model Options

| Model | Provider | Notes |
| :--- | :--- | :--- |
| `meta-llama/Llama-3.2-90B-Vision-Instruct` | DeepInfra | Best balance of quality and speed |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | DeepInfra | Faster, good for large batches |
| `openai/gpt-4o` | OpenAI | Highest quality, more expensive |
| `google/gemini-1.5-pro` | OpenRouter | Good alternative |

### Vision Mode Comparison with OCR Engine

| Feature | Vision Mode | OCR Mode |
| :--- | :--- | :--- |
| Speed | Faster (no text extraction) | Slower |
| Accuracy | Depends on model | Good for clear text |
| Cost | API calls per file | API calls per file |
| Best for | Scanned PDFs, images | Text-based PDFs |

---

---

## 🛠️ Development

```bash
# Linting & Formatting
uv run ruff check .
uv run ruff format .

# Type Checking
uv run pyright

# Run Tests
uv run pytest tests/ -v
```

---

renAI - Organize your digital library with AI
