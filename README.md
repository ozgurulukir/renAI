# renAI: AI-Powered Document and Image Organizer

`renAI` is a Python tool that automatically renames and organizes your digital files using Large Language Models (LLMs) and Optical Character Recognition (OCR). It extracts meaningful metadata from your files using either text extraction or multimodal AI vision, creating consistent, searchable filenames.

Last updated: 2026-04-09

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
Title -- (Language).ext
```

*Example: `Deep Learning Research Paper -- (en).pdf`*

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
| **Smart Caching** | MD5-based caching reduces API costs and processing time |
| **Custom Categories** | Define your own categories per language in `categories.toml` |

### Operating Modes

- **`rename`**: Renames files based on extracted metadata
- **`organize`**: Moves renamed files into category folders
- **`benchmark`**: Compares PDF extraction library performance
- **`eval-model`**: Tests different LLM models for accuracy
- **`--fallback-mode ocr`**: Tries text extraction first. If it fails, falls back to Tesseract OCR (default behavior).
- **`--fallback-mode vision`**: Tries text extraction first (no OCR). If it fails, falls back to vision-capable LLM for PDFs.
- **`--fallback-mode move`**: Moves files where text extraction fails to `Needs_Scan/` folder.
- **`--rename-images`**: Renames image files using vision-capable LLM. Also processes non-image files with text extraction + selected fallback.

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
```

### API Keys

Set your API keys in `.env` file:

```bash
# Windows: %APPDATA%\renAI\renAI\.env
# Linux/Mac: ~/.config/renAI/.env
# Or copy from project root:
cp .env.example .env
```

Then edit `.env` with your API keys:

- `DEEPINFRA_API_KEY`
- `OPENROUTER_API_KEY`
- `OPENAI_API_KEY`

Get keys from: [DeepInfra](https://deepinfra.com) | [OpenRouter](https://openrouter.ai) | [OpenAI](https://platform.openai.com/api-keys)

### Configuration

Configuration options in `.env`:

- `RENAI_PROVIDER` - LLM provider (deepinfra, openrouter, openai, custom)
- `RENAI_WORKERS` - Number of concurrent workers (default: 4)
- `RENAI_TEXT_LENGTH` - Max text length for LLM (default: 10000)
- `RENAI_EXTRACTOR` - PDF extractor (pymupdf, pdfplumber, pypdf, pypdfium2)

### Custom Categories

Custom categories are now managed via `categories.toml`. See [Configuration](#-configuration) for details.

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

## Controlling Text Length for Local Models

When running local models on consumer hardware, you can reduce the text length sent to the LLM to decrease memory usage and improve performance:

```bash
# Reduce text length to 5000 characters (default: 10000)
renai process "C:/Path/To/Books" --mode rename --provider custom --text-length 5000

# Increase for better context if your model can handle it
renai process "C:/Path/To/Books" --mode rename --provider custom --text-length 20000
```

| Option | Description | Default |
| :--- | :--- | :--- |
| `--text-length` | Target text length sent to LLM | 10000 |

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
```

#### LLM Model Evaluation

Test multiple models from a provider to find the best balance between cost and accuracy for your specific document collection.

```bash
renai process "C:/Path/To/Books" --eval-model --provider deepinfra
```

**How it works:**

- Models are loaded from provider-specific configuration files: `eval_models_deepinfra.txt`, `eval_models_openai.txt`, or `eval_models_openrouter.txt`.
- Each file should contain a list of model names, one per line.
- The tool processes the first 2 files in your directory using **all** listed models.
- Results and confidence scores are logged to `evaluation.log` for detailed comparison.

### Cache Management

```bash
# Clear all cache
renai process --clear-cache

# Clear cache for specific directory
renai process "C:/Path/To/Books" --clear-cache
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

| Feature | Vision Mode (`--vision`) | OCR Mode (default) |
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
| **Text Length** | Up to 8000 characters sent to LLM per file |
| **API Costs** | Each file processed sends text to an LLM API |
| **Rate Limits** | Providers may enforce rate limits |
| **OCR Quality** | Scanned documents require good quality scans |
| **Language Support** | Optimized for Turkish and English |
| **Vision Mode** | PDF only (images supported with `--rename-images`) |

### OCR Handling Options

- **`--fallback-mode move`**: Moves files with insufficient text to `Needs_Scan` folder (skips fallback processing)
- **`--move-scan-failures`**: Moves files that fail both standard and fallback extraction to `Scan_Failures` folder

---

## ✅ Best Practices

1. **Regular Backups**: Always keep a backup of your original files
2. **Start Small**: Test with 5-10 files first to verify quality
3. **Monitor Costs**: Set up usage alerts with your LLM provider
4. **Use Caching**: Avoid `--update-metadata` unless necessary
5. **Choose OCR Wisely**: Use `--ocr-enhanced --use-mutool` only for scanned documents
6. **Vision for PDFs**: Use `--fallback-mode vision` for text-based PDFs (faster, more accurate)
7. **Compare Providers**: OpenRouter often offers competitive pricing
8. **Clear Cache**: Use `--clear-cache` when switching models or settings
9. **Protect Privacy**: For sensitive or private documents, use a local LLM (Ollama, LM Studio) with the `custom` provider. This keeps your document content on your machine rather than sending it to cloud services.

### Caching Strategy

renAI uses an intelligent caching system that includes provider-model awareness:

| Cache Type | Key Components | Invalidation |
| :--- | :--- | :--- |
| **Text Cache** | File hash + OCR settings | When file content or OCR mode changes |
| **Metadata Cache** | File hash + provider + model | When file content, provider, or model changes |

**How it works:**

- Each file is identified by its MD5 hash
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

**Clear Cache:**

```bash
# Clear all cache (metadata and text)
renai cache

# Or clear with the process command
renai process --clear-cache
```

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

- **Cache Data:** Managed in the OS cache directory.
- **Log and Debug Data:** Managed in the OS user data directory.
- **Configuration (like `categories.txt`):** Managed in the OS config directory.

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
| `--update-metadata` | Force refresh metadata cache |
| `--fallback-mode` | Strategy when extraction fails (`none`, `ocr`, `vision`, `move`) |
| `--move-scan-failures` | Move failed files to `Scan_Failures` folder |
| `--clear-cache` | Clear cache and exit |

> **Important:** When using `--fallback-mode vision` or `--rename-images`, you must use a vision-capable model. renAI does not validate vision capability - please ensure your model supports multimodal/vision input.

---

## 🏗️ Architecture

```text
renAI/
├── src/
│   └── renai/
│       ├── cli.py               # CLI entry point (Typer)
│       ├── settings.py          # Configuration & Pydantic settings
│       ├── utils.py             # Helper functions
│       ├── logging_config.py    # Structured logging
│       ├── engines/             # Vision & OCR engines
│       ├── extractors/          # Document parsers
│       ├── processors/          # Core processing logic
│       │   ├── book.py
│       │   ├── orchestrator.py  # Pipeline orchestration
│       │   ├── text_extractor.py
│       │   └── visual.py
│       ├── file_ops/            # File manipulation logic
│       └── filename/            # Naming conventions
├── tests/                       # Unit tests
├── scripts/                     # Benchmark & Utility scripts
├── pyproject.toml               # Package metadata & dependencies
└── requirements.txt             # Dependency list
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

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
uv run pytest tests/ -v

# Run vision-specific tests
uv run pytest tests/test_vision_*.py -v

# Run with coverage
uv run pytest tests/test_file_processor.py --cov=src/renai/processors/book
```

---

renAI - Organize your digital library with AI
