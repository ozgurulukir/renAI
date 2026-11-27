# renAI: AI-Powered Document and Image Organizer

`renAI` is a Python tool that automatically renames and organizes your digital files using Large Language Models (LLMs) and Optical Character Recognition (OCR). It extracts meaningful metadata from your files using either text extraction or multimodal AI vision, creating consistent, searchable filenames.

Last updated: 2026-11-24

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
Title -- Subtitle -- Series #Vol -- 'Edition' -- Author -- (Year).ext
```

_Example: `Introduction to Machine Learning -- 3rd Edition -- Andrew Ng -- (2018).pdf`_ (missing parts are skipped; series/volume renders as `Series #3`, bare volume as `Vol. 3`)_

**Documents (Vision Mode):**

```text
Title -- Author -- (Year) -- (Language).ext
```

_Example: `Deep Learning Research -- Ian Goodfellow -- (2016) -- (en).pdf`_

**Images (Vision Mode):**

```text
Description -- [Date] -- (Language).ext
```

_Example: `Sunset -- [2024-07-15] -- (en).jpg`_

---

## ✨ Features

| Feature                     | Description                                                                                                      |
| :-------------------------- | :--------------------------------------------------------------------------------------------------------------- |
| **Multi-Format Support**    | Processes PDF, EPUB, MOBI, DOCX, PPTX, XLSX, TXT, MD, and image files (JPG, PNG, HEIC, etc.)                     |
| **Multiple PDF Extractors** | Choose from PyMuPDF, pdfplumber, pypdf, or pypdfium2                                                             |
| **AI-Driven Metadata**      | Uses LLMs to extract titles, authors, years, and categories                                                      |
| **Vision Mode**             | Tries text extraction first; falls back to vision-capable LLM for PDFs if text extraction fails (instead of OCR) |
| **Image Renaming**          | Rename image files using AI vision analysis                                                                      |
| **Advanced OCR**            | Tesseract integration with Enhanced Mode for low-quality scans                                                   |
| **mutool Support**          | Extracts high-quality images from fixed-layout PDFs                                                              |
| **Smart Caching**           | SHA256-based caching system (separate layers for both text and metadata).                                        |
| **AsyncIO-First**           | Fully asynchronous architecture using `asyncio` for non-blocking I/O and high concurrency                        |
| **Async RPM Limiting**      | Integrated `AsyncRateLimiter` ensures provider compliance across concurrent tasks                                |
| **Ruff Powered**            | ultra-fast linting and formatting for consistent code style                                                      |
| **SSoT Architecture**       | Single Source of Truth for metadata using strictly-typed Pydantic models                                         |
| **Modular Prompts**         | Centralized, reusable prompt components with dynamic JSON schema injection                                       |
| **Unified Refinement**      | Standardized second-pass 'Senior Editor' logic for both text and vision modes                                    |
| **Token Optimization**      | Context-free refinement pass reduces secondary API costs by ~90%                                                 |
| **Custom Categories**       | Define your own categories per language in `categories.toml`                                                     |
| **Folder Rules**            | Restrict categories and override filename templates per folder via `folder_rules.toml`                           |

### Operating Modes

- **`rename`**: Renames files based on extracted metadata
- **`organize`**: Moves renamed files into category folders
- **`copy`**: Writes the renamed copy next to the original and keeps the source file untouched (non-destructive)
- **`evaluate`**: Tests multiple providers and models for accuracy
- **`--fallback-mode ocr`**: Tries text extraction first. If it fails, falls back to Tesseract OCR.
- **`--fallback-mode vision`**: Tries text extraction first (no OCR). If it fails, falls back to vision-capable LLM for PDFs.
- **`--fallback-mode move`**: Moves files where text extraction fails to `Needs_Scan/` folder.
- **`--rename-images`**: Renames image files using vision-capable LLM. Also processes non-image files with text extraction + selected fallback.
- **`--benchmark`**: Compares PDF extraction library performance on a sample of files (default: first 3, via the `benchmark_sample_count` setting).
- **`--dry-run`**: Simulate processing without renaming or moving files.
- **`undo` / `undo-last`**: Revert file operations from the most recent renaming or organizing run.

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

# Install using uv
uv sync --all-extras

# Initialize renAI (Interactive Wizard)
# This will guide you through provider setup and create necessary config files
uv run renai init
```

#### Global Install (run `renai` from any folder)

`uv run renai` only works from inside the project directory. To run `renai` from anywhere in your terminal, install it as a global tool (editable, so source changes apply immediately):

```bash
# From the project root
# "[pdf-extra]" installs the optional backends (pdfplumber, pypdf, pypdfium2),
# matching `uv sync --all-extras`; omit it to install only the base dependencies.
uv tool install -e ".[pdf-extra]"
```

After this, `renai` is available globally:

```bash
renai process "path/to/files" --mode rename
renai init
```

> To update after pulling new changes, re-run `uv tool install -e ".[pdf-extra]"` from the project root.

### Setup & API Keys

renAI uses an interactive wizard for first-time setup. Run the following command:

```bash
uv run renai init
```

This will create your configuration files in the appropriate OS directory:

- **Windows**: `%LOCALAPPDATA%\renAI`
- **Linux**: `~/.local/share/renAI`
- **macOS**: `~/Library/Application Support/renAI`

The wizard helps you configure:

- **LLM Provider**: DeepInfra (default), OpenRouter, OpenAI, or Custom
- **API Keys**: Stored securely in `secrets.toml` (the wizard automatically validates entered keys against the provider's API during setup, prompting for correction on authentication failure).
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
- `folder_rules.toml`: Per-folder category restrictions and filename template overrides.
- `eval_models.toml`: List models to test with the `evaluate` command.

Get keys from: [DeepInfra](https://deepinfra.com) | [OpenRouter](https://openrouter.ai) | [OpenAI](https://platform.openai.com/api-keys)

### Environment Variables

Every `AppSettings` field can be overridden with an environment variable using the `RENAI_` prefix: pydantic-settings maps `RENAI_<FIELD>` (upper snake case) to the `<field>` defined in `settings.default.toml` — e.g. `RENAI_OCR_LANG` overrides `ocr_lang`, `RENAI_VISION_PROVIDER` overrides `vision_provider`. See [`settings.default.toml`](src/renai/config/settings.default.toml) for the authoritative list of fields and their defaults.

Precedence (highest wins):

1. **CLI flags** (e.g., `--workers`, `--provider`)
2. **Environment variables** (`RENAI_*` and the plain variables below)
3. **User config**: `settings.toml` / `secrets.toml` in your user config directory
4. **Factory defaults**: `settings.default.toml`
5. **Internal hardcoded defaults**

#### General

| Variable         | Description                                                                                                              | Default            |
| :--------------- | :----------------------------------------------------------------------------------------------------------------------- | :----------------- |
| `TESSERACT_PATH` | Full path to the Tesseract executable. Checked before `PATH` lookup and platform-specific defaults.                      | auto-detected      |
| `RENAI_LOG_LEVEL`| Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).                                                         | `INFO`             |

#### Provider API Keys & Endpoints

These plain (unprefixed) variables take highest precedence over `secrets.toml` and `settings.toml` (`environment` > `secrets.toml` > `settings.toml`):

| Variable              | Description                                                        | Default                    |
| :-------------------- | :----------------------------------------------------------------- | :------------------------- |
| `DEEPINFRA_API_KEY`   | DeepInfra API key                                                  | unset (required for use)   |
| `OPENROUTER_API_KEY`  | OpenRouter API key                                                 | unset (required for use)   |
| `OPENAI_API_KEY`      | OpenAI API key                                                     | unset (required for use)   |
| `CUSTOM_API_KEY`      | Custom provider API key. **See the sentinel warning below.**       | `not-required`             |
| `CUSTOM_API_BASE_URL` | Base URL of your OpenAI-compatible custom server                   | `http://localhost:8000/v1` |
| `CUSTOM_MODEL`        | Model name for the custom provider                                 | `llama3`                   |

> **Warning — the `not-required` sentinel:** `CUSTOM_API_KEY` defaults to the sentinel value `not-required` (also baked into `settings.default.toml`). This means that with the `custom` provider, a missing key does **not** fail fast — renAI silently proceeds and sends requests with the literal string `not-required` as the API key. Local servers (Ollama, LM Studio, LocalAI) usually ignore the key, so this works there; but if your custom endpoint requires authentication, you **must** set `CUSTOM_API_KEY` (or `custom_api_key` in `secrets.toml`) to a real key. The sentinel exists purely to allow keyless local usage out of the box.

#### Useful `RENAI_*` Field Overrides

Any field from `settings.default.toml` works; the most commonly used:

| Variable                     | Field            | Description                                              | Default     |
| :--------------------------- | :--------------- | :------------------------------------------------------- | :---------- |
| `RENAI_PROVIDER`             | `provider`       | LLM provider (`deepinfra`, `openrouter`, `openai`, `custom`) | `deepinfra` |
| `RENAI_MODEL`                | `model`          | Default model name                                       | provider default |
| `RENAI_VISION_PROVIDER`      | `vision_provider`| Provider used for vision fallback                        | unset       |
| `RENAI_WORKERS`              | `workers`        | Number of concurrent workers                             | `4`         |
| `RENAI_OCR_LANG`             | `ocr_lang`       | Tesseract language string                                | `tur+eng`   |
| `RENAI_FALLBACK_MODE`        | `fallback_mode`  | Extraction-failure strategy: `ocr`, `vision`, `move`     | `none`      |
| `RENAI_EXTRACTOR`            | `extractor`      | PDF text extractor (`pymupdf`, `pdfplumber`, `pypdf`, `pypdfium2`) | `pymupdf` |
| `RENAI_TEXT_LENGTH`          | `text_length`    | Target text length sent to the LLM                       | `10000`     |

`RENAI_FALLBACK_MODE` accepts `ocr` (OCR the file), `vision` (send pages as images to the vision LLM), or `move` (move the file to the `Scan_Failures` folder). The internal `none` sentinel is accepted for backward compatibility but is remapped to OCR by the document pipeline; the `--fallback-mode none` CLI flag is rejected.

### Zero-Code Defaults

renAI follows a "Zero-Code Defaults" philosophy. You don't _need_ to edit configuration files to start. Baseline factory defaults are included within the package, allowing it to "just work" after simple initialization.

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

| Parameter                                | PDF/Supported            | Image            |
| :--------------------------------------- | :----------------------- | :--------------- |
| `--fallback-mode vision`                 | Normal → Vision fallback | ⏭️ Skipped       |
| `--fallback-mode vision --rename-images` | Normal → Vision fallback | 🔄 Vision rename |
| `--rename-images`                        | Normal (no fallback by default) | 🔄 Vision rename |

```bash
# PDFs with vision, images with vision
renai process "C:/Path/To/Mixed" --fallback-mode vision --rename-images --provider openai --model gpt-4o

# Only rename images (PDFs processed normally, no fallback)
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

## Controlling Extraction Length & Sampling

When running local models on consumer hardware, or to save tokens, you can control how much text is extracted and sent to the LLM:

```bash
# Reduce text length to 5000 characters (default: 10000)
renai process "C:/Path/To/Books" --mode rename --provider custom --text-length 5000

# Increase for better context if your model can handle it
renai process "C:/Path/To/Books" --mode rename --provider custom --text-length 20000

# Regular mode: Read first N pages sequentially, no head-tail split
renai process "C:/Path/To/Books" --mode rename --text-extraction regular --extraction-pages 5
```

| Option               | Description                                   | Default     |
| :------------------- | :-------------------------------------------- | :---------- |
| `--text-length`      | Target text length sent to LLM                | 10000       |
| `--text-extraction`  | Extraction mode (`head-tail` or `regular`)    | `head-tail` |
| `--extraction-pages` | Max pages to process per side of the document | 10          |

### Head-Tail Sampling

The default `head-tail` mode reads pages procedurally from the start (head) and end (tail) of the document. This captures metadata-rich content (title, author, copyright) from the beginning and references/ISBN from the end, while skipping the metadata-poor middle section.

- **`head_ratio`** (default: 0.7): Controls the head/tail character split. 0.7 means 70% of the target text comes from the head, 30% from the tail.
- **`"mixed"`** is silently normalized to `"head-tail"` for backward compatibility.

### Environment Variables for Custom Provider

| Variable              | Description                            | Default                    |
| :-------------------- | :------------------------------------- | :------------------------- |
| `CUSTOM_API_BASE_URL` | Base URL of your local API server      | `http://localhost:8000/v1` |
| `CUSTOM_MODEL`        | Model name to use                      | `llama3`                   |
| `CUSTOM_API_KEY`      | API key (usually not needed for local; see the `not-required` sentinel warning above) | `not-required` |

#### Common Local Server URLs

| Tool                  | URL                         |
| :-------------------- | :-------------------------- |
| Ollama                | `http://localhost:11434/v1` |
| LM Studio             | `http://localhost:1234/v1`  |
| LocalAI               | `http://localhost:8080/v1`  |
| Text Generation WebUI | `http://localhost:5001/v1`  |

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

#### Advanced Model Evaluation

The `evaluate` command performs a competitive test:

1. It takes the sample files from your directory.
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

The cache is automatically managed based on file SHA256 hashes. Metadata rows record which engine produced them; changing that engine triggers re-extraction and overwrites the row. See the [Caching Strategy](docs/CACHING_STRATEGY.md) reference for details.

```bash
# Manually clear all cache (text, metadata, and ocr_debug)
renai cache --yes
```

---

## 🤖 Vision Models

renAI supports vision-capable LLMs that can process PDFs and images as visual content. By default, it tries text extraction first to save costs, falling back to vision only if necessary.

### Recommended Vision Models

| Provider       | Model                                      | Best For                |
| :------------- | :----------------------------------------- | :---------------------- |
| **OpenAI**     | `gpt-4o`                                   | Best accuracy, reliable |
| **OpenAI**     | `gpt-4o-mini`                              | Fast, cost-effective    |
| **OpenRouter** | `openai/gpt-4o`                            | Flexible pricing        |
| **OpenRouter** | `google/gemini-1.5-flash`                  | Free tier available     |
| **DeepInfra**  | `meta-llama/Llama-3.2-90B-Vision-Instruct` | Open source, powerful   |
| **DeepInfra**  | `Qwen/Qwen2-VL-72B-Instruct`               | Excellent value         |
| **Custom**     | `llava`, `qwen-vl`                         | Local inference         |

### Vision Mode vs OCR

| Feature          | Vision Fallback (`--fallback-mode vision`) | OCR Fallback (`--fallback-mode ocr`) |
| :--------------- | :----------------------------------------- | :----------------------------------- |
| **Speed**        | Faster for text PDFs                       | Slower                               |
| **Accuracy**     | Higher for clear text                      | Depends on scan quality              |
| **Scanned Docs** | May struggle                               | Optimized for scans                  |
| **Languages**    | Limited to model                           | Turkish+English optimized            |
| **API Cost**     | Image tokens                               | Text tokens                          |

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
- Use `--mode copy` to preview the pipeline non-destructively: the renamed copy is written next to the original, which is left untouched (copy runs are not recorded in undo history — there is nothing to undo)
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

| Limitation           | Description                                          |
| :------------------- | :--------------------------------------------------- |
| **Text Length**      | Configurable via `--text-length` (e.g., 10000 chars) |
| **API Costs**        | Each file processed sends text to an LLM API         |
| **Rate Limits**      | Providers may enforce rate limits                    |
| **OCR Quality**      | Scanned documents require good quality scans         |
| **Language Support** | Optimized for Turkish and English                    |
| **Vision Mode**      | PDF only (images supported with `--rename-images`)   |

### OCR Handling Options

- **`--fallback-mode move`**: Moves files with insufficient text to `Needs_Scan` folder (skips fallback processing)
- **`--move-scan-failures`**: Moves files that fail both standard and fallback extraction to `Scan_Failures` folder
- **`--benchmark`**: Run a benchmark comparing PDF extractors on a sample of files (default: first 3)

---

## ✅ Best Practices

1. **Regular Backups**: Always keep a backup of your original files
2. **Start Small**: Test with 5-10 files first to verify quality
3. **Monitor Costs**: Set up usage alerts with your LLM provider
4. **Use Caching**: Avoid `--update-metadata` unless necessary (cached rows are validated against the engine that produced them)
5. **Choose OCR Wisely**: Use `--ocr-enhanced --use-mutool` only for scanned documents
6. **Vision for PDFs**: Use `--fallback-mode vision` for text-based PDFs (faster, more accurate)
7. **Compare Providers**: OpenRouter often offers competitive pricing
8. **Protect Privacy**: For sensitive or private documents, use a local LLM (Ollama, LM Studio) with the `custom` provider. This keeps your document content on your machine rather than sending it to cloud services.

### Caching Strategy

renAI uses a two-layer cache backed by SQLite WAL (Write-Ahead Logging) mode for concurrent-safe, reliable caching.

| Cache Type         | Key Components                                         | Invalidation                                                                                       |
| :----------------- | :----------------------------------------------------- | :------------------------------------------------------------------------------------------------- |
| **Text Cache**     | File hash + OCR/mutool flags                           | File content or extraction settings change                                                          |
| **Metadata Cache** | Document/image mode + file hash + optional `_vision` suffix | File content changes, producing-engine identity changes, folder-rule signature changes, or `--update-metadata` |

**How it works:**

- Each file is identified by its SHA256 hash.
- Metadata keys are identity-free: the engine that actually _produced_ a record is stored in the row and validated on read. A provider/model change re-extracts and overwrites the same row rather than creating a separate entry.
- Under `--fallback-mode vision`, PDFs use a separate `_vision` namespace, so text-based and vision-based results do not overwrite each other.
- Use `--update-metadata` to force fresh extraction regardless of cache.
- SQLite WAL mode keeps concurrent reads/writes safe and performant.

**Example:**

```bash
# First run with DeepInfra - caches metadata
renai process "C:/Books" --provider deepinfra

# Same files with OpenRouter - producing engine changed, so re-extract and overwrite
renai process "C:/Books" --provider openrouter

# Same provider/model - uses cached metadata
renai process "C:/Books" --provider openrouter

# Force refresh - bypasses cache
renai process "C:/Books" --provider openrouter --update-metadata
```

For key formats, provenance validation, legacy-row handling, and troubleshooting, see the [Caching Strategy](docs/CACHING_STRATEGY.md) reference.

**Cache Location:** Managed via `platformdirs` user data directory (e.g., `~/.local/share/renAI/Cache` on Linux, `~/Library/Application Support/renAI/Cache` on macOS, `%LOCALAPPDATA%\renAI\Cache` on Windows). Internally uses SQLite WAL mode for concurrent-safe, high-performance caching.

---

### Privacy Considerations

| Scenario                     | Recommended Provider                               |
| :--------------------------- | :------------------------------------------------- |
| Public documents             | Any cloud provider (DeepInfra, OpenRouter, OpenAI) |
| Private/personal files       | `custom` provider with local LLM                   |
| Sensitive business documents | `custom` provider with local LLM                   |
| Testing/experimentation      | Cloud providers for convenience                    |

For detailed privacy guidance, data retention policies, and compliance considerations, see [PRIVACY.md](docs/PRIVACY.md).

### Cost Estimation

Costs depend on the provider, model, file count, and whether vision fallback is used (image tokens are billed differently from text tokens). Check your provider's current pricing before large runs; use `--benchmark` and `--dry-run` to estimate workload, and set usage alerts with your provider.

---

## 🔧 Configuration

### OS-Agnostic File Management

renAI now utilizes modern OS-agnostic path management via the `platformdirs` library.

- **Cache Data**: Managed in `%LOCALAPPDATA%\renAI\Cache` (Windows) or `~/.local/share/renAI/Cache` (Linux).
- **Log and Debug Data**: Managed in the OS user data directory.
- **Configuration**: Managed via `settings.toml`, `secrets.toml`, `categories.toml`, and `folder_rules.toml` in the flat OS config directory (no nested folders).

### Defining Custom Categories

Create `categories.toml` in your OS's respective user config directory for `renAI` (e.g., `~/.local/share/renAI/categories.toml` on Linux, `%LOCALAPPDATA%\renAI\categories.toml` on Windows):

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

### Folder Rules (Per-Folder Categories & Filename Templates)

Create `folder_rules.toml` in your user config directory (e.g., `~/.local/share/renAI/folder_rules.toml` on Linux) to change how renAI behaves for specific folders. Rules match by path prefix: files processed from the rule folder — or any of its subfolders — use the rule. When several rules match, the most specific (deepest) path wins.

Each rule supports:

| Key                | Effect                                                                                             |
| :----------------- | :------------------------------------------------------------------------------------------------- |
| `path`             | (required) Folder the rule applies to. `~` is expanded.                                             |
| `categories`       | Replaces the document category list offered to the LLM for that folder.                             |
| `image_categories` | Same, but for image files (`--rename-images`).                                                      |
| `doc_template`     | Filename template for text-extracted documents.                                                     |
| `vision_template`  | Filename template for vision-fallback documents.                                                    |
| `image_template`   | Filename template for image files.                                                                  |

```toml
[[rules]]
path = "~/Documents/Akademik"
categories = ["Tez", "Makale", "Araştırma"]
doc_template = "{author} -- {title} -- ({year})"

[[rules]]
path = "~/Pictures/Screenshots"
image_categories = ["Screenshot"]
image_template = "{description} -- ({language})"
```

**Template tokens**: every metadata field — `{title}`, `{subtitle}`, `{edition}`, `{author}`, `{year}`, `{series}`, `{volume}`, `{publisher}`, `{isbn}`, `{doi}`, `{translator}`, `{page_count}`, `{description}`, `{visual_context}`, `{exif_date}` — plus the composites `{language}` (normalized to `tr`/`en`), `{category}`, and `{series_volume}` (renders `Series #3` or `Vol. 3`).

Behavior notes:

- Empty fields collapse away; dangling `()`/`[]` pairs and repeated ` -- ` separators are cleaned automatically.
- If a template produces an empty name, renAI falls back to the built-in filename pattern.
- File types without a template in the rule keep the built-in pattern.
- Templates with unknown tokens are ignored at load time (a warning is logged).
- Rule categories fully replace the global list for that folder; if the model still returns an out-of-list category, it is remapped to `Other`/`Diğer` when allowed, otherwise to the first category in the rule.
- Metadata rows carry a rule signature: changing a rule's categories or templates invalidates its stale rows, so affected files are re-extracted automatically. Legacy rows without a signature are treated as global and re-extracted under rule-scoped runs.
- Rule loading is layered: the first config file that _exists_ wins (user → default → package). A user file that exists but defines no valid rules disables packaged defaults; only a missing file lets the next layer be consulted.

### Command Options

| Option                 | Description                                                                                            |
| :--------------------- | :----------------------------------------------------------------------------------------------------- |
| `--provider`           | `deepinfra`, `openrouter`, `openai`, or `custom`                                                       |
| `--model`              | Model name (for text processing or vision if --fallback-mode vision is used)                           |
| `--extractor`          | `pymupdf`, `pypdf`, `pdfplumber`, `pypdfium2`                                                          |
| `--workers`            | Concurrent workers (default: 4)                                                                        |
| `--ocr-enhanced`       | Enable enhanced OCR                                                                                    |
| `--use-mutool`         | Use mutool for high-quality PDF-to-image conversion during OCR fallback (recommended for scanned PDFs) |
| `--mutool-path`        | Path to the mutool binary (default from settings)                                                      |
| `--debug-ocr`          | Save debug images                                                                                      |
| `--recursive`          | Process subdirectories (default: True)                                                                 |
| `--no-recursive`       | Process root only                                                                                      |
| `--exif`               | Add EXIF date to image filenames when available                                                        |
| `--text-length`        | Target text length for LLM (default: 10000)                                                            |
| `--text-extraction`    | Extraction mode (`head-tail` or `regular`) (default: `head-tail`)                                      |
| `--extraction-pages`   | Pages to process per document                                                                          |
| `--update-metadata`    | Force refresh metadata cache                                                                           |
| `--fallback-mode`      | Strategy when extraction fails (`ocr`, `vision`, `move`)                                               |
| `--move-scan-failures` | Move failed files to `Scan_Failures` folder                                                            |
| `--benchmark`          | Benchmark PDF extractors on a sample of files (default: first 3)                                       |
| `--vision-max-pages`   | Max pages to sample for vision PDF extraction (default: 3)                                             |
| `--vision-provider`    | LLM provider for vision operations (default: from settings)                                            |
| `--vision-model`       | Vision-capable model for vision operations (default: from settings)                                    |
| `--dry-run`            | Simulate processing without renaming or moving files                                                   |
| `undo` / `undo-last`   | Revert file operations from the most recent run                                                         |
| `cache`                | Command to clear cache                                                                                 |
| `init`                 | Command to start interactive setup wizard                                                              |
| `providers`            | Command to list configured providers                                                                   |
| `--version`            | Show renAI version                                                                                     |

> **Important:** When using `--fallback-mode vision` or `--rename-images`, you must use a vision-capable model. renAI validates this at startup (via the models.dev catalog) and exits with a clear error if the configured vision provider/model is missing, unknown, or not vision-capable.

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
│       ├── messages/            # [NEW] Centralized user-facing messages
│       │   ├── cli.py           # CLI warnings, prompts, help texts
│       │   ├── orchestrator.py  # Pipeline status, progress, summary
│       │   ├── setup.py         # Wizard, preflight, save messages
│       │   └── errors.py        # Exception analysis (why/how-to-fix)
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
