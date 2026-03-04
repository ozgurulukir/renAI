# renAI: AI-Powered Document Renamer and Organizer

`renAI` is a powerful Python tool that automatically renames and organizes your digital documents using Large Language Models (LLMs) and Optical Character Recognition (OCR). It extracts metadata directly from document content to create consistent, searchable filenames.

---

## 📖 Introduction

Managing large digital document collections can be challenging. Files often arrive with inconsistent or meaningless names, making them difficult to find and organize. renAI solves this by:

- **Reading document content** to identify titles, authors, and publication years
- **Classifying documents** into customizable categories
- **Applying consistent naming conventions** automatically
- **Organizing files** into category-based folders when desired

The tool supports multiple document formats (PDF, EPUB, MOBI, DOCX, PPTX, XLSX, TXT, MD, Images) and works with various LLM providers (DeepInfra, OpenRouter, OpenAI).

### Naming Convention

**Documents:**
```
Title -- Subtitle -- 'Edition' -- Author -- (Year).ext
```

Example: `Introduction to Machine Learning -- A Practical Approach -- '1st Edition' -- Andrew Ng -- (2018).pdf`

**Images:**
```
Title -- Category -- (Language).ext
```

Example: `Sunset at Beach -- Photo -- (en).jpg`

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Multi-Format Support** | Processes PDF, EPUB, MOBI, DOCX, PPTX, XLSX, TXT, MD, and image files (JPG, PNG, HEIC, etc.) |
| **Multiple PDF Extractors** | Choose from PyMuPDF, pdfplumber, pypdf, or pypdfium2 |
| **AI-Driven Metadata** | Uses LLMs to extract titles, authors, years, and categories |
| **Vision Mode** | Process PDFs directly with vision-capable LLMs (no OCR needed) |
| **Image Renaming** | Rename image files using AI vision analysis |
| **Advanced OCR** | Tesseract integration with Enhanced Mode for low-quality scans |
| **mutool Support** | Extracts high-quality images from fixed-layout PDFs |
| **Smart Caching** | MD5-based caching reduces API costs and processing time |
| **Custom Categories** | Define your own categories per language in `categories.txt` |

### Operating Modes

- **`rename`**: Renames files based on extracted metadata
- **`organize`**: Moves renamed files into category folders
- **`benchmark`**: Compares PDF extraction library performance
- **`eval-model`**: Tests different LLM models for accuracy
- **`--vision`**: Uses vision-capable LLM to process PDFs (no OCR)
- **`--rename-images`**: Renames image files using AI vision

---

## 🛠️ Installation

### Prerequisites

1. **Python 3.10+**
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

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### API Keys

Set your API keys as environment variables:

```bash
# Windows (PowerShell)
$env:DEEPINFRA_API_KEY = "your_key_here"
$env:OPENROUTER_API_KEY = "your_key_here"
$env:OPENAI_API_KEY = "your_key_here"

# Linux/Mac
export DEEPINFRA_API_KEY="your_key_here"
export OPENROUTER_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"
```

Get keys from: [DeepInfra](https://deepinfra.com) | [OpenRouter](https://openrouter.ai) | [OpenAI](https://platform.openai.com/api-keys)

---

## 📖 Usage Examples

### Basic Renaming

```bash
# Rename all files in directory
python main.py "C:/Path/To/Books" --mode rename
```

### Organizing into Folders

```bash
# Rename and organize into category folders
python main.py "C:/Path/To/Books" --mode organize
```

### Vision Mode (PDF without OCR)

Vision mode uses vision-capable LLMs to process PDFs directly - no OCR needed! This is faster and more accurate for text-based PDFs.

```bash
# Process PDFs with vision model (no OCR) - uses default provider/model
python main.py "C:/Path/To/Books" --vision

# Use specific vision model
python main.py "C:/Path/To/Books" --vision --vision-provider openrouter --vision-model openai/gpt-4o

# DeepInfra with Llama Vision
python main.py "C:/Path/To/Books" --vision --vision-provider deepinfra --vision-model meta-llama/Llama-3.2-90B-Vision-Instruct

# OpenAI GPT-4o
python main.py "C:/Path/To/Books" --vision --vision-provider openai --vision-model gpt-4o
```

### Image Renaming

Rename image files using AI vision analysis. Requires `--vision-model` to be specified:

```bash
# Rename images with vision model (required: --vision-model)
python main.py "C:/Path/To/Photos" --rename-images --vision-model openai/gpt-4o

# With explicit provider
python main.py "C:/Path/To/Images" --rename-images --vision-provider deepinfra --vision-model Qwen/Qwen2-VL-72B-Instruct
```

### Mixed Folders (PDF + Images)

Process both PDFs and images in the same folder using vision AI:

```bash
# PDFs processed with vision, images renamed with vision
python main.py "C:/Path/To/Mixed" --vision --rename-images --vision-model openai/gpt-4o
```

### High-Quality OCR for Scanned Documents

```bash
# Enhanced OCR with image preprocessing
python main.py "C:/Path/To/Books" --ocr-enhanced --use-mutool
```

### Using Different```bash
# LLM Providers

 DeepInfra (default)
python main.py "C:/Path/To/Books" --mode rename --provider deepinfra

# OpenRouter
python main.py "C:/Path/To/Books" --mode rename --provider openrouter

# OpenAI
python main.py "C:/Path/To/Books" --mode rename --provider openai
```

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
python main.py "C:/Path/To/Books" --mode rename --provider custom

# Or override model directly
python main.py "C:/Path/To/Books" --mode rename --provider custom --model mistral
```

#### Controlling Text Length for Local Models

When running local models on consumer hardware, you can reduce the text length sent to the LLM to decrease memory usage and improve performance:

```bash
# Reduce text length to 5000 characters (default: 10000)
python main.py "C:/Path/To/Books" --mode rename --provider custom --text-length 5000

# Increase for better context if your model can handle it
python main.py "C:/Path/To/Books" --mode rename --provider custom --text-length 20000
```

| Option | Description | Default |
|--------|-------------|---------|
| `--text-length` | Target text length sent to LLM | 10000 |

#### Environment Variables for Custom Provider

| Variable | Description | Default |
|----------|-------------|---------|
| `CUSTOM_API_BASE_URL` | Base URL of your local API server | `http://localhost:8000/v1` |
| `CUSTOM_MODEL` | Model name to use | `llama3` |
| `CUSTOM_API_KEY` | API key (usually not needed for local) | `not-required` |

#### Common Local Server URLs

| Tool | URL |
|------|-----|
| Ollama | `http://localhost:11434/v1` |
| LM Studio | `http://localhost:1234/v1` |
| LocalAI | `http://localhost:8080/v1` |
| Text Generation WebUI | `http://localhost:5001/v1` |

### Different Models

```bash
# DeepInfra with Qwen model
python main.py "C:/Path/To/Books" --mode rename --model Qwen/Qwen2.5-72B-Instruct

# OpenRouter with Gemini
python main.py "C:/Path/To/Books" --mode rename --provider openrouter --model google/gemini-2.0-flash-exp:free
```

### Faster Processing

```bash
# Increase concurrent workers
python main.py "C:/Path/To/Books" --mode rename --workers 8
```

### Force Refresh Metadata

```bash
# Bypass cache and re-extract
python main.py "C:/Path/To/Books" --mode rename --update-metadata
```

### Benchmarking & Model Evaluation

#### PDF Extractor Benchmark
Compare the performance and accuracy of different PDF extraction libraries on your documents.
```bash
python main.py "C:/Path/To/Books" --benchmark
```

#### LLM Model Evaluation
Test multiple models from a provider to find the best balance between cost and accuracy for your specific document collection.
```bash
python main.py "C:/Path/To/Books" --eval-model --provider deepinfra
```

**How it works:**
- Models are loaded from provider-specific configuration files: `eval_models_deepinfra.txt`, `eval_models_openai.txt`, or `eval_models_openrouter.txt`.
- Each file should contain a list of model names, one per line.
- The tool processes the first 2 files in your directory using **all** listed models.
- Results and confidence scores are logged to `evaluation.log` for detailed comparison.

### Cache Management

```bash
# Clear all cache
python main.py --clear-cache

# Clear cache for specific directory
python main.py "C:/Path/To/Books" --clear-cache
```

---

## 🤖 Vision Models

renAI supports vision-capable LLMs that can process PDFs and images directly without OCR. This is faster and more accurate for text-based documents.

### Recommended Vision Models

| Provider | Model | Best For |
|----------|-------|----------|
| **OpenAI** | `gpt-4o` | Best accuracy, reliable |
| **OpenAI** | `gpt-4o-mini` | Fast, cost-effective |
| **OpenRouter** | `openai/gpt-4o` | Flexible pricing |
| **OpenRouter** | `google/gemini-1.5-flash` | Free tier available |
| **DeepInfra** | `meta-llama/Llama-3.2-90B-Vision-Instruct` | Open source, powerful |
| **DeepInfra** | `Qwen/Qwen2-VL-72B-Instruct` | Excellent value |
| **Custom** | `llava`, `qwen-vl` | Local inference |

### Vision Mode vs OCR

| Feature | Vision Mode (`--vision`) | OCR Mode (default) |
|---------|--------------------------|-------------------|
| **Speed** | Faster for text PDFs | Slower |
| **Accuracy** | Higher for clear text | Depends on scan quality |
| **Scanned Docs** | May struggle | Optimized for scans |
| **Languages** | Limited to model | Turkish+English optimized |
| **API Cost** | Image tokens | Text tokens |

**When to use Vision Mode:**
- Text-based PDFs (books, articles)
- When you don't have Tesseract installed
- When OCR is too slow
- When you want better accuracy on clean documents

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
|------------|-------------|
| **Text Length** | Up to 8000 characters sent to LLM per file |
| **API Costs** | Each file processed sends text to an LLM API |
| **Rate Limits** | Providers may enforce rate limits |
| **OCR Quality** | Scanned documents require good quality scans |
| **Language Support** | Optimized for Turkish and English |
| **Vision Mode** | PDF only (images supported with `--rename-images`) |

### OCR Handling Options

- **`--skip-needs-ocr`**: Moves files with insufficient text to `Needs_OCR` folder (skips OCR)
- **`--move-ocr-failures`**: Moves completely unprocessable files to `OCR_Failures` folder

---

## ✅ Best Practices

1. **Regular Backups**: Always keep a backup of your original files
2. **Start Small**: Test with 5-10 files first to verify quality
3. **Monitor Costs**: Set up usage alerts with your LLM provider
4. **Use Caching**: Avoid `--update-metadata` unless necessary
5. **Choose OCR Wisely**: Use `--ocr-enhanced --use-mutool` only for scanned documents
6. **Vision for PDFs**: Use `--vision` for text-based PDFs (faster, more accurate)
7. **Compare Providers**: OpenRouter often offers competitive pricing
8. **Clear Cache**: Use `--clear-cache` when switching models or settings
9. **Protect Privacy**: For sensitive or private documents, use a local LLM (Ollama, LM Studio) with the `custom` provider. This keeps your document content on your machine rather than sending it to cloud services.

### Caching Strategy

renAI uses an intelligent caching system that includes provider-model awareness:

| Cache Type | Key Components | Invalidation |
|------------|----------------|--------------|
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
python main.py "C:/Books" --provider deepinfra

# Same files with OpenRouter - NEW cache entry (old one preserved)
python main.py "C:/Books" --provider openrouter

# Same provider/model - uses cached metadata
python main.py "C:/Books" --provider deepinfra

# Force refresh - bypasses cache
python main.py "C:/Books" --provider deepinfra --update-metadata
```

**Cache Location:** Managed automatically by the OS via `platformdirs` (e.g., `~/.cache/renAI` on Linux, `~/Library/Caches/renAI` on macOS, `%LOCALAPPDATA%\renAI\Cache` on Windows).

**Clear Cache:**
```bash
# Clear all cache
python main.py --clear-cache

# Clear cache for specific directory
python main.py "C:/Books" --clear-cache
```

### Privacy Considerations

| Scenario | Recommended Provider |
|----------|---------------------|
| Public documents | Any cloud provider (DeepInfra, OpenRouter, OpenAI) |
| Private/personal files | `custom` provider with local LLM |
| Sensitive business documents | `custom` provider with local LLM |
| Testing/experimentation | Cloud providers for convenience |

For detailed privacy guidance, data retention policies, and compliance considerations, see [PRIVACY.md](PRIVACY.md).

### Cost Estimation

| Files | Est. Cost (~$0.001/file) |
|-------|--------------------------|
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

### Custom Categories

Create `categories.txt` in your OS's respective user config directory for `renAI` (e.g., `~/.config/renAI/categories.txt` on Linux, `%APPDATA%\renAI` on Windows):

```txt
# English Categories
LANG: en
Article
Fiction
Non-Fiction
Textbook
Report
Gov
Presentation
Manual
Thesis
Research
Other
Dictionary

# Turkish Categories
LANG: tr
Makale
Kurgu
Kurgu Dışı
Ders Kitabı
Rapor
Devlet
Sunum
Kılavuz
Tez
Araştırma
Diğer
Sözlük

# Image Categories (for --rename-images)
LANG: en
Photo
Screenshot
Design
Art
Diagram
Chart
Document
Other

LANG: tr
Fotoğraf
Ekran Görüntüsü
Tasarım
Sanat
Diyagram
Grafik
Belge
Diğer
```

### Command Options

| Option | Description |
|--------|-------------|
| `--provider` | `deepinfra`, `openrouter`, `openai`, or `custom` |
| `--model` | Model name for text processing |
| `--vision-provider` | Provider for vision mode (default: same as --provider) |
| `--vision-model` | Vision-capable model for --vision and --rename-images |
| `--extractor` | `pymupdf`, `pypdf`, `pdfplumber`, `pypdfium2` |
| `--workers` | Concurrent workers (default: 4) |
| `--ocr-enhanced` | Enable enhanced OCR |
| `--use-mutool` | Use mutool for extraction |
| `--debug-ocr` | Save debug images |
| `--recursive` | Process subdirectories (default: True) |
| `--no-recursive` | Process root only |
| `--vision` | Use vision AI for PDFs (no OCR needed) |
| `--rename-images` | Rename images using vision AI (requires --vision-model) |
| `--text-length` | Target text length for LLM (default: 10000) |
| `--update-metadata` | Force refresh metadata cache |
| `--skip-needs-ocr` | Move files needing OCR to folder |
| `--move-ocr-failures` | Move failed files to folder |
| `--clear-cache` | Clear cache and exit |

---

## 🏗️ Architecture

```
renAI/
├── main.py              # Entry point & CLI
├── config.py            # Configuration & API settings
├── utils.py             # Helper functions
├── llm_engine.py        # LLM provider integration (including vision)
├── ocr_engine.py        # OCR processing
├── file_processor.py    # Core processing logic (including vision processing)
├── extractors/         # Document parsers
│   ├── base.py         # Abstract base class
│   └── pdf_extractors.py # PDF extraction implementations
├── test_*.py          # Unit tests
└── requirements.txt    # Dependencies
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

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest test_*.py -v

# Run vision-specific tests
pytest test_vision_*.py -v

# Run with coverage
pytest test_file_processor.py --cov=file_processor
```

---

*renAI - Organize your digital library with AI*