# renAI: AI-Powered Document Renamer and Organizer

`renAI` is a powerful Python tool that automatically renames and organizes your digital documents using Large Language Models (LLMs) and Optical Character Recognition (OCR). It extracts metadata directly from document content to create consistent, searchable filenames.

---

## 📖 Introduction

Managing large digital document collections can be challenging. Files often arrive with inconsistent or meaningless names, making them difficult to find and organize. renAI solves this by:

- **Reading document content** to identify titles, authors, and publication years
- **Classifying documents** into customizable categories
- **Applying consistent naming conventions** automatically
- **Organizing files** into category-based folders when desired

The tool supports multiple document formats (PDF, EPUB, MOBI, DOCX, PPTX, XLSX, TXT, MD) and works with various LLM providers (DeepInfra, OpenRouter, OpenAI).

### Naming Convention

Files are renamed using the format:
```
Title -- Subtitle -- 'Edition' -- Author -- (Year).ext
```

Example: `Introduction to Machine Learning -- A Practical Approach -- '1st Edition' -- Andrew Ng -- (2018).pdf`

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Multi-Format Support** | Processes PDF, EPUB, MOBI, DOCX, PPTX, XLSX, TXT, and MD files |
| **Multiple PDF Extractors** | Choose from PyMuPDF, pdfplumber, pypdf, or pypdfium2 |
| **AI-Driven Metadata** | Uses LLMs to extract titles, authors, years, and categories |
| **Advanced OCR** | Tesseract integration with Enhanced Mode for low-quality scans |
| **mutool Support** | Extracts high-quality images from fixed-layout PDFs |
| **Smart Caching** | MD5-based caching reduces API costs and processing time |
| **Custom Categories** | Define your own categories per language in `categories.txt` |

### Operating Modes

- **`rename`**: Renames files based on extracted metadata
- **`organize`**: Moves renamed files into category folders
- **`benchmark`**: Compares PDF extraction library performance
- **`eval-model`**: Tests different LLM models for accuracy

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

### High-Quality OCR for Scanned Documents

```bash
# Enhanced OCR with image preprocessing
python main.py "C:/Path/To/Books" --ocr-enhanced --use-mutool
```

### Using Different LLM Providers

```bash
# DeepInfra (default)
python main.py "C:/Path/To/Books" --mode rename --provider deepinfra

# OpenRouter
python main.py "C:/Path/To/Books" --mode rename --provider openrouter

# OpenAI
python main.py "C:/Path/To/Books" --mode rename --provider openai
```

### Local Inference (Ollama / LM Studio)

Use the `custom` provider to connect to local inference servers like [Ollama](https://ollama.com/) or [LM Studio](https://lmstudio.ai/). These tools provide OpenAI-compatible APIs.

```bash
# Set environment variables (Windows PowerShell)
$env:CUSTOM_API_BASE_URL = "http://localhost:11434/v1"  # Ollama default
$env:CUSTOM_MODEL = "llama3"  # Your model name

# Run with custom provider
python main.py "C:/Path/To/Books" --mode rename --provider custom

# Or override model directly
python main.py "C:/Path/To/Books" --mode rename --provider custom --model mistral
```

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
6. **Compare Providers**: OpenRouter often offers competitive pricing
7. **Clear Cache**: Use `--clear-cache` when switching models or settings
8. **Protect Privacy**: For sensitive or private documents, use a local LLM (Ollama, LM Studio) with the `custom` provider. This keeps your document content on your machine rather than sending it to cloud services.

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

**Cache Location:** `.renamer_cache/` directory

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

Actual costs vary by provider and model.

---

## 🔧 Configuration

### Custom Categories

Create `categories.txt` in the project root:

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
```

### Command Options

| Option | Description |
|--------|-------------|
| `--provider` | `deepinfra`, `openrouter`, `openai`, or `custom` |
| `--extractor` | `pymupdf`, `pypdf`, `pdfplumber`, `pypdfium2` |
| `--workers` | Concurrent workers (default: 4) |
| `--ocr-enhanced` | Enable enhanced OCR |
| `--use-mutool` | Use mutool for extraction |
| `--debug-ocr` | Save debug images |
| `--recursive` | Process subdirectories (default: True) |
| `--no-recursive` | Process root only |

---

## 🏗️ Architecture

```
renAI/
├── main.py              # Entry point & CLI
├── config.py            # Configuration & API settings
├── utils.py             # Helper functions
├── llm_engine.py        # LLM provider integration
├── ocr_engine.py        # OCR processing
├── file_processor.py    # Core processing logic
├── extractors/          # Document parsers
│   ├── base.py          # Abstract base class
│   └── pdf_extractors.py # PDF extraction implementations
├── test_file_processor.py # Unit tests
└── requirements.txt     # Dependencies
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
pytest test_file_processor.py -v

# Run with coverage
pytest test_file_processor.py --cov=file_processor
```

---

*renAI - Organize your digital library with AI*
