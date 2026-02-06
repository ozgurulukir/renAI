# renAI: AI-Powered Book Renamer and Organizer

`renAI` is a powerful Python tool designed to automatically rename and organize your digital book collection using LLMs (Large Language Models) and OCR (Optical Character Recognition). It extracts metadata directly from the content of your PDF, EPUB, MOBI, and other document formats.

---



## ⚙️ Custom Categories Configuration

You can define custom categories by creating a `categories.txt` file in the project root directory.

### Dictionary Structure:
Categories are stored as a dictionary with language codes as keys and lists of categories as values:
```json
{
  "en": ["Article", "Fiction", "Non-Fiction", ...],
  "tr": ["Makale", "Kurgu", "Kurgu Dışı", ...]
}
```

### File Format:
- Lines starting with `#` are comments
- Empty lines are ignored
- Use `LANG: language_code` to specify language (tr for Turkish, en for English)
- Categories are listed under their respective language

### Example `categories.txt`:
```
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
Biography
Memoir
Science
History
Philosophy
Psychology

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

If no `categories.txt` file exists, the tool uses default categories.

---

### Data Loss Prevention
- **Always backup your files** before running renAI in `rename` or `organize` mode. The tool will rename and move files, and while it includes safety checks, unexpected issues can occur.
- **Review the naming convention** before applying to your collection. The format is: `Title -- Subtitle -- Author (Year).ext`
- **Test with a small directory first** to verify the tool behaves as expected with your specific files.

### Token Usage & Budget Considerations
- **LLM API costs apply**: Each file processed sends text to an LLM API. Monitor your API usage regularly.
- **Text length limits**: Up to 8000 characters are sent to the LLM per file. Large documents may be truncated.
- **Caching helps reduce costs**: Extracted text and metadata are cached. Use `--update-metadata` only when necessary.
- **Estimate your budget**: For 1000 files at ~$0.001 per file (approximate), expect ~$1 in API costs. Actual costs vary by provider and model.
- **Rate limits**: Some providers enforce rate limits. Use `--workers` to control concurrency if needed.

### Recommended Best Practices
1. **Regular Backups**: Keep at least one backup of your original file collection before processing.
2. **Start Small**: Test with 5-10 files first to verify metadata extraction quality.
3. **Monitor API Usage**: Set up usage alerts with your LLM provider to avoid unexpected charges.
4. **Cache Management**: Use `--clear-cache` to reset cached data when switching models or extraction methods.
5. **OCR Quality**: For scanned documents, use `--ocr-enhanced --use-mutool` for better results (but expect higher API costs due to more text).
6. **Provider Selection**: Compare costs between DeepInfra, OpenRouter, and OpenAI. OpenRouter often offers competitive pricing.

---

## ⚠️ Important Warnings & Best Practices

### Directory Layout Warning for Organize Mode
**WARNING:** When using `--mode organize`, the script will rename and move files into category folders at the target directory root. This may disrupt your existing directory layout:
- Files from all subdirectories will be moved to category folders (e.g., `Fiction/`, `Makale/`)
- Empty subdirectories may remain after files are moved
- Category folders are created at the root level, not within subdirectories
- Running organize mode multiple times will re-process files and may change folder structure

**Recommendation:** Always backup your files before running organize mode, and test with a small directory first.

### Data Loss Prevention
- **Always backup your files** before running renAI in `rename` or `organize` mode. The tool will rename and move files, and while it includes safety checks, unexpected issues can occur.
- **Review the naming convention** before applying to your collection. The format is: `Title -- Subtitle -- Author (Year).ext`
- **Test with a small directory first** to verify the tool behaves as expected with your specific files.

### Token Usage & Budget Considerations
- **LLM API costs apply**: Each file processed sends text to an LLM API. Monitor your API usage regularly.
- **Text length limits**: Up to 8000 characters are sent to the LLM per file. Large documents may be truncated.
- **Caching helps reduce costs**: Extracted text and metadata are cached. Use `--update-metadata` only when necessary.
- **Estimate your budget**: For 1000 files at ~$0.001 per file (approximate), expect ~$1 in API costs. Actual costs vary by provider and model.
- **Rate limits**: Some providers enforce rate limits. Use `--workers` to control concurrency if needed.


5. **OCR Quality**: When working with scanned documents, use `--ocr-enhanced --use-mutool` for better results (but expect higher API costs due to more text).
6. **Skew Correction**: For rotated/skewed documents, the enhanced OCR mode includes automatic skew correction using OpenCV (install with `pip install opencv-python`).

---

## 🚀 Features

- **Multi-Format Support:** Processes PDF, EPUB, MOBI, DOCX, PPTX, XLSX, TXT, and MD files.
- **Multiple PDF Extractors:** Choose from PyMuPDF, pdfplumber, pypdf, or pypdfium2 for text extraction.
- **AI-Driven Metadata Extraction:** Uses DeepInfra, OpenRouter (OpenAI-compatible), or OpenAI to identify titles, authors, years, and categories.
- **Advanced OCR:** Integrated Tesseract OCR with an "Enhanced Mode" for cleaning and sharpening low-quality scans.
- **mutool Integration:** Support for `mutool` to extract high-quality images from fixed-layout PDFs.
- **Smart Caching:** Uses MD5 hashing to cache extracted text and metadata, saving API costs and time.
- **Flexible Modes:**
  - `rename`: Renames files based on extracted metadata.
  - `organize`: Moves renamed files into automatically created category folders.
  - `benchmark`: Compares different PDF extraction libraries (`PyMuPDF`, `pdfplumber`, etc.).
  - `eval-model`: Evaluates different LLM models for metadata accuracy. (Models are managed via `eval_models_{provider}.txt` files).

---

## 🛠️ Installation

### 1. Prerequisites
- **Python 3.10+**
- **Tesseract OCR:** [Download and install](https://github.com/UB-Mannheim/tesseract/wiki/Tesseract-at-UB-Mannheim).
- **mutool (Optional but recommended):** Part of [MuPDF](https://mupdf.com/releases/index.html).

### 2. API Keys
Get your API keys from:
- **DeepInfra:** [https://deepinfra.com](https://deepinfra.com)
- **OpenRouter:** [https://openrouter.ai](https://openrouter.ai)
- **OpenAI:** [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### 3. Clone the Repository
```bash
git clone https://github.com/ozgurulukir/renAI.git
cd renAI
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

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

You can also customize settings in `config.py`, such as default models and Tesseract paths.

---

## 📖 Usage

### Basic Renaming
```bash
python main.py "C:/Path/To/Your/Books" --mode rename
```

### Organizing into Folders
```bash
python main.py "C:/Path/To/Your/Books" --mode organize
```

### High-Quality OCR (for scans)
```bash
python main.py "C:/Path/To/Your/Books" --ocr-enhanced --use-mutool
```

### Using Different AI Providers

#### DeepInfra (default)
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --provider deepinfra
```

#### OpenRouter
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --provider openrouter
```

#### OpenAI
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --provider openai
```

### Using Different PDF Extractors

#### Using pdfplumber instead of default PyMuPDF
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --extractor pdfplumber
```

#### Using pypdf
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --extractor pypdf
```

#### Using pypdfium2
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --extractor pypdfium2
```

### Custom Model Selection

#### Specify a custom model for DeepInfra
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --model Qwen/Qwen2.5-72B-Instruct
```

#### Specify a custom model for OpenRouter
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --provider openrouter --model google/gemini-2.0-flash-exp:free
```

#### Specify a custom model for OpenAI
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --provider openai --model gpt-4o
```

### Processing Multiple Files Concurrently

#### Increase workers for faster processing
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --workers 8
```

### Force Refresh Metadata

#### Bypass cache and re-extract metadata
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --update-metadata
```

### Debug OCR Processing

#### Save OCR debug images for troubleshooting
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --debug-ocr
```

#### Skip OCR processing for files that require OCR fallback
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --skip-needs-ocr
```

### Benchmark PDF Extractors

#### Compare performance of different extractors
```bash
python main.py "C:/Path/To/Your/Books" --benchmark
```

### Evaluate LLM Models

#### Test multiple models for metadata accuracy
```bash
python main.py "C:/Path/To/Your/Books" --eval-model --provider deepinfra
```

#### Evaluate OpenAI models
```bash
python main.py "C:/Path/To/Your/Books" --eval-model --provider openai
```

### Advanced Options
| Option | Description |
| :--- | :--- |
| `--provider` | Choose between `deepinfra`, `openrouter`, or `openai`. |
| `--extractor` | Choose PDF extractor (`pymupdf`, `pypdf`, `pdfplumber`, `pypdfium2`). |
| `--workers` | Number of concurrent files to process (default: 4). |
| `--update-metadata` | Force fresh extraction and LLM call, bypassing cache. |
| `--ocr-enhanced` | Enable high-quality OCR with image enhancement. |
| `--use-mutool` | Use mutool for higher quality PDF image extraction. |
| `--debug-ocr` | Save OCR debug images to `.ocr_debug/` folder. |
| `--benchmark` | Run extraction benchmark on first 3 files. |
| `--eval-model` | Evaluate multiple models for the selected provider. |
| `--clear-cache` | Clear all cache or cache for files in the specified directory (removes text and metadata). |
| `--skip-needs-ocr` | Skip OCR processing for files that require OCR fallback (text extraction failed) and move them directly to Needs_OCR folder. |
| `--move-ocr-failures` | Move files that cannot be processed (no text extracted, even with OCR) to OCR_Failures folder. |
| `--recursive` | Recursively process subdirectories (default: True). |
| `--no-recursive` | Process only the root directory (no recursion). |

### Clearing the Cache

#### Clear all cached text and metadata (no directory required)
```bash
python main.py --clear-cache
```

#### Clear cache for a directory (removes text and metadata for files to be processed)
```bash
python main.py "C:/Path/To/Your/Books" --clear-cache
```

#### Clear cache for a single file
```bash
python main.py "C:/Path/To/Your/Books/book.pdf" --clear-cache
```

This will remove cache files:
- Cached extracted text files (e.g., `hash_enhanced.txt`, `hash_basic.txt`)
- Cached metadata files (e.g., `hash_metadata.json`)

---

### Recursive Processing

#### Process only root directory (no recursion)
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --no-recursive
```

#### Process recursively (default behavior)
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --recursive
```

For `organize` mode, recursive is enabled by default to organize files from all subdirectories into category folders at the root level.

#### OCR Handling Options

renAI provides two distinct OCR handling options:

- **`--skip-needs-ocr`**: Skips OCR processing entirely for files that require OCR fallback and moves them directly to a `Needs_OCR` folder. These files have insufficient text from direct extraction. Use this option to save processing time and API costs by avoiding OCR attempts on files that are likely to need manual review.

- **`--move-ocr-failures`**: Moves files that completely fail processing (no text extracted, even with OCR) to an `OCR_Failures` folder. These files cannot be processed and require manual review.

**Key Difference**: `--skip-needs-ocr` prevents OCR processing entirely for efficiency, while `--move-ocr-failures` handles files that have *completely failed* processing after OCR was attempted.

#### OCR Failures Folder
When using `--move-ocr-failures`, files that cannot be processed (no extractable text, even with OCR fallback) are moved to an `OCR_Failures` folder within the target directory. This helps identify problematic files that may need manual review.

#### Re-processing Behavior:
The script supports re-running organize mode on the same directory. Files in category folders will be re-evaluated and potentially renamed/moved again. This is useful when:
- Switching to a different LLM provider or model
- Updating category definitions in `categories.txt`
- Correcting manually moved files
- Re-processing with different settings

The caching mechanism still applies - text and metadata extraction is skipped if cached results exist for a file.

---

## 🏗️ Technical Architecture

- **`main.py`:** Entry point and CLI manager.
- **`config.py`:** Configuration for API keys, paths, and evaluation models.
- **`utils.py`:** Helper functions for hashing, sanitization, and text chunking.
- **`file_processor.py`:** Orchestrates the flow: extract → LLM → rename/move.
- **`ocr_engine.py`:** Manages image preprocessing and Tesseract integration.
- **`llm_engine.py`:** Handles communication with AI providers (DeepInfra, OpenRouter, OpenAI) and validates JSON output.
- **`extractors/`:** Modular package for various document parsing engines.

---

## 📝 License

[MIT License](LICENSE)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Support

For support, please open an issue in the GitHub repository or contact the maintainers.
