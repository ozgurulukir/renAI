
# Privacy & Data Retention Guide

This document outlines privacy considerations, data handling practices, and recommendations for using renAI securely.

---

## 📋 Overview

renAI processes documents locally and sends text content to Large Language Model (LLM) providers for metadata extraction. Understanding how data flows through the system is essential for making informed decisions about your document privacy.

### Data Flow

```text
Local Document → Text Extraction → LLM API → Metadata Response
                    ↓
              Local Cache Storage
```

1. **Text Extraction**: Document content is extracted locally using PDF parsers or OCR.
2. **Vision Processing (Optional)**: If Vision Mode is enabled and text extraction fails (or if renaming images), pages/images are sampled and converted to Base64.
3. **API Transmission**: Text or sampled images are sent to LLM providers (cloud or local).
4. **Processing**: LLM analyzes content and returns metadata.
5. **Caching**: Extracted text, sampled vision data, and final metadata are stored locally for efficiency.

---

## 🔒 Data Sent to LLM Providers

When using cloud LLM providers (DeepInfra, OpenRouter, OpenAI), the following data is transmitted:

| Data Type | Description | Sent to LLM |
| :--- | :---: | :---: |
| **Document Text** | Extracted text content (up to 10000 chars) | ✅ Yes |
| **Sampled Images** | Base64 encoded page samples (**Vision Mode only**) | ✅ Yes |
| **Original Images** | Full image files when using `--rename-images` | ✅ Yes |
| **File Metadata** | Filename, file size, page count | ❌ No |
| **File Content** | Full document binary/source | ❌ No |
| **Category List** | Your custom categories from `categories.toml` | ✅ Yes (in prompt) |

### What is NOT Sent

- Original document binary files (except when sampled for Vision)
- Embedded files/attachments inside PDFs
- File paths (unless they appear in extracted text)
- Metadata not relevant to the renaming task
- Any personally identifiable information from filenames

### What IS Sent

- Extracted text content (truncated to `RENAI_TEXT_LENGTH`, default 10000 characters)
- **Base64 encoded images** of document pages or image files (**Vision Mode only**)
- System prompt with category definitions and naming instructions
- Intermittent conversation history during extraction refinement

---

## 🏠 Local Processing vs. Cloud Processing

### Cloud Providers (Default)

| Provider | Data Location | Privacy Level |
| :------- | :------------ | :------------ |
| DeepInfra | US-based servers | Standard |
| OpenRouter | Various providers | Standard |
| OpenAI | US-based servers | Standard |

**Characteristics:**

- Text content leaves your machine
- Subject to provider's data retention policies
- Typically 30-day retention for API logs
- May be used for model improvement (check provider terms)

### Local Providers (Recommended for Privacy)

| Tool | Setup Required | Model Options |
| :------- | :------------- | :------------ |
| Ollama | Install Ollama, run `ollama serve` | Llama 3, Mistral, Gemma, etc. |
| LM Studio | Install LM Studio, enable API | Any GGUF model |
| LocalAI | Docker or binary installation | Many options |
| vLLM | Server installation | High-performance |

**Characteristics:**

- All processing happens on your machine
- No data leaves your network
- You control data retention entirely
- No third-party access to your documents

---

## 📁 Local Cache Storage

### Cache Location

renAI utilizes modern OS-agnostic path management via the `platformdirs` library. Cache files are stored in your operating system's standard cache location:

- **Windows**: `%LOCALAPPDATA%\renAI\Cache`
- **macOS**: `~/Library/Caches/renAI`
- **Linux**: `~/.cache/renAI`

The directory structure is organized as follows:

```text
Cache/
├── text/        # Extracted text files
└── metadata/    # metadata files
```

### Cache Management

```bash
# Clear all cache (text, metadata, and ocr_debug)
renai cache
```

### Cache Security Recommendations

1. **Use built-in tools**: Use `renai cache` to safely clear cached data.
2. **Exclude from backups**: Consider excluding the OS-specific cache directory from automated backups.
3. **Regular cleanup**: Clear cache periodically, especially after processing sensitive documents.
4. **Encrypted storage**: Store cache on encrypted drives if possible.

---

## 🔐 Privacy by Scenario

### Scenario 1: Public Domain Books

**Risk Level**: 🟢 Low

**Recommendation**: Any provider is suitable

**Considerations:**

- Content is already public domain
- No personal or sensitive information
- Cloud processing is convenient and cost-effective

```bash
# Example: Using OpenRouter for public books
renai process "C:/Public/Books" --mode rename --provider openrouter
```

### Scenario 2: Personal Documents

**Risk Level**: 🟡 Medium

**Recommendation**: Use local LLM or trusted cloud provider

**Considerations:**

- May contain personal information
- Consider using local processing for sensitive content
- Review provider's data handling policies

```bash
# Example: Using local Ollama for personal documents
# Set these in your OS-specific .env file or environment
RENAI_PROVIDER=custom
CUSTOM_API_BASE_URL=http://localhost:11434/v1
CUSTOM_MODEL=llama3
uv run renai process "C:/Personal/Documents" --mode rename
```

### Scenario 3: Business/Professional Documents

**Risk Level**: 🔴 High

**Recommendation**: Use local LLM only

**Considerations:**

- May contain trade secrets, client information
- Subject to confidentiality obligations
- Compliance requirements (GDPR, HIPAA, etc.)
- Legal liability for data breaches

```bash
# Example: Using LM Studio for business documents
$env:CUSTOM_API_BASE_URL = "http://localhost:1234/v1"
$env:CUSTOM_MODEL = "your-model"
renai process "C:/Work/Documents" --mode rename --provider custom
```

### Scenario 4: Healthcare/Medical Records

**Risk Level**: 🔴🔴 Very High

**Recommendation**: Local LLM with additional safeguards

**Considerations:**

- Subject to strict regulations (HIPAA, etc.)
- May require audit trails
- Consider air-gapped systems
- Consult compliance officer before use

---

## 🛡️ Best Practices for Privacy

### 1. Use Local Processing for Sensitive Content

```bash
export CUSTOM_API_BASE_URL="http://localhost:11434/v1"
export CUSTOM_MODEL="llama3"
renai process "C:/Sensitive/Files" --mode rename --provider custom
```

### 2. Clear Cache After Processing Sensitive Files

```bash
# Immediately clear cache after processing
renai cache
```

### 3. Review Provider Terms of Service

Before using any cloud provider, review their:

- Data retention policies
- Data usage for model training
- Privacy policy
- Security certifications

### 4. Implement Network Controls

For local inference servers:

```bash
# Bind to localhost only (prevent external access)
# Ollama: OLLAMA_HOST=127.0.0.1:11434 ollama serve

# Configure firewall to block external access
```

### 5. Use Environment Variables for Sensitive Config

```bash
# Instead of command-line arguments (which may be logged)
# Use the .env file in your OS configuration directory

RENAI_PROVIDER=deepinfra
DEEPINFRA_API_KEY=your-api-key-here

uv run renai process "C:/Books" --mode rename
```

### 6. Audit Log Review

Enable logging and review regularly:

```bash
# Check evaluation.log for processing history
cat evaluation.log

# Review console output for errors
```

---

## 📊 Provider Comparison

| Provider | Data Retention | Training Data | Privacy Rating |
| :------- | :------------- | :------------ | :------------- |
| **Local (Ollama/LM Studio)** | You control | N/A | ⭐⭐⭐⭐⭐ |
| DeepInfra | 30 days | Check policy | ⭐⭐⭐ |
| OpenRouter | Varies by provider | Check policy | ⭐⭐⭐ |
| OpenAI | 30 days | May use | ⭐⭐⭐ |

---

## 🧹 Data Cleanup Checklist

After processing sensitive documents:

- [ ] Clear renAI cache: `renai cache`
- [ ] Delete any debug images in OS-specific data directory
- [ ] Review and clear shell history
- [ ] Check provider API dashboards for usage logs
- [ ] Verify cache directory is empty

---

## ⚖️ Compliance Considerations

### GDPR (EU)

- **Right to be Forgotten**: Use `--clear-cache` to remove cached data
- **Data Minimization**: Only 8000 characters sent per file
- **Consent**: Ensure you have rights to process documents

### HIPAA (Healthcare - US)

- **Local processing strongly recommended**
- **BAA required** for cloud providers
- **Audit trails** may be necessary
- **Consult compliance officer** before use

### SOC 2 / ISO 27001

- **Local processing** preferred for sensitive workloads
- **Access controls** on cache directory
- **Encryption at rest** recommended

---

## ❓ Frequently Asked Questions

### Q: Can I use renAI offline?

**A**: Yes, by using a local LLM provider (Ollama, LM Studio). The tool requires an LLM API to function, but it can be entirely local.

### Q: Does renAI upload my documents to the cloud?

**A**: Only the extracted text (up to 8000 characters) is sent to the LLM provider. The original document files are never uploaded.

### Q: How long does the cloud provider keep my data?

**A**: This varies by provider. Check their terms of service:

- OpenAI: 30 days for API data
- DeepInfra: Check provider policy
- OpenRouter: Varies by underlying provider

### Q: Can I delete my data from the cloud provider?

**A**: For cloud providers, you can request data deletion per their policies. For local processing, you have complete control.

### Q: Is the cache encrypted?

**A**: No, the cache is stored as plain text files. Use `renai cache` to manage your data or store the OS-specific cache directory on encrypted storage if processing sensitive documents.

### Q: What happens to data during retries?

**A**: If the initial LLM request fails, the system may retry with a modified prompt. Each retry sends the text content again.

---

## 📞 Additional Resources

- **Ollama Privacy**: <https://ollama.com/privacy>
- **OpenAI Data Usage**: <https://platform.openai.com/docs/data-usage>
- **DeepInfra Privacy**: <https://deepinfra.com/privacy>
- **LocalAI Security**: <https://localai.io/security/>

---

## 📝 Changelog

| Date | Change |
| :--------- | :----- |
| 2025-01-26 | Initial privacy guide |

---

| 2026-03-14 | Updated to include platformdirs and new CLI commands |
| 2026-04-09 | Updated for Vision Mode data sharing transparency, uv integration, and .env security |
