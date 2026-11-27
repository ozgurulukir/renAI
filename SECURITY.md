# Security Policy

## Supported Versions

Only the latest version of renAI is supported for security updates.

| Version | Supported          |
| ------- | ------------------ |
| 2.4.x   | :white_check_mark: |
| < 2.3.0 | :x:                |

## Reporting a Vulnerability

We take the security of renAI seriously. If you believe you have found a security vulnerability, please do NOT open a public issue. Instead, please report it via one of the following methods:

1. **Private Email**: Send a detailed report to `ozgur.ulukir@gmail.com`.
2. **GitHub Private Vulnerability Reporting**: Use the "Report a vulnerability" button on the GitHub Security tab if enabled.

### What to include in your report

- A descriptive title for the vulnerability.
- A summary of the vulnerability and its potential impact.
- Steps to reproduce the issue (links to code, proof of concept scripts, etc.).
- Your preferred name/handle for credit in the changelog (if desired).

We aim to acknowledge your report within 48 hours and provide a fix or mitigation strategy within 7 days for CRITICAL vulnerabilities.

## API Key Handling

### Where API keys may live

renAI reads provider API keys from, in order of precedence (highest first):

1. **Environment variables**: `<PROVIDER>_API_KEY` (e.g. `DEEPINFRA_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, `CUSTOM_API_KEY`).
2. **`secrets.toml`** in your user config directory (`<provider>_api_key` fields, e.g. `deepinfra_api_key`, `groq_api_key`, `custom_api_key`), written by the `renai init` wizard.
3. **`settings.toml`** (`[providers.<name>].api_key` entries) for registered providers.

- `CUSTOM_API_KEY` has a default sentinel value of `not-required`, intended for keyless local servers (Ollama, LM Studio, etc.). If your custom endpoint requires authentication, always set a real key — the sentinel is **not** a credential.

### Never commit keys

- Do **not** put API keys in `settings.toml`, `settings.default.toml`, or any file inside the repository.
- `secrets.toml` lives in your user config directory (outside the repo) and must never be committed, shared, or copied into dotfiles you sync publicly.
- If a key is accidentally committed, revoke and rotate it immediately at the provider, then purge it from git history (e.g., with `git filter-repo` or BFG Repo-Cleaner).
