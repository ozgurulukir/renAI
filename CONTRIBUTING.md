# Contributing to renAI

Thank you for your interest in contributing to renAI! We welcome all contributions, from bug reports and documentation updates to new features and architectural improvements.

## 🛠️ Development Environment

This project uses [Astral uv](https://docs.astral.sh/uv/) for dependency management and project orchestration.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ozgurulukir/renAI.git
    cd renAI
    ```

2.  **Synchronize dependencies:**
    ```bash
    uv sync --all-extras
    ```

3.  **Activate the virtual environment:**
    ```bash
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

## 🧪 Running Tests

We value stability. Please ensure all tests pass before submitting a pull request.

```bash
uv run pytest tests/ -v
```

To run tests with coverage:
```bash
uv run pytest tests/ --cov=renai
```

## 🧹 Code Quality

We use `ruff` for linting and formatting, and `mypy` for type checking.

```bash
# Linting & Formatting
uv run ruff check .
uv run ruff format .

# Type Checking
uv run mypy src/renai
```

## 🚀 Pull Request Process

1.  **Create a branch** for your feature or bugfix: `git checkout -b feature/your-feature-name`.
2.  **Commit your changes** following clear, descriptive messages.
3.  **Push to your fork** and submit a Pull Request.
4.  **Ensure CI passes**. Our GitHub Actions will automatically run tests and linting.

## 📝 Documentation

If you add a new feature, please update:
- `README.md` (and `okubeni.md` for Turkish localization if possible).
- `CHANGELOG.md` following the [Keep a Changelog](https://keepachangelog.com/) format.

---

*"Good code is a love letter to your future self. Clarity and kindness come before cleverness."*
