# Contributing to AI-Powered Fashion Design Assistant

First off, thank you for considering contributing! This project is a showcase of AI, generative models, and fashion technology. Every contribution helps make it better.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)
- [Testing](#testing)

---

## Code of Conduct

This project and everyone participating in it is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

---

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the [issue list](https://github.com/Jemin29/AI-Fashion-Agent/issues) to avoid duplicates.

When you are creating a bug report, please include:
- **Clear, descriptive title**
- **Exact steps to reproduce** the problem
- **Expected behavior** and what actually happened
- **Python version**, OS, and relevant package versions
- **Code snippets** or error tracebacks

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. Include:
- Clear, descriptive title
- Step-by-step description of the enhancement
- Current behavior vs expected behavior
- Why this enhancement would be useful

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. Ensure the test suite passes (`pytest tests/ --cov=src`)
4. Make sure your code follows the style guide (PEP8)
5. Update documentation if needed
6. Issue that pull request!

---

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Jemin29/AI-Fashion-Agent.git
cd AI-Fashion-Agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/ --cov=src --cov-report=html

# Run the Gradio app
python apps/gradio_app/app.py

# Run the FastAPI server
uvicorn apps.api.main:app --reload
```

---

## Pull Request Process

1. Update the `README.md` with details of interface changes
2. Update `CHANGELOG.md` with your changes under "Unreleased"
3. The PR will be merged once you have the sign-off of at least one maintainer
4. Make sure CI passes (GitHub Actions)

### Branch Naming Conventions

- `feature/your-feature-name` — New features
- `fix/bug-description` — Bug fixes
- `docs/what-you-documented` — Documentation only
- `refactor/what-you-refactored` — Code refactoring
- `test/what-you-tested` — Test additions

---

## Style Guide

### Python

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints for all function signatures
- Write docstrings for all public functions and classes (Google style)
- Maximum line length: 120 characters
- Use `loguru` for logging (not `print` statements)

### Docstring Format

```python
def recommend_styles(preferences: dict, n: int = 5) -> List[str]:
    """Return top-N style recommendations for given user preferences.

    Args:
        preferences: Dict containing 'favorite_style' and 'favorite_color'.
        n: Number of recommendations to return.

    Returns:
        List of recommended style names sorted by relevance score.

    Raises:
        ValueError: If preferences dict is missing required keys.
    """
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: add style similarity search`
- `fix: correct ChromaDB collection initialization`
- `docs: update RAG pipeline documentation`
- `test: add unit tests for TrendForecaster`
- `refactor: extract embedding logic to base class`
- `ci: add coverage enforcement step`

---

## Testing

All new code must include tests:

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific module tests
pytest tests/test_fashion_retriever.py -v

# Enforce 80% threshold
pytest tests/ --cov=src --cov-fail-under=80
```

**Coverage target**: ≥80% for all new modules.

### Test Structure

```
tests/
├── conftest.py          # Shared fixtures
├── test_<module>.py     # Unit tests per module
└── integration_*.py     # Integration tests
```

---

## Questions?

Open a [GitHub Discussion](https://github.com/Jemin29/AI-Fashion-Agent/discussions) or file an issue.

Thank you for your contribution! 🎨
