# Contributing to Deep-Space Photonics Thermal Advisor

Thank you for your interest in contributing! This document covers setup,
testing, and pull request guidelines.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ATaylorAerospace/Thermal-Agent.git
cd Thermal-Agent

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest

# Configure environment
cp .env.example .env
# Edit .env with your AWS credentials (optional — only needed for Bedrock features)
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only simulator tests
pytest tests/test_simulator.py -v

# Run only classifier tests
pytest tests/test_classifier.py -v
```

## Pull Request Checklist

- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] New code includes docstrings with Args/Returns/Raises sections
- [ ] No unused imports (check with `pylint --disable=all --enable=W0611 src/`)
- [ ] Commit messages follow conventional format (e.g., `feat:`, `fix:`, `refactor:`, `docs:`)

## Code Style

- Follow PEP 8
- Use type hints for public method signatures
- All imports at the top of the file (no inline imports except for optional dependencies)
- Docstrings in Google style

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `src/` | Core library code |
| `app/` | Streamlit web application |
| `tests/` | Pytest test suites |
| `config/` | YAML configuration files |
| `notebooks/` | Jupyter notebooks for EDA and fine-tuning |
| `scripts/` | Shell scripts for pipeline automation |
| `results/` | Model artifacts and output files (gitignored) |

## Questions?

Open an issue or reach out to the maintainer.
