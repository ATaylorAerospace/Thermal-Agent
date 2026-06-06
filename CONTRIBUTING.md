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

# Install dependencies (core: agent + tests)
pip install -r requirements.txt
pip install pytest

# Optional: heavy open-weight fine-tuning stack (GPU host only)
# pip install -r requirements-finetune.txt

# Configure environment
cp .env.example .env
# Edit .env with your AWS credentials (optional — only needed for Bedrock features)
```

> The core `requirements.txt` is enough to run the agent and the full test
> suite. The QLoRA + GGUF fine-tuning modules (`src/finetune.py`,
> `src/quantize.py`) need `requirements-finetune.txt` and a GPU; their heavy
> dependencies are imported lazily so the rest of the project stays light.

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
| `src/` | Core library code (agent, backends, tools, data store, simulator, classifier, fine-tuning) |
| `app/` | Streamlit web application |
| `tests/` | Pytest test suites |
| `config/` | YAML configuration files (agent + fine-tuning) |
| `notebooks/` | Jupyter notebooks for EDA, the agent, and fine-tuning |
| `scripts/` | Shell scripts for building artifacts and fine-tuning |
| `results/` | Model, index & adapter artifacts (gitignored) |

## Questions?

Open an issue or reach out to the maintainer.
