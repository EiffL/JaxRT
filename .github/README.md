# GitHub Actions CI/CD

This directory contains GitHub Actions workflows for automated testing, code quality checks, and performance benchmarking.

## Workflows

### 🧪 `test.yml` - Automated Testing
- **Triggers**: Push to main/develop, Pull requests
- **Matrix**: Ubuntu/macOS × Python 3.9/3.10/3.11
- **Features**:
  - Installs JAX and jax-cosmo dependencies
  - Runs pytest with coverage reporting
  - Attempts LensTools installation on Ubuntu for comparison tests
  - Uploads coverage to Codecov
  - Tests basic demo script functionality

### 🔍 `lint.yml` - Code Quality
- **Triggers**: Push to main/develop, Pull requests  
- **Checks**:
  - Code formatting with Black
  - Import sorting with isort
  - Linting with flake8
  - Type checking with mypy (non-blocking)

### ⚡ `benchmark.yml` - Performance Benchmarks
- **Triggers**: Manual dispatch, commits with `[benchmark]` tag
- **Features**:
  - Runs performance tests across different resolutions
  - Measures time per ray and total computation time
  - Reports convergence statistics
  - Uploads benchmark artifacts

### 📚 `docs.yml` - Documentation Checks
- **Triggers**: Push to main/develop, Pull requests
- **Validates**:
  - Presence of required documentation files
  - README completeness (key sections)
  - Package metadata and exports
  - Example code in README

## Usage

### Running Tests Locally

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest

# Run without LensTools comparison
pytest -m "not lenstools"

# Run with coverage
pytest --cov=jaxrt --cov-report=html
```

### Code Quality Checks

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
black jaxrt tests
isort jaxrt tests

# Lint code
flake8 jaxrt tests

# Type check
mypy jaxrt --ignore-missing-imports
```

### Manual Benchmark

```bash
# Trigger benchmark workflow
gh workflow run benchmark.yml

# Or add [benchmark] to commit message
git commit -m "Optimize interpolation [benchmark]"
```

## CI/CD Strategy

1. **Every PR**: Full test matrix + code quality checks + docs validation
2. **Main branch**: All checks + optional performance benchmarks
3. **Release tags**: Full test suite + benchmark + documentation build

## Dependencies in CI

- **Core**: JAX, jax-cosmo, numpy, scipy, astropy
- **Testing**: pytest, pytest-cov
- **Quality**: black, isort, flake8, mypy
- **Optional**: LensTools (Ubuntu only, for comparison tests)
- **Benchmarks**: matplotlib (for plots)

## Badges

Add these badges to your main README:

```markdown
[![Tests](https://github.com/EiffL/JaxRT/workflows/Tests/badge.svg)](https://github.com/EiffL/JaxRT/actions/workflows/test.yml)
[![Code Quality](https://github.com/EiffL/JaxRT/workflows/Code%20Quality/badge.svg)](https://github.com/EiffL/JaxRT/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/EiffL/JaxRT/branch/main/graph/badge.svg)](https://codecov.io/gh/EiffL/JaxRT)
```