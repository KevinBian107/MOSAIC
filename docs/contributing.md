# Contributing Guide

## Development Workflow

### 1. Create a Feature Branch

Never work directly on `main`. Always create a scoped feature branch:

```bash
# For new features
git checkout -b feature/my-new-feature

# For bug fixes
git checkout -b fix/bug-description
```

### 2. Make Your Changes

Follow these guidelines:

**Code Style:**
- Use `ruff format` for formatting (max line length: 88)
- Use Google-style docstrings with "Attributes" in class docstrings
- Always include type hints for function arguments and return types
- Import order: standard library, third-party, local (enforced by ruff)

**Example:**
```python
def compute_metric(
    graphs: list[Data],
    reference: list[Data],
    sigma: float = 1.0,
) -> dict[str, float]:
    """Compute evaluation metrics.

    Args:
        graphs: List of generated graphs.
        reference: List of reference graphs.
        sigma: Kernel bandwidth.

    Returns:
        Dictionary mapping metric names to values.
    """
    ...
```

### 3. Write Tests

- Place tests in `tests/` mirroring the source structure
- Use fixtures from `tests/fixtures/` when possible
- Create minimal synthetic data rather than large test files
- Use `tmp_path` for any file I/O
- Use pytest function-style (not unittest classes)

```python
# Good: pytest function style
def test_tokenize_triangle(triangle_graph: Data) -> None:
    """Test tokenization of a triangle graph."""
    tokenizer = SENTTokenizer(seed=42)
    tokenizer.set_num_nodes(10)
    tokens = tokenizer.tokenize(triangle_graph)
    assert tokens[0] == tokenizer.sos
```

### 4. Run Tests and Linting

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run linting
ruff check src/ tests/

# Format code
ruff format src/ tests/
```

### 5. Commit Your Changes

Create focused commits with clear messages:

```bash
# Stage specific files/hunks
git add -p

# Commit with descriptive message
git commit -m "Add triangle detection to MotifDetector

- Implement efficient triangle enumeration
- Add tests for triangle detection
- Update docstrings"
```

**Important:** Never commit all changes at once. Carefully analyze and create specific commits.

### 6. Create a Pull Request

```bash
# Push your branch
git push -u origin feature/my-new-feature

# Create PR using GitHub CLI
gh pr create --title "Add triangle detection" --body "..."
```

PR checklist:
- [ ] Tests pass
- [ ] Linting passes
- [ ] Documentation updated (if applicable)
- [ ] Changelog updated (if applicable)

### 7. Wait for CI

After creating the PR, wait for CI to complete:

```bash
# Check CI status
gh pr checks
```

## Project Structure

```
motif-preserving-tokenization/
├── src/
│   ├── data/           # Data loading and generation
│   │   ├── motif.py    # Motif detection
│   │   ├── synthetic.py # Synthetic generators
│   │   └── datamodule.py # PyTorch Lightning datamodule
│   ├── tokenizers/     # Graph tokenization
│   │   ├── base.py     # Abstract interface
│   │   └── sent.py     # SENT tokenizer
│   ├── models/         # Neural network models
│   │   └── transformer.py # HuggingFace wrapper
│   └── evaluation/     # Metrics
│       ├── dist_helper.py # Distance computation
│       ├── metrics.py     # Standard graph metrics
│       └── motif_metrics.py # Motif-specific metrics
├── configs/            # Hydra configs
├── scripts/            # Entry points
├── tests/              # Test suite
│   ├── fixtures/       # Shared fixtures
│   └── test_*.py       # Test modules
└── docs/               # Documentation
```

## Adding New Components

### New Tokenizer

1. Implement the `Tokenizer` interface in `src/tokenizers/`
2. Add exports to `src/tokenizers/__init__.py`
3. Add tests in `tests/test_tokenizer.py`
4. Document in `docs/api/tokenizers.md`

### New Metric

1. Implement in `src/evaluation/`
2. Add exports to `src/evaluation/__init__.py`
3. Add tests in `tests/test_metrics.py`
4. Document in `docs/api/evaluation.md`

### New Graph Generator

1. Add to `SyntheticGraphGenerator.GENERATORS` in `src/data/synthetic.py`
2. Add tests in `tests/test_synthetic.py`

## Testing Rules

1. Use existing fixtures when possible
2. Create minimal synthetic data for new tests
3. Use `tmp_path` for file I/O
4. Write focused tests (one assertion per test when feasible)
5. Never mock internal code - if it's hard to test, refactor it

## Questions?

Open an issue on GitHub for:
- Bug reports
- Feature requests
- Questions about the codebase
