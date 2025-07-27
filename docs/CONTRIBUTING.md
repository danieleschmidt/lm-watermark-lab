# Contributing to LM Watermark Lab

We welcome contributions to LM Watermark Lab! This guide will help you understand our development process and how to contribute effectively.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Contribution Guidelines](#contribution-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Review Process](#review-process)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- Docker (optional, for containerized development)
- Basic understanding of watermarking and NLP concepts

### Development Setup

1. **Fork and Clone**
```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/lm-watermark-lab.git
cd lm-watermark-lab
```

2. **Set up Development Environment**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

3. **Verify Setup**
```bash
# Run tests to ensure everything works
pytest tests/unit/

# Run linting
pre-commit run --all-files
```

### Development with Docker

```bash
# Build development container
docker-compose up --build dev

# Run tests in container
docker-compose exec dev pytest

# Access development shell
docker-compose exec dev bash
```

### Using DevContainer

If you use VS Code, open the project in a DevContainer for a fully configured development environment:

1. Install the "Remote - Containers" extension
2. Open the project in VS Code
3. Click "Reopen in Container" when prompted

## Development Workflow

### Branching Strategy

We use a modified Git Flow branching model:

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Feature development branches
- `bugfix/*`: Bug fix branches
- `release/*`: Release preparation branches
- `hotfix/*`: Critical production fixes

### Creating a Feature Branch

```bash
# Start from develop branch
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes
# Commit your changes
git add .
git commit -m "feat: add new watermarking algorithm"

# Push branch
git push origin feature/your-feature-name
```

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes

**Examples:**
```bash
feat(watermarking): add support for LLaMA models
fix(detection): resolve false positive issue in statistical test
docs(api): update authentication documentation
test(integration): add end-to-end API tests
```

## Contribution Guidelines

### What We Welcome

**High Priority:**
- New watermarking algorithms
- Improved detection methods
- Performance optimizations
- Bug fixes
- Documentation improvements
- Test coverage improvements

**Medium Priority:**
- New attack methods
- Additional evaluation metrics
- UI/UX improvements
- Integration with new models

**Please Discuss First:**
- Major architectural changes
- New dependencies
- Breaking API changes
- Large feature additions

### Code Style

We use automated code formatting and linting:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Style checking
- **mypy**: Type checking
- **bandit**: Security linting

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check style
flake8 src/ tests/
mypy src/

# Security check
bandit -r src/
```

### Adding New Watermarking Methods

When adding a new watermarking method:

1. **Create the algorithm class**:
```python
# src/watermark_lab/methods/your_method.py
from watermark_lab.methods.base import BaseWatermark

class YourMethodWatermark(BaseWatermark):
    """Your watermarking method implementation."""
    
    def __init__(self, model: str, **kwargs):
        super().__init__(model)
        self.configure(**kwargs)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate watermarked text."""
        # Implementation here
        pass
    
    def get_config(self) -> dict:
        """Return watermark configuration."""
        # Implementation here
        pass
```

2. **Register the method**:
```python
# src/watermark_lab/methods/__init__.py
from .your_method import YourMethodWatermark

WATERMARK_METHODS = {
    "your_method": YourMethodWatermark,
    # ... other methods
}
```

3. **Add tests**:
```python
# tests/unit/test_your_method.py
import pytest
from watermark_lab.methods import YourMethodWatermark

def test_your_method_generation():
    """Test watermark generation."""
    watermark = YourMethodWatermark("mock-model")
    result = watermark.generate("test prompt")
    assert isinstance(result, str)
    assert len(result) > 0
```

4. **Add documentation**:
```python
# Add docstrings and update README
# Include paper reference and parameter descriptions
```

### Adding Detection Methods

When adding detection methods:

1. **Create detector class**:
```python
# src/watermark_lab/detection/your_detector.py
from watermark_lab.detection.base import BaseDetector

class YourDetector(BaseDetector):
    """Your detection method."""
    
    def detect(self, text: str, config: dict) -> dict:
        """Detect watermark in text."""
        # Implementation here
        pass
```

2. **Add comprehensive tests**:
```python
# Test with known watermarked/clean texts
# Test edge cases and error conditions
# Test performance characteristics
```

### Performance Considerations

- **Efficiency**: Optimize for speed and memory usage
- **Scalability**: Consider batch processing capabilities
- **Caching**: Implement appropriate caching strategies
- **GPU Support**: Add CUDA support where beneficial

```python
# Example performance optimization
@lru_cache(maxsize=128)
def compute_expensive_operation(key: str) -> Result:
    """Cache expensive computations."""
    pass

# Batch processing support
def process_batch(self, texts: List[str]) -> List[Result]:
    """Process multiple texts efficiently."""
    # Use vectorized operations when possible
    pass
```

## Testing

### Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── unit/                    # Unit tests
│   ├── test_watermarking.py
│   ├── test_detection.py
│   └── test_*.py
├── integration/             # Integration tests
│   ├── test_api.py
│   └── test_workflows.py
├── e2e/                     # End-to-end tests
│   └── test_complete_flow.py
├── performance/             # Performance tests
│   └── test_benchmarks.py
└── smoke/                   # Smoke tests
    └── test_health.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only
pytest -m "not slow"                 # Skip slow tests
pytest -m "gpu"                      # GPU tests only

# Run with coverage
pytest --cov=src/watermark_lab --cov-report=html

# Run performance tests
pytest tests/performance/ --benchmark-only
```

### Writing Tests

**Unit Test Example:**
```python
import pytest
from unittest.mock import MagicMock, patch
from watermark_lab.methods import KirchenbauerWatermark

class TestKirchenbauerWatermark:
    """Test Kirchenbauer watermarking method."""
    
    @pytest.fixture
    def watermark(self, mock_model):
        """Create watermark instance."""
        return KirchenbauerWatermark(
            model="mock-model",
            gamma=0.25,
            delta=2.0,
            seed=42
        )
    
    def test_generate_returns_string(self, watermark):
        """Test that generate returns a string."""
        result = watermark.generate("test prompt")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_generate_with_parameters(self, watermark):
        """Test generation with custom parameters."""
        result = watermark.generate(
            "test prompt",
            max_length=50,
            temperature=0.5
        )
        assert isinstance(result, str)
        # Add more specific assertions
    
    @patch('watermark_lab.models.ModelManager.load_model')
    def test_model_loading(self, mock_load, watermark):
        """Test model loading behavior."""
        watermark.generate("test")
        mock_load.assert_called_once()
```

**Integration Test Example:**
```python
def test_complete_watermarking_workflow(api_client):
    """Test complete watermarking workflow via API."""
    # Generate watermarked text
    generate_response = api_client.post("/api/v1/generate", json={
        "method": "kirchenbauer",
        "model": "gpt2-medium",
        "prompt": "test prompt"
    })
    assert generate_response.status_code == 200
    
    generated_data = generate_response.json()["data"]
    
    # Detect watermark
    detect_response = api_client.post("/api/v1/detect", json={
        "text": generated_data["text"],
        "watermark_config": generated_data["watermark_config"]
    })
    assert detect_response.status_code == 200
    
    detection_data = detect_response.json()["data"]
    assert detection_data["is_watermarked"] is True
```

### Test Fixtures

We provide comprehensive test fixtures in `conftest.py`:

```python
@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return {
        "short": "Brief text.",
        "medium": "Medium length text with multiple sentences.",
        "long": "Very long text with many sentences and paragraphs..."
    }

@pytest.fixture
def watermark_configs():
    """Sample watermark configurations."""
    return {
        "kirchenbauer": {"gamma": 0.25, "delta": 2.0},
        "aaronson": {"key": "test_key"},
    }
```

## Documentation

### Documentation Types

1. **Code Documentation**: Docstrings for all public APIs
2. **User Guides**: How-to guides and tutorials
3. **API Documentation**: Complete API reference
4. **Architecture Documentation**: System design and decisions

### Writing Docstrings

Follow the [NumPy docstring convention](https://numpydoc.readthedocs.io/):

```python
def detect_watermark(text: str, config: dict, return_details: bool = False) -> dict:
    """Detect watermark in given text.
    
    This function analyzes the input text to determine if it contains
    a watermark based on the provided configuration.
    
    Parameters
    ----------
    text : str
        The text to analyze for watermarks.
    config : dict
        Watermark configuration containing method parameters.
    return_details : bool, optional
        Whether to return detailed analysis results, by default False.
    
    Returns
    -------
    dict
        Detection results with the following keys:
        - is_watermarked : bool
            Whether watermark was detected.
        - confidence : float
            Confidence score between 0 and 1.
        - p_value : float
            Statistical p-value of detection.
    
    Raises
    ------
    ValueError
        If text is empty or config is invalid.
    ModelNotFoundError
        If specified model is not available.
    
    Examples
    --------
    >>> config = {"method": "kirchenbauer", "gamma": 0.25}
    >>> result = detect_watermark("Sample text", config)
    >>> print(result["is_watermarked"])
    True
    
    Notes
    -----
    Detection accuracy depends on text length and watermark strength.
    Minimum recommended text length is 50 tokens.
    
    References
    ----------
    .. [1] Kirchenbauer, J. et al. "A Watermark for Large Language Models."
           arXiv preprint arXiv:2301.10226 (2023).
    """
    # Implementation here
    pass
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build HTML documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## Review Process

### Pull Request Guidelines

1. **Before Submitting**:
   - Ensure all tests pass
   - Update documentation
   - Add or update tests for new functionality
   - Run pre-commit hooks

2. **PR Description Template**:
```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or documented)
```

3. **Review Process**:
   - Automated checks must pass
   - At least one maintainer review required
   - All feedback addressed
   - Final approval from code owner

### Review Criteria

**Code Quality:**
- Follows project conventions
- Properly documented
- Efficient implementation
- Error handling

**Testing:**
- Adequate test coverage
- Tests are meaningful
- Edge cases covered
- Performance considerations

**Documentation:**
- Clear and comprehensive
- Examples provided
- API changes documented
- User-facing changes noted

## Community

### Getting Help

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Discord/Slack**: Real-time community chat
- **Email**: security@watermark-lab.com for security issues

### Contribution Recognition

We recognize contributors through:

- GitHub contributor listings
- Release notes mentions
- Annual contributor highlights
- Swag for significant contributions

### Maintainer Guidelines

For maintainers and frequent contributors:

1. **Issue Triage**: Label and prioritize issues
2. **Code Review**: Provide constructive feedback
3. **Release Management**: Help with version releases
4. **Community Support**: Help answer questions

### Becoming a Maintainer

Regular contributors may be invited to become maintainers based on:

- Consistent, quality contributions
- Community involvement
- Technical expertise
- Commitment to project values

Thank you for contributing to LM Watermark Lab! 🚀