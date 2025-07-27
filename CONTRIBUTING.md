# Contributing to LM Watermark Lab

Thank you for your interest in contributing to LM Watermark Lab! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to conduct@terragonlabs.com.

## Getting Started

### Prerequisites

- Python 3.9+ (3.11 recommended)
- Git
- Docker and Docker Compose (for full development environment)
- Basic understanding of machine learning and NLP concepts

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/lm-watermark-lab.git
   cd lm-watermark-lab
   ```

2. **Set up your development environment**
   ```bash
   make dev-setup
   ```

3. **Verify your setup**
   ```bash
   make test-fast
   python scripts/health_check.py
   ```

For detailed setup instructions, see our [Development Guide](docs/DEVELOPMENT.md).

## Development Process

### 1. Choose What to Work On

- **Good First Issues**: Look for issues labeled `good-first-issue`
- **Help Wanted**: Issues labeled `help-wanted` need community help
- **Feature Requests**: Propose new features via GitHub issues
- **Bug Reports**: Fix existing bugs

### 2. Create a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/feature-name` - New features
- `bugfix/bug-description` - Bug fixes
- `docs/documentation-update` - Documentation changes
- `refactor/component-name` - Code refactoring

### 3. Make Your Changes

- Write clean, readable code
- Follow our coding standards
- Add tests for new functionality
- Update documentation as needed
- Commit with clear, descriptive messages

### 4. Test Your Changes

```bash
# Run the full test suite
make test

# Check code quality
make quality

# Verify with different Python versions (if applicable)
tox
```

## Pull Request Process

### Before Creating a PR

1. **Rebase on latest develop**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout your-feature-branch
   git rebase develop
   ```

2. **Run quality checks**
   ```bash
   make quality
   make test
   ```

3. **Update documentation**
   - Update relevant documentation
   - Add docstrings to new functions/classes
   - Update API documentation if needed

### Creating the PR

1. **Push your branch**
   ```bash
   git push origin your-feature-branch
   ```

2. **Create pull request**
   - Use the GitHub web interface
   - Fill out the PR template completely
   - Link to related issues
   - Add appropriate labels

### PR Requirements

- [ ] All tests pass
- [ ] Code coverage maintained or improved
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Follows coding standards
- [ ] Includes appropriate tests
- [ ] Has clear commit messages

### Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: Team members review your code
3. **Feedback**: Address any requested changes
4. **Approval**: PR approved by maintainers
5. **Merge**: Merged into develop branch

## Issue Guidelines

### Reporting Bugs

Use the bug report template and include:

- **Environment**: Python version, OS, dependencies
- **Steps to Reproduce**: Clear, step-by-step instructions
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Logs/Screenshots**: Any relevant output

### Requesting Features

Use the feature request template and include:

- **Use Case**: Why is this feature needed?
- **Proposed Solution**: How should it work?
- **Alternatives**: Other approaches considered
- **Examples**: Similar features in other tools

### Security Issues

**Do not create public issues for security vulnerabilities!**
See our [Security Policy](SECURITY.md) for reporting procedures.

## Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with these tools:

```bash
# Auto-format code
black src tests

# Sort imports
isort src tests

# Lint code
flake8 src tests

# Type checking
mypy src
```

### Code Organization

```python
# Good: Clear imports and structure
from typing import Dict, List, Optional
import logging

from watermark_lab.base import BaseWatermark
from watermark_lab.utils import validate_input

logger = logging.getLogger(__name__)

class MyWatermark(BaseWatermark):
    """Clear docstring explaining the class."""
    
    def __init__(self, model: str, **kwargs) -> None:
        """Initialize with type hints and docstring."""
        super().__init__(model)
        self.config = self._validate_config(kwargs)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate with proper error handling."""
        try:
            return self._generate_impl(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
```

### Documentation Standards

```python
def detect_watermark(text: str, config: Dict[str, Any]) -> DetectionResult:
    """Detect watermark in the given text.
    
    Args:
        text: Input text to analyze
        config: Watermark configuration parameters
        
    Returns:
        DetectionResult with is_watermarked, confidence, and details
        
    Raises:
        ValueError: If text is empty or config is invalid
        
    Example:
        >>> result = detect_watermark("sample text", {"method": "kirchenbauer"})
        >>> print(result.is_watermarked)
        True
    """
```

## Testing Requirements

### Test Categories

1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Verify performance requirements

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

class TestMyWatermark:
    """Test class for MyWatermark."""
    
    @pytest.fixture
    def watermark(self):
        """Create test watermark instance."""
        return MyWatermark("test-model", param=42)
    
    def test_initialization(self, watermark):
        """Test proper initialization."""
        assert watermark.model == "test-model"
        assert watermark.config["param"] == 42
    
    @pytest.mark.parametrize("prompt,expected", [
        ("short", True),
        ("", False),
        ("very long prompt" * 100, True),
    ])
    def test_generate_various_inputs(self, watermark, prompt, expected):
        """Test generation with various inputs."""
        if expected:
            result = watermark.generate(prompt)
            assert isinstance(result, str)
        else:
            with pytest.raises(ValueError):
                watermark.generate(prompt)
    
    @patch('watermark_lab.models.load_model')
    def test_with_mocked_model(self, mock_load, watermark):
        """Test with mocked dependencies."""
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        result = watermark.generate("test")
        mock_load.assert_called_once()
```

### Test Coverage

- **Minimum**: 80% line coverage
- **Target**: 90% line coverage
- **Critical paths**: 100% coverage for security-sensitive code

```bash
# Run with coverage
pytest --cov=src/watermark_lab --cov-report=html tests/

# View coverage report
open htmlcov/index.html
```

## Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings and inline comments
2. **API Documentation**: Auto-generated from docstrings
3. **User Guides**: How-to guides for users
4. **Developer Guides**: Technical documentation
5. **Tutorials**: Step-by-step learning materials

### Writing Guidelines

- **Clear and Concise**: Use simple language
- **Examples**: Include code examples
- **Up-to-Date**: Keep documentation current
- **Searchable**: Use good headings and structure

### Building Documentation

```bash
# Build documentation locally
cd docs
make html

# Serve documentation
make serve
```

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Discord**: Real-time chat with the community
- **Email**: dev@terragonlabs.com for direct contact

### Getting Help

1. **Search existing issues** before creating new ones
2. **Check documentation** for common questions
3. **Ask in discussions** for general help
4. **Join Discord** for real-time assistance

### Recognition

Contributors are recognized in several ways:

- **Contributors file**: Listed in CONTRIBUTORS.md
- **Release notes**: Major contributions highlighted
- **Hall of fame**: Special recognition for significant contributions
- **Swag**: Stickers and other goodies for active contributors

## Contributor License Agreement

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License). You certify that:

1. You have the right to submit the contribution
2. The contribution is your original work or properly attributed
3. You understand the contribution will be publicly available

## Questions?

If you have questions about contributing:

1. Check the [Development Guide](docs/DEVELOPMENT.md)
2. Search [GitHub Issues](https://github.com/terragon-labs/lm-watermark-lab/issues)
3. Ask in [GitHub Discussions](https://github.com/terragon-labs/lm-watermark-lab/discussions)
4. Join our [Discord server](https://discord.gg/terragon-labs)
5. Email us at dev@terragonlabs.com

Thank you for contributing to LM Watermark Lab! 🚀