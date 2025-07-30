# Mutation Testing Workflow Documentation

## Overview

Mutation testing is a advanced testing technique that evaluates the quality of your test suite by introducing small changes (mutations) to the source code and checking if your tests can detect these changes.

## Implementation Guide

### Step 1: Install Required Tools

```bash
pip install mutmut coverage pytest
```

### Step 2: Basic Mutation Testing

```bash
# Run mutation testing on core modules
mutmut run --paths-to-mutate src/watermark_lab/

# Generate HTML report
mutmut html

# View results
mutmut show
```

### Step 3: Advanced Configuration

Create `.mutmut-config.yml`:

```yaml
paths_to_mutate: src/watermark_lab/
tests_dir: tests/
runner: python -m pytest -x
use_coverage: true
coverage_data: .coverage
```

### Step 4: GitHub Actions Integration

```yaml
name: Mutation Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  mutation-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.9"
    
    - name: Install dependencies
      run: |
        pip install -e ".[test]"
        pip install mutmut coverage
    
    - name: Run baseline tests
      run: pytest tests/ -v
    
    - name: Run mutation testing
      run: |
        mutmut run --paths-to-mutate src/watermark_lab/
        mutmut html
    
    - name: Check mutation score
      run: |
        SCORE=$(mutmut results | grep -o "killed: [0-9]*" | cut -d' ' -f2)
        TOTAL=$(mutmut results | grep -o "total: [0-9]*" | cut -d' ' -f2)
        PERCENTAGE=$((SCORE * 100 / TOTAL))
        echo "Mutation Score: $PERCENTAGE%"
        if [ $PERCENTAGE -lt 80 ]; then
          echo "Mutation score below 80% threshold"
          exit 1
        fi
    
    - name: Upload reports
      uses: actions/upload-artifact@v3
      with:
        name: mutation-reports
        path: html/
```

## Best Practices

### 1. Target Critical Code Paths

Focus mutation testing on:
- Core watermarking algorithms
- Detection logic
- Security-critical functions
- Error handling code

### 2. Interpret Results

- **Killed**: Mutant detected by tests (good)
- **Survived**: Mutant not detected (test gap)
- **Timeout**: Test took too long (possible infinite loop)
- **Suspicious**: Unexpected behavior

### 3. Improve Test Coverage

When mutants survive:
1. Add specific test cases for that code path
2. Improve assertion specificity
3. Test edge cases and error conditions
4. Add property-based tests

### 4. Performance Optimization

```bash
# Use coverage to focus on tested code only
coverage run -m pytest
mutmut run --use-coverage

# Parallel execution (experimental)
mutmut run --processes 4

# Target specific files
mutmut run --paths-to-mutate src/watermark_lab/core/detector.py
```

## Expected Mutation Score Thresholds

| Component | Minimum Score | Target Score |
|-----------|---------------|--------------|
| Core Algorithms | 85% | 95% |
| Detection Logic | 80% | 90% |
| API Endpoints | 70% | 85% |
| Utility Functions | 75% | 85% |
| Overall Project | 80% | 90% |

## Common Mutation Types

1. **Arithmetic**: `+` → `-`, `*` → `/`
2. **Comparison**: `>` → `>=`, `==` → `!=`
3. **Boolean**: `and` → `or`, `True` → `False`
4. **Constants**: `0` → `1`, `""` → `"X"`
5. **Statements**: Remove lines, change returns

## Troubleshooting

### High Number of Timeouts
- Reduce test timeout values
- Optimize slow tests
- Use `--timeout` parameter

### Low Mutation Score
- Add missing test cases
- Improve assertion quality
- Test error conditions
- Add boundary value tests

### Memory Issues
- Use `--processes 1` for single-threaded execution
- Increase system memory
- Target smaller code sections

## Integration with CI/CD

### Pre-commit Hook

```bash
# .pre-commit-config.yaml
- repo: local
  hooks:
  - id: mutation-test
    name: Mutation Testing
    entry: mutmut run --CI
    language: system
    files: \.py$
    stages: [pre-push]
```

### Quality Gates

Set mutation score requirements:
- Development: 70%+ 
- Staging: 80%+
- Production: 85%+

## Reporting and Analysis

### Generate Comprehensive Reports

```python
#!/usr/bin/env python3
import json
import subprocess

def analyze_mutation_results():
    # Get results
    result = subprocess.run(['mutmut', 'junitxml'], 
                          capture_output=True, text=True)
    
    # Parse and analyze
    # ... implementation details
    
    return {
        'score': mutation_score,
        'recommendations': recommendations,
        'critical_gaps': critical_gaps
    }
```

### Dashboard Integration

Integrate with Grafana for trend monitoring:
- Mutation score over time
- Test coverage correlation
- Failed mutant categories
- Performance impact analysis

## Resources

- [Mutmut Documentation](https://mutmut.readthedocs.io/)
- [Mutation Testing Best Practices](https://en.wikipedia.org/wiki/Mutation_testing)
- [Property-Based Testing with Hypothesis](https://hypothesis.readthedocs.io/)

## Maintenance

### Regular Tasks

1. **Weekly**: Review surviving mutants
2. **Monthly**: Update mutation test configuration
3. **Quarterly**: Analyze trends and adjust thresholds
4. **Per Release**: Ensure mutation score meets requirements