# Release Process

This document outlines the release process for the LM Watermark Lab project.

## Release Types

We follow [Semantic Versioning](https://semver.org/) (SemVer):
- **MAJOR** (X.0.0): Incompatible API changes
- **MINOR** (X.Y.0): New functionality, backwards compatible
- **PATCH** (X.Y.Z): Bug fixes, backwards compatible

## Release Schedule

- **Major releases**: Quarterly (every 3 months)
- **Minor releases**: Monthly
- **Patch releases**: As needed for critical bugs

## Pre-Release Checklist

### 1. Code Quality
- [ ] All tests pass (`pytest tests/`)
- [ ] Code coverage ≥ 80% (`pytest --cov=src/watermark_lab`)
- [ ] Linting passes (`ruff check src/ tests/`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Security scan passes (`bandit -r src/`)

### 2. Documentation
- [ ] CHANGELOG.md updated with new features/fixes
- [ ] README.md updated if needed
- [ ] API documentation generated (`sphinx-build -b html docs/ docs/_build/`)
- [ ] Performance benchmarks updated (if applicable)

### 3. Dependencies
- [ ] Dependencies are up to date (`pip-compile requirements.in`)
- [ ] Security vulnerabilities addressed (`safety check`)
- [ ] License compatibility verified

### 4. Testing
- [ ] All test suites pass
  - [ ] Unit tests
  - [ ] Integration tests
  - [ ] Performance tests
  - [ ] Contract tests
- [ ] Manual testing completed
- [ ] Smoke tests on production-like environment

## Release Process

### 1. Prepare Release Branch

```bash
# Start from clean main branch
git checkout main
git pull origin main

# Create release branch
git checkout -b release/v1.2.0

# Update version in pyproject.toml
sed -i 's/version = "1.1.0"/version = "1.2.0"/' pyproject.toml

# Update version in __init__.py
sed -i 's/__version__ = "1.1.0"/__version__ = "1.2.0"/' src/watermark_lab/__init__.py
```

### 2. Update CHANGELOG.md

```markdown
## [1.2.0] - 2024-02-15

### Added
- New MarkLLM integration for advanced watermarking
- Performance optimization for batch processing
- CLI command for benchmark comparison

### Changed
- Improved detection accuracy by 3%
- Updated dependency versions

### Fixed
- Memory leak in long-running detection tasks
- Inconsistent results with certain model configurations

### Security
- Updated cryptography dependency to address CVE-2024-XXXX
```

### 3. Run Pre-Release Tests

```bash
# Full test suite
pytest tests/ -v --cov=src/watermark_lab --cov-report=html

# Security scan
bandit -r src/ -f json -o security-report.json

# Performance benchmarks
python benchmarks/run_all.py --output benchmarks-v1.2.0.json

# Build and test package
python -m build
twine check dist/*
```

### 4. Create Release PR

```bash
# Commit changes
git add .
git commit -m "chore: prepare release v1.2.0"

# Push release branch
git push origin release/v1.2.0

# Create PR via GitHub CLI
gh pr create \
  --title "Release v1.2.0" \
  --body "$(cat CHANGELOG.md | grep -A 20 '## \[1.2.0\]')" \
  --label "release" \
  --assignee "@me"
```

### 5. Review and Merge

- [ ] Code review by at least 2 maintainers
- [ ] All CI checks pass
- [ ] Documentation review
- [ ] Security review (for major/minor releases)
- [ ] Performance impact assessment

### 6. Tag and Release

```bash
# After PR is merged, tag the release
git checkout main
git pull origin main

# Create annotated tag
git tag -a v1.2.0 -m "Release v1.2.0

### Added
- New MarkLLM integration
- Performance optimizations
- CLI benchmark commands

### Changed
- Improved detection accuracy
- Updated dependencies

### Fixed
- Memory leak fixes
- Configuration consistency

See CHANGELOG.md for full details."

# Push tag
git push origin v1.2.0
```

### 7. GitHub Release

Create GitHub release using the web interface or CLI:

```bash
gh release create v1.2.0 \
  --title "LM Watermark Lab v1.2.0" \
  --notes "$(cat CHANGELOG.md | grep -A 30 '## \[1.2.0\]')" \
  --target main \
  dist/*
```

### 8. PyPI Release

```bash
# Build distribution packages
python -m build

# Upload to PyPI (requires API token)
twine upload dist/*

# Verify upload
pip install lm-watermark-lab==1.2.0
python -c "import watermark_lab; print(watermark_lab.__version__)"
```

### 9. Docker Release

```bash
# Build Docker image
docker build -t terragon-labs/watermark-lab:1.2.0 .
docker build -t terragon-labs/watermark-lab:latest .

# Push to registry
docker push terragon-labs/watermark-lab:1.2.0
docker push terragon-labs/watermark-lab:latest
```

### 10. Post-Release Tasks

- [ ] Update project documentation links
- [ ] Announce release on community channels
- [ ] Update dependent projects/examples
- [ ] Monitor for issues in first 24 hours
- [ ] Create milestone for next release

## Hotfix Process

For critical bugs requiring immediate release:

### 1. Create Hotfix Branch

```bash
# Branch from main (or release tag)
git checkout main
git checkout -b hotfix/v1.2.1

# Make minimal fix
# Update version to 1.2.1
# Update CHANGELOG.md
```

### 2. Fast-Track Release

```bash
# Abbreviated testing
pytest tests/unit/ tests/integration/
python -m build
twine check dist/*

# Create PR with "hotfix" label
gh pr create --title "Hotfix v1.2.1" --label "hotfix"

# After approval and merge
git checkout main
git pull origin main
git tag -a v1.2.1 -m "Hotfix v1.2.1 - Critical bug fixes"
git push origin v1.2.1

# Release to PyPI
python -m build
twine upload dist/*
```

## Release Automation

### GitHub Actions Integration

Our CI/CD pipeline automates several release steps:

1. **On PR to main**: Run full test suite, security scans
2. **On tag creation**: Build and publish to PyPI, create GitHub release
3. **On release**: Update Docker images, notify team

### Release Scripts

Use the provided release scripts for consistency:

```bash
# Prepare release
scripts/prepare-release.sh 1.2.0

# Publish release (after manual approval)
scripts/publish-release.sh 1.2.0
```

## Version Management

### Pre-release Versions

For testing and development:

```bash
# Alpha release
1.2.0a1, 1.2.0a2, ...

# Beta release  
1.2.0b1, 1.2.0b2, ...

# Release candidate
1.2.0rc1, 1.2.0rc2, ...
```

### Development Versions

Between releases, use development versions:

```bash
# After 1.2.0 release, next dev version
1.3.0.dev0
```

## Rollback Procedure

If a release has critical issues:

### 1. Immediate Response
- [ ] Document the issue
- [ ] Assess impact and affected users
- [ ] Decide on rollback vs. hotfix

### 2. PyPI Rollback
```bash
# Remove problematic version (use sparingly)
# Contact PyPI support if needed

# Or release new version with fixes
```

### 3. Communication
- [ ] Update GitHub release with warning
- [ ] Notify users via GitHub discussions
- [ ] Update documentation

## Release Metrics

Track these metrics for each release:

- **Development time**: From start to release
- **Bug count**: Issues found post-release
- **Adoption rate**: Download statistics
- **User feedback**: GitHub issues, discussions

## Emergency Contact

For release-related emergencies:
- **Release Manager**: @terragon-labs/maintainers
- **Security Issues**: security@terragonlabs.com
- **Infrastructure**: @terragon-labs/devops-team

---

This process ensures consistent, high-quality releases while maintaining project stability and user trust.