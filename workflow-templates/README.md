# 🔧 GitHub Actions Workflow Templates

This directory contains production-ready GitHub Actions workflow templates for comprehensive SDLC automation.

## Quick Setup

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy all workflow templates (remove .txt extension)
for file in workflow-templates/*.yml.txt; do
  cp "$file" ".github/workflows/$(basename "$file" .txt)"
done

# Add and commit workflows
git add .github/workflows/
git commit -m "feat: add comprehensive CI/CD workflows"
git push

# Clean up templates
rm -rf workflow-templates/
```

## Workflow Overview

### 🚀 Available Workflows

| File | Purpose | Triggers |
|------|---------|----------|
| `ci.yml.txt` | Comprehensive CI pipeline | Push, PR |
| `cd.yml.txt` | Production deployment | Release tags |
| `security.yml.txt` | Security scanning | Push, Schedule |
| `container-security.yml.txt` | Docker security | Dockerfile changes |
| `sbom.yml.txt` | Software Bill of Materials | Release, Schedule |
| `contract-testing.yml.txt` | API contract validation | Push, PR |
| `mutation-testing.yml.txt` | Test quality verification | Weekly |
| `maintenance.yml.txt` | Automated maintenance | Schedule |
| `release.yml.txt` | Semantic release | Push to main |

### 🔒 Required Secrets

Configure these in Repository Settings → Secrets and variables → Actions:

```bash
# Container Registry
DOCKER_USERNAME=your-username
DOCKER_PASSWORD=your-password

# Package Publishing  
NPM_TOKEN=your-npm-token
PYPI_TOKEN=your-pypi-token

# Monitoring (optional)
SLACK_WEBHOOK_URL=your-slack-webhook
CODECOV_TOKEN=your-codecov-token
```

### 📋 Features Included

- **Multi-platform testing** (Ubuntu, Windows, macOS)
- **Matrix testing** across Python 3.9, 3.10, 3.11
- **Security scanning** (CodeQL, Trivy, Bandit, Safety)
- **Container security** (Docker Scout, Hadolint)
- **Quality gates** (test coverage, code quality)
- **Performance benchmarking** with regression detection
- **Contract testing** with Pact for API validation
- **Mutation testing** for test suite quality
- **SBOM generation** for compliance
- **Blue-green deployments** with rollback
- **Automated dependency updates**

### 🎯 Quality Thresholds

- **Test Coverage**: 80% minimum
- **Security**: Zero critical vulnerabilities
- **Performance**: No regression > 10%
- **Code Quality**: Pre-commit hooks passing
- **Dependencies**: No known high/critical CVEs

The workflows are production-ready and include comprehensive error handling, notifications, and artifact management.