# Manual Setup Requirements

## GitHub Actions Workflows
Due to permission limitations, these items require manual setup:

### 1. Copy Workflow Files
```bash
mkdir -p .github/workflows
cp docs/workflows/*.yml .github/workflows/
git add .github/workflows/ && git commit -m "feat: add CI/CD workflows"
```

### 2. Configure Repository Settings
- **Topics**: Add `watermarking`, `llm`, `security`, `nlp`
- **Description**: "Comprehensive toolkit for watermarking and detecting LLM-generated text"
- **Homepage**: Set to documentation URL

### 3. Branch Protection Rules
Enable protection for `main` branch requiring:
- Status checks: CI, Security Scanning, Code Quality
- Pull request reviews (1 minimum)
- Up-to-date branches

### 4. Required Secrets
Configure in repository settings:
- `PYPI_API_TOKEN` (for publishing)
- `GITGUARDIAN_API_KEY` (for security scanning)

### 5. Enable GitHub Features
- **Issues**: Enable with templates
- **Discussions**: Enable for community
- **Security**: Enable vulnerability reporting
- **Pages**: Configure for documentation

For detailed setup instructions, see [Workflow Setup Guide](workflows/README.md)