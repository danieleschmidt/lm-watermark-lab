# 🔧 GitHub Workflows Setup Instructions

Due to GitHub App security restrictions, the workflow files need to be manually added to `.github/workflows/`. 

## Quick Setup

The workflow templates are located in `workflow-templates/` and need to be copied to `.github/workflows/`:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy all workflow templates (remove .txt extension)
for file in workflow-templates/*.yml.txt; do
  cp "$file" ".github/workflows/$(basename "$file" .txt)"
done

# Add and commit workflows
git add .github/workflows/
git commit -m "feat: add comprehensive CI/CD workflows

🚀 Complete GitHub Actions workflow suite:
- ci.yml: Comprehensive CI pipeline with quality gates
- cd.yml: Production deployment automation
- security.yml: Multi-layer security scanning  
- container-security.yml: Docker image security validation
- sbom.yml: Software Bill of Materials generation
- contract-testing.yml: API contract validation
- mutation-testing.yml: Test quality verification
- maintenance.yml: Automated maintenance tasks
- release.yml: Semantic release automation

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push the workflows
git push

# Clean up templates (optional)
rm -rf workflow-templates/
```

## Workflow Overview

### Core CI/CD Workflows
- **`ci.yml`** - Main CI pipeline with testing, quality checks, and security scanning
- **`cd.yml`** - Continuous deployment with blue-green strategy and rollback
- **`release.yml`** - Automated semantic versioning and release management

### Security & Compliance  
- **`security.yml`** - CodeQL analysis, dependency scanning, and vulnerability detection
- **`container-security.yml`** - Docker image scanning with Trivy and Docker Scout
- **`sbom.yml`** - Software Bill of Materials generation for compliance

### Quality Assurance
- **`contract-testing.yml`** - API contract validation and backward compatibility
- **`mutation-testing.yml`** - Test quality verification with mutation coverage

### Operations
- **`maintenance.yml`** - Scheduled maintenance, cleanup, and health monitoring

## Required Secrets

Set these in GitHub repository settings → Secrets and variables → Actions:

```bash
# Container Registry (if using)
DOCKER_USERNAME=your-docker-username
DOCKER_PASSWORD=your-docker-password

# Package Registry (if publishing)
NPM_TOKEN=your-npm-token
PYPI_TOKEN=your-pypi-token

# Monitoring & Alerting (optional)
SLACK_WEBHOOK_URL=your-slack-webhook
DISCORD_WEBHOOK_URL=your-discord-webhook

# Security Scanning (optional)
SNYK_TOKEN=your-snyk-token
CODECOV_TOKEN=your-codecov-token
```

## Branch Protection Setup

Apply branch protection rules using the template in `.github/branch-protection.json`:

1. Go to Repository Settings → Branches
2. Add rule for `main` branch
3. Configure according to the JSON template
4. Repeat for `develop` branch if used

## Dependabot Configuration

The `.github/dependabot.yml` file is already configured for:
- Python dependencies (weekly updates)
- GitHub Actions (weekly updates)  
- Docker base images (weekly updates)

## Next Steps

1. **Add workflow files** using the commands above
2. **Configure repository secrets** for integrations
3. **Set up branch protection** rules for main/develop
4. **Enable Dependabot** security updates
5. **Configure monitoring** dashboards and alerts

The SDLC automation is now ready for production use! 🚀