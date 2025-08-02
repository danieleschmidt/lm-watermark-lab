# GitHub Actions Workflow Setup Guide

This guide provides step-by-step instructions for setting up the comprehensive CI/CD workflows for LM Watermark Lab.

## Prerequisites

Before setting up the workflows, ensure you have:

- [ ] Repository admin access
- [ ] GitHub Actions enabled for your repository
- [ ] Required secrets configured (see [Required Secrets](#required-secrets))
- [ ] Appropriate repository permissions (see [Permissions](#permissions))

## Quick Setup (Recommended)

For most users, the standard workflow setup is sufficient:

```bash
# 1. Create workflows directory
mkdir -p .github/workflows

# 2. Copy essential workflows
cp docs/workflows/ci.yml .github/workflows/
cp docs/workflows/security.yml .github/workflows/
cp docs/workflows/cd.yml .github/workflows/

# 3. Commit and push
git add .github/workflows/
git commit -m "feat: add essential CI/CD workflows"
git push
```

## Advanced Setup

For advanced features and comprehensive automation:

```bash
# 1. Copy all workflow files
cp docs/workflows/*.yml .github/workflows/

# 2. Copy advanced examples if needed
cp docs/workflows/examples/advanced-ci.yml .github/workflows/
cp docs/workflows/examples/deployment-pipeline.yml .github/workflows/

# 3. Customize for your environment
# Edit workflow files to match your deployment setup

# 4. Commit workflows
git add .github/workflows/
git commit -m "feat: add comprehensive CI/CD pipeline"
git push
```

## Workflow Configuration

### 1. Environment-Specific Settings

Update these values in your workflow files:

```yaml
# In ci.yml, cd.yml, etc.
env:
  PYTHON_VERSION: "3.11"          # Your Python version
  REGISTRY: ghcr.io               # Your container registry
  IMAGE_NAME: ${{ github.repository }}
  
# In deployment workflows
environment:
  name: production
  url: https://your-domain.com     # Your production URL
```

### 2. Branch Protection Rules

Configure branch protection for your main branches:

1. Go to Settings → Branches
2. Add rule for `main` branch:
   ```yaml
   Require status checks before merging: ✓
   Required status checks:
     - "CI / Code Quality Checks"
     - "CI / Test Suite" 
     - "Security / Security Scan"
   Require branches to be up to date: ✓
   Require pull request reviews: ✓
   Required approving review count: 1
   Dismiss stale reviews: ✓
   ```

### 3. Repository Permissions

Ensure your repository has these permissions:

- Settings → Actions → General:
  - Actions permissions: "Allow all actions and reusable workflows"
  - Workflow permissions: "Read and write permissions"
  - Allow GitHub Actions to create and approve pull requests: ✓

## Required Secrets

Configure these secrets in Settings → Secrets and variables → Actions:

### Essential Secrets
- `GITHUB_TOKEN` - Automatically provided by GitHub

### Optional Secrets (for enhanced functionality)

#### Container Registry
- `DOCKER_REGISTRY_TOKEN` - For pushing to container registries
- `REGISTRY_USERNAME` - Registry username (if not using GitHub)

#### Deployment
- `PRODUCTION_SSH_KEY` - SSH key for production deployments
- `STAGING_SSH_KEY` - SSH key for staging deployments
- `KUBE_CONFIG` - Kubernetes configuration for deployments

#### External Services
- `SLACK_WEBHOOK_URL` - For Slack notifications
- `DISCORD_WEBHOOK_URL` - For Discord notifications
- `PYPI_API_TOKEN` - For PyPI package publishing
- `CODECOV_TOKEN` - For Codecov integration

#### Security Scanning
- `GITGUARDIAN_API_KEY` - For GitGuardian secret scanning
- `SNYK_TOKEN` - For Snyk vulnerability scanning
- `SONAR_TOKEN` - For SonarCloud analysis

## Environment Variables

Set these environment variables for your organization:

### Repository Variables
Go to Settings → Secrets and variables → Actions → Variables:

```
PYTHON_VERSION=3.11
NODE_VERSION=18
CONTAINER_REGISTRY=ghcr.io
PRODUCTION_URL=https://watermark-lab.terragon.ai
STAGING_URL=https://staging.watermark-lab.terragon.ai
```

### Organization Variables (if applicable)
For consistency across repositories:

```
ORG_DOCKER_REGISTRY=ghcr.io/your-org
ORG_SLACK_CHANNEL=#deployments
ORG_SECURITY_TEAM=@security-team
```

## Workflow Customization

### 1. Adding Custom Jobs

Add custom jobs to existing workflows:

```yaml
# In ci.yml
jobs:
  # ... existing jobs ...
  
  custom-analysis:
    name: Custom Analysis
    runs-on: ubuntu-latest
    steps:
    - name: Checkout
      uses: actions/checkout@v4
    
    - name: Run custom analysis
      run: |
        # Your custom analysis steps
        echo "Running custom analysis..."
```

### 2. Modifying Triggers

Customize when workflows run:

```yaml
# Run on different branches
on:
  push:
    branches: [ main, develop, 'release/*' ]
  
# Add scheduled runs
on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM
  
# Add manual triggers
on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Target environment'
        required: true
        type: choice
        options:
        - staging
        - production
```

### 3. Adding Notifications

Integrate with your communication tools:

```yaml
# Slack notification job
notify-slack:
  name: Notify Slack
  runs-on: ubuntu-latest
  if: always()
  steps:
  - name: Send Slack notification
    uses: 8398a7/action-slack@v3
    with:
      status: ${{ job.status }}
      channel: '#ci-cd'
      text: 'Workflow completed'
    env:
      SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

## Testing Your Setup

### 1. Workflow Validation

Test your workflows without deployment:

```bash
# Validate workflow syntax
github-actions-validator .github/workflows/

# Test with act (local GitHub Actions runner)
act -n  # Dry run
act pull_request  # Test PR workflow
```

### 2. Gradual Rollout

Enable workflows gradually:

1. **Week 1**: Enable CI workflow only
2. **Week 2**: Add security scanning
3. **Week 3**: Enable staging deployments
4. **Week 4**: Enable production deployments

### 3. Monitor Initial Runs

Watch the first few workflow runs:

1. Check Actions tab in GitHub
2. Review workflow logs for errors
3. Verify all jobs complete successfully
4. Test notifications work correctly

## Troubleshooting

### Common Issues

#### 1. Permission Denied
```
Error: Permission denied
```
**Solution**: Check repository permissions and secrets configuration.

#### 2. Workflow Not Triggering
```
Workflow doesn't run on push/PR
```
**Solution**: Verify branch names and trigger conditions in workflow files.

#### 3. Secret Not Found
```
Error: Secret DOCKER_TOKEN not found
```
**Solution**: Ensure secrets are configured in repository settings.

#### 4. Container Build Fails
```
Error: unable to prepare context
```
**Solution**: Check Dockerfile and .dockerignore configuration.

### Getting Help

1. **Check Workflow Logs**: Review detailed logs in the Actions tab
2. **Validate Syntax**: Use GitHub's workflow validator
3. **Community Support**: Check GitHub Actions community forums
4. **Documentation**: Review official GitHub Actions documentation

## Maintenance

### Regular Tasks

#### Monthly
- [ ] Review workflow performance metrics
- [ ] Update action versions to latest
- [ ] Review and rotate secrets
- [ ] Check for new security vulnerabilities

#### Quarterly
- [ ] Audit workflow permissions
- [ ] Review and update notification channels
- [ ] Optimize workflow performance
- [ ] Update documentation

### Action Updates

Keep actions up to date:

```bash
# Use Dependabot for automatic updates
# Add to .github/dependabot.yml:
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

## Security Considerations

### 1. Secret Management
- Use repository secrets for sensitive data
- Rotate secrets regularly
- Use environment-specific secrets
- Never log secret values

### 2. Workflow Security
- Pin action versions (use SHA instead of tags for security-critical workflows)
- Review third-party actions before use
- Use minimal required permissions
- Enable branch protection rules

### 3. Container Security
- Scan containers for vulnerabilities
- Use minimal base images
- Keep dependencies updated
- Sign container images (optional)

## Advanced Features

### 1. Matrix Builds
Test across multiple environments:

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: ["3.9", "3.10", "3.11"]
```

### 2. Conditional Execution
Run jobs based on conditions:

```yaml
if: github.ref == 'refs/heads/main'
```

### 3. Artifact Management
Share data between jobs:

```yaml
- name: Upload artifacts
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: test-results.xml
```

### 4. Caching
Speed up workflows with caching:

```yaml
- name: Cache dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
```

## Next Steps

After setting up workflows:

1. **Monitor Performance**: Track workflow execution times and success rates
2. **Optimize**: Identify and optimize slow steps
3. **Expand**: Add more advanced features as needed
4. **Document**: Keep this setup guide updated with your customizations

For questions or issues, refer to the project documentation or create an issue in the repository.