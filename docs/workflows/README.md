# GitHub Actions Workflows

This directory contains the GitHub Actions workflow files for the LM Watermark Lab SDLC automation.

## Setup Instructions

To activate these workflows, manually copy them to the `.github/workflows/` directory in your repository:

```bash
# Create the workflows directory if it doesn't exist
mkdir -p .github/workflows

# Copy all workflow files
cp docs/workflows/*.yml .github/workflows/

# Commit the workflows
git add .github/workflows/
git commit -m "feat: add comprehensive CI/CD workflows"
git push
```

## Workflow Files

### 1. `ci.yml` - Continuous Integration
Comprehensive CI pipeline with:
- Code quality checks (linting, formatting, type checking)
- Multi-platform testing (Ubuntu, Windows, macOS)
- Security scanning (CodeQL, Bandit, Safety)
- Performance benchmarking
- Docker image building and testing
- Dependency review

**Triggers**: Push to main/develop, Pull requests
**Duration**: ~15-20 minutes

### 2. `cd.yml` - Continuous Deployment
Automated deployment pipeline with:
- Staging environment deployment
- Production deployment on release
- Container registry publishing
- Post-deployment health checks
- Automated rollback capabilities

**Triggers**: Release published, Manual workflow dispatch
**Duration**: ~10-15 minutes

### 3. `security.yml` - Security Scanning
Comprehensive security automation:
- Dependency vulnerability scanning
- Static Application Security Testing (SAST)
- Secrets detection
- Container security scanning
- License compliance checking
- Infrastructure security validation

**Triggers**: Push, Pull requests, Daily schedule (3 AM UTC)
**Duration**: ~8-12 minutes

### 4. `maintenance.yml` - Automated Maintenance
Ongoing maintenance automation:
- Daily dependency security checks
- Weekly repository health assessments
- Monthly technical debt analysis
- Automated dependency updates
- Health metrics collection

**Triggers**: Scheduled (daily/weekly/monthly), Manual dispatch
**Duration**: ~5-10 minutes

### 5. `release.yml` - Release Management
Automated release process:
- Semantic versioning with conventional commits
- Automated changelog generation
- GitHub release creation
- Package publishing preparation
- Documentation updates

**Triggers**: Push to main branch
**Duration**: ~3-5 minutes

## Required Secrets

To use these workflows, configure the following GitHub secrets in your repository:

### Required Secrets
- `GITHUB_TOKEN` (automatically provided)

### Optional Secrets (for enhanced functionality)
- `PYPI_API_TOKEN` - For PyPI package publishing
- `GITGUARDIAN_API_KEY` - For enhanced secrets scanning
- `SLACK_WEBHOOK_URL` - For Slack notifications
- `DOCKER_REGISTRY_TOKEN` - For container registry publishing

## Permissions Required

Ensure your repository has the following permissions:
- `actions: write` - For workflow execution
- `contents: write` - For creating releases and commits
- `packages: write` - For container registry publishing
- `security-events: write` - For security scanning results
- `pull-requests: write` - For PR comments and reviews

## Customization

### Environment-Specific Configuration
Update the workflow files for your specific environment:

1. **Repository URLs**: Update container registry URLs and repository references
2. **Notification Channels**: Configure Slack/Discord/email notifications
3. **Deployment Targets**: Update staging/production deployment configurations
4. **Security Policies**: Adjust security scanning policies and thresholds

### Branch Protection Rules
Configure branch protection rules to enforce workflow requirements:

```yaml
# Suggested branch protection for main branch
required_status_checks:
  strict: true
  contexts:
    - "CI / Code Quality Checks"
    - "CI / Test Suite"
    - "CI / Security Scanning"
    - "CI / Build Package"

enforce_admins: true
required_pull_request_reviews:
  required_approving_review_count: 1
  dismiss_stale_reviews: true
restrictions: null
```

## Workflow Dependencies

The workflows are designed to work together:

1. **CI workflow** validates all changes
2. **Security workflow** ensures security compliance
3. **CD workflow** deploys validated changes
4. **Release workflow** manages version releases
5. **Maintenance workflow** keeps the system healthy

## Monitoring and Alerts

The workflows include comprehensive monitoring:
- Workflow execution metrics
- Success/failure notifications
- Performance benchmarking
- Security vulnerability alerts
- Dependency update notifications

## Troubleshooting

Common issues and solutions:

### Workflow Fails with Permission Error
- Check repository permissions
- Verify required secrets are configured
- Ensure branch protection rules allow the action

### Security Scans Fail
- Review security policy configurations
- Check for new vulnerabilities in dependencies
- Verify secret scanning configurations

### Deployment Fails
- Check deployment target accessibility
- Verify deployment credentials
- Review environment-specific configurations

## Performance Optimization

To optimize workflow performance:
- Use workflow concurrency controls
- Cache dependencies where possible
- Run jobs in parallel when appropriate
- Use self-hosted runners for faster execution

## Support

For issues with these workflows:
1. Check the GitHub Actions logs
2. Review the workflow documentation
3. Create an issue with detailed error information
4. Contact the development team

---

**Note**: These workflows provide enterprise-grade CI/CD automation. Start with the CI workflow and gradually enable others based on your needs.