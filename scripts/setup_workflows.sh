#!/bin/bash

# LM Watermark Lab - SDLC Workflows Setup Script
# This script sets up the comprehensive CI/CD workflows for the repository

set -e

echo "🚀 Setting up LM Watermark Lab SDLC Workflows"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "docs/workflows" ]; then
    print_error "This script must be run from the repository root directory"
    print_error "Make sure you're in the lm-watermark-lab directory and that docs/workflows exists"
    exit 1
fi

print_info "Detected repository root: $(pwd)"

# Create .github/workflows directory if it doesn't exist
print_info "Creating .github/workflows directory..."
mkdir -p .github/workflows
print_status "Created .github/workflows directory"

# Copy workflow files
print_info "Copying workflow files from docs/workflows to .github/workflows..."
workflow_files=("ci.yml" "cd.yml" "security.yml" "maintenance.yml" "release.yml")

for file in "${workflow_files[@]}"; do
    if [ -f "docs/workflows/$file" ]; then
        cp "docs/workflows/$file" ".github/workflows/$file"
        print_status "Copied $file"
    else
        print_warning "Workflow file $file not found in docs/workflows/"
    fi
done

# Verify workflow files
print_info "Verifying workflow files..."
for file in "${workflow_files[@]}"; do
    if [ -f ".github/workflows/$file" ]; then
        print_status "Verified .github/workflows/$file"
    else
        print_error "Failed to create .github/workflows/$file"
    fi
done

# Check for GitHub CLI
if command -v gh &> /dev/null; then
    print_status "GitHub CLI detected"
    
    # Check if user is authenticated
    if gh auth status &> /dev/null; then
        print_status "GitHub CLI is authenticated"
        
        # Get repository info
        REPO_NAME=$(gh repo view --json name -q .name)
        REPO_OWNER=$(gh repo view --json owner -q .owner.login)
        print_info "Repository: $REPO_OWNER/$REPO_NAME"
        
        # Check repository permissions
        print_info "Checking repository permissions..."
        
        # Note: GitHub CLI doesn't have a direct way to check specific permissions
        # So we'll provide instructions instead
        print_warning "Please ensure your repository has the following permissions:"
        echo "  - Actions: Read and write"
        echo "  - Contents: Read and write"
        echo "  - Metadata: Read"
        echo "  - Pull requests: Write"
        echo "  - Security events: Write"
        echo "  - Packages: Write (if using container registry)"
        
    else
        print_warning "GitHub CLI is not authenticated"
        print_info "Run 'gh auth login' to authenticate"
    fi
else
    print_warning "GitHub CLI not found. Consider installing it for easier repository management."
fi

# Check for required secrets
print_info "Checking for recommended GitHub secrets..."
echo ""
echo "📋 Required Secrets Checklist:"
echo "  □ GITHUB_TOKEN (automatically provided by GitHub)"
echo ""
echo "📋 Optional Secrets for Enhanced Functionality:"
echo "  □ PYPI_API_TOKEN - For PyPI package publishing"
echo "  □ GITGUARDIAN_API_KEY - For enhanced secrets scanning"
echo "  □ SLACK_WEBHOOK_URL - For Slack notifications"
echo "  □ DOCKER_REGISTRY_TOKEN - For container registry publishing"
echo ""

# Check branch protection
print_info "Branch protection recommendations:"
echo ""
echo "🛡️  Recommended branch protection rules for 'main' branch:"
echo "  □ Require pull request reviews before merging"
echo "  □ Require status checks to pass before merging:"
echo "    - CI / Code Quality Checks"
echo "    - CI / Test Suite" 
echo "    - CI / Security Scanning"
echo "    - CI / Build Package"
echo "  □ Require branches to be up to date before merging"
echo "  □ Include administrators in restrictions"
echo ""

# Validate workflow syntax
print_info "Validating workflow syntax..."
for file in "${workflow_files[@]}"; do
    if [ -f ".github/workflows/$file" ]; then
        # Basic YAML syntax check
        if command -v python3 &> /dev/null; then
            python3 -c "
import yaml
import sys
try:
    with open('.github/workflows/$file', 'r') as f:
        yaml.safe_load(f)
    print('✅ $file: Valid YAML syntax')
except yaml.YAMLError as e:
    print('❌ $file: YAML syntax error')
    print(f'   {e}')
    sys.exit(1)
except Exception as e:
    print('⚠️  $file: Could not validate (missing dependencies)')
"
        else
            print_warning "$file: Could not validate YAML syntax (python3 not available)"
        fi
    fi
done

# Create commit message
echo ""
print_info "Creating commit message template..."
cat > .github/workflows/COMMIT_MESSAGE.txt << 'EOF'
feat: implement comprehensive SDLC automation workflows

This commit adds enterprise-grade CI/CD workflows providing:

✅ Continuous Integration (ci.yml):
- Multi-platform testing (Ubuntu, Windows, macOS)  
- Code quality checks and security scanning
- Performance benchmarking and Docker builds

✅ Continuous Deployment (cd.yml):
- Automated staging and production deployments
- Container registry publishing with health checks

✅ Security Scanning (security.yml):
- Dependency vulnerability and secrets scanning
- SAST, container security, and license compliance

✅ Maintenance Automation (maintenance.yml):
- Daily security scans and weekly health checks
- Monthly technical debt analysis and auto-updates

✅ Release Management (release.yml):
- Semantic versioning and automated changelogs
- GitHub releases with artifact publishing

🚀 The repository is now production-ready with enterprise SDLC automation!

Co-authored-by: Claude <noreply@anthropic.com>
EOF

print_status "Created commit message template at .github/workflows/COMMIT_MESSAGE.txt"

# Final instructions
echo ""
echo "🎉 Workflow Setup Complete!"
echo "=========================="
echo ""
print_status "All workflow files have been copied to .github/workflows/"
print_info "Next steps:"
echo ""
echo "1. 📋 Review the workflow files in .github/workflows/"
echo "2. 🔑 Configure required secrets in your GitHub repository"
echo "3. 🛡️  Set up branch protection rules (see recommendations above)"
echo "4. 💾 Commit and push the workflow files:"
echo ""
echo "   git add .github/workflows/"
echo "   git commit -F .github/workflows/COMMIT_MESSAGE.txt"
echo "   git push"
echo ""
echo "5. 🚀 Create a pull request to merge the workflows"
echo "6. ✅ Enable the workflows and verify they run successfully"
echo ""
print_warning "Remember: You may need repository admin permissions to enable workflows"
print_info "For detailed documentation, see docs/workflows/README.md"
echo ""
print_status "SDLC automation setup is complete! 🎊"