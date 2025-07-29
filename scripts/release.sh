#!/usr/bin/env bash

# Release automation script for LM Watermark Lab
# Usage: ./scripts/release.sh [version] [--dry-run]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_VERSION="3.9"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."
    
    # Check if running from project root
    if [[ ! -f "pyproject.toml" ]]; then
        log_error "Must run from project root directory"
        exit 1
    fi
    
    # Check required tools
    local tools=("git" "python" "gh" "docker")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check git status
    if [[ -n "$(git status --porcelain)" ]]; then
        log_error "Working directory is not clean. Commit or stash changes first."
        exit 1
    fi
    
    # Check current branch
    local current_branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [[ "$current_branch" != "main" ]]; then
        log_error "Must be on main branch to create release"
        exit 1
    fi
    
    log_success "Requirements check passed"
}

validate_version() {
    local version="$1"
    
    if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        log_error "Version must follow semantic versioning (e.g., 1.2.3)"
        exit 1
    fi
    
    # Check if tag already exists
    if git tag -l | grep -q "^v$version$"; then
        log_error "Tag v$version already exists"
        exit 1
    fi
    
    log_success "Version $version is valid"
}

update_version_files() {
    local version="$1"
    local dry_run="$2"
    
    log_info "Updating version files to $version..."
    
    if [[ "$dry_run" == "false" ]]; then
        # Update pyproject.toml
        sed -i.bak "s/version = \".*\"/version = \"$version\"/" pyproject.toml
        
        # Update __init__.py
        sed -i.bak "s/__version__ = \".*\"/__version__ = \"$version\"/" src/watermark_lab/__init__.py
        
        # Remove backup files
        rm -f pyproject.toml.bak src/watermark_lab/__init__.py.bak
        
        log_success "Version files updated"
    else
        log_info "DRY RUN: Would update version files to $version"
    fi
}

run_tests() {
    local dry_run="$1"
    
    log_info "Running test suite..."
    
    if [[ "$dry_run" == "false" ]]; then
        # Install dependencies
        python -m pip install -e ".[dev,test]"
        
        # Run tests
        python -m pytest tests/ -v --cov=src/watermark_lab --cov-report=term-missing
        
        # Run linting
        python -m ruff check src/ tests/
        
        # Run type checking
        python -m mypy src/
        
        # Security scan
        python -m bandit -r src/
        
        log_success "All tests passed"
    else
        log_info "DRY RUN: Would run full test suite"
    fi
}

build_package() {
    local dry_run="$1"
    
    log_info "Building package..."
    
    if [[ "$dry_run" == "false" ]]; then
        # Clean previous builds
        rm -rf dist/ build/ *.egg-info/
        
        # Build package
        python -m build
        
        # Check package
        python -m twine check dist/*
        
        log_success "Package built successfully"
    else
        log_info "DRY RUN: Would build package"
    fi
}

create_release_notes() {
    local version="$1"
    local dry_run="$2"
    
    log_info "Creating release notes..."
    
    local release_notes_file="/tmp/release-notes-$version.md"
    
    if [[ "$dry_run" == "false" ]]; then
        # Extract changelog entry for this version
        awk "/## \[$version\]/,/## \[/{if(/## \[/ && !/## \[$version\]/) exit; print}" CHANGELOG.md > "$release_notes_file"
        
        if [[ ! -s "$release_notes_file" ]]; then
            log_error "No changelog entry found for version $version"
            exit 1
        fi
        
        log_success "Release notes created at $release_notes_file"
    else
        log_info "DRY RUN: Would create release notes"
    fi
}

create_git_tag() {
    local version="$1"
    local dry_run="$2"
    
    log_info "Creating git tag v$version..."
    
    if [[ "$dry_run" == "false" ]]; then
        # Commit version changes
        git add pyproject.toml src/watermark_lab/__init__.py
        git commit -m "chore: bump version to $version"
        
        # Create annotated tag
        git tag -a "v$version" -m "Release v$version"
        
        log_success "Git tag created"
    else
        log_info "DRY RUN: Would create git tag v$version"
    fi
}

push_changes() {
    local version="$1"
    local dry_run="$2"
    
    log_info "Pushing changes to remote..."
    
    if [[ "$dry_run" == "false" ]]; then
        git push origin main
        git push origin "v$version"
        
        log_success "Changes pushed to remote"
    else
        log_info "DRY RUN: Would push changes to remote"
    fi
}

create_github_release() {
    local version="$1"
    local dry_run="$2"
    
    log_info "Creating GitHub release..."
    
    if [[ "$dry_run" == "false" ]]; then
        local release_notes_file="/tmp/release-notes-$version.md"
        
        gh release create "v$version" \
            --title "LM Watermark Lab v$version" \
            --notes-file "$release_notes_file" \
            --target main \
            dist/*
        
        log_success "GitHub release created"
    else
        log_info "DRY RUN: Would create GitHub release"
    fi
}

publish_to_pypi() {
    local dry_run="$1"
    
    log_info "Publishing to PyPI..."
    
    if [[ "$dry_run" == "false" ]]; then
        # Check if PYPI_TOKEN is set
        if [[ -z "${PYPI_TOKEN:-}" ]]; then
            log_warning "PYPI_TOKEN not set. Skipping PyPI publication."
            log_info "To publish manually: python -m twine upload dist/*"
            return
        fi
        
        python -m twine upload dist/* --username __token__ --password "$PYPI_TOKEN"
        
        log_success "Published to PyPI"
    else
        log_info "DRY RUN: Would publish to PyPI"
    fi
}

build_docker_image() {
    local version="$1"
    local dry_run="$2"
    
    log_info "Building Docker image..."
    
    if [[ "$dry_run" == "false" ]]; then
        docker build -t "terragon-labs/watermark-lab:$version" .
        docker build -t "terragon-labs/watermark-lab:latest" .
        
        log_success "Docker images built"
        
        # Push to registry if credentials available
        if docker info | grep -q "Username:"; then
            docker push "terragon-labs/watermark-lab:$version"
            docker push "terragon-labs/watermark-lab:latest"
            log_success "Docker images pushed to registry"
        else
            log_warning "Docker not logged in. Skipping push."
            log_info "To push manually: docker push terragon-labs/watermark-lab:$version"
        fi
    else
        log_info "DRY RUN: Would build and push Docker images"
    fi
}

cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f /tmp/release-notes-*.md
    log_success "Cleanup completed"
}

show_help() {
    cat << EOF
LM Watermark Lab Release Script

Usage: $0 [VERSION] [OPTIONS]

Arguments:
    VERSION     Semantic version number (e.g., 1.2.3)

Options:
    --dry-run   Show what would be done without making changes
    --help      Show this help message

Examples:
    $0 1.2.3                # Create release 1.2.3
    $0 1.2.3 --dry-run      # Preview release 1.2.3 without changes

Environment Variables:
    PYPI_TOKEN              PyPI API token for publishing (optional)
    DOCKER_USERNAME         Docker Hub username (optional)
    DOCKER_PASSWORD         Docker Hub password (optional)

Prerequisites:
    - Clean git working directory
    - Current branch must be 'main'
    - All required tools installed (git, python, gh, docker)
    - CHANGELOG.md updated with release notes

EOF
}

main() {
    local version=""
    local dry_run="false"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                dry_run="true"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
            *)
                if [[ -z "$version" ]]; then
                    version="$1"
                else
                    log_error "Multiple versions specified"
                    show_help
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Check if version is provided
    if [[ -z "$version" ]]; then
        log_error "Version number is required"
        show_help
        exit 1
    fi
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    log_info "Starting release process for version $version (dry-run: $dry_run)"
    
    # Execute release steps
    check_requirements
    validate_version "$version"
    update_version_files "$version" "$dry_run"
    run_tests "$dry_run"
    build_package "$dry_run"
    create_release_notes "$version" "$dry_run"
    create_git_tag "$version" "$dry_run"
    push_changes "$version" "$dry_run"
    create_github_release "$version" "$dry_run"
    publish_to_pypi "$dry_run"
    build_docker_image "$version" "$dry_run"
    
    if [[ "$dry_run" == "false" ]]; then
        log_success "Release $version completed successfully!"
        log_info "Don't forget to:"
        log_info "  - Announce the release on community channels"
        log_info "  - Update dependent projects"
        log_info "  - Monitor for issues in the first 24 hours"
    else
        log_success "Dry run completed. No changes were made."
    fi
}

# Run main function with all arguments
main "$@"