#!/bin/bash

# LM Watermark Lab - Production Deployment Script
# This script handles the complete deployment process

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VERSION="${VERSION:-1.0.0}"
ENVIRONMENT="${ENVIRONMENT:-production}"
NAMESPACE="${NAMESPACE:-production}"
REGISTRY="${REGISTRY:-lm-watermark-lab}"

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

check_dependencies() {
    log_info "Checking dependencies..."
    
    local deps=("docker" "kubectl" "helm")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "$dep is not installed"
            exit 1
        fi
    done
    
    log_success "All dependencies are installed"
}

build_docker_image() {
    log_info "Building Docker image..."
    
    docker build \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
        --build-arg VERSION="$VERSION" \
        --target production \
        -t "$REGISTRY:$VERSION" \
        -t "$REGISTRY:latest" \
        .
    
    log_success "Docker image built successfully"
}

run_tests() {
    log_info "Running tests..."
    
    # Run unit tests
    docker run --rm "$REGISTRY:$VERSION" python -m pytest tests/unit/ -v
    
    # Run integration tests
    docker run --rm "$REGISTRY:$VERSION" python -m pytest tests/integration/ -v
    
    # Run security tests
    docker run --rm "$REGISTRY:$VERSION" bandit -r src/
    
    log_success "All tests passed"
}

push_image() {
    log_info "Pushing Docker image to registry..."
    
    if [ -n "${DOCKER_REGISTRY:-}" ]; then
        docker tag "$REGISTRY:$VERSION" "$DOCKER_REGISTRY/$REGISTRY:$VERSION"
        docker tag "$REGISTRY:latest" "$DOCKER_REGISTRY/$REGISTRY:latest"
        
        docker push "$DOCKER_REGISTRY/$REGISTRY:$VERSION"
        docker push "$DOCKER_REGISTRY/$REGISTRY:latest"
        
        log_success "Image pushed to registry"
    else
        log_warning "DOCKER_REGISTRY not set, skipping image push"
    fi
}

deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply ConfigMaps and Secrets
    kubectl apply -f kubernetes/deployment.yaml -n "$NAMESPACE"
    
    # Wait for deployment to be ready
    kubectl rollout status deployment/lm-watermark-lab -n "$NAMESPACE" --timeout=300s
    kubectl rollout status deployment/lm-watermark-lab-worker -n "$NAMESPACE" --timeout=300s
    
    log_success "Kubernetes deployment completed"
}

deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    # Load environment variables
    if [ -f ".env.$ENVIRONMENT" ]; then
        export $(cat ".env.$ENVIRONMENT" | xargs)
    fi
    
    # Deploy services
    docker-compose -f docker-compose.yml up -d --build
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    timeout 60 bash -c 'until docker-compose ps | grep -q "healthy"; do sleep 5; done'
    
    log_success "Docker Compose deployment completed"
}

run_database_migrations() {
    log_info "Running database migrations..."
    
    if [ "$DEPLOYMENT_METHOD" = "kubernetes" ]; then
        kubectl exec -n "$NAMESPACE" deployment/lm-watermark-lab -- python -m alembic upgrade head
    else
        docker-compose exec app python -m alembic upgrade head
    fi
    
    log_success "Database migrations completed"
}

run_health_check() {
    log_info "Running health checks..."
    
    local health_url
    if [ "$DEPLOYMENT_METHOD" = "kubernetes" ]; then
        health_url="http://$(kubectl get service lm-watermark-lab-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')/health"
    else
        health_url="http://localhost:8080/health"
    fi
    
    # Wait for service to be ready
    timeout 60 bash -c "until curl -f $health_url; do sleep 5; done"
    
    log_success "Health checks passed"
}

setup_monitoring() {
    log_info "Setting up monitoring..."
    
    if [ "$DEPLOYMENT_METHOD" = "kubernetes" ]; then
        # Deploy Prometheus and Grafana using Helm
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
        helm repo update
        
        helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
            --namespace monitoring \
            --create-namespace \
            --values monitoring/prometheus-values.yaml
    else
        # Start monitoring services with Docker Compose
        docker-compose --profile monitoring up -d
    fi
    
    log_success "Monitoring setup completed"
}

cleanup_old_resources() {
    log_info "Cleaning up old resources..."
    
    # Remove old Docker images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    if [ "$DEPLOYMENT_METHOD" = "kubernetes" ]; then
        # Clean up old ReplicaSets
        kubectl delete replicaset --all -n "$NAMESPACE" --cascade=background
    fi
    
    log_success "Cleanup completed"
}

main() {
    log_info "Starting deployment of LM Watermark Lab v$VERSION"
    
    # Parse command line arguments
    DEPLOYMENT_METHOD="${1:-docker-compose}"
    RUN_TESTS="${2:-true}"
    
    case "$DEPLOYMENT_METHOD" in
        "kubernetes"|"k8s")
            DEPLOYMENT_METHOD="kubernetes"
            ;;
        "docker-compose"|"compose")
            DEPLOYMENT_METHOD="docker-compose"
            ;;
        *)
            log_error "Invalid deployment method: $DEPLOYMENT_METHOD"
            echo "Usage: $0 [kubernetes|docker-compose] [true|false (run tests)]"
            exit 1
            ;;
    esac
    
    # Run deployment steps
    check_dependencies
    build_docker_image
    
    if [ "$RUN_TESTS" = "true" ]; then
        run_tests
    fi
    
    push_image
    
    if [ "$DEPLOYMENT_METHOD" = "kubernetes" ]; then
        deploy_kubernetes
    else
        deploy_docker_compose
    fi
    
    run_database_migrations
    run_health_check
    setup_monitoring
    cleanup_old_resources
    
    log_success "Deployment completed successfully!"
    log_info "Application is available at:"
    
    if [ "$DEPLOYMENT_METHOD" = "kubernetes" ]; then
        log_info "  - API: https://api.watermark-lab.com"
        log_info "  - Monitoring: kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring"
    else
        log_info "  - API: http://localhost:8080"
        log_info "  - Dashboard: http://localhost:8501"
        log_info "  - Monitoring: http://localhost:3000"
    fi
}

# Trap signals and cleanup
trap 'log_error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"