# LM Watermark Lab - Production Deployment Guide

This document provides comprehensive instructions for deploying LM Watermark Lab to production environments.

## 🚀 Quick Start

### Docker Compose (Recommended for small to medium deployments)

```bash
# 1. Clone the repository
git clone https://github.com/terragon-labs/lm-watermark-lab.git
cd lm-watermark-lab

# 2. Configure environment
cp .env.production .env
# Edit .env with your configuration

# 3. Deploy
./scripts/deploy.sh docker-compose

# 4. Verify deployment
curl http://localhost:8080/health
```

### Kubernetes (Recommended for large-scale deployments)

```bash
# 1. Configure Kubernetes context
kubectl config current-context

# 2. Deploy
./scripts/deploy.sh kubernetes

# 3. Verify deployment
kubectl get pods -n production
```

## 📋 Prerequisites

### System Requirements

- **CPU**: Minimum 4 cores, Recommended 8+ cores
- **Memory**: Minimum 8GB RAM, Recommended 16GB+ RAM
- **Storage**: Minimum 100GB, Recommended 500GB+ SSD
- **Network**: High-bandwidth internet connection for model downloads

### Software Dependencies

- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.24+ (for K8s deployment)
- kubectl CLI
- Helm 3.0+ (for K8s deployment)
- Git

### Optional Dependencies

- NVIDIA Docker (for GPU acceleration)
- cert-manager (for automatic SSL certificates)
- Prometheus Operator (for advanced monitoring)

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   Nginx Proxy   │────│   Application   │
│    (Optional)   │    │                 │    │    Instances    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                              ┌────────────────────────┼────────────────────────┐
                              │                        │                        │
                    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
                    │      Redis      │    │   PostgreSQL    │    │   Celery        │
                    │    (Cache &     │    │   (Database)    │    │   Workers       │
                    │  Message Queue) │    │                 │    │                 │
                    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Configuration

### Environment Variables

#### Core Application Settings

```bash
# Application
VERSION=1.0.0
DEBUG=false
LOG_LEVEL=INFO
API_PORT=8080

# Database
POSTGRES_DB=watermark_lab
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
DATABASE_URL=postgresql://postgres:password@postgres:5432/watermark_lab

# Cache & Queue
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0

# Security
SECRET_KEY=your-256-bit-secret-key
JWT_SECRET_KEY=your-jwt-secret-key
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
```

#### Performance & Scaling

```bash
# Resource Limits
MAX_WORKERS=4
CELERY_CONCURRENCY=2
MODEL_CACHE_SIZE=10GB
BATCH_SIZE=16

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_HOUR=1000
RATE_LIMIT_PER_DAY=10000
```

#### Privacy & Compliance

```bash
# Compliance Features
GDPR_ENABLED=true
CCPA_ENABLED=true
PDPA_ENABLED=true
DATA_RETENTION_DAYS=365

# Internationalization
DEFAULT_LOCALE=en
SUPPORTED_LOCALES=en,es,fr,de,zh,ja
```

### SSL/TLS Configuration

#### Self-Signed Certificates (Development)

```bash
# Generate self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout nginx/ssl/server.key \
    -out nginx/ssl/server.crt \
    -config <(
    echo '[dn]'
    echo 'CN=localhost'
    echo '[req]'
    echo 'distinguished_name = dn'
    echo '[EXT]'
    echo 'subjectAltName=DNS:localhost'
    echo 'keyUsage=keyEncipherment,dataEncipherment'
    echo 'extendedKeyUsage=serverAuth'
    ) -extensions EXT
```

#### Let's Encrypt Certificates (Production)

For Kubernetes deployments, cert-manager handles automatic certificate provisioning.

For Docker Compose deployments:

```bash
# Install certbot
sudo apt-get install certbot

# Obtain certificate
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/
```

## 🐳 Docker Deployment

### Single Server Deployment

```bash
# 1. Configure environment
cp .env.production .env
# Edit .env with your settings

# 2. Start core services
docker-compose up -d app worker redis postgres

# 3. Run database migrations
docker-compose exec app python -m alembic upgrade head

# 4. Start additional services
docker-compose --profile production up -d

# 5. Verify deployment
docker-compose ps
curl http://localhost:8080/health
```

### Multi-Server Deployment

For multi-server deployments, use Docker Swarm:

```bash
# 1. Initialize swarm
docker swarm init

# 2. Deploy stack
docker stack deploy -c docker-compose.yml watermark-lab

# 3. Scale services
docker service scale watermark-lab_app=3
docker service scale watermark-lab_worker=2
```

## ☸️ Kubernetes Deployment

### Cluster Setup

#### Prerequisites

- Kubernetes cluster (1.24+)
- Ingress controller (nginx recommended)
- Storage class for persistent volumes
- cert-manager for SSL certificates

#### Namespace Setup

```bash
# Create production namespace
kubectl create namespace production

# Set default namespace
kubectl config set-context --current --namespace=production
```

### Deployment Steps

#### 1. Configure Secrets

```bash
# Create secrets
kubectl create secret generic watermark-lab-secrets \
    --from-literal=redis-url="redis://redis:6379/0" \
    --from-literal=database-url="postgresql://postgres:password@postgres:5432/watermark_lab" \
    --from-literal=secret-key="your-secret-key" \
    --from-literal=jwt-secret="your-jwt-secret" \
    -n production
```

#### 2. Deploy Dependencies

```bash
# Deploy Redis
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install redis bitnami/redis \
    --set auth.enabled=false \
    --set master.persistence.size=8Gi \
    -n production

# Deploy PostgreSQL
helm install postgres bitnami/postgresql \
    --set auth.postgresPassword=your-password \
    --set primary.persistence.size=50Gi \
    -n production
```

#### 3. Deploy Application

```bash
# Apply deployment manifests
kubectl apply -f kubernetes/deployment.yaml

# Wait for rollout
kubectl rollout status deployment/lm-watermark-lab -n production
```

#### 4. Configure Ingress

```bash
# Install nginx ingress controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install nginx-ingress ingress-nginx/ingress-nginx

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Apply ingress configuration
kubectl apply -f kubernetes/deployment.yaml
```

### Scaling and High Availability

#### Horizontal Pod Autoscaler

```bash
# Enable metrics server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# HPA is automatically created by deployment.yaml
kubectl get hpa -n production
```

#### Cluster Autoscaler

For cloud providers, configure cluster autoscaler:

```yaml
# cluster-autoscaler-values.yaml
autoDiscovery:
  clusterName: your-cluster-name
rbac:
  create: true
```

## 📊 Monitoring & Observability

### Prometheus & Grafana Setup

#### Docker Compose

```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Access Grafana
open http://localhost:3000
# Default credentials: admin/admin
```

#### Kubernetes

```bash
# Install kube-prometheus-stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --create-namespace

# Access Grafana
kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring
```

### Available Metrics

- **Application Metrics**: Request rate, response time, error rate
- **System Metrics**: CPU, memory, disk usage
- **Business Metrics**: Watermark generation/detection rates
- **Compliance Metrics**: GDPR consent rates, data retention

### Log Aggregation

#### ELK Stack (Optional)

```bash
# Start ELK stack
docker-compose --profile monitoring up -d elasticsearch logstash kibana

# Access Kibana
open http://localhost:5601
```

## 🔒 Security Considerations

### Network Security

- Use HTTPS/TLS for all external communications
- Implement network policies in Kubernetes
- Use private networks for inter-service communication
- Enable firewall rules for production deployments

### Data Security

- Encrypt data at rest using database encryption
- Use secure key management systems
- Implement proper access controls
- Regular security audits and penetration testing

### Authentication & Authorization

```bash
# Enable authentication
export ENABLE_AUTH=true

# Configure JWT settings
export JWT_SECRET_KEY="your-secure-jwt-secret"
export JWT_EXPIRATION_HOURS=24
```

## 🗄️ Backup & Recovery

### Database Backup

#### Automated Backups

```bash
# PostgreSQL backup script
#!/bin/bash
BACKUP_FILE="backup_$(date +%Y%m%d_%H%M%S).sql"
pg_dump $DATABASE_URL > $BACKUP_FILE
gzip $BACKUP_FILE

# Upload to cloud storage
aws s3 cp $BACKUP_FILE.gz s3://your-backup-bucket/
```

#### Backup Scheduling

```yaml
# Kubernetes CronJob for backups
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15
            command: ["/bin/sh", "-c"]
            args:
            - pg_dump $DATABASE_URL | gzip > /backup/backup_$(date +%Y%m%d).sql.gz
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

### Model Cache Backup

```bash
# Backup model cache
tar -czf models_backup_$(date +%Y%m%d).tar.gz ./models/

# Restore model cache
tar -xzf models_backup_20240127.tar.gz
```

## 🚦 Load Testing

### Using Locust

```bash
# Start load testing
docker-compose --profile loadtest up -d

# Access Locust UI
open http://localhost:8089

# Configure load test:
# - Number of users: 100
# - Spawn rate: 10/second
# - Host: http://app:8080
```

### Custom Load Test

```python
# tests/load/locustfile.py
from locust import HttpUser, task, between

class WatermarkUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def generate_watermark(self):
        self.client.post("/api/v1/watermark", json={
            "method": "kirchenbauer",
            "text": "Sample text for watermarking",
            "max_length": 100
        })
    
    @task(2)
    def detect_watermark(self):
        self.client.post("/api/v1/detect", json={
            "text": "Sample watermarked text",
            "method": "kirchenbauer"
        })
    
    @task(1)
    def health_check(self):
        self.client.get("/health")
```

## 🔧 Troubleshooting

### Common Issues

#### High Memory Usage

```bash
# Check memory usage
docker stats
kubectl top pods -n production

# Reduce batch size
export BATCH_SIZE=8
export CELERY_CONCURRENCY=1
```

#### Database Connection Issues

```bash
# Check database connectivity
docker-compose exec app python -c "
import psycopg2
conn = psycopg2.connect('$DATABASE_URL')
print('Database connection successful')
"

# Check PostgreSQL logs
docker-compose logs postgres
kubectl logs deployment/postgres -n production
```

#### Redis Connection Issues

```bash
# Test Redis connection
docker-compose exec redis redis-cli ping
kubectl exec deployment/redis -n production -- redis-cli ping

# Check Redis memory usage
docker-compose exec redis redis-cli info memory
```

### Log Analysis

```bash
# Application logs
docker-compose logs app
kubectl logs deployment/lm-watermark-lab -n production

# Worker logs
docker-compose logs worker
kubectl logs deployment/lm-watermark-lab-worker -n production

# Follow logs in real-time
docker-compose logs -f app worker
kubectl logs -f deployment/lm-watermark-lab -n production
```

### Performance Optimization

#### Database Optimization

```sql
-- Create indexes for frequently queried columns
CREATE INDEX idx_processing_records_subject_id ON processing_records(subject_id);
CREATE INDEX idx_processing_records_date ON processing_records(processing_date);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM processing_records WHERE subject_id = 'user123';
```

#### Cache Optimization

```bash
# Monitor Redis memory usage
redis-cli info memory

# Configure cache eviction policy
redis-cli config set maxmemory-policy allkeys-lru
```

## 🔄 Updates & Maintenance

### Rolling Updates

#### Docker Compose

```bash
# Build new image
docker build -t lm-watermark-lab:1.1.0 .

# Update services one by one
docker-compose up -d --no-deps app
docker-compose up -d --no-deps worker
```

#### Kubernetes

```bash
# Update image
kubectl set image deployment/lm-watermark-lab \
    watermark-lab=lm-watermark-lab:1.1.0 \
    -n production

# Monitor rollout
kubectl rollout status deployment/lm-watermark-lab -n production
```

### Maintenance Windows

```bash
# Scale down workers during maintenance
kubectl scale deployment lm-watermark-lab-worker --replicas=0 -n production

# Perform maintenance tasks
kubectl exec deployment/lm-watermark-lab -n production -- python manage.py maintenance

# Scale workers back up
kubectl scale deployment lm-watermark-lab-worker --replicas=2 -n production
```

## 📞 Support

### Getting Help

- **Documentation**: [https://lm-watermark-lab.readthedocs.io](https://lm-watermark-lab.readthedocs.io)
- **Issues**: [https://github.com/terragon-labs/lm-watermark-lab/issues](https://github.com/terragon-labs/lm-watermark-lab/issues)
- **Discussions**: [https://github.com/terragon-labs/lm-watermark-lab/discussions](https://github.com/terragon-labs/lm-watermark-lab/discussions)

### Professional Support

For enterprise support, training, and consulting services, contact:
- **Email**: support@terragonlabs.com
- **Website**: [https://terragonlabs.com](https://terragonlabs.com)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.