# LM Watermark Lab Deployment Guide

This guide covers deployment strategies, configuration, and operational procedures for LM Watermark Lab.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Deployment Methods](#deployment-methods)
- [Configuration](#configuration)
- [Monitoring and Health Checks](#monitoring-and-health-checks)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)
- [Scaling](#scaling)

## Overview

LM Watermark Lab supports multiple deployment strategies:

- **Docker Compose**: For development and small-scale deployments
- **Kubernetes**: For production-scale deployments
- **Cloud Providers**: AWS, GCP, Azure with managed services
- **Bare Metal**: Traditional server deployments

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB SSD
- Network: 1Gbps

**Recommended for Production:**
- CPU: 8+ cores
- RAM: 32GB+
- Storage: 500GB+ NVMe SSD
- Network: 10Gbps
- GPU: NVIDIA GPU with 8GB+ VRAM (optional, for acceleration)

### Software Dependencies

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.9+
- Git
- SSL certificates (for HTTPS)

## Deployment Methods

### 1. Docker Compose Deployment

#### Quick Start

```bash
# Clone the repository
git clone https://github.com/terragon-labs/lm-watermark-lab.git
cd lm-watermark-lab

# Copy environment file
cp .env.example .env
# Edit .env with your configuration

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

#### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  watermark-lab:
    image: ghcr.io/terragon-labs/lm-watermark-lab:latest
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://user:pass@postgres:5432/watermark_lab
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    ports:
      - "8080:8080"
    restart: unless-stopped
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - watermark-lab
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=watermark_lab
      - POSTGRES_USER=watermark_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### 2. Kubernetes Deployment

#### Namespace and Configuration

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: watermark-lab
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: watermark-lab-config
  namespace: watermark-lab
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  API_WORKERS: "4"
```

#### Application Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: watermark-lab
  namespace: watermark-lab
spec:
  replicas: 3
  selector:
    matchLabels:
      app: watermark-lab
  template:
    metadata:
      labels:
        app: watermark-lab
    spec:
      containers:
      - name: watermark-lab
        image: ghcr.io/terragon-labs/lm-watermark-lab:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: watermark-lab-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: watermark-lab-secrets
              key: redis-url
        envFrom:
        - configMapRef:
            name: watermark-lab-config
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: watermark-lab-data
      - name: logs-volume
        persistentVolumeClaim:
          claimName: watermark-lab-logs
```

#### Service and Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: watermark-lab-service
  namespace: watermark-lab
spec:
  selector:
    app: watermark-lab
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: watermark-lab-ingress
  namespace: watermark-lab
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.watermark-lab.com
    secretName: watermark-lab-tls
  rules:
  - host: api.watermark-lab.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: watermark-lab-service
            port:
              number: 80
```

### 3. Cloud Provider Deployments

#### AWS ECS

```json
{
  "family": "watermark-lab",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "watermark-lab",
      "image": "ghcr.io/terragon-labs/lm-watermark-lab:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:watermark-lab/db-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/watermark-lab",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### Google Cloud Run

```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: watermark-lab
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "2"
    spec:
      containerConcurrency: 100
      containers:
      - image: ghcr.io/terragon-labs/lm-watermark-lab:latest
        env:
        - name: ENVIRONMENT
          value: production
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: watermark-lab-secrets
              key: database-url
        ports:
        - containerPort: 8080
        resources:
          limits:
            memory: "2Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Deployment environment | `development` | Yes |
| `SECRET_KEY` | Application secret key | - | Yes |
| `DATABASE_URL` | Database connection string | - | Yes |
| `REDIS_URL` | Redis connection string | - | Yes |
| `API_WORKERS` | Number of API workers | `4` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `CORS_ORIGINS` | Allowed CORS origins | `[]` | No |

### Database Configuration

#### PostgreSQL Setup

```sql
-- Create database and user
CREATE DATABASE watermark_lab;
CREATE USER watermark_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE watermark_lab TO watermark_user;

-- Create extensions
\c watermark_lab;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
```

#### Redis Configuration

```conf
# redis.conf
bind 127.0.0.1
port 6379
timeout 0
tcp-keepalive 300
daemonize yes
supervised no
pidfile /var/run/redis/redis-server.pid
loglevel notice
logfile /var/log/redis/redis-server.log
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /var/lib/redis
maxmemory 2gb
maxmemory-policy allkeys-lru
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
```

### SSL/TLS Configuration

#### Nginx SSL Configuration

```nginx
# nginx.conf
server {
    listen 80;
    server_name api.watermark-lab.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.watermark-lab.com;

    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    location / {
        proxy_pass http://watermark-lab:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /health {
        proxy_pass http://watermark-lab:8080/health;
        access_log off;
    }
}
```

## Monitoring and Health Checks

### Health Check Endpoints

- `/health` - Basic health check
- `/ready` - Readiness check
- `/metrics` - Prometheus metrics
- `/api/v1/status` - Detailed status

### Monitoring Setup

```bash
# Start monitoring stack
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d

# Access monitoring services
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000 (admin/admin)"
echo "AlertManager: http://localhost:9093"
```

### Custom Metrics

```python
# Custom application metrics
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
watermark_generation_total = Counter('watermark_generation_total', 'Total watermark generations')
watermark_generation_duration = Histogram('watermark_generation_duration_seconds', 'Watermark generation duration')
active_models = Gauge('active_models_count', 'Number of active models')
```

## Security Considerations

### Container Security

1. **Non-root user**: All containers run as non-root users
2. **Read-only filesystem**: Most filesystem mounts are read-only
3. **Resource limits**: CPU and memory limits are enforced
4. **Security scanning**: Regular vulnerability scans

### Network Security

1. **TLS encryption**: All external traffic uses TLS 1.2+
2. **Network policies**: Kubernetes network policies restrict traffic
3. **Rate limiting**: API rate limiting prevents abuse
4. **Firewall rules**: Restrictive firewall rules

### Secrets Management

```bash
# Kubernetes secrets
kubectl create secret generic watermark-lab-secrets \
  --from-literal=database-url="postgresql://user:pass@host:5432/db" \
  --from-literal=redis-url="redis://host:6379/0" \
  --from-literal=secret-key="your-secret-key"

# AWS Secrets Manager
aws secretsmanager create-secret \
  --name "watermark-lab/database-url" \
  --secret-string "postgresql://user:pass@host:5432/db"
```

## Troubleshooting

### Common Issues

#### Application Won't Start

```bash
# Check logs
docker-compose logs watermark-lab

# Check environment variables
docker-compose exec watermark-lab env

# Test database connection
docker-compose exec watermark-lab python -c "
import os
import psycopg2
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
print('Database connection successful')
"
```

#### High Memory Usage

```bash
# Check memory usage
docker stats

# Analyze memory allocation
docker-compose exec watermark-lab python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

#### Slow API Responses

```bash
# Check API metrics
curl http://localhost:8080/metrics | grep http_request_duration

# Profile specific endpoints
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8080/api/v1/generate"
```

### Debug Mode

```bash
# Enable debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG

# Start with debug
docker-compose up watermark-lab
```

## Scaling

### Horizontal Scaling

#### Docker Compose

```bash
# Scale API workers
docker-compose up --scale watermark-lab=3

# Load balancer configuration
# Add nginx upstream configuration
```

#### Kubernetes

```bash
# Scale deployment
kubectl scale deployment watermark-lab --replicas=5

# Horizontal Pod Autoscaler
kubectl autoscale deployment watermark-lab --cpu-percent=70 --min=2 --max=10
```

### Vertical Scaling

#### Resource Optimization

```yaml
# Kubernetes resource tuning
resources:
  requests:
    memory: "4Gi"
    cpu: "1000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

### Database Scaling

#### Read Replicas

```python
# Database connection configuration
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'watermark_lab',
        'USER': 'watermark_user',
        'PASSWORD': 'password',
        'HOST': 'postgres-master',
        'PORT': '5432',
    },
    'read_replica': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'watermark_lab',
        'USER': 'watermark_reader',
        'PASSWORD': 'password',
        'HOST': 'postgres-replica',
        'PORT': '5432',
    }
}
```

### Caching Strategy

```python
# Redis cluster configuration
REDIS_CLUSTERS = {
    'cache': {
        'hosts': [
            {'host': 'redis-1', 'port': 6379},
            {'host': 'redis-2', 'port': 6379},
            {'host': 'redis-3', 'port': 6379},
        ]
    }
}
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup-db.sh
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump $DATABASE_URL > "backup_${DATE}.sql"
aws s3 cp "backup_${DATE}.sql" s3://watermark-lab-backups/
```

### Data Recovery

```bash
#!/bin/bash
# restore-db.sh
BACKUP_FILE=$1
psql $DATABASE_URL < $BACKUP_FILE
```

## Performance Tuning

### Application Tuning

```python
# Optimize model loading
MODEL_CACHE_SIZE = 5
BATCH_SIZE = 32
WORKER_CONNECTIONS = 1000
```

### Database Tuning

```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '1GB';
ALTER SYSTEM SET effective_cache_size = '3GB';
ALTER SYSTEM SET maintenance_work_mem = '256MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
```

This deployment guide provides comprehensive coverage of deployment strategies, configuration options, and operational procedures for LM Watermark Lab in various environments.