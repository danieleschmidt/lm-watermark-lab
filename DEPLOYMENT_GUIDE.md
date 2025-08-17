# Production Deployment Guide - Enhanced LM Watermark Lab

This guide covers deploying the enhanced LM Watermark Lab with all three generations of improvements: robustness, scalability, and enterprise features.

## 🚀 Quick Start Deployment

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Kubernetes (optional, for production)
- 4GB+ RAM, 2+ CPU cores

### Local Development Setup

```bash
# Clone and setup
git clone <repository-url>
cd lm-watermark-lab

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[all]"

# Run basic tests
pytest tests/ -v

# Start local development server
uvicorn watermark_lab.api.main:app --reload --port 8080
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check services
docker-compose ps

# View logs
docker-compose logs -f watermark-lab
```

## 🏗️ Architecture Overview

### Enhanced Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer / WAF                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│                API Gateway                                  │
│  - Enhanced Security (Threat Detection)                    │
│  - Rate Limiting & Circuit Breakers                        │
│  - Real-time Analytics                                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
│  Worker 1   │ │  Worker 2 │ │  Worker N │
│             │ │           │ │           │
│ - Resilience│ │ - Scaling │ │ - Monitor │
│ - Security  │ │ - Distrib │ │ - Analytics│
└─────────────┘ └───────────┘ └───────────┘
        │             │             │
        └─────────────┼─────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 Data Layer                                  │
│  - Redis (Caching, Session)                               │
│  - PostgreSQL (Metrics, Audit)                            │
│  - File Storage (Models, Artifacts)                       │
└─────────────────────────────────────────────────────────────┘
```

## 🛡️ Security Configuration

### Threat Detection Setup

```python
# config/security.py
from watermark_lab.security.threat_detection import get_global_threat_engine

# Configure threat detection
engine = get_global_threat_engine()

# Custom threat response
def block_malicious_ip(threat, response):
    if response.action == "block":
        # Integrate with firewall/WAF
        update_firewall_rules(threat.source_ip, duration=response.duration)

engine.register_response_handler(ThreatType.SQL_INJECTION, block_malicious_ip)
```

### Enhanced Authentication

```python
# config/auth.py
from watermark_lab.security.advanced_security import AdvancedSecurityManager

security = AdvancedSecurityManager()
security.enable_mfa()
security.configure_session_security(
    session_timeout=3600,
    secure_cookies=True,
    csrf_protection=True
)
```

## 📊 Monitoring & Analytics

### Real-time Analytics Setup

```python
# config/monitoring.py
from watermark_lab.monitoring.realtime_analytics import get_global_analytics
from watermark_lab.monitoring.realtime_analytics import AlertRule, AlertCondition

analytics = get_global_analytics()

# Configure critical alerts
cpu_alert = AlertRule(
    name="high_cpu_usage",
    metric_name="system_cpu_percent",
    condition=AlertCondition.GREATER_THAN,
    threshold=80.0,
    time_window=300  # 5 minutes
)

memory_alert = AlertRule(
    name="high_memory_usage", 
    metric_name="system_memory_percent",
    condition=AlertCondition.GREATER_THAN,
    threshold=90.0,
    time_window=300
)

analytics.add_alert_rule(cpu_alert)
analytics.add_alert_rule(memory_alert)
```

### Prometheus Integration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'watermark-lab'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    
  - job_name: 'system-metrics'
    static_configs:
      - targets: ['localhost:9100']  # Node exporter
```

## ⚡ Scaling Configuration

### Adaptive Auto-Scaling

```python
# config/scaling.py
from watermark_lab.optimization.adaptive_scaling import AdaptiveAutoScaler, ScalingRule
from watermark_lab.optimization.adaptive_scaling import ScalingTrigger

# Configure auto-scaling rules
cpu_scaling_rule = ScalingRule(
    name="cpu_based_scaling",
    trigger=ScalingTrigger.CPU_USAGE,
    scale_up_threshold=75.0,
    scale_down_threshold=25.0,
    scale_up_adjustment=2,
    scale_down_adjustment=1,
    min_instances=2,
    max_instances=20,
    cooldown_period=300
)

queue_scaling_rule = ScalingRule(
    name="queue_based_scaling",
    trigger=ScalingTrigger.QUEUE_LENGTH,
    scale_up_threshold=100.0,  # Queue length
    scale_down_threshold=10.0,
    scale_up_adjustment=3,
    scale_down_adjustment=1,
    min_instances=1,
    max_instances=50,
    cooldown_period=180
)

# Initialize auto-scaler
scaler = AdaptiveAutoScaler(metric_provider, resource_manager)
scaler.add_scaling_rule(cpu_scaling_rule)
scaler.add_scaling_rule(queue_scaling_rule)
scaler.start()
```

### Distributed Processing

```python
# config/distributed.py
from watermark_lab.optimization.distributed_processing import get_global_engine

# Configure distributed processing
engine = get_global_engine()

# Register watermarking functions for distribution
@distributed_task(priority=TaskPriority.HIGH)
def distributed_watermark_generation(text, config):
    from watermark_lab import WatermarkFactory
    watermarker = WatermarkFactory.create(**config)
    return watermarker.generate(text)

@distributed_task(priority=TaskPriority.NORMAL)
def distributed_detection(text, config):
    from watermark_lab import WatermarkDetector
    detector = WatermarkDetector(config)
    return detector.detect(text)
```

## 🐳 Container Deployment

### Production Dockerfile

```dockerfile
# Dockerfile.production
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY tests/ tests/
COPY pyproject.toml .

# Install application
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["uvicorn", "watermark_lab.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Docker Compose for Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  watermark-lab:
    build:
      context: .
      dockerfile: Dockerfile.production
    ports:
      - "8080:8080"
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@postgres:5432/watermark_lab
      - LOG_LEVEL=INFO
      - WORKERS=4
    depends_on:
      - redis
      - postgres
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=watermark_lab
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:
```

## ☸️ Kubernetes Deployment

### Production Deployment

```yaml
# k8s/deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: watermark-lab
  labels:
    app: watermark-lab
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
        image: watermark-lab:latest
        ports:
        - containerPort: 8080
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: watermark-lab-secrets
              key: redis-url
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: watermark-lab-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
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
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: watermark-lab-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: watermark-lab
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 🔧 Configuration Management

### Environment Variables

```bash
# .env.production
# Application
APP_NAME=LM-Watermark-Lab
APP_VERSION=2.0.0
LOG_LEVEL=INFO
DEBUG=false

# Security
SECRET_KEY=your-super-secret-key
JWT_SECRET=your-jwt-secret
ENCRYPTION_KEY=your-encryption-key

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/watermark_lab
REDIS_URL=redis://localhost:6379

# Scaling
AUTO_SCALING_ENABLED=true
MIN_WORKERS=2
MAX_WORKERS=20
WORKER_TIMEOUT=300

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=30

# Security Features
THREAT_DETECTION_ENABLED=true
RATE_LIMITING_ENABLED=true
AUDIT_LOGGING_ENABLED=true

# Performance
DISTRIBUTED_PROCESSING_ENABLED=true
CIRCUIT_BREAKER_ENABLED=true
CACHING_ENABLED=true
```

### Feature Flags

```python
# config/features.py
FEATURE_FLAGS = {
    'enhanced_security': True,
    'predictive_scaling': True,
    'distributed_processing': True,
    'real_time_analytics': True,
    'advanced_monitoring': True,
    'threat_detection': True,
    'behavioral_analysis': True,
    'adaptive_resilience': True
}
```

## 📈 Performance Optimization

### Resource Allocation

```yaml
# Resource recommendations for different scales

# Small deployment (< 1000 req/day)
resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "512Mi"
    cpu: "500m"
replicas: 1-2

# Medium deployment (< 100k req/day)  
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
replicas: 2-5

# Large deployment (< 1M req/day)
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
replicas: 5-20
```

### Database Optimization

```sql
-- PostgreSQL optimization for metrics storage
CREATE INDEX CONCURRENTLY idx_metrics_timestamp ON metrics(timestamp);
CREATE INDEX CONCURRENTLY idx_threats_source_ip ON threat_events(source_ip);
CREATE INDEX CONCURRENTLY idx_tasks_status ON distributed_tasks(status);

-- Partitioning for large datasets
CREATE TABLE metrics_2024 PARTITION OF metrics
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

## 🚨 Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
kubectl top pods
docker stats

# Scale down if needed
kubectl scale deployment watermark-lab --replicas=2
```

#### Circuit Breaker Triggering
```python
# Check circuit breaker status
from watermark_lab.utils.enhanced_resilience import get_global_resilience_manager
manager = get_global_resilience_manager()
metrics = manager.get_comprehensive_metrics()
print(metrics['circuit_breakers'])
```

#### Threat Detection False Positives
```python
# Adjust threat detection thresholds
from watermark_lab.security.threat_detection import get_global_threat_engine
engine = get_global_threat_engine()

# Whitelist trusted IPs
engine.threat_intelligence.add_trusted_ip("192.168.1.0/24")
```

### Health Checks

```bash
# API health check
curl http://localhost:8080/health

# Detailed system status
curl http://localhost:8080/status

# Metrics endpoint
curl http://localhost:8080/metrics
```

## 📋 Deployment Checklist

### Pre-deployment
- [ ] Security configuration reviewed
- [ ] Resource limits configured
- [ ] Database migrations applied
- [ ] SSL certificates installed
- [ ] Monitoring configured
- [ ] Backup strategy implemented

### Post-deployment
- [ ] Health checks passing
- [ ] Metrics collection working
- [ ] Alerts configured
- [ ] Auto-scaling tested
- [ ] Security scans completed
- [ ] Performance baseline established

### Production Readiness
- [ ] Load testing completed
- [ ] Disaster recovery tested
- [ ] Documentation updated
- [ ] Team trained on operations
- [ ] Incident response procedures ready
- [ ] Security audit completed

## 🔗 Additional Resources

- [API Documentation](docs/API.md)
- [Security Guide](docs/SECURITY.md) 
- [Performance Tuning](docs/PERFORMANCE.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
- [Architecture Decisions](docs/adr/)

---

This deployment guide covers the enhanced LM Watermark Lab with all three generations of improvements. For specific deployment scenarios or advanced configurations, refer to the detailed documentation in the `docs/` directory.