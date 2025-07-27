# Development Guide

Welcome to the LM Watermark Lab development guide! This document will help you get started with contributing to the project.

## Quick Start

### Prerequisites

- Python 3.9+ (3.11 recommended)
- Docker and Docker Compose
- Git
- 8GB+ RAM (16GB recommended for model testing)
- GPU optional but recommended for performance testing

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/terragon-labs/lm-watermark-lab.git
   cd lm-watermark-lab
   ```

2. **Set up development environment**
   ```bash
   # Quick setup with make
   make dev-setup
   
   # Or manual setup
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   pre-commit install
   ```

3. **Start development services**
   ```bash
   # Start all services with Docker Compose
   docker-compose --profile dev up -d
   
   # Or start just the dependencies
   docker-compose up -d redis postgres
   
   # Run the app locally
   make dev
   ```

4. **Verify setup**
   ```bash
   # Run health check
   python scripts/health_check.py
   
   # Run quick tests
   make test-fast
   ```

## Development Workflow

### 1. Branch Strategy

We follow a Git Flow model:

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical production fixes

```bash
# Start a new feature
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name

# Work on your feature...

# Create pull request to develop
```

### 2. Code Quality

We maintain high code quality standards:

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Security scan
make security

# All quality checks
make quality
```

### 3. Testing

Comprehensive testing is essential:

```bash
# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# All tests except slow ones
make test-fast

# Full test suite
make test

# With coverage
make test-cov
```

### 4. Pre-commit Hooks

Pre-commit hooks ensure code quality:

```bash
# Install hooks (done automatically in dev-setup)
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Skip hooks (not recommended)
git commit --no-verify
```

## Project Structure

```
lm-watermark-lab/
├── src/watermark_lab/          # Main source code
│   ├── watermarking/           # Watermarking algorithms
│   ├── detection/              # Detection methods
│   ├── attacks/                # Attack simulations
│   ├── evaluation/             # Quality metrics
│   ├── api/                    # REST API
│   └── dashboard/              # Web dashboard
├── tests/                      # Test suites
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── e2e/                    # End-to-end tests
├── docs/                       # Documentation
├── scripts/                    # Utility scripts
├── monitoring/                 # Monitoring configs
└── configs/                    # Configuration files
```

## Core Components

### Watermarking Engine

The watermarking engine implements various algorithms:

```python
from watermark_lab.watermarking import WatermarkFactory

# Create a watermarker
watermarker = WatermarkFactory.create(
    method="kirchenbauer",
    model="gpt2-medium",
    gamma=0.25,
    delta=2.0
)

# Generate watermarked text
result = watermarker.generate("Your prompt here")
```

### Detection System

The detection system identifies watermarked text:

```python
from watermark_lab.detection import StatisticalDetector

# Create detector
detector = StatisticalDetector(watermark_config)

# Detect watermark
result = detector.detect(text)
print(f"Watermarked: {result.is_watermarked}")
print(f"Confidence: {result.confidence}")
```

### API Framework

The API provides HTTP endpoints:

```python
# Start development server
uvicorn watermark_lab.api.main:app --reload

# Test endpoints
curl -X POST http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"method": "kirchenbauer", "prompts": ["test"]}'
```

## Adding New Features

### 1. Watermarking Algorithm

To add a new watermarking algorithm:

1. **Create algorithm class**
   ```python
   # src/watermark_lab/watermarking/algorithms/my_algorithm.py
   from ..base import BaseWatermark
   
   class MyWatermark(BaseWatermark):
       def __init__(self, model: str, **kwargs):
           super().__init__(model)
           self.configure(**kwargs)
       
       def generate(self, prompt: str, **kwargs) -> str:
           # Implementation here
           pass
   ```

2. **Register in factory**
   ```python
   # src/watermark_lab/watermarking/factory.py
   from .algorithms.my_algorithm import MyWatermark
   
   @WatermarkFactory.register("my_algorithm")
   class MyWatermarkFactory:
       def create(self, **kwargs) -> MyWatermark:
           return MyWatermark(**kwargs)
   ```

3. **Add tests**
   ```python
   # tests/unit/test_my_algorithm.py
   def test_my_algorithm():
       watermark = MyWatermark("test-model")
       result = watermark.generate("test prompt")
       assert isinstance(result, str)
   ```

4. **Add configuration**
   ```yaml
   # configs/my_algorithm.yaml
   method: my_algorithm
   model: gpt2-medium
   parameters:
     param1: value1
     param2: value2
   ```

### 2. Detection Method

To add a new detection method:

1. **Create detector class**
   ```python
   # src/watermark_lab/detection/my_detector.py
   from .base import BaseDetector
   from .result import DetectionResult
   
   class MyDetector(BaseDetector):
       def detect(self, text: str) -> DetectionResult:
           # Implementation here
           pass
   ```

2. **Add tests**
   ```python
   # tests/unit/test_my_detector.py
   def test_my_detector():
       detector = MyDetector(config)
       result = detector.detect("test text")
       assert isinstance(result, DetectionResult)
   ```

### 3. API Endpoint

To add a new API endpoint:

1. **Create router**
   ```python
   # src/watermark_lab/api/routes/my_endpoint.py
   from fastapi import APIRouter
   
   router = APIRouter()
   
   @router.post("/my-endpoint")
   async def my_endpoint(request: MyRequest) -> MyResponse:
       # Implementation here
       pass
   ```

2. **Register router**
   ```python
   # src/watermark_lab/api/main.py
   from .routes.my_endpoint import router as my_router
   
   app.include_router(my_router, prefix="/api/v1")
   ```

3. **Add tests**
   ```python
   # tests/integration/test_my_endpoint.py
   def test_my_endpoint(api_client):
       response = api_client.post("/api/v1/my-endpoint", json={})
       assert response.status_code == 200
   ```

## Debugging

### Local Debugging

1. **VS Code Setup**
   ```json
   // .vscode/launch.json (already configured)
   {
     "name": "Python: Current File",
     "type": "python",
     "request": "launch",
     "program": "${file}",
     "console": "integratedTerminal"
   }
   ```

2. **Debug API Server**
   ```bash
   # Start with debugger
   python -m debugpy --listen 5678 --wait-for-client \
     -m uvicorn watermark_lab.api.main:app --reload
   ```

3. **Debug Tests**
   ```bash
   # Debug specific test
   python -m pytest tests/unit/test_specific.py::test_function -s -vv
   ```

### Docker Debugging

1. **Debug in container**
   ```bash
   # Start container with debug mode
   docker-compose -f docker-compose.yml -f docker-compose.debug.yml up
   ```

2. **Attach to running container**
   ```bash
   docker exec -it watermark-lab-app bash
   ```

### Performance Profiling

1. **CPU Profiling**
   ```bash
   python -m cProfile -o profile.stats scripts/profile_generation.py
   python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
   ```

2. **Memory Profiling**
   ```bash
   pip install memory-profiler
   mprof run scripts/profile_memory.py
   mprof plot
   ```

## Database Migrations

If you add database features:

1. **Create migration**
   ```bash
   alembic revision --autogenerate -m "Add new table"
   ```

2. **Apply migration**
   ```bash
   alembic upgrade head
   ```

3. **Rollback migration**
   ```bash
   alembic downgrade -1
   ```

## Configuration Management

### Environment Variables

Key environment variables for development:

```bash
# .env (copy from .env.example)
DEBUG=true
LOG_LEVEL=DEBUG
API_HOST=0.0.0.0
API_PORT=8080
REDIS_URL=redis://localhost:6379/0
DATABASE_URL=postgresql://postgres:password@localhost:5432/watermark_lab
MODEL_CACHE_PATH=./data/models
```

### Configuration Files

1. **Algorithm configs**: `configs/*.yaml`
2. **Development overrides**: `configs/development.yaml`
3. **Test configs**: `configs/test.yaml`

## Performance Guidelines

### Memory Management

1. **Model Loading**
   ```python
   # Good: Use context managers
   with ModelManager.load_model(model_name) as model:
       result = model.generate(prompt)
   
   # Bad: Manual memory management
   model = load_model(model_name)
   result = model.generate(prompt)
   del model  # May not free GPU memory
   ```

2. **Batch Processing**
   ```python
   # Good: Process in batches
   for batch in chunk_texts(texts, batch_size=32):
       results.extend(detector.detect_batch(batch))
   
   # Bad: One at a time
   for text in texts:
       results.append(detector.detect(text))
   ```

### GPU Utilization

1. **CUDA Best Practices**
   ```python
   import torch
   
   # Check GPU availability
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   # Clear cache when needed
   torch.cuda.empty_cache()
   
   # Use autocast for mixed precision
   with torch.autocast(device_type="cuda"):
       output = model(input)
   ```

## Contributing Guidelines

### Pull Request Process

1. **Create feature branch**
2. **Make changes with tests**
3. **Update documentation**
4. **Run quality checks**
5. **Create pull request**
6. **Address review feedback**
7. **Merge when approved**

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance impact considered
- [ ] Breaking changes documented

### Release Process

1. **Version Bump**: Update version in `pyproject.toml`
2. **Changelog**: Update `CHANGELOG.md`
3. **Tag Release**: Create git tag
4. **GitHub Release**: Create release on GitHub
5. **Package Upload**: Automatic via CI/CD

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure PYTHONPATH is set
   export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
   ```

2. **Model Download Errors**
   ```bash
   # Set HuggingFace cache directory
   export HF_HOME=./data/hf_cache
   ```

3. **GPU Memory Issues**
   ```bash
   # Monitor GPU usage
   nvidia-smi
   
   # Clear PyTorch cache
   python -c "import torch; torch.cuda.empty_cache()"
   ```

4. **Test Failures**
   ```bash
   # Run tests with verbose output
   pytest -xvs tests/unit/test_failing.py
   
   # Skip slow tests
   pytest -m "not slow" tests/
   ```

### Getting Help

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Discord**: Join our development community
- **Email**: Contact the team at dev@terragonlabs.com

## Resources

- **Python Style Guide**: [PEP 8](https://pep8.org/)
- **FastAPI Documentation**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)
- **PyTorch Documentation**: [pytorch.org/docs](https://pytorch.org/docs/)
- **Docker Best Practices**: [docs.docker.com/develop/best-practices](https://docs.docker.com/develop/best-practices/)

Happy coding! 🚀