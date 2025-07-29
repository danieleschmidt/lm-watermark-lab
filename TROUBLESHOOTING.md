# Troubleshooting Guide

This guide helps resolve common issues encountered when using the LM Watermark Lab.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Model Loading Problems](#model-loading-problems)
- [Memory Issues](#memory-issues)
- [Performance Problems](#performance-problems)
- [Detection Issues](#detection-issues)
- [API and CLI Problems](#api-and-cli-problems)
- [Docker Issues](#docker-issues)
- [Development Setup Issues](#development-setup-issues)

## Installation Issues

### ImportError: No module named 'watermark_lab'

**Problem**: Package not installed or not in Python path.

**Solutions**:
```bash
# Install in development mode
pip install -e .

# Or install from PyPI
pip install lm-watermark-lab

# Check installation
python -c "import watermark_lab; print(watermark_lab.__version__)"
```

### CUDA/GPU Issues

**Problem**: PyTorch not detecting GPU or CUDA version mismatch.

**Solutions**:
```bash
# Check CUDA version
nvidia-smi

# Install correct PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU detection
python -c "import torch; print(torch.cuda.is_available())"
```

### Dependency Conflicts

**Problem**: Package version conflicts during installation.

**Solutions**:
```bash
# Create fresh environment
conda create -n watermark-lab python=3.9
conda activate watermark-lab

# Or use virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with specific versions
pip install -r requirements.txt
```

## Model Loading Problems

### HuggingFace Model Download Issues

**Problem**: Unable to download models from HuggingFace Hub.

**Solutions**:
```python
# Set cache directory
import os
os.environ['TRANSFORMERS_CACHE'] = '/path/to/cache'

# Use offline mode if model already cached
from transformers import AutoModel
model = AutoModel.from_pretrained('gpt2', local_files_only=True)

# Manual download
from huggingface_hub import snapshot_download
snapshot_download(repo_id="gpt2", cache_dir="/path/to/cache")
```

### Model Loading Timeout

**Problem**: Model loading takes too long or times out.

**Solutions**:
```python
# Increase timeout
import torch
torch.hub.set_dir('/path/to/cache')

# Use smaller model for testing
model_name = "gpt2"  # Instead of "gpt2-xl"

# Load with reduced precision
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
```

## Memory Issues

### Out of Memory (OOM) Errors

**Problem**: GPU or RAM runs out of memory during processing.

**Solutions**:

#### GPU Memory
```python
# Reduce batch size
batch_size = 1  # Start small

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Clear cache between operations
import torch
torch.cuda.empty_cache()

# Use mixed precision
from transformers import TrainingArguments
training_args = TrainingArguments(
    fp16=True,  # or bf16=True for newer GPUs
    dataloader_pin_memory=False
)
```

#### System Memory
```python
# Process in chunks
def process_large_dataset(dataset, chunk_size=1000):
    for i in range(0, len(dataset), chunk_size):
        chunk = dataset[i:i+chunk_size]
        yield process_chunk(chunk)

# Use memory mapping
import numpy as np
data = np.memmap('large_file.dat', dtype='float32', mode='r')
```

### Memory Leaks

**Problem**: Memory usage increases over time.

**Solutions**:
```python
# Explicit cleanup
import gc
import torch

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Call cleanup periodically
for i, batch in enumerate(dataloader):
    process_batch(batch)
    if i % 100 == 0:
        cleanup()
```

## Performance Problems

### Slow Generation

**Problem**: Text generation is slower than expected.

**Solutions**:
```python
# Use appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimize generation parameters
generation_config = {
    "max_length": 100,        # Reduce if too long
    "num_beams": 1,          # Use greedy search
    "do_sample": False,      # Disable sampling
    "pad_token_id": tokenizer.eos_token_id,
    "use_cache": True        # Enable KV caching
}

# Batch processing
texts = ["prompt1", "prompt2", "prompt3"]
inputs = tokenizer(texts, return_tensors="pt", padding=True)
outputs = model.generate(**inputs, **generation_config)
```

### Slow Detection

**Problem**: Watermark detection is taking too long.

**Solutions**:
```python
# Use statistical detection (faster than neural)
from watermark_lab.detection import StatisticalDetector
detector = StatisticalDetector(config)

# Batch processing
results = detector.detect_batch(texts, batch_size=32)

# Reduce text length for testing
test_text = full_text[:500]  # First 500 characters
```

## Detection Issues

### False Positives/Negatives

**Problem**: Detection results are unreliable.

**Solutions**:
```python
# Check watermark configuration
print(f"Watermark config: {watermarker.get_config()}")
print(f"Detector config: {detector.config}")

# Verify configs match
assert watermarker.get_config()['method'] == detector.config['method']

# Test with known watermarked text
test_text = watermarker.generate("Test prompt")
result = detector.detect(test_text)
print(f"Should be detected: {result.is_watermarked}")

# Adjust detection threshold
detector.threshold = 0.95  # More conservative
```

### Inconsistent Results

**Problem**: Detection results vary between runs.

**Solutions**:
```python
# Set random seeds
import random
import numpy as np
import torch

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seeds(42)

# Use deterministic algorithms
torch.use_deterministic_algorithms(True)

# Check for non-deterministic operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## API and CLI Problems

### API Server Won't Start

**Problem**: FastAPI server fails to start.

**Solutions**:
```bash
# Check port availability
netstat -an | grep 8000

# Use different port
uvicorn watermark_lab.api.main:app --port 8080

# Check dependencies
pip install "fastapi[all]" uvicorn

# Debug mode
uvicorn watermark_lab.api.main:app --reload --log-level debug
```

### CLI Commands Not Working

**Problem**: Command-line interface commands fail.

**Solutions**:
```bash
# Check installation
pip show lm-watermark-lab

# Reinstall CLI entry points
pip install -e . --force-reinstall

# Use module invocation
python -m watermark_lab.cli.main generate --help

# Check PATH
echo $PATH
which watermark-lab
```

## Docker Issues

### Container Build Failures

**Problem**: Docker build fails with errors.

**Solutions**:
```bash
# Clear Docker cache
docker system prune -f

# Build with no cache
docker build --no-cache -t watermark-lab .

# Check for space issues
docker system df

# Use multi-stage build optimization
# (Check Dockerfile for COPY order)
```

### Container Runtime Issues

**Problem**: Container runs but functionality is broken.

**Solutions**:
```bash
# Check container logs
docker logs watermark-lab

# Interactive debugging
docker run -it watermark-lab /bin/bash

# Mount local code for development
docker run -v $(pwd):/app watermark-lab

# Check environment variables
docker run watermark-lab env
```

## Development Setup Issues

### Pre-commit Hooks Failing

**Problem**: Pre-commit hooks prevent commits.

**Solutions**:
```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Update hooks
pre-commit autoupdate

# Run manually to fix issues
pre-commit run --all-files

# Skip hooks if necessary (use sparingly)
git commit --no-verify
```

### Test Failures

**Problem**: Tests fail when running pytest.

**Solutions**:
```bash
# Install test dependencies
pip install -e ".[test]"

# Run specific test
pytest tests/test_detection.py -v

# Run with coverage
pytest --cov=src/watermark_lab

# Skip slow tests
pytest -m "not slow"

# Debug failing test
pytest tests/test_detection.py::test_function -pdb
```

### Import Errors in Development

**Problem**: Cannot import modules during development.

**Solutions**:
```bash
# Install in editable mode
pip install -e .

# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Verify package structure
ls -la src/watermark_lab/

# Check __init__.py files exist
find src/ -name "__init__.py"
```

## Getting Help

### Debug Information Collection

When reporting issues, please provide:

```python
# System information
import sys
import torch
import transformers
import watermark_lab

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Watermark Lab: {watermark_lab.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
```

### Log Configuration

Enable detailed logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable transformers logging
import transformers
transformers.logging.set_verbosity_debug()
```

### Performance Profiling

Profile slow operations:

```python
import cProfile
import pstats

def profile_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    return result
```

## Community Support

- **GitHub Issues**: [Report bugs and request features](https://github.com/terragon-labs/lm-watermark-lab/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/terragon-labs/lm-watermark-lab/discussions)
- **Discord**: [Join our community chat](https://discord.gg/terragon-labs)
- **Documentation**: [Full documentation](https://lm-watermark-lab.readthedocs.io)

---

If you encounter an issue not covered here, please [open an issue](https://github.com/terragon-labs/lm-watermark-lab/issues) with:
1. Detailed description of the problem
2. Steps to reproduce
3. Error messages and stack traces
4. System information (Python version, OS, hardware)
5. Minimal code example that reproduces the issue