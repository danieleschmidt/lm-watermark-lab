# Performance Guide

This document outlines performance considerations, optimizations, and benchmarks for the LM Watermark Lab.

## Performance Overview

The watermarking process involves several computational steps:
1. **Text Generation**: Model inference with watermark injection
2. **Detection**: Statistical analysis of token distributions
3. **Attack Simulation**: Text transformation and analysis
4. **Evaluation**: Metric computation across datasets

## Benchmarks

### Generation Performance

| Method | Tokens/sec | Memory (GB) | Model Size |
|--------|------------|-------------|------------|
| Kirchenbauer | 45 | 2.1 | GPT2-Medium |
| MarkLLM-KGW | 38 | 2.3 | GPT2-Medium |
| Aaronson | 42 | 2.0 | GPT2-Medium |
| Zhao | 40 | 2.2 | GPT2-Medium |

### Detection Performance

| Method | Texts/sec | Accuracy | Memory (MB) |
|--------|-----------|----------|-------------|
| Statistical | 1250 | 99.2% | 50 |
| Neural | 850 | 97.8% | 150 |
| Multi-method | 320 | 99.5% | 200 |

## Optimization Strategies

### 1. Model Optimization

```python
# Use quantized models for faster inference
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "gpt2-medium",
    torch_dtype=torch.float16,  # Half precision
    device_map="auto",          # Automatic device placement
    load_in_8bit=True          # 8-bit quantization
)
```

### 2. Batch Processing

```python
# Process multiple texts simultaneously
detector = WatermarkDetector(config)
results = detector.detect_batch(texts, batch_size=32)
```

### 3. Caching

```python
# Cache model outputs and intermediate results
from functools import lru_cache

@lru_cache(maxsize=1000)
def compute_watermark_score(text_hash):
    # Expensive computation
    return score
```

### 4. Parallel Processing

```python
# Use multiprocessing for CPU-bound tasks
from multiprocessing import Pool
import concurrent.futures

def process_texts_parallel(texts, num_workers=4):
    with Pool(num_workers) as pool:
        results = pool.map(detect_watermark, texts)
    return results
```

### 5. GPU Acceleration

```python
# Leverage GPU for tensor operations
import torch

# Ensure CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
```

## Memory Management

### 1. Model Memory

- **Model Sharding**: Split large models across multiple GPUs
- **Gradient Checkpointing**: Trade compute for memory during training
- **Model Pruning**: Remove unnecessary parameters

### 2. Data Memory

- **Streaming**: Process data in chunks rather than loading everything
- **Memory Mapping**: Use memory-mapped files for large datasets
- **Garbage Collection**: Explicit cleanup of large objects

```python
import gc
import torch

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()
```

## Profiling and Monitoring

### 1. Performance Profiling

```python
# Profile code execution
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
detector.detect(text)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```

### 2. Memory Profiling

```python
# Monitor memory usage
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function to profile
    pass
```

### 3. GPU Monitoring

```python
# Monitor GPU utilization
import GPUtil

gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f"GPU {gpu.id}: {gpu.load*100:.1f}% load, {gpu.memoryUtil*100:.1f}% memory")
```

## Configuration Tuning

### Environment Variables

```bash
# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4

# HuggingFace settings
export TRANSFORMERS_CACHE=/path/to/cache
export HF_DATASETS_CACHE=/path/to/cache
```

### Model Configuration

```python
# Optimize model configuration
config = {
    "max_length": 512,          # Shorter sequences = faster
    "num_beams": 1,             # Greedy search = faster
    "do_sample": False,         # Deterministic = faster
    "pad_token_id": tokenizer.eos_token_id,
    "use_cache": True,          # Enable KV caching
}
```

## Benchmarking Tools

### 1. Generation Benchmarks

```python
import time
from watermark_lab.benchmarks import GenerationBenchmark

benchmark = GenerationBenchmark()
results = benchmark.measure_generation_speed(
    methods=["kirchenbauer", "markllm"],
    text_lengths=[100, 500, 1000],
    num_samples=100
)
```

### 2. Detection Benchmarks

```python
from watermark_lab.benchmarks import DetectionBenchmark

benchmark = DetectionBenchmark()
results = benchmark.measure_detection_speed(
    texts=test_texts,
    methods=["statistical", "neural"],
    batch_sizes=[1, 8, 32]
)
```

### 3. End-to-End Benchmarks

```python
from watermark_lab.benchmarks import E2EBenchmark

benchmark = E2EBenchmark()
results = benchmark.run_full_pipeline(
    num_texts=1000,
    include_attacks=True,
    include_evaluation=True
)
```

## Performance Best Practices

### 1. Model Selection
- Use the smallest model that meets accuracy requirements
- Consider distilled models for production deployment
- Evaluate trade-offs between speed and quality

### 2. Batch Processing
- Always use batching for multiple inputs
- Find optimal batch size through experimentation
- Consider memory constraints when setting batch size

### 3. Hardware Optimization
- Use GPUs for model inference
- Use multiple CPUs for parallel detection
- Consider specialized hardware (TPUs) for large-scale deployment

### 4. Algorithmic Optimization
- Choose appropriate watermark methods for use case
- Use early stopping in iterative algorithms
- Implement efficient data structures

### 5. System-Level Optimization
- Use appropriate Python version and libraries
- Enable compiler optimizations
- Consider using compiled languages for critical paths

## Scaling Considerations

### 1. Horizontal Scaling
- Distribute workload across multiple machines
- Use message queues for async processing
- Implement load balancing

### 2. Vertical Scaling
- Optimize single-node performance first
- Add more powerful hardware as needed
- Monitor resource utilization

### 3. Caching Strategies
- Cache frequent computations
- Use Redis for distributed caching
- Implement cache invalidation strategies

## Monitoring in Production

### 1. Metrics to Track
- Throughput (texts/second)
- Latency (time per text)
- Error rates
- Resource utilization (CPU, GPU, memory)

### 2. Alerting
- Set up alerts for performance degradation
- Monitor for memory leaks
- Track error rate spikes

### 3. Logging
- Log performance metrics
- Include timing information in logs
- Use structured logging for analysis

## Troubleshooting Performance Issues

### Common Issues
1. **Memory Leaks**: Monitor memory usage over time
2. **GPU OOM**: Reduce batch size or model size
3. **Slow Detection**: Profile detection algorithms
4. **High Latency**: Check for blocking operations

### Debugging Tools
- `torch.profiler` for PyTorch operations
- `py-spy` for Python profiling
- `nvidia-smi` for GPU monitoring
- `htop` for system resource monitoring

---

For specific performance questions or issues, please refer to our [troubleshooting guide](docs/TROUBLESHOOTING.md) or open an issue on GitHub.