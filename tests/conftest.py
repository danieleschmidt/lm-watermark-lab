"""Pytest configuration and fixtures for LM Watermark Lab."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Generator, Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

# Test configuration
os.environ["TESTING"] = "1"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["MODEL_CACHE_PATH"] = str(Path(tempfile.gettempdir()) / "test_models")


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Test configuration settings."""
    return {
        "model_cache_path": Path(tempfile.gettempdir()) / "test_models",
        "max_length": 100,
        "test_timeout": 30,
        "mock_models": True,
        "use_small_models": True,
    }


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_texts() -> Dict[str, str]:
    """Sample texts for testing."""
    return {
        "short": "The quick brown fox jumps over the lazy dog.",
        "medium": " ".join([
            "The future of artificial intelligence holds immense promise.",
            "Machine learning algorithms continue to evolve and improve.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models can recognize patterns in complex data."
        ]),
        "long": " ".join([
            "In the rapidly evolving landscape of artificial intelligence,",
            "researchers and developers are constantly pushing the boundaries",
            "of what's possible with machine learning and deep learning techniques.",
            "From natural language processing to computer vision,",
            "AI systems are becoming increasingly sophisticated and capable.",
            "The integration of large language models has revolutionized",
            "how we interact with technology and process information.",
            "As these systems become more powerful, the need for robust",
            "security measures and watermarking techniques becomes paramount.",
            "Watermarking provides a crucial mechanism for identifying",
            "AI-generated content and ensuring accountability in AI systems."
        ])
    }


@pytest.fixture
def mock_model():
    """Mock language model for testing."""
    mock = MagicMock()
    mock.generate.return_value = "This is a mock generated text."
    mock.config = {"model_type": "mock", "vocab_size": 50000}
    mock.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock.tokenizer.decode.return_value = "decoded text"
    return mock


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    mock = MagicMock()
    mock.encode.return_value = [1, 2, 3, 4, 5]
    mock.decode.return_value = "decoded text"
    mock.vocab_size = 50000
    mock.pad_token_id = 0
    mock.eos_token_id = 2
    return mock


@pytest.fixture
def watermark_configs() -> Dict[str, Dict[str, Any]]:
    """Sample watermark configurations."""
    return {
        "kirchenbauer": {
            "method": "kirchenbauer",
            "model": "mock-model",
            "gamma": 0.25,
            "delta": 2.0,
            "seed": 42
        },
        "aaronson": {
            "method": "aaronson",
            "model": "mock-model",
            "key": "test_key_123",
            "hash_length": 256
        },
        "simple": {
            "method": "simple",
            "model": "mock-model",
            "pattern": [0, 1, 0, 1]
        }
    }


@pytest.fixture
def detection_results():
    """Sample detection results."""
    return {
        "positive": {
            "is_watermarked": True,
            "confidence": 0.95,
            "p_value": 0.001,
            "method": "kirchenbauer",
            "details": {"test_statistic": 4.5, "threshold": 2.0}
        },
        "negative": {
            "is_watermarked": False,
            "confidence": 0.12,
            "p_value": 0.8,
            "method": "kirchenbauer",
            "details": {"test_statistic": 0.5, "threshold": 2.0}
        }
    }


@pytest.fixture
def api_client() -> Generator[TestClient, None, None]:
    """FastAPI test client."""
    # Import here to avoid circular imports during test collection
    try:
        from watermark_lab.api.main import app
        with TestClient(app) as client:
            yield client
    except ImportError:
        # If API module doesn't exist yet, create a mock client
        mock_app = MagicMock()
        yield TestClient(mock_app)


@pytest.fixture
def redis_mock():
    """Mock Redis client for testing."""
    with patch('redis.Redis') as mock_redis:
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        yield mock_client


@pytest.fixture
def celery_mock():
    """Mock Celery for testing."""
    with patch('celery.Celery') as mock_celery:
        mock_app = MagicMock()
        mock_celery.return_value = mock_app
        yield mock_app


@pytest.fixture
def wandb_mock():
    """Mock Weights & Biases for testing."""
    with patch('wandb.init') as mock_wandb:
        yield mock_wandb


@pytest.fixture(autouse=True)
def mock_torch_models():
    """Mock torch model loading to avoid downloading during tests."""
    with patch('torch.load') as mock_load, \
         patch('transformers.AutoModel.from_pretrained') as mock_model, \
         patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
        
        # Configure mock model
        mock_model_instance = MagicMock()
        mock_model_instance.config.vocab_size = 50000
        mock_model.return_value = mock_model_instance
        
        # Configure mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.vocab_size = 50000
        mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer_instance.decode.return_value = "mock decoded text"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        yield {
            'model': mock_model,
            'tokenizer': mock_tokenizer,
            'load': mock_load
        }


@pytest.fixture
def gpu_available() -> bool:
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture
def skip_if_no_gpu(gpu_available):
    """Skip test if no GPU is available."""
    if not gpu_available:
        pytest.skip("GPU not available")


@pytest.fixture
def benchmark_data():
    """Sample data for benchmarking tests."""
    return {
        "prompts": [
            "Write a story about artificial intelligence.",
            "Explain the concept of machine learning.",
            "Describe the future of technology.",
            "What are the benefits of automation?",
            "How does natural language processing work?"
        ],
        "expected_lengths": [100, 150, 200, 120, 180],
        "quality_thresholds": {
            "perplexity": 50.0,
            "bleu": 0.3,
            "bertscore": 0.8
        }
    }


@pytest.fixture
def attack_scenarios():
    """Sample attack scenarios for robustness testing."""
    return {
        "paraphrase_light": {
            "strength": "light",
            "success_rate_threshold": 0.3
        },
        "paraphrase_heavy": {
            "strength": "heavy",
            "success_rate_threshold": 0.7
        },
        "truncation": {
            "truncate_ratio": 0.5,
            "success_rate_threshold": 0.5
        },
        "synonym_replacement": {
            "replacement_ratio": 0.1,
            "success_rate_threshold": 0.2
        }
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API test"
    )
    config.addinivalue_line(
        "markers", "cli: mark test as CLI test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid or any(keyword in item.nodeid.lower() 
                                       for keyword in ["benchmark", "load", "stress"]):
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark GPU tests
        if any(keyword in item.nodeid.lower() 
               for keyword in ["gpu", "cuda", "torch"]):
            item.add_marker(pytest.mark.gpu)
        
        # Mark API tests
        if "api" in item.nodeid or "test_api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        
        # Mark CLI tests
        if "cli" in item.nodeid or "test_cli" in item.nodeid:
            item.add_marker(pytest.mark.cli)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # Ensure test directories exist
    test_dirs = [
        Path(tempfile.gettempdir()) / "test_models",
        Path(tempfile.gettempdir()) / "test_data",
        Path(tempfile.gettempdir()) / "test_cache"
    ]
    
    for test_dir in test_dirs:
        test_dir.mkdir(exist_ok=True)
    
    yield
    
    # Cleanup (optional - temp dirs are usually cleaned automatically)
    for test_dir in test_dirs:
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)