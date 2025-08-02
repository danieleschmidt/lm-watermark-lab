"""Test fixtures for LM Watermark Lab.

This module provides utilities for loading test data and configurations
used across the test suite.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Union

# Get the fixtures directory path
FIXTURES_DIR = Path(__file__).parent


def load_sample_configs() -> Dict[str, Any]:
    """Load sample watermarking configurations."""
    config_file = FIXTURES_DIR / "sample_configs.yaml"
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_sample_texts() -> Dict[str, Any]:
    """Load sample texts for testing."""
    texts_file = FIXTURES_DIR / "sample_texts.json"
    with open(texts_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_test_config(config_name: str) -> Dict[str, Any]:
    """Get a specific test configuration by name."""
    configs = load_sample_configs()
    if config_name not in configs:
        available = list(configs.keys())
        raise ValueError(f"Config '{config_name}' not found. Available: {available}")
    return configs[config_name]


def get_test_texts(category: str) -> List[str]:
    """Get test texts by category."""
    texts = load_sample_texts()
    if category not in texts:
        available = list(texts.keys())
        raise ValueError(f"Category '{category}' not found. Available: {available}")
    return texts[category]


def get_prompts_and_completions() -> List[Dict[str, str]]:
    """Get prompt-completion pairs for testing."""
    texts = load_sample_texts()
    prompts = texts.get("test_prompts", [])
    completions = texts.get("clean_texts", [])
    
    # Pair them up (cycle if lengths don't match)
    pairs = []
    for i, prompt in enumerate(prompts):
        completion_idx = i % len(completions)
        pairs.append({
            "prompt": prompt,
            "completion": completions[completion_idx]
        })
    
    return pairs


# Common test data that can be imported directly
TEST_CONFIGS = load_sample_configs()
TEST_TEXTS = load_sample_texts()

# Frequently used configurations
KIRCHENBAUER_CONFIG = TEST_CONFIGS["kirchenbauer_default"]
AARONSON_CONFIG = TEST_CONFIGS["aaronson_default"]
SIMPLE_CONFIG = TEST_CONFIGS["simple_test"]

# Frequently used text categories
SHORT_TEXTS = TEST_TEXTS["short_texts"]
CLEAN_TEXTS = TEST_TEXTS["clean_texts"]
WATERMARKED_TEXTS = TEST_TEXTS["potentially_watermarked"]
LONG_TEXTS = TEST_TEXTS["long_texts"]