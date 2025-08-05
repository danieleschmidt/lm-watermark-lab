"""Input validation utilities."""

import re
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

from .exceptions import ValidationError


def validate_text(
    text: str,
    min_length: int = 1,
    max_length: Optional[int] = None,
    allow_empty: bool = False,
    check_encoding: bool = True
) -> str:
    """Validate input text."""
    
    if text is None:
        raise ValidationError("Text cannot be None")
    
    if not isinstance(text, str):
        raise ValidationError(f"Text must be a string, got {type(text)}")
    
    if not allow_empty and len(text.strip()) == 0:
        raise ValidationError("Text cannot be empty")
    
    if len(text) < min_length:
        raise ValidationError(f"Text must be at least {min_length} characters long")
    
    if max_length and len(text) > max_length:
        raise ValidationError(f"Text must be at most {max_length} characters long")
    
    if check_encoding:
        try:
            text.encode('utf-8')
        except UnicodeEncodeError as e:
            raise ValidationError(f"Text contains invalid UTF-8 characters: {e}")
    
    return text.strip()


def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration against schema."""
    
    if not isinstance(config, dict):
        raise ValidationError(f"Configuration must be a dictionary, got {type(config)}")
    
    validated_config = {}
    
    for key, requirements in schema.items():
        value = config.get(key)
        
        # Check required fields
        if requirements.get("required", False) and value is None:
            raise ValidationError(f"Required field '{key}' is missing")
        
        # Skip validation if value is None and not required
        if value is None:
            continue
        
        # Type validation
        expected_type = requirements.get("type")
        if expected_type and not isinstance(value, expected_type):
            raise ValidationError(f"Field '{key}' must be of type {expected_type.__name__}, got {type(value).__name__}")
        
        # Range validation for numbers
        if isinstance(value, (int, float)):
            min_val = requirements.get("min")
            max_val = requirements.get("max")
            
            if min_val is not None and value < min_val:
                raise ValidationError(f"Field '{key}' must be >= {min_val}, got {value}")
            
            if max_val is not None and value > max_val:
                raise ValidationError(f"Field '{key}' must be <= {max_val}, got {value}")
        
        # Choice validation
        choices = requirements.get("choices")
        if choices and value not in choices:
            raise ValidationError(f"Field '{key}' must be one of {choices}, got {value}")
        
        # Length validation for strings and lists
        if isinstance(value, (str, list)):
            min_length = requirements.get("min_length")
            max_length = requirements.get("max_length")
            
            if min_length is not None and len(value) < min_length:
                raise ValidationError(f"Field '{key}' must have at least {min_length} items/characters")
            
            if max_length is not None and len(value) > max_length:
                raise ValidationError(f"Field '{key}' must have at most {max_length} items/characters")
        
        # Custom validation function
        validator = requirements.get("validator")
        if validator and callable(validator):
            try:
                value = validator(value)
            except Exception as e:
                raise ValidationError(f"Validation failed for field '{key}': {e}")
        
        validated_config[key] = value
    
    # Check for unexpected fields
    unexpected = set(config.keys()) - set(schema.keys())
    if unexpected:
        raise ValidationError(f"Unexpected fields in configuration: {unexpected}")
    
    return validated_config


def validate_watermark_method(method: str) -> str:
    """Validate watermark method."""
    valid_methods = ["kirchenbauer", "markllm", "aaronson", "zhao"]
    
    if method not in valid_methods:
        raise ValidationError(f"Invalid watermark method: {method}. Valid methods: {valid_methods}")
    
    return method


def validate_probability(value: float, name: str = "probability") -> float:
    """Validate probability value (0.0 to 1.0)."""
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number")
    
    if not 0.0 <= value <= 1.0:
        raise ValidationError(f"{name} must be between 0.0 and 1.0, got {value}")
    
    return float(value)


def validate_positive_integer(value: int, name: str = "value") -> int:
    """Validate positive integer."""
    if not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer")
    
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")
    
    return value


def validate_file_path(path: Union[str, Path], must_exist: bool = False, must_be_file: bool = True) -> Path:
    """Validate file path."""
    if isinstance(path, str):
        path = Path(path)
    
    if not isinstance(path, Path):
        raise ValidationError(f"Path must be a string or Path object, got {type(path)}")
    
    if must_exist and not path.exists():
        raise ValidationError(f"Path does not exist: {path}")
    
    if must_exist and must_be_file and not path.is_file():
        raise ValidationError(f"Path is not a file: {path}")
    
    return path


def validate_email(email: str) -> str:
    """Validate email address."""
    if not isinstance(email, str):
        raise ValidationError("Email must be a string")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        raise ValidationError(f"Invalid email format: {email}")
    
    return email.lower()


def validate_url(url: str, allowed_schemes: Optional[List[str]] = None) -> str:
    """Validate URL."""
    if not isinstance(url, str):
        raise ValidationError("URL must be a string")
    
    if allowed_schemes is None:
        allowed_schemes = ["http", "https"]
    
    url_pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
    if not re.match(url_pattern, url):
        raise ValidationError(f"Invalid URL format: {url}")
    
    scheme = url.split('://')[0].lower()
    if scheme not in allowed_schemes:
        raise ValidationError(f"URL scheme '{scheme}' not allowed. Allowed schemes: {allowed_schemes}")
    
    return url


def validate_json_string(json_str: str) -> Dict[str, Any]:
    """Validate and parse JSON string."""
    import json
    
    if not isinstance(json_str, str):
        raise ValidationError("JSON must be a string")
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON format: {e}")


def validate_yaml_string(yaml_str: str) -> Dict[str, Any]:
    """Validate and parse YAML string."""
    import yaml
    
    if not isinstance(yaml_str, str):
        raise ValidationError("YAML must be a string")
    
    try:
        return yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        raise ValidationError(f"Invalid YAML format: {e}")


def validate_attack_strength(strength: str) -> str:
    """Validate attack strength."""
    valid_strengths = ["light", "medium", "heavy"]
    
    if strength not in valid_strengths:
        raise ValidationError(f"Invalid attack strength: {strength}. Valid strengths: {valid_strengths}")
    
    return strength


def validate_detection_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate detection result structure."""
    required_fields = ["is_watermarked", "confidence", "p_value", "method"]
    
    for field in required_fields:
        if field not in result:
            raise ValidationError(f"Detection result missing required field: {field}")
    
    # Validate specific fields
    if not isinstance(result["is_watermarked"], bool):
        raise ValidationError("is_watermarked must be a boolean")
    
    result["confidence"] = validate_probability(result["confidence"], "confidence")
    result["p_value"] = validate_probability(result["p_value"], "p_value")
    
    if not isinstance(result["method"], str):
        raise ValidationError("method must be a string")
    
    return result


def validate_batch_size(batch_size: int, max_batch_size: int = 1000) -> int:
    """Validate batch size."""
    batch_size = validate_positive_integer(batch_size, "batch_size")
    
    if batch_size > max_batch_size:
        raise ValidationError(f"Batch size too large: {batch_size}. Maximum allowed: {max_batch_size}")
    
    return batch_size


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe filesystem operations."""
    if not isinstance(filename, str):
        raise ValidationError("Filename must be a string")
    
    # Remove or replace dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure filename is not empty
    if not filename:
        raise ValidationError("Filename cannot be empty after sanitization")
    
    # Check for reserved names (Windows)
    reserved_names = ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in range(1, 10)]
    if filename.upper() in reserved_names:
        filename = f"_{filename}"
    
    return filename


def validate_memory_limit(limit_mb: int) -> int:
    """Validate memory limit in MB."""
    limit_mb = validate_positive_integer(limit_mb, "memory_limit")
    
    # Reasonable limits (1MB to 100GB)
    if limit_mb < 1:
        raise ValidationError("Memory limit must be at least 1 MB")
    
    if limit_mb > 100 * 1024:  # 100 GB
        raise ValidationError("Memory limit too high: maximum 100 GB")
    
    return limit_mb


def validate_timeout(timeout_seconds: int) -> int:
    """Validate timeout value."""
    timeout_seconds = validate_positive_integer(timeout_seconds, "timeout")
    
    # Reasonable limits (1 second to 1 hour)
    if timeout_seconds > 3600:
        raise ValidationError("Timeout too long: maximum 1 hour")
    
    return timeout_seconds


# Validation schemas for common configurations
WATERMARK_CONFIG_SCHEMA = {
    "method": {"type": str, "required": True, "validator": validate_watermark_method},
    "max_length": {"type": int, "required": False, "min": 1, "max": 4096},
    "seed": {"type": int, "required": False},
    "gamma": {"type": float, "required": False, "min": 0.0, "max": 1.0},
    "delta": {"type": float, "required": False, "min": 0.0},
    "watermark_strength": {"type": float, "required": False, "min": 0.0},
    "threshold": {"type": float, "required": False, "min": 0.0, "max": 1.0},
    "redundancy": {"type": int, "required": False, "min": 1}
}

DETECTION_CONFIG_SCHEMA = {
    "method": {"type": str, "required": True, "validator": validate_watermark_method},
    "threshold": {"type": float, "required": False, "min": 0.0, "max": 1.0},
    "confidence_level": {"type": float, "required": False, "min": 0.0, "max": 1.0},
    "min_tokens": {"type": int, "required": False, "min": 1}
}

ATTACK_CONFIG_SCHEMA = {
    "attack_type": {"type": str, "required": True},
    "strength": {"type": str, "required": False, "validator": validate_attack_strength},
    "replacement_probability": {"type": float, "required": False, "min": 0.0, "max": 1.0},
    "truncation_ratio": {"type": float, "required": False, "min": 0.0, "max": 1.0}
}