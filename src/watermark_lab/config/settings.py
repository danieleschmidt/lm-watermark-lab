"""Application settings and configuration management."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from pydantic import BaseSettings, Field, validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = Field(default="LM Watermark Lab", env="APP_NAME")
    version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8080, env="API_PORT")
    api_reload: bool = Field(default=False, env="API_RELOAD")
    api_workers: int = Field(default=1, env="API_WORKERS")
    cors_origins: list = Field(default=["*"], env="CORS_ORIGINS")
    
    # Database
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # Model Configuration
    default_model_cache: str = Field(default="./models", env="MODEL_CACHE_PATH")
    model_timeout: int = Field(default=300, env="MODEL_TIMEOUT")
    max_sequence_length: int = Field(default=2048, env="MAX_SEQUENCE_LENGTH")
    
    # Watermarking Defaults
    default_watermark_method: str = Field(default="kirchenbauer", env="DEFAULT_WATERMARK_METHOD")
    default_gamma: float = Field(default=0.25, env="DEFAULT_GAMMA")
    default_delta: float = Field(default=2.0, env="DEFAULT_DELTA")
    default_max_length: int = Field(default=100, env="DEFAULT_MAX_LENGTH")
    
    # Detection Defaults
    default_detection_threshold: float = Field(default=0.05, env="DEFAULT_DETECTION_THRESHOLD")
    default_confidence_level: float = Field(default=0.95, env="DEFAULT_CONFIDENCE_LEVEL")
    
    # Security
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Performance
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    jaeger_endpoint: Optional[str] = Field(default=None, env="JAEGER_ENDPOINT")
    
    # File Paths
    config_file: Optional[str] = Field(default=None, env="CONFIG_FILE")
    data_dir: str = Field(default="./data", env="DATA_DIR")
    logs_dir: str = Field(default="./logs", env="LOGS_DIR")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @validator("default_watermark_method")
    def validate_watermark_method(cls, v):
        """Validate watermark method."""
        valid_methods = ["kirchenbauer", "markllm", "aaronson", "zhao"]
        if v not in valid_methods:
            raise ValueError(f"Watermark method must be one of: {valid_methods}")
        return v
    
    def load_from_file(self, config_file: Union[str, Path]) -> None:
        """Load configuration from file (JSON or YAML)."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        # Update settings with loaded configuration
        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_to_file(self, config_file: Union[str, Path], format: str = "yaml") -> None:
        """Save current configuration to file."""
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = self.dict()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if format.lower() in ['yml', 'yaml']:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                json.dump(config_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def get_watermark_config(self, method: Optional[str] = None) -> Dict[str, Any]:
        """Get watermark configuration for specified method."""
        method = method or self.default_watermark_method
        
        base_config = {
            "method": method,
            "max_length": self.default_max_length
        }
        
        if method == "kirchenbauer":
            base_config.update({
                "gamma": self.default_gamma,
                "delta": self.default_delta
            })
        elif method == "markllm":
            base_config.update({
                "algorithm": "KGW",
                "watermark_strength": 2.0
            })
        elif method == "aaronson":
            base_config.update({
                "threshold": 0.5,
                "secret_key": self.secret_key
            })
        elif method == "zhao":
            base_config.update({
                "message_bits": "101010",
                "redundancy": 3
            })
        
        return base_config
    
    def get_detection_config(self, method: Optional[str] = None) -> Dict[str, Any]:
        """Get detection configuration for specified method."""
        method = method or self.default_watermark_method
        
        return {
            "method": method,
            "threshold": self.default_detection_threshold,
            "confidence_level": self.default_confidence_level
        }
    
    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        directories = [
            self.default_model_cache,
            self.data_dir,
            self.logs_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    settings = Settings()
    
    # Load from config file if specified
    if settings.config_file and Path(settings.config_file).exists():
        settings.load_from_file(settings.config_file)
    
    # Ensure required directories exist
    settings.ensure_directories()
    
    return settings


def create_default_config(config_file: Union[str, Path]) -> None:
    """Create a default configuration file."""
    settings = Settings()
    settings.save_to_file(config_file, format="yaml")


def load_config_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    config = {}
    
    # Map environment variables to config keys
    env_mapping = {
        "WATERMARK_METHOD": "default_watermark_method",
        "GAMMA": "default_gamma", 
        "DELTA": "default_delta",
        "MAX_LENGTH": "default_max_length",
        "DETECTION_THRESHOLD": "default_detection_threshold",
        "MODEL_CACHE": "default_model_cache",
        "DEBUG": "debug",
        "LOG_LEVEL": "log_level"
    }
    
    for env_var, config_key in env_mapping.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            
            # Type conversion
            if config_key in ["default_gamma", "default_delta", "default_detection_threshold"]:
                config[config_key] = float(value)
            elif config_key in ["default_max_length"]:
                config[config_key] = int(value)
            elif config_key == "debug":
                config[config_key] = value.lower() in ["true", "1", "yes", "on"]
            else:
                config[config_key] = value
    
    return config