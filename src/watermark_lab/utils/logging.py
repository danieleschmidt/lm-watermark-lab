"""Logging configuration and utilities."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    json_format: bool = False,
    include_trace_id: bool = True
) -> None:
    """Set up application logging configuration."""
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Default log format
    if log_format is None:
        if json_format:
            log_format = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        elif include_trace_id:
            log_format = "%(asctime)s [%(levelname)s] %(name)s [%(trace_id)s]: %(message)s"
        else:
            log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set third-party library log levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with optional trace ID support."""
    logger = logging.getLogger(name)
    
    # Add trace ID filter if not already present
    if not any(isinstance(f, TraceIDFilter) for f in logger.filters):
        logger.addFilter(TraceIDFilter())
    
    return logger


class TraceIDFilter(logging.Filter):
    """Add trace ID to log records."""
    
    def filter(self, record):
        """Add trace ID to log record."""
        import contextvars
        
        # Try to get trace ID from context
        try:
            trace_id = contextvars.copy_context().get("trace_id", "unknown")
        except:
            trace_id = "unknown"
        
        record.trace_id = trace_id
        return True


class StructuredLogger:
    """Structured logger for better observability."""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
    
    def log_event(
        self,
        event: str,
        level: str = "INFO",
        **kwargs
    ) -> None:
        """Log a structured event."""
        log_data = {
            "event": event,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(log_level, f"Event: {event}", extra=log_data)
    
    def log_watermark_generation(
        self,
        method: str,
        prompt_length: int,
        output_length: int,
        generation_time: float,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """Log watermark generation event."""
        self.log_event(
            "watermark_generation",
            level="INFO" if success else "ERROR",
            method=method,
            prompt_length=prompt_length,
            output_length=output_length,
            generation_time=generation_time,
            success=success,
            error=error
        )
    
    def log_detection(
        self,
        method: str,
        text_length: int,
        is_watermarked: bool,
        confidence: float,
        detection_time: float,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """Log detection event."""
        self.log_event(
            "watermark_detection",
            level="INFO" if success else "ERROR",
            method=method,
            text_length=text_length,
            is_watermarked=is_watermarked,
            confidence=confidence,
            detection_time=detection_time,
            success=success,
            error=error
        )
    
    def log_attack(
        self,
        attack_type: str,
        original_length: int,
        attacked_length: int,
        quality_score: float,
        similarity_score: float,
        attack_time: float,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """Log attack event."""
        self.log_event(
            "attack_simulation",
            level="INFO" if success else "ERROR",
            attack_type=attack_type,
            original_length=original_length,
            attacked_length=attacked_length,
            quality_score=quality_score,
            similarity_score=similarity_score,
            attack_time=attack_time,
            success=success,
            error=error
        )
    
    def log_benchmark(
        self,
        methods: list,
        num_samples: int,
        metrics: list,
        total_time: float,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """Log benchmark event."""
        self.log_event(
            "benchmark_run",
            level="INFO" if success else "ERROR",
            methods=methods,
            num_samples=num_samples,
            metrics=metrics,
            total_time=total_time,
            success=success,
            error=error
        )
    
    def log_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        request_size: Optional[int] = None,
        response_size: Optional[int] = None,
        user_id: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """Log API request event."""
        self.log_event(
            "api_request",
            level="INFO" if status_code < 400 else "ERROR",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time=response_time,
            request_size=request_size,
            response_size=response_size,
            user_id=user_id,
            error=error
        )


# Performance monitoring utilities
class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, name: str):
        self.logger = get_logger(f"{name}.performance")
        self.metrics = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        import time
        self.metrics[operation] = {"start_time": time.time()}
    
    def end_timer(self, operation: str, **metadata) -> float:
        """End timing an operation and log the duration."""
        import time
        
        if operation not in self.metrics:
            self.logger.warning(f"Timer for operation '{operation}' was not started")
            return 0.0
        
        end_time = time.time()
        duration = end_time - self.metrics[operation]["start_time"]
        
        self.logger.info(
            f"Operation '{operation}' completed in {duration:.3f}s",
            extra={
                "operation": operation,
                "duration": duration,
                **metadata
            }
        )
        
        # Clean up
        del self.metrics[operation]
        
        return duration
    
    def log_memory_usage(self, operation: str) -> None:
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.logger.info(
                f"Memory usage for '{operation}': {memory_info.rss / 1024 / 1024:.2f} MB",
                extra={
                    "operation": operation,
                    "memory_rss_mb": memory_info.rss / 1024 / 1024,
                    "memory_vms_mb": memory_info.vms / 1024 / 1024
                }
            )
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")


# Context managers for logging
class LoggingContext:
    """Context manager for operation logging."""
    
    def __init__(self, logger: StructuredLogger, operation: str, **metadata):
        self.logger = logger
        self.operation = operation
        self.metadata = metadata
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.log_event(
            f"{self.operation}_started",
            level="DEBUG",
            **self.metadata
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time if self.start_time else 0
        
        if exc_type is None:
            self.logger.log_event(
                f"{self.operation}_completed",
                level="INFO",
                duration=duration,
                **self.metadata
            )
        else:
            self.logger.log_event(
                f"{self.operation}_failed",
                level="ERROR",
                duration=duration,
                error=str(exc_val),
                exception_type=exc_type.__name__,
                **self.metadata
            )