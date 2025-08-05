"""Custom exceptions for LM Watermark Lab."""

from typing import Optional, Dict, Any


class WatermarkLabError(Exception):
    """Base exception for LM Watermark Lab."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class WatermarkError(WatermarkLabError):
    """Exception raised during watermarking operations."""
    pass


class DetectionError(WatermarkLabError):
    """Exception raised during detection operations."""
    pass


class ValidationError(WatermarkLabError):
    """Exception raised during input validation."""
    pass


class ConfigurationError(WatermarkLabError):
    """Exception raised for configuration issues."""
    pass


class ModelError(WatermarkLabError):
    """Exception raised for model-related issues."""
    pass


class AttackError(WatermarkLabError):
    """Exception raised during attack operations."""
    pass


class BenchmarkError(WatermarkLabError):
    """Exception raised during benchmarking operations."""
    pass


class APIError(WatermarkLabError):
    """Exception raised during API operations."""
    
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.status_code = status_code


class TimeoutError(WatermarkLabError):
    """Exception raised when operations timeout."""
    pass


class ResourceError(WatermarkLabError):
    """Exception raised when resources are unavailable."""
    pass


class AuthenticationError(WatermarkLabError):
    """Exception raised for authentication issues."""
    pass


class AuthorizationError(WatermarkLabError):
    """Exception raised for authorization issues."""
    pass


class RateLimitError(WatermarkLabError):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.retry_after = retry_after


class ComplianceError(WatermarkLabError):
    """Exception raised for privacy compliance issues."""
    pass


# Error handling utilities
def format_error_response(error: Exception, include_traceback: bool = False) -> Dict[str, Any]:
    """Format error for API responses."""
    response = {
        "error": type(error).__name__,
        "message": str(error)
    }
    
    if isinstance(error, WatermarkLabError):
        response["details"] = error.details
        
        if isinstance(error, APIError):
            response["status_code"] = error.status_code
        
        if isinstance(error, RateLimitError) and error.retry_after:
            response["retry_after"] = error.retry_after
    
    if include_traceback:
        import traceback
        response["traceback"] = traceback.format_exc()
    
    return response


def create_error_chain(*errors: Exception) -> WatermarkLabError:
    """Create an error chain for debugging."""
    if not errors:
        return WatermarkLabError("Unknown error")
    
    main_error = errors[-1]
    if len(errors) == 1:
        if isinstance(main_error, WatermarkLabError):
            return main_error
        return WatermarkLabError(str(main_error))
    
    # Create chain
    error_chain = []
    for i, error in enumerate(errors):
        error_chain.append(f"#{i+1}: {type(error).__name__}: {error}")
    
    return WatermarkLabError(
        f"Error chain: {main_error}",
        details={"error_chain": error_chain}
    )