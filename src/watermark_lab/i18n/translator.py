"""Translation and localization utilities."""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from threading import Lock

from ..utils.logging import get_logger


class Translator:
    """Thread-safe translator for internationalization."""
    
    def __init__(self, locale_dir: Optional[Path] = None):
        """Initialize translator with locale directory."""
        self.logger = get_logger("translator")
        self._lock = Lock()
        self._current_locale = "en"
        self._translations: Dict[str, Dict[str, str]] = {}
        self._locale_dir = locale_dir or Path(__file__).parent / "locales"
        self._fallback_locale = "en"
        
        # Load translations
        self._load_translations()
    
    def _load_translations(self):
        """Load translation files from locale directory."""
        if not self._locale_dir.exists():
            self.logger.warning(f"Locale directory not found: {self._locale_dir}")
            # Create basic English translations
            self._translations["en"] = self._get_default_translations()
            return
        
        for locale_file in self._locale_dir.glob("*.json"):
            locale = locale_file.stem
            try:
                with open(locale_file, 'r', encoding='utf-8') as f:
                    self._translations[locale] = json.load(f)
                self.logger.info(f"Loaded translations for locale: {locale}")
            except Exception as e:
                self.logger.error(f"Failed to load translations for {locale}: {e}")
        
        # Ensure fallback locale exists
        if self._fallback_locale not in self._translations:
            self._translations[self._fallback_locale] = self._get_default_translations()
    
    def _get_default_translations(self) -> Dict[str, str]:
        """Get default English translations."""
        return {
            # Core functionality
            "watermark.generation.success": "Watermark generated successfully",
            "watermark.generation.failed": "Failed to generate watermark",
            "watermark.detection.success": "Watermark detection completed",
            "watermark.detection.failed": "Failed to detect watermark",
            "watermark.detection.positive": "Watermark detected",
            "watermark.detection.negative": "No watermark detected",
            
            # Validation messages
            "validation.text.empty": "Text cannot be empty",
            "validation.text.too_long": "Text exceeds maximum length of {max_length} characters",
            "validation.text.too_short": "Text must be at least {min_length} characters",
            "validation.config.invalid": "Invalid configuration provided",
            "validation.method.unknown": "Unknown watermark method: {method}",
            
            # API messages
            "api.request.invalid": "Invalid request format",
            "api.response.success": "Request processed successfully",
            "api.response.error": "Request processing failed",
            "api.rate_limit.exceeded": "Rate limit exceeded. Please try again later",
            
            # CLI messages
            "cli.command.success": "Command executed successfully",
            "cli.command.failed": "Command execution failed",
            "cli.help.description": "LM Watermark Lab - Comprehensive toolkit for LLM watermarking",
            
            # File operations
            "file.not_found": "File not found: {filename}",
            "file.permission_denied": "Permission denied: {filename}",
            "file.save.success": "File saved successfully: {filename}",
            "file.save.failed": "Failed to save file: {filename}",
            
            # Configuration
            "config.loaded": "Configuration loaded from {source}",
            "config.save.success": "Configuration saved successfully",
            "config.save.failed": "Failed to save configuration",
            
            # Processing status
            "processing.started": "Processing started",
            "processing.completed": "Processing completed in {duration:.2f} seconds",
            "processing.failed": "Processing failed: {error}",
            "processing.cancelled": "Processing cancelled by user",
            
            # Batch operations
            "batch.processing.started": "Processing batch of {count} items",
            "batch.processing.progress": "Processed {completed}/{total} items ({percent:.1f}%)",
            "batch.processing.completed": "Batch processing completed",
            "batch.processing.failed": "Batch processing failed",
            
            # Model operations
            "model.loading": "Loading model: {model_name}",
            "model.loaded": "Model loaded successfully: {model_name}",
            "model.failed": "Failed to load model: {model_name}",
            "model.not_found": "Model not found: {model_name}",
            
            # Cache operations
            "cache.hit": "Cache hit for key: {key}",
            "cache.miss": "Cache miss for key: {key}",
            "cache.cleared": "Cache cleared successfully",
            "cache.size": "Cache contains {count} entries",
            
            # Error messages
            "error.internal": "Internal server error occurred",
            "error.timeout": "Operation timed out",
            "error.network": "Network connection error",
            "error.authentication": "Authentication failed",
            "error.authorization": "Access denied",
            "error.not_implemented": "Feature not implemented",
            
            # Compliance messages
            "compliance.gdpr.consent_required": "GDPR consent required for data processing",
            "compliance.ccpa.opt_out_honored": "CCPA opt-out request honored",
            "compliance.pdpa.notice_provided": "PDPA privacy notice provided",
            "compliance.data_retention.expired": "Data retention period expired",
            "compliance.data_export.ready": "Data export ready for download",
            
            # Performance metrics
            "metrics.throughput": "Throughput: {rate:.2f} items/second",
            "metrics.latency": "Latency: {time:.2f}ms",
            "metrics.accuracy": "Accuracy: {score:.2f}%",
            "metrics.memory_usage": "Memory usage: {usage:.1f}MB",
            
            # System status
            "system.ready": "System ready",
            "system.maintenance": "System under maintenance",
            "system.overloaded": "System overloaded, please try again later",
            "system.healthy": "All systems operational",
            
            # User interface
            "ui.welcome": "Welcome to LM Watermark Lab",
            "ui.goodbye": "Thank you for using LM Watermark Lab",
            "ui.help": "Type 'help' for available commands",
            "ui.progress": "Progress: {percent}% complete",
            
            # Security
            "security.key_generated": "Security key generated successfully",
            "security.signature_valid": "Digital signature verified",
            "security.signature_invalid": "Invalid digital signature",
            "security.encryption_enabled": "Encryption enabled",
            
            # Export/Import
            "export.started": "Export started",
            "export.completed": "Export completed: {filename}",
            "import.started": "Import started",
            "import.completed": "Import completed: {count} items processed",
            
            # Quality assurance
            "qa.test_passed": "Quality test passed",
            "qa.test_failed": "Quality test failed: {reason}",
            "qa.benchmark_completed": "Benchmark completed",
            "qa.validation_successful": "Validation successful"
        }
    
    def set_locale(self, locale: str) -> bool:
        """Set current locale."""
        with self._lock:
            if locale in self._translations:
                self._current_locale = locale
                self.logger.info(f"Locale set to: {locale}")
                return True
            else:
                self.logger.warning(f"Locale not available: {locale}")
                return False
    
    def get_locale(self) -> str:
        """Get current locale."""
        return self._current_locale
    
    def get_supported_locales(self) -> List[str]:
        """Get list of supported locales."""
        return list(self._translations.keys())
    
    def translate(self, key: str, locale: Optional[str] = None, **kwargs) -> str:
        """Translate a message key to current or specified locale."""
        target_locale = locale or self._current_locale
        
        with self._lock:
            # Try target locale first
            if target_locale in self._translations:
                translations = self._translations[target_locale]
                if key in translations:
                    try:
                        return translations[key].format(**kwargs)
                    except KeyError as e:
                        self.logger.warning(f"Missing format parameter {e} for key '{key}'")
                        return translations[key]
            
            # Fallback to default locale
            if self._fallback_locale in self._translations:
                translations = self._translations[self._fallback_locale]
                if key in translations:
                    try:
                        return translations[key].format(**kwargs)
                    except KeyError as e:
                        self.logger.warning(f"Missing format parameter {e} for key '{key}'")
                        return translations[key]
            
            # Final fallback - return key
            self.logger.warning(f"Translation key not found: {key}")
            return key
    
    def add_translation(self, locale: str, key: str, value: str):
        """Add or update a translation."""
        with self._lock:
            if locale not in self._translations:
                self._translations[locale] = {}
            self._translations[locale][key] = value
            self.logger.debug(f"Added translation: {locale}.{key} = {value}")
    
    def has_translation(self, key: str, locale: Optional[str] = None) -> bool:
        """Check if translation exists for key."""
        target_locale = locale or self._current_locale
        with self._lock:
            return (target_locale in self._translations and 
                   key in self._translations[target_locale])


# Global translator instance
_translator = None
_translator_lock = Lock()


def get_translator() -> Translator:
    """Get global translator instance."""
    global _translator
    if _translator is None:
        with _translator_lock:
            if _translator is None:
                _translator = Translator()
    return _translator


def translate(key: str, locale: Optional[str] = None, **kwargs) -> str:
    """Convenience function for translation."""
    return get_translator().translate(key, locale, **kwargs)


def set_locale(locale: str) -> bool:
    """Set current locale globally."""
    return get_translator().set_locale(locale)


def get_supported_locales() -> List[str]:
    """Get supported locales."""
    return get_translator().get_supported_locales()


def init_translations(locale_dir: Optional[Path] = None, default_locale: str = "en"):
    """Initialize translation system."""
    global _translator
    with _translator_lock:
        _translator = Translator(locale_dir)
        _translator.set_locale(default_locale)


class LocalizationContext:
    """Context manager for temporary locale switching."""
    
    def __init__(self, locale: str):
        self.locale = locale
        self.previous_locale = None
    
    def __enter__(self):
        self.previous_locale = get_translator().get_locale()
        get_translator().set_locale(self.locale)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_locale:
            get_translator().set_locale(self.previous_locale)


# Translation decorator for functions
def localized(key_prefix: str = ""):
    """Decorator to add localization support to functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get locale from kwargs if provided
            locale = kwargs.pop('locale', None)
            
            try:
                result = func(*args, **kwargs)
                
                # If result is a string and looks like a translation key, translate it
                if isinstance(result, str) and result.startswith(key_prefix):
                    return translate(result, locale)
                
                return result
                
            except Exception as e:
                # Translate error messages
                error_key = f"{key_prefix}.error.{type(e).__name__.lower()}"
                error_msg = translate(error_key, locale, error=str(e))
                
                # If translation not found, use original error
                if error_msg == error_key:
                    raise
                else:
                    raise type(e)(error_msg) from e
        
        return wrapper
    return decorator