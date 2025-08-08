"""Production-grade input sanitization with comprehensive security controls."""

import re
import html
import json
import hashlib
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass
from enum import Enum

try:
    import bleach
    BLEACH_AVAILABLE = True
except ImportError:
    BLEACH_AVAILABLE = False

from ..utils.logging import get_logger
from ..utils.exceptions import SecurityError, ValidationError

logger = get_logger("security.sanitization")


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SanitizationConfig:
    """Configuration for input sanitization."""
    
    # Text sanitization
    max_text_length: int = 10000
    max_line_length: int = 1000
    allowed_characters: Optional[Set[str]] = None
    blocked_patterns: List[str] = None
    
    # HTML sanitization
    strip_html: bool = True
    allowed_tags: List[str] = None
    allowed_attributes: Dict[str, List[str]] = None
    
    # Script detection
    detect_scripts: bool = True
    detect_sql_injection: bool = True
    detect_xss: bool = True
    detect_path_traversal: bool = True
    
    # Content validation
    require_printable: bool = True
    normalize_unicode: bool = True
    detect_binary: bool = True
    
    # Rate limiting
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 100
    
    def __post_init__(self):
        """Initialize default values."""
        if self.blocked_patterns is None:
            self.blocked_patterns = [
                r'<script[^>]*>.*?</script>',  # Script tags
                r'javascript:',  # JavaScript protocols
                r'vbscript:',    # VBScript protocols
                r'on\w+\s*=',    # Event handlers
                r'eval\s*\(',    # Eval calls
                r'exec\s*\(',    # Exec calls
                r'\.\./',        # Path traversal
                r'union\s+select',  # SQL injection
                r'drop\s+table',    # SQL injection
                r'<iframe',      # Iframe injections
                r'<object',      # Object injections
                r'<embed',       # Embed injections
            ]
        
        if self.allowed_tags is None:
            self.allowed_tags = []  # No HTML allowed by default
        
        if self.allowed_attributes is None:
            self.allowed_attributes = {}


class InputSanitizer:
    """Advanced input sanitization with security threat detection."""
    
    def __init__(self, config: Optional[SanitizationConfig] = None):
        self.config = config or SanitizationConfig()
        self.logger = get_logger("input_sanitizer")
        
        # Compile regex patterns for performance
        self.blocked_pattern_regex = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL) 
            for pattern in self.config.blocked_patterns
        ]
        
        # Request tracking for rate limiting
        self.request_history = {}
        
    def sanitize_text(self, text: str, context: str = "general") -> str:
        """Comprehensive text sanitization with threat detection."""
        
        if not isinstance(text, str):
            raise ValidationError(f"Input must be string, got {type(text)}")
        
        try:
            # Check rate limiting
            if self.config.enable_rate_limiting:
                self._check_rate_limit(context)
            
            # Basic validation
            self._validate_basic_constraints(text)
            
            # Threat detection
            threats = self._detect_threats(text)
            if threats:
                self._handle_threats(threats, text, context)
            
            # Sanitization pipeline
            sanitized = text
            
            # 1. Length validation
            if len(sanitized) > self.config.max_text_length:
                self.logger.warning(f"Text truncated from {len(sanitized)} to {self.config.max_text_length}")
                sanitized = sanitized[:self.config.max_text_length]
            
            # 2. Line length validation
            lines = sanitized.split('\n')
            if any(len(line) > self.config.max_line_length for line in lines):
                sanitized = '\n'.join(
                    line[:self.config.max_line_length] if len(line) > self.config.max_line_length else line
                    for line in lines
                )
                self.logger.warning("Long lines truncated")
            
            # 3. Unicode normalization
            if self.config.normalize_unicode:
                sanitized = self._normalize_unicode(sanitized)
            
            # 4. Character filtering
            if self.config.allowed_characters:
                sanitized = ''.join(c for c in sanitized if c in self.config.allowed_characters)
            
            # 5. HTML sanitization
            if self.config.strip_html:
                sanitized = self._sanitize_html(sanitized)
            
            # 6. Remove blocked patterns
            sanitized = self._remove_blocked_patterns(sanitized)
            
            # 7. Printable character enforcement
            if self.config.require_printable:
                sanitized = self._ensure_printable(sanitized)
            
            # 8. Binary detection
            if self.config.detect_binary and self._is_binary(sanitized):
                raise SecurityError("Binary content detected in text input")
            
            self.logger.debug(f"Text sanitized: {len(text)} -> {len(sanitized)} chars")
            return sanitized
            
        except Exception as e:
            if isinstance(e, (SecurityError, ValidationError)):
                raise
            else:
                self.logger.error(f"Sanitization failed: {e}")
                raise SecurityError(f"Input sanitization failed: {e}")
    
    def sanitize_json(self, data: Union[str, Dict, List]) -> Union[Dict, List]:
        """Sanitize JSON data with deep inspection."""
        
        try:
            # Parse JSON if string
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError as e:
                    raise ValidationError(f"Invalid JSON: {e}")
            
            # Recursively sanitize
            return self._sanitize_json_recursive(data, depth=0)
            
        except Exception as e:
            if isinstance(e, (SecurityError, ValidationError)):
                raise
            else:
                self.logger.error(f"JSON sanitization failed: {e}")
                raise SecurityError(f"JSON sanitization failed: {e}")
    
    def validate_file_upload(self, filename: str, content: bytes, max_size: int = 10 * 1024 * 1024) -> bool:
        """Validate file uploads for security threats."""
        
        try:
            # Size check
            if len(content) > max_size:
                raise SecurityError(f"File too large: {len(content)} > {max_size}")
            
            # Extension validation
            allowed_extensions = {'.txt', '.json', '.csv', '.md'}
            ext = '.' + filename.lower().split('.')[-1] if '.' in filename else ''
            
            if ext not in allowed_extensions:
                raise SecurityError(f"File type not allowed: {ext}")
            
            # Content validation
            if self._contains_malicious_content(content):
                raise SecurityError("Malicious content detected in file")
            
            # Magic number validation
            if not self._validate_file_type(content, ext):
                raise SecurityError("File type mismatch")
            
            self.logger.info(f"File validated: {filename} ({len(content)} bytes)")
            return True
            
        except SecurityError:
            raise
        except Exception as e:
            self.logger.error(f"File validation failed: {e}")
            raise SecurityError(f"File validation failed: {e}")
    
    def _validate_basic_constraints(self, text: str):
        """Validate basic input constraints."""
        
        if len(text) == 0:
            return  # Allow empty strings
        
        if len(text) > self.config.max_text_length:
            # Don't raise error here, will be truncated later
            self.logger.warning(f"Input exceeds maximum length: {len(text)}")
        
        # Check for null bytes
        if '\x00' in text:
            raise SecurityError("Null bytes detected in input")
        
        # Check for extremely long lines
        max_line = max((len(line) for line in text.split('\n')), default=0)
        if max_line > self.config.max_line_length * 10:  # 10x limit for blocking
            raise SecurityError(f"Extremely long line detected: {max_line} characters")
    
    def _detect_threats(self, text: str) -> List[Dict[str, Any]]:
        """Detect security threats in input text."""
        
        threats = []
        
        # Script injection detection
        if self.config.detect_scripts:
            script_threats = self._detect_script_injection(text)
            threats.extend(script_threats)
        
        # SQL injection detection  
        if self.config.detect_sql_injection:
            sql_threats = self._detect_sql_injection(text)
            threats.extend(sql_threats)
        
        # XSS detection
        if self.config.detect_xss:
            xss_threats = self._detect_xss(text)
            threats.extend(xss_threats)
        
        # Path traversal detection
        if self.config.detect_path_traversal:
            path_threats = self._detect_path_traversal(text)
            threats.extend(path_threats)
        
        return threats
    
    def _detect_script_injection(self, text: str) -> List[Dict[str, Any]]:
        """Detect script injection attempts."""
        
        threats = []
        script_patterns = [
            r'<script[^>]*>',
            r'javascript\s*:',
            r'vbscript\s*:',
            r'eval\s*\(',
            r'exec\s*\(',
            r'function\s*\(',
            r'setTimeout\s*\(',
            r'setInterval\s*\('
        ]
        
        for pattern in script_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                threats.append({
                    'type': 'script_injection',
                    'level': ThreatLevel.HIGH,
                    'pattern': pattern,
                    'match': match.group(),
                    'position': match.start()
                })
        
        return threats
    
    def _detect_sql_injection(self, text: str) -> List[Dict[str, Any]]:
        """Detect SQL injection attempts."""
        
        threats = []
        sql_patterns = [
            r'union\s+select',
            r'drop\s+table',
            r'delete\s+from',
            r'insert\s+into',
            r'update\s+\w+\s+set',
            r'or\s+1\s*=\s*1',
            r'and\s+1\s*=\s*1',
            r'--\s*$',
            r'/\*.*\*/',
            r';\s*drop\s+'
        ]
        
        for pattern in sql_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                threats.append({
                    'type': 'sql_injection', 
                    'level': ThreatLevel.CRITICAL,
                    'pattern': pattern,
                    'match': match.group(),
                    'position': match.start()
                })
        
        return threats
    
    def _detect_xss(self, text: str) -> List[Dict[str, Any]]:
        """Detect XSS attempts."""
        
        threats = []
        xss_patterns = [
            r'<\s*img[^>]+onerror\s*=',
            r'<\s*input[^>]+onfocus\s*=',
            r'<\s*body[^>]+onload\s*=',
            r'<\s*iframe[^>]*>',
            r'<\s*object[^>]*>',
            r'<\s*embed[^>]*>',
            r'on\w+\s*=\s*["\'][^"\']*["\']'
        ]
        
        for pattern in xss_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                threats.append({
                    'type': 'xss',
                    'level': ThreatLevel.HIGH,
                    'pattern': pattern,
                    'match': match.group(),
                    'position': match.start()
                })
        
        return threats
    
    def _detect_path_traversal(self, text: str) -> List[Dict[str, Any]]:
        """Detect path traversal attempts."""
        
        threats = []
        path_patterns = [
            r'\.\./',
            r'\.\.\/',
            r'\.\.\\',
            r'%2e%2e%2f',
            r'%2e%2e/',
            r'..%2f',
            r'%252e%252e%252f'
        ]
        
        for pattern in path_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                threats.append({
                    'type': 'path_traversal',
                    'level': ThreatLevel.MEDIUM,
                    'pattern': pattern,
                    'match': match.group(),
                    'position': match.start()
                })
        
        return threats
    
    def _handle_threats(self, threats: List[Dict[str, Any]], text: str, context: str):
        """Handle detected security threats."""
        
        critical_threats = [t for t in threats if t['level'] == ThreatLevel.CRITICAL]
        high_threats = [t for t in threats if t['level'] == ThreatLevel.HIGH]
        
        # Log all threats
        for threat in threats:
            self.logger.warning(
                f"Security threat detected: {threat['type']} "
                f"(level: {threat['level'].value}) "
                f"in context: {context}"
            )
        
        # Block critical threats
        if critical_threats:
            threat_types = {t['type'] for t in critical_threats}
            raise SecurityError(f"Critical security threats detected: {', '.join(threat_types)}")
        
        # Warn about high threats but allow with sanitization
        if high_threats:
            self.logger.error(f"High-level threats will be sanitized: {len(high_threats)} found")
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        import unicodedata
        return unicodedata.normalize('NFKC', text)
    
    def _sanitize_html(self, text: str) -> str:
        """Sanitize HTML content."""
        
        if BLEACH_AVAILABLE:
            return bleach.clean(
                text, 
                tags=self.config.allowed_tags,
                attributes=self.config.allowed_attributes,
                strip=True
            )
        else:
            # Fallback HTML escaping
            return html.escape(text)
    
    def _remove_blocked_patterns(self, text: str) -> str:
        """Remove blocked patterns from text."""
        
        for pattern_regex in self.blocked_pattern_regex:
            text = pattern_regex.sub('', text)
        
        return text
    
    def _ensure_printable(self, text: str) -> str:
        """Ensure text contains only printable characters."""
        
        import string
        printable_chars = set(string.printable)
        return ''.join(c for c in text if c in printable_chars)
    
    def _is_binary(self, text: str) -> bool:
        """Check if text contains binary data."""
        
        try:
            text.encode('utf-8')
            # Check for high ratio of non-printable characters
            import string
            printable = set(string.printable)
            non_printable = sum(1 for c in text if c not in printable)
            return non_printable / max(1, len(text)) > 0.3
        except UnicodeEncodeError:
            return True
    
    def _sanitize_json_recursive(self, obj: Any, depth: int = 0, max_depth: int = 10) -> Any:
        """Recursively sanitize JSON objects."""
        
        if depth > max_depth:
            raise SecurityError("JSON structure too deep")
        
        if isinstance(obj, dict):
            return {
                self._sanitize_json_key(k): self._sanitize_json_recursive(v, depth + 1)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._sanitize_json_recursive(item, depth + 1) for item in obj]
        elif isinstance(obj, str):
            return self.sanitize_text(obj, context="json")
        else:
            return obj
    
    def _sanitize_json_key(self, key: str) -> str:
        """Sanitize JSON object keys."""
        if not isinstance(key, str):
            key = str(key)
        
        # Remove potentially dangerous characters from keys
        key = re.sub(r'[<>"\']', '', key)
        key = key.strip()
        
        return key[:100]  # Limit key length
    
    def _check_rate_limit(self, context: str):
        """Check rate limiting constraints."""
        
        import time
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old entries
        if context in self.request_history:
            self.request_history[context] = [
                t for t in self.request_history[context] if t > minute_ago
            ]
        else:
            self.request_history[context] = []
        
        # Check rate limit
        if len(self.request_history[context]) >= self.config.max_requests_per_minute:
            raise SecurityError("Rate limit exceeded")
        
        # Record this request
        self.request_history[context].append(current_time)
    
    def _contains_malicious_content(self, content: bytes) -> bool:
        """Check file content for malicious patterns."""
        
        try:
            # Convert to text for pattern matching
            text = content.decode('utf-8', errors='ignore')
            
            # Check for embedded scripts or executables
            malicious_patterns = [
                b'\x4d\x5a',  # PE header
                b'\x7f\x45\x4c\x46',  # ELF header
                b'#!/bin/bash',
                b'#!/bin/sh',
                b'<script',
                b'javascript:',
                b'eval(',
                b'exec(',
            ]
            
            for pattern in malicious_patterns:
                if pattern in content:
                    return True
            
            return False
            
        except Exception:
            # If we can't analyze, assume it's suspicious
            return True
    
    def _validate_file_type(self, content: bytes, expected_ext: str) -> bool:
        """Validate file type matches extension."""
        
        # Simple magic number validation
        magic_numbers = {
            '.txt': None,  # Text files don't have magic numbers
            '.json': None,  # JSON files don't have magic numbers
            '.csv': None,   # CSV files don't have magic numbers
            '.md': None,    # Markdown files don't have magic numbers
        }
        
        if expected_ext in magic_numbers:
            expected_magic = magic_numbers[expected_ext]
            if expected_magic is None:
                return True  # No magic number to check
            return content.startswith(expected_magic)
        
        return False  # Unknown file type
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of detected threats."""
        
        return {
            'rate_limit_history': dict(self.request_history),
            'config': {
                'max_text_length': self.config.max_text_length,
                'detect_scripts': self.config.detect_scripts,
                'detect_sql_injection': self.config.detect_sql_injection,
                'detect_xss': self.config.detect_xss,
                'strip_html': self.config.strip_html,
            }
        }


# Export main classes
__all__ = [
    "InputSanitizer",
    "SanitizationConfig",
    "ThreatLevel"
]