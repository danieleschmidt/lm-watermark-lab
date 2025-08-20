"""Robust security implementation with comprehensive protection measures."""

import re
import hashlib
import hmac
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import secrets
import unicodedata

class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    """Types of security threats."""
    INJECTION = "injection"
    XSS = "xss"
    PATH_TRAVERSAL = "path_traversal"
    EXCESSIVE_LENGTH = "excessive_length"
    MALICIOUS_PATTERN = "malicious_pattern"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"

@dataclass
class SecurityViolation:
    """Security violation details."""
    threat_type: ThreatType
    severity: SecurityLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    client_id: Optional[str] = None
    blocked: bool = True

@dataclass
class SecurityConfig:
    """Security configuration."""
    max_text_length: int = 100000
    max_prompt_length: int = 10000
    max_requests_per_minute: int = 60
    blocked_patterns: List[str] = field(default_factory=lambda: [
        r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
        r'javascript:',  # JavaScript protocols
        r'vbscript:',    # VBScript protocols
        r'on\w+\s*=',    # Event handlers
        r'\.\./.*',      # Path traversal
        r'file://',      # File protocols
        r'ftp://',       # FTP protocols
    ])
    sensitive_keywords: List[str] = field(default_factory=lambda: [
        'password', 'secret', 'token', 'key', 'credential', 'auth',
        'private', 'confidential', 'internal', 'admin'
    ])
    max_unicode_categories: Set[str] = field(default_factory=lambda: {
        'Lu', 'Ll', 'Lt', 'Lm', 'Lo',  # Letters
        'Nd', 'Nl', 'No',              # Numbers
        'Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po',  # Punctuation
        'Sm', 'Sc', 'Sk', 'So',        # Symbols
        'Zs', 'Zl', 'Zp',             # Separators
    })

class RobustSecurityManager:
    """Comprehensive security manager for watermark operations."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize security manager."""
        self.config = config or SecurityConfig()
        self.logger = logging.getLogger("security.manager")
        
        # Rate limiting storage
        self._request_history: Dict[str, List[float]] = {}
        
        # Threat detection
        self._compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                 for pattern in self.config.blocked_patterns]
        
        # Security metrics
        self._violation_count = 0
        self._blocked_requests = 0
        
        self.logger.info("Security manager initialized")
    
    def validate_text_input(self, text: str, context: str = "general", 
                          max_length: Optional[int] = None) -> Tuple[str, List[SecurityViolation]]:
        """
        Comprehensively validate and sanitize text input.
        
        Args:
            text: Input text to validate
            context: Context of the input (prompt, content, etc.)
            max_length: Maximum allowed length (overrides config)
            
        Returns:
            Tuple of (sanitized_text, violations_list)
        """
        violations = []
        
        # Input type validation
        if not isinstance(text, str):
            violations.append(SecurityViolation(
                threat_type=ThreatType.INJECTION,
                severity=SecurityLevel.HIGH,
                message=f"Invalid input type: expected str, got {type(text)}"
            ))
            return "", violations
        
        # Length validation
        max_len = max_length or self.config.max_text_length
        if context == "prompt":
            max_len = min(max_len, self.config.max_prompt_length)
            
        if len(text) > max_len:
            violations.append(SecurityViolation(
                threat_type=ThreatType.EXCESSIVE_LENGTH,
                severity=SecurityLevel.MEDIUM,
                message=f"Text too long: {len(text)} > {max_len}"
            ))
            text = text[:max_len]  # Truncate instead of blocking
        
        # Unicode validation
        sanitized_text = self._validate_unicode(text)
        if sanitized_text != text:
            violations.append(SecurityViolation(
                threat_type=ThreatType.MALICIOUS_PATTERN,
                severity=SecurityLevel.LOW,
                message="Suspicious unicode characters removed"
            ))
            text = sanitized_text
        
        # Pattern matching for malicious content
        pattern_violations = self._detect_malicious_patterns(text)
        violations.extend(pattern_violations)
        
        # Remove detected malicious patterns
        for pattern in self._compiled_patterns:
            if pattern.search(text):
                text = pattern.sub('', text)
        
        # Sensitive information detection
        sensitive_violations = self._detect_sensitive_content(text)
        violations.extend(sensitive_violations)
        
        # Final sanitization
        text = self._final_sanitization(text)
        
        # Log violations
        for violation in violations:
            self._log_violation(violation, context)
        
        return text, violations
    
    def _validate_unicode(self, text: str) -> str:
        """Validate and sanitize unicode characters."""
        sanitized_chars = []
        
        for char in text:
            category = unicodedata.category(char)
            
            # Allow standard categories
            if category in self.config.max_unicode_categories:
                sanitized_chars.append(char)
            # Allow some control characters
            elif category == 'Cc' and char in '\n\r\t':
                sanitized_chars.append(char)
            else:
                # Replace suspicious characters
                sanitized_chars.append('?')
        
        return ''.join(sanitized_chars)
    
    def _detect_malicious_patterns(self, text: str) -> List[SecurityViolation]:
        """Detect malicious patterns in text."""
        violations = []
        
        for i, pattern in enumerate(self._compiled_patterns):
            if pattern.search(text):
                violations.append(SecurityViolation(
                    threat_type=ThreatType.MALICIOUS_PATTERN,
                    severity=SecurityLevel.HIGH,
                    message=f"Malicious pattern detected: pattern_{i}"
                ))
        
        # Additional heuristic checks
        
        # Excessive script-like content
        if text.count('<') > 10 and text.count('>') > 10:
            violations.append(SecurityViolation(
                threat_type=ThreatType.XSS,
                severity=SecurityLevel.MEDIUM,
                message="Excessive HTML-like brackets detected"
            ))
        
        # Path traversal attempts
        if '../' in text or '..\\' in text:
            violations.append(SecurityViolation(
                threat_type=ThreatType.PATH_TRAVERSAL,
                severity=SecurityLevel.HIGH,
                message="Path traversal attempt detected"
            ))
        
        # Excessive URL-like content
        url_count = len(re.findall(r'https?://', text, re.IGNORECASE))
        if url_count > 5:
            violations.append(SecurityViolation(
                threat_type=ThreatType.MALICIOUS_PATTERN,
                severity=SecurityLevel.MEDIUM,
                message=f"Excessive URLs detected: {url_count}"
            ))
        
        return violations
    
    def _detect_sensitive_content(self, text: str) -> List[SecurityViolation]:
        """Detect potentially sensitive content."""
        violations = []
        text_lower = text.lower()
        
        # Check for sensitive keywords
        found_keywords = []
        for keyword in self.config.sensitive_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        if found_keywords:
            violations.append(SecurityViolation(
                threat_type=ThreatType.MALICIOUS_PATTERN,
                severity=SecurityLevel.MEDIUM,
                message=f"Sensitive keywords detected: {', '.join(found_keywords)}"
            ))
        
        # Check for patterns that look like secrets
        
        # API key patterns
        if re.search(r'[a-zA-Z0-9]{32,}', text):
            violations.append(SecurityViolation(
                threat_type=ThreatType.MALICIOUS_PATTERN,
                severity=SecurityLevel.HIGH,
                message="Potential API key or token detected"
            ))
        
        # Email patterns (might be PII)
        email_count = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        if email_count > 2:
            violations.append(SecurityViolation(
                threat_type=ThreatType.MALICIOUS_PATTERN,
                severity=SecurityLevel.LOW,
                message=f"Multiple email addresses detected: {email_count}"
            ))
        
        return violations
    
    def _final_sanitization(self, text: str) -> str:
        """Final sanitization pass."""
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove zero-width characters
        text = ''.join(char for char in text if unicodedata.category(char) != 'Cf')
        
        return text
    
    def check_rate_limit(self, client_id: str) -> bool:
        """
        Check if client has exceeded rate limits.
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            True if request is allowed, False if rate limited
        """
        current_time = time.time()
        
        # Clean old entries
        if client_id in self._request_history:
            self._request_history[client_id] = [
                timestamp for timestamp in self._request_history[client_id]
                if current_time - timestamp < 60  # Keep last minute
            ]
        else:
            self._request_history[client_id] = []
        
        # Check rate limit
        request_count = len(self._request_history[client_id])
        if request_count >= self.config.max_requests_per_minute:
            self._log_violation(SecurityViolation(
                threat_type=ThreatType.RATE_LIMIT,
                severity=SecurityLevel.MEDIUM,
                message=f"Rate limit exceeded: {request_count} requests/minute",
                client_id=client_id
            ))
            return False
        
        # Record this request
        self._request_history[client_id].append(current_time)
        return True
    
    def validate_api_key(self, api_key: str, expected_hash: str) -> bool:
        """
        Validate API key using secure comparison.
        
        Args:
            api_key: Provided API key
            expected_hash: Expected hash of the API key
            
        Returns:
            True if valid, False otherwise
        """
        if not api_key or not expected_hash:
            return False
        
        try:
            # Hash the provided key
            key_hash = hashlib.sha256(api_key.encode('utf-8')).hexdigest()
            
            # Secure comparison to prevent timing attacks
            return hmac.compare_digest(key_hash, expected_hash)
            
        except Exception as e:
            self.logger.error(f"API key validation error: {e}")
            return False
    
    def generate_secure_key(self, length: int = 32) -> str:
        """Generate cryptographically secure random key."""
        return secrets.token_hex(length)
    
    def hash_key(self, key: str, salt: Optional[str] = None) -> str:
        """Hash a key with optional salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 for key derivation
        import hashlib
        key_bytes = hashlib.pbkdf2_hmac('sha256', key.encode('utf-8'), salt.encode('utf-8'), 100000)
        return salt + key_bytes.hex()
    
    def verify_key_hash(self, key: str, key_hash: str) -> bool:
        """Verify a key against its hash."""
        try:
            salt = key_hash[:32]  # First 32 chars are salt
            stored_hash = key_hash[32:]
            
            key_bytes = hashlib.pbkdf2_hmac('sha256', key.encode('utf-8'), salt.encode('utf-8'), 100000)
            computed_hash = key_bytes.hex()
            
            return hmac.compare_digest(stored_hash, computed_hash)
            
        except Exception as e:
            self.logger.error(f"Key verification error: {e}")
            return False
    
    def _log_violation(self, violation: SecurityViolation, context: str = "") -> None:
        """Log security violation."""
        self._violation_count += 1
        
        log_message = f"Security violation [{violation.threat_type.value}]: {violation.message}"
        if context:
            log_message += f" (context: {context})"
        if violation.client_id:
            log_message += f" (client: {violation.client_id})"
        
        if violation.severity == SecurityLevel.CRITICAL:
            self.logger.critical(log_message)
        elif violation.severity == SecurityLevel.HIGH:
            self.logger.error(log_message)
        elif violation.severity == SecurityLevel.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics."""
        return {
            "total_violations": self._violation_count,
            "blocked_requests": self._blocked_requests,
            "active_clients": len(self._request_history),
            "rate_limited_clients": sum(
                1 for requests in self._request_history.values()
                if len(requests) >= self.config.max_requests_per_minute
            ),
            "config": {
                "max_text_length": self.config.max_text_length,
                "max_prompt_length": self.config.max_prompt_length,
                "max_requests_per_minute": self.config.max_requests_per_minute,
                "blocked_patterns_count": len(self.config.blocked_patterns),
                "sensitive_keywords_count": len(self.config.sensitive_keywords)
            }
        }
    
    def reset_client_history(self, client_id: Optional[str] = None) -> None:
        """Reset rate limiting history for a client or all clients."""
        if client_id:
            self._request_history.pop(client_id, None)
        else:
            self._request_history.clear()
    
    def update_config(self, **kwargs) -> None:
        """Update security configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated security config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown config key: {key}")
        
        # Recompile patterns if updated
        if 'blocked_patterns' in kwargs:
            self._compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                     for pattern in self.config.blocked_patterns]

# Global security manager instance
_security_manager = None

def get_security_manager(config: Optional[SecurityConfig] = None) -> RobustSecurityManager:
    """Get global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = RobustSecurityManager(config)
    return _security_manager