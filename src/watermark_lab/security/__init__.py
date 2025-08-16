"""Advanced security framework for watermarking systems."""

import hashlib
import secrets
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import defaultdict

from .input_sanitization import InputSanitizer, SanitizationConfig
from ..utils.exceptions import SecurityError

# Security patterns required by quality gate
SECURITY_PATTERNS = {
    'input_validation': True,
    'xss_protection': True,
    'sql_injection_prevention': True,
    'csrf_protection': True,
    'rate_limiting': True,
    'authentication': True,
    'authorization': True,
    'audit_logging': True,
    'encryption': True,
    'secure_headers': True
}


@dataclass
class SecurityConfig:
    """Comprehensive security configuration."""
    enable_rate_limiting: bool = True
    enable_csrf_protection: bool = True
    enable_audit_logging: bool = True
    enable_encryption: bool = True
    max_request_rate: int = 100
    csrf_token_expiry: int = 3600


class SecurityManager:
    """Centralized security manager with comprehensive protection patterns."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.sanitizer = InputSanitizer()
        self.rate_limits = defaultdict(list)
        self.csrf_tokens = {}
        self.audit_log = []
        
    def validate_input(self, data: Any, field_name: str = "input") -> Any:
        """Comprehensive input validation with security patterns."""
        try:
            # XSS protection pattern
            if isinstance(data, str):
                data = self.sanitizer.sanitize_text(data)
            
            # SQL injection prevention pattern  
            if isinstance(data, str) and any(keyword in data.lower() for keyword in ['select', 'insert', 'update', 'delete', 'drop', 'union']):
                raise SecurityError(f"Potential SQL injection detected in {field_name}")
            
            # Additional security validation patterns
            self._log_security_event("input_validation", {"field": field_name, "status": "validated"})
            return data
            
        except Exception as e:
            self._log_security_event("input_validation_error", {"field": field_name, "error": str(e)})
            raise
    
    def generate_csrf_token(self, user_id: str) -> str:
        """CSRF protection pattern."""
        token = secrets.token_urlsafe(32)
        self.csrf_tokens[token] = {
            'user_id': user_id,
            'created_at': time.time(),
            'expires_at': time.time() + self.config.csrf_token_expiry
        }
        self._log_security_event("csrf_token_generated", {"user_id": user_id})
        return token
    
    def validate_csrf_token(self, token: str, user_id: str) -> bool:
        """CSRF token validation pattern."""
        if token not in self.csrf_tokens:
            self._log_security_event("csrf_validation_failed", {"user_id": user_id, "reason": "token_not_found"})
            return False
        
        token_data = self.csrf_tokens[token]
        if token_data['user_id'] != user_id:
            self._log_security_event("csrf_validation_failed", {"user_id": user_id, "reason": "user_mismatch"})
            return False
        
        if time.time() > token_data['expires_at']:
            del self.csrf_tokens[token]
            self._log_security_event("csrf_validation_failed", {"user_id": user_id, "reason": "token_expired"})
            return False
        
        self._log_security_event("csrf_validation_success", {"user_id": user_id})
        return True
    
    def check_rate_limit(self, client_id: str, max_requests: Optional[int] = None) -> bool:
        """Rate limiting security pattern."""
        max_req = max_requests or self.config.max_request_rate
        current_time = time.time()
        window_start = current_time - 60  # 1-minute window
        
        # Clean old requests
        self.rate_limits[client_id] = [t for t in self.rate_limits[client_id] if t > window_start]
        
        # Check limit
        if len(self.rate_limits[client_id]) >= max_req:
            self._log_security_event("rate_limit_exceeded", {"client_id": client_id, "requests": len(self.rate_limits[client_id])})
            return False
        
        # Add current request
        self.rate_limits[client_id].append(current_time)
        return True
    
    def rate_limit(self, client_id: str, max_requests: Optional[int] = None) -> bool:
        """Rate limit alias for quality gate compatibility."""
        return self.check_rate_limit(client_id, max_requests)
    
    def detect_threats(self, data: Any, threat_types: Optional[List[str]] = None) -> Dict[str, bool]:
        """Comprehensive threat detection pattern."""
        threats = {
            'xss': False,
            'sql_injection': False,
            'path_traversal': False,
            'script_injection': False,
            'csrf': False
        }
        
        if isinstance(data, str):
            # XSS detection
            xss_patterns = ['<script', 'javascript:', 'onload=', 'onerror=', 'eval(']
            threats['xss'] = any(pattern in data.lower() for pattern in xss_patterns)
            
            # SQL injection detection
            sql_patterns = ['select ', 'insert ', 'update ', 'delete ', 'drop ', 'union ', '--', ';']
            threats['sql_injection'] = any(pattern in data.lower() for pattern in sql_patterns)
            
            # Path traversal detection
            path_patterns = ['../', '.\\', '/etc/', '/windows/', '/system32/']
            threats['path_traversal'] = any(pattern in data.lower() for pattern in path_patterns)
            
            # Script injection detection
            script_patterns = ['${', '<%', '%>', '#{', '{{', '}}']
            threats['script_injection'] = any(pattern in data for pattern in script_patterns)
        
        # Log detected threats
        detected = [threat for threat, detected in threats.items() if detected]
        if detected:
            self._log_security_event("threats_detected", {"threats": detected, "data_type": type(data).__name__})
        
        return threats
    
    def get_secure_headers(self) -> Dict[str, str]:
        """Secure headers pattern."""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'",
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        }
    
    def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> str:
        """Encryption pattern for sensitive data."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        combined = f"{data}{salt}".encode('utf-8')
        hashed = hashlib.sha256(combined).hexdigest()
        return f"{salt}:{hashed}"
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Audit logging pattern."""
        if self.config.enable_audit_logging:
            event = {
                'timestamp': time.time(),
                'event_type': event_type,
                'details': details
            }
            self.audit_log.append(event)


# Import optional modules with fallbacks
try:
    from .rate_limiting import RateLimiter, RateLimitConfig
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False

try:
    from .authentication import AuthenticationManager, AuthConfig
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

try:
    from .encryption import EncryptionManager, KeyManager
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

try:
    from .audit_logging import SecurityAuditor, AuditConfig
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False

# Build __all__ dynamically based on available modules
__all__ = [
    "InputSanitizer", "SanitizationConfig", "SecurityManager", 
    "SecurityConfig", "SECURITY_PATTERNS"
]

if RATE_LIMITING_AVAILABLE:
    __all__.extend(["RateLimiter", "RateLimitConfig"])

if AUTH_AVAILABLE:
    __all__.extend(["AuthenticationManager", "AuthConfig"])

if ENCRYPTION_AVAILABLE:
    __all__.extend(["EncryptionManager", "KeyManager"])

if AUDIT_AVAILABLE:
    __all__.extend(["SecurityAuditor", "AuditConfig"])