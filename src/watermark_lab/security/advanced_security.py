"""Advanced security system with authentication, authorization, and threat protection."""

import time
import hashlib
import hmac
import secrets
import threading
from typing import Dict, Any, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
from enum import Enum
import logging
import json
import re
from pathlib import Path
import ipaddress
from functools import wraps

class SecurityError(Exception):
    """Security-related exception."""
    pass

class SecurityLevel(Enum):
    """Security levels for operations."""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    PRIVILEGED = "privileged"
    ADMIN = "admin"


class ThreatLevel(Enum):
    """Threat levels for security events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class User:
    """User identity and permissions."""
    
    user_id: str
    username: str
    email: str = ""
    roles: Set[str] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    login_attempts: int = 0
    locked_until: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role."""
        return role in self.roles
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions or "admin" in self.roles
    
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        return self.locked_until is not None and time.time() < self.locked_until
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)."""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'roles': list(self.roles),
            'permissions': list(self.permissions),
            'created_at': self.created_at,
            'last_login': self.last_login,
            'locked': self.is_locked()
        }


@dataclass
class SecurityEvent:
    """Security event record."""
    
    event_id: str = field(default_factory=lambda: secrets.token_hex(8))
    event_type: str = ""
    threat_level: ThreatLevel = ThreatLevel.LOW
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    description: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'threat_level': self.threat_level.value,
            'user_id': self.user_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'description': self.description,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


@dataclass
class AccessToken:
    """JWT-like access token."""
    
    token_id: str = field(default_factory=lambda: secrets.token_hex(16))
    user_id: str = ""
    issued_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 3600)  # 1 hour
    scopes: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return time.time() > self.expires_at
    
    def has_scope(self, scope: str) -> bool:
        """Check if token has specific scope."""
        return scope in self.scopes or "admin" in self.scopes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'token_id': self.token_id,
            'user_id': self.user_id,
            'issued_at': self.issued_at,
            'expires_at': self.expires_at,
            'scopes': list(self.scopes),
            'expired': self.is_expired()
        }


class PasswordValidator:
    """Password strength validation."""
    
    def __init__(self):
        self.min_length = 12
        self.require_uppercase = True
        self.require_lowercase = True
        self.require_digits = True
        self.require_special = True
        self.special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    def validate(self, password: str) -> tuple[bool, List[str]]:
        """Validate password strength."""
        
        issues = []
        
        if len(password) < self.min_length:
            issues.append(f"Password must be at least {self.min_length} characters long")
        
        if self.require_uppercase and not re.search(r'[A-Z]', password):
            issues.append("Password must contain at least one uppercase letter")
        
        if self.require_lowercase and not re.search(r'[a-z]', password):
            issues.append("Password must contain at least one lowercase letter")
        
        if self.require_digits and not re.search(r'\d', password):
            issues.append("Password must contain at least one digit")
        
        if self.require_special and not any(c in self.special_chars for c in password):
            issues.append("Password must contain at least one special character")
        
        # Check for common patterns
        if password.lower() in ['password', '123456789', 'qwerty', 'admin']:
            issues.append("Password contains common patterns")
        
        return len(issues) == 0, issues


class RateLimiter:
    """Rate limiting for security protection."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)
        self._lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for identifier."""
        
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        with self._lock:
            # Clean old requests
            request_times = self.requests[identifier]
            while request_times and request_times[0] < window_start:
                request_times.popleft()
            
            # Check limit
            if len(request_times) >= self.max_requests:
                return False
            
            # Record current request
            request_times.append(current_time)
            return True
    
    def get_stats(self, identifier: str) -> Dict[str, Any]:
        """Get rate limiting stats for identifier."""
        
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        with self._lock:
            request_times = self.requests[identifier]
            recent_requests = [t for t in request_times if t >= window_start]
            
            return {
                'identifier': identifier,
                'requests_in_window': len(recent_requests),
                'max_requests': self.max_requests,
                'window_seconds': self.window_seconds,
                'remaining_requests': max(0, self.max_requests - len(recent_requests))
            }


class InputSanitizer:
    """Input sanitization and validation."""
    
    def __init__(self):
        self.sql_injection_patterns = [
            r"('|(\\')|(;)|(\-\-)|(\s+(or|and)\s+)",
            r"(union\s+select)",
            r"(drop\s+table)",
            r"(delete\s+from)",
            r"(insert\s+into)",
            r"(update\s+.+set)"
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"onload\s*=",
            r"onerror\s*=",
            r"onclick\s*="
        ]
        
        self.path_traversal_patterns = [
            r"\.\.\/",
            r"\.\.\\",
            r"%2e%2e%2f",
            r"%2e%2e%5c"
        ]
    
    def sanitize_text(self, text: str, max_length: int = 10000) -> str:
        """Sanitize text input."""
        
        if not isinstance(text, str):
            text = str(text)
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Basic HTML escaping
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&#x27;')
        
        return text
    
    def detect_injection_attempt(self, text: str) -> tuple[bool, List[str]]:
        """Detect potential injection attempts."""
        
        threats = []
        text_lower = text.lower()
        
        # SQL injection detection
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                threats.append("sql_injection")
                break
        
        # XSS detection
        for pattern in self.xss_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                threats.append("xss")
                break
        
        # Path traversal detection
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                threats.append("path_traversal")
                break
        
        return len(threats) > 0, threats
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        
        # Remove path components
        filename = filename.split('/')[-1].split('\\')[-1]
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Ensure not empty
        if not filename:
            filename = "unnamed_file"
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            max_name_length = 250 - len(ext)
            filename = name[:max_name_length] + ('.' + ext if ext else '')
        
        return filename


class AdvancedSecurity:
    """Advanced security system."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # User management
        self.users: Dict[str, User] = {}
        self.active_tokens: Dict[str, AccessToken] = {}
        
        # Security components
        self.password_validator = PasswordValidator()
        self.rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
        self.input_sanitizer = InputSanitizer()
        
        # Security monitoring
        self.security_events: deque = deque(maxlen=10000)
        self.failed_login_attempts = defaultdict(int)
        self.blocked_ips: Dict[str, float] = {}  # IP -> unblock_time
        
        # Security configuration
        self.max_login_attempts = 5
        self.lockout_duration = 900  # 15 minutes
        self.token_expiration = 3600  # 1 hour
        self.require_2fa = False
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Setup default admin user
        self._setup_default_users()
        
        self.logger.info("Advanced security system initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup security logging."""
        logger = logging.getLogger("advanced_security")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def _setup_default_users(self):
        """Setup default users."""
        
        admin_user = User(
            user_id="admin_001",
            username="admin",
            email="admin@watermark-lab.com",
            roles={"admin"},
            permissions={"*"}
        )
        
        self.users[admin_user.user_id] = admin_user
        
        # Create demo user
        demo_user = User(
            user_id="demo_001",
            username="demo",
            email="demo@watermark-lab.com",
            roles={"user"},
            permissions={"watermark", "detect", "experiment"}
        )
        
        self.users[demo_user.user_id] = demo_user
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: Optional[Set[str]] = None,
        permissions: Optional[Set[str]] = None
    ) -> tuple[bool, str, Optional[User]]:
        """Create a new user."""
        
        with self._lock:
            # Validate password
            password_valid, password_issues = self.password_validator.validate(password)
            if not password_valid:
                return False, "; ".join(password_issues), None
            
            # Check if username exists
            if any(user.username == username for user in self.users.values()):
                return False, "Username already exists", None
            
            # Create user
            user_id = f"user_{secrets.token_hex(8)}"
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                roles=roles or {"user"},
                permissions=permissions or {"watermark", "detect"}
            )
            
            self.users[user_id] = user
            
            # Store password hash (simplified - would use proper hashing)
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), b'salt', 100000)
            user.metadata['password_hash'] = password_hash.hex()
            
            # Log security event
            self._log_security_event(
                event_type="user_created",
                threat_level=ThreatLevel.LOW,
                user_id=user_id,
                description=f"User created: {username}"
            )
            
            self.logger.info(f"Created user: {username} ({user_id})")
            return True, "User created successfully", user
    
    def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> tuple[bool, str, Optional[AccessToken]]:
        """Authenticate user and return access token."""
        
        with self._lock:
            # Check IP blocking
            if ip_address and self._is_ip_blocked(ip_address):
                self._log_security_event(
                    event_type="blocked_ip_access",
                    threat_level=ThreatLevel.HIGH,
                    ip_address=ip_address,
                    description=f"Access attempt from blocked IP: {ip_address}"
                )
                return False, "IP address is temporarily blocked", None
            
            # Rate limiting
            if ip_address and not self.rate_limiter.is_allowed(f"auth_{ip_address}"):
                self._log_security_event(
                    event_type="rate_limit_exceeded",
                    threat_level=ThreatLevel.MEDIUM,
                    ip_address=ip_address,
                    description="Authentication rate limit exceeded"
                )
                return False, "Too many authentication attempts", None
            
            # Find user
            user = None
            for u in self.users.values():
                if u.username == username:
                    user = u
                    break
            
            if not user:
                self._log_security_event(
                    event_type="invalid_username",
                    threat_level=ThreatLevel.LOW,
                    ip_address=ip_address,
                    description=f"Invalid username: {username}"
                )
                return False, "Invalid credentials", None
            
            # Check if user is locked
            if user.is_locked():
                self._log_security_event(
                    event_type="locked_account_access",
                    threat_level=ThreatLevel.MEDIUM,
                    user_id=user.user_id,
                    ip_address=ip_address,
                    description=f"Access attempt to locked account: {username}"
                )
                return False, "Account is temporarily locked", None
            
            # Validate password (simplified)
            stored_hash = user.metadata.get('password_hash', '')
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), b'salt', 100000)
            
            if stored_hash != password_hash.hex():
                # Failed authentication
                user.login_attempts += 1
                
                if user.login_attempts >= self.max_login_attempts:
                    user.locked_until = time.time() + self.lockout_duration
                    
                    self._log_security_event(
                        event_type="account_locked",
                        threat_level=ThreatLevel.HIGH,
                        user_id=user.user_id,
                        ip_address=ip_address,
                        description=f"Account locked after {self.max_login_attempts} failed attempts"
                    )
                    
                    return False, "Account has been locked due to multiple failed login attempts", None
                
                self._log_security_event(
                    event_type="failed_login",
                    threat_level=ThreatLevel.MEDIUM,
                    user_id=user.user_id,
                    ip_address=ip_address,
                    description=f"Failed login attempt ({user.login_attempts}/{self.max_login_attempts})"
                )
                
                return False, "Invalid credentials", None
            
            # Successful authentication
            user.login_attempts = 0
            user.last_login = time.time()
            
            # Create access token
            token = AccessToken(
                user_id=user.user_id,
                expires_at=time.time() + self.token_expiration,
                scopes=user.permissions
            )
            
            self.active_tokens[token.token_id] = token
            
            self._log_security_event(
                event_type="successful_login",
                threat_level=ThreatLevel.LOW,
                user_id=user.user_id,
                ip_address=ip_address,
                description=f"Successful login: {username}"
            )
            
            self.logger.info(f"User authenticated: {username}")
            return True, "Authentication successful", token
    
    def validate_token(self, token_id: str) -> tuple[bool, Optional[User], Optional[AccessToken]]:
        """Validate access token and return user."""
        
        with self._lock:
            if token_id not in self.active_tokens:
                return False, None, None
            
            token = self.active_tokens[token_id]
            
            if token.is_expired():
                del self.active_tokens[token_id]
                return False, None, None
            
            user = self.users.get(token.user_id)
            if not user:
                del self.active_tokens[token_id]
                return False, None, None
            
            if user.is_locked():
                return False, None, None
            
            return True, user, token
    
    def revoke_token(self, token_id: str) -> bool:
        """Revoke access token."""
        
        with self._lock:
            if token_id in self.active_tokens:
                token = self.active_tokens[token_id]
                del self.active_tokens[token_id]
                
                self._log_security_event(
                    event_type="token_revoked",
                    threat_level=ThreatLevel.LOW,
                    user_id=token.user_id,
                    description=f"Token revoked: {token_id}"
                )
                
                return True
            
            return False
    
    def check_permission(
        self,
        user: User,
        permission: str,
        resource: Optional[str] = None
    ) -> bool:
        """Check if user has permission for operation."""
        
        # Admin has all permissions
        if "admin" in user.roles:
            return True
        
        # Check specific permission
        if user.has_permission(permission):
            return True
        
        # Check resource-specific permissions
        if resource:
            resource_permission = f"{permission}:{resource}"
            if user.has_permission(resource_permission):
                return True
        
        return False
    
    def sanitize_input(self, input_data: Any) -> Any:
        """Sanitize input data."""
        
        if isinstance(input_data, str):
            # Check for injection attempts
            has_threats, threats = self.input_sanitizer.detect_injection_attempt(input_data)
            
            if has_threats:
                self._log_security_event(
                    event_type="injection_attempt",
                    threat_level=ThreatLevel.HIGH,
                    description=f"Injection attempt detected: {', '.join(threats)}"
                )
            
            return self.input_sanitizer.sanitize_text(input_data)
        
        elif isinstance(input_data, dict):
            return {k: self.sanitize_input(v) for k, v in input_data.items()}
        
        elif isinstance(input_data, list):
            return [self.sanitize_input(item) for item in input_data]
        
        else:
            return input_data
    
    def _is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        
        if ip_address in self.blocked_ips:
            unblock_time = self.blocked_ips[ip_address]
            
            if time.time() < unblock_time:
                return True
            else:
                # Unblock expired IP
                del self.blocked_ips[ip_address]
        
        return False
    
    def block_ip(self, ip_address: str, duration: int = 3600):
        """Block IP address for specified duration."""
        
        self.blocked_ips[ip_address] = time.time() + duration
        
        self._log_security_event(
            event_type="ip_blocked",
            threat_level=ThreatLevel.HIGH,
            ip_address=ip_address,
            description=f"IP blocked for {duration} seconds"
        )
        
        self.logger.warning(f"Blocked IP address: {ip_address}")
    
    def _log_security_event(
        self,
        event_type: str,
        threat_level: ThreatLevel,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log security event."""
        
        event = SecurityEvent(
            event_type=event_type,
            threat_level=threat_level,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            description=description,
            metadata=metadata or {}
        )
        
        self.security_events.append(event)
        
        # Log based on threat level
        if threat_level == ThreatLevel.CRITICAL:
            self.logger.critical(f"SECURITY ALERT: {description}")
        elif threat_level == ThreatLevel.HIGH:
            self.logger.error(f"SECURITY WARNING: {description}")
        elif threat_level == ThreatLevel.MEDIUM:
            self.logger.warning(f"Security event: {description}")
        else:
            self.logger.info(f"Security event: {description}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security system summary."""
        
        with self._lock:
            # Count events by threat level
            threat_counts = defaultdict(int)
            for event in self.security_events:
                threat_counts[event.threat_level.value] += 1
            
            # Recent events (last 24 hours)
            day_ago = time.time() - 86400
            recent_events = [e for e in self.security_events if e.timestamp > day_ago]
            
            return {
                'users_count': len(self.users),
                'active_tokens': len(self.active_tokens),
                'blocked_ips': len(self.blocked_ips),
                'total_security_events': len(self.security_events),
                'recent_events_24h': len(recent_events),
                'threat_level_counts': dict(threat_counts),
                'max_login_attempts': self.max_login_attempts,
                'token_expiration_hours': self.token_expiration / 3600,
                'rate_limiting_enabled': True
            }
    
    def get_user_activity(self, user_id: str) -> Dict[str, Any]:
        """Get user activity summary."""
        
        user = self.users.get(user_id)
        if not user:
            return {'error': 'User not found'}
        
        # Get user's security events
        user_events = [e for e in self.security_events if e.user_id == user_id]
        
        # Get active tokens for user
        user_tokens = [t for t in self.active_tokens.values() if t.user_id == user_id]
        
        return {
            'user': user.to_dict(),
            'security_events_count': len(user_events),
            'active_tokens_count': len(user_tokens),
            'last_login': user.last_login,
            'login_attempts': user.login_attempts,
            'account_locked': user.is_locked()
        }
    
    @contextmanager
    def security_context(
        self,
        user: Optional[User] = None,
        ip_address: Optional[str] = None,
        operation: str = ""
    ):
        """Security context manager."""
        
        start_time = time.time()
        
        try:
            yield
            
            # Log successful operation
            if user and operation:
                self._log_security_event(
                    event_type="operation_success",
                    threat_level=ThreatLevel.LOW,
                    user_id=user.user_id,
                    ip_address=ip_address,
                    description=f"Operation completed: {operation}",
                    metadata={'duration': time.time() - start_time}
                )
                
        except Exception as e:
            # Log failed operation
            if user and operation:
                self._log_security_event(
                    event_type="operation_failure",
                    threat_level=ThreatLevel.MEDIUM,
                    user_id=user.user_id,
                    ip_address=ip_address,
                    description=f"Operation failed: {operation} - {str(e)}",
                    metadata={'duration': time.time() - start_time, 'error': str(e)}
                )
            
            raise


# Security decorators
def require_authentication(func: Callable) -> Callable:
    """Decorator to require authentication."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Enhanced authentication check with proper error handling
        try:
            # Extract token from kwargs or request context
            token_id = kwargs.pop('auth_token', None)
            if not token_id:
                raise SecurityError("Authentication required")
            
            is_valid, user, token = advanced_security.validate_token(token_id)
            if not is_valid:
                raise SecurityError("Invalid or expired token")
            
            # Add user context to kwargs
            kwargs['authenticated_user'] = user
            kwargs['access_token'] = token
            
            return func(*args, **kwargs)
            
        except SecurityError:
            raise
        except Exception as e:
            raise SecurityError(f"Authentication error: {str(e)}")
    
    return wrapper


def require_permission(permission: str):
    """Decorator to require specific permission."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Enhanced permission check
            try:
                user = kwargs.get('authenticated_user')
                if not user:
                    raise SecurityError("User authentication required")
                
                if not advanced_security.check_permission(user, permission):
                    advanced_security._log_security_event(
                        event_type="permission_denied",
                        threat_level=ThreatLevel.MEDIUM,
                        user_id=user.user_id,
                        description=f"Permission denied: {permission}"
                    )
                    raise SecurityError(f"Permission denied: {permission}")
                
                return func(*args, **kwargs)
                
            except SecurityError:
                raise
            except Exception as e:
                raise SecurityError(f"Permission check error: {str(e)}")
                
        return wrapper
    
    return decorator


# Global security instance
advanced_security = AdvancedSecurity()


# Convenience functions
def authenticate(username: str, password: str, ip_address: Optional[str] = None) -> tuple[bool, str, Optional[AccessToken]]:
    """Authenticate user."""
    return advanced_security.authenticate_user(username, password, ip_address)


def validate_token(token_id: str) -> tuple[bool, Optional[User], Optional[AccessToken]]:
    """Validate access token."""
    return advanced_security.validate_token(token_id)


def sanitize_input(input_data: Any) -> Any:
    """Sanitize input data."""
    return advanced_security.sanitize_input(input_data)


def get_security_summary() -> Dict[str, Any]:
    """Get security summary."""
    return advanced_security.get_security_summary()


__all__ = [
    'AdvancedSecurity',
    'User',
    'AccessToken',
    'SecurityEvent',
    'SecurityLevel',
    'ThreatLevel',
    'PasswordValidator',
    'RateLimiter',
    'InputSanitizer',
    'advanced_security',
    'require_authentication',
    'require_permission',
    'authenticate',
    'validate_token',
    'sanitize_input',
    'get_security_summary'
]