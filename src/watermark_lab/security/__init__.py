"""Advanced security framework for watermarking systems."""

from .input_sanitization import InputSanitizer, SanitizationConfig

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
__all__ = ["InputSanitizer", "SanitizationConfig"]

if RATE_LIMITING_AVAILABLE:
    __all__.extend(["RateLimiter", "RateLimitConfig"])

if AUTH_AVAILABLE:
    __all__.extend(["AuthenticationManager", "AuthConfig"])

if ENCRYPTION_AVAILABLE:
    __all__.extend(["EncryptionManager", "KeyManager"])

if AUDIT_AVAILABLE:
    __all__.extend(["SecurityAuditor", "AuditConfig"])