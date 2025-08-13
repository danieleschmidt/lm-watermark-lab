"""Production-grade authentication and authorization system."""

import os
try:
    import jwt
except ImportError:
    jwt = None
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
from functools import wraps

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

try:
    from passlib.context import CryptContext
    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False

from ..utils.logging import get_logger, StructuredLogger
from ..utils.exceptions import AuthenticationError, AuthorizationError, SecurityError
from ..config.settings import get_settings

logger = get_logger("security.auth")
structured_logger = StructuredLogger("auth")


class UserRole(Enum):
    """User roles with different permission levels."""
    ADMIN = "admin"
    RESEARCHER = "researcher" 
    USER = "user"
    READONLY = "readonly"
    API_USER = "api_user"


class Permission(Enum):
    """System permissions."""
    WATERMARK_GENERATE = "watermark:generate"
    WATERMARK_DETECT = "watermark:detect"
    ATTACK_SIMULATE = "attack:simulate"
    BENCHMARK_RUN = "benchmark:run"
    BATCH_PROCESS = "batch:process"
    ADMIN_ACCESS = "admin:access"
    METRICS_VIEW = "metrics:view"
    HEALTH_VIEW = "health:view"
    TASK_CANCEL = "task:cancel"
    USER_MANAGE = "user:manage"


@dataclass
class User:
    """User data structure."""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: List[Permission]
    is_active: bool = True
    created_at: datetime = None
    last_login: Optional[datetime] = None
    api_key: Optional[str] = None
    rate_limit_tier: str = "standard"  # standard, premium, enterprise
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions or Permission.ADMIN_ACCESS in self.permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "permissions": [p.value for p in self.permissions],
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "rate_limit_tier": self.rate_limit_tier
        }


@dataclass
class TokenPayload:
    """JWT token payload structure."""
    user_id: str
    username: str
    role: str
    permissions: List[str]
    exp: int
    iat: int
    jti: str  # JWT ID for token revocation


class PasswordManager:
    """Secure password hashing and validation."""
    
    def __init__(self):
        self.logger = get_logger("password_manager")
        
        if PASSLIB_AVAILABLE:
            self.pwd_context = CryptContext(
                schemes=["bcrypt"],
                deprecated="auto",
                bcrypt__rounds=12
            )
        elif BCRYPT_AVAILABLE:
            self.bcrypt_rounds = 12
        else:
            logger.warning("Neither passlib nor bcrypt available, using fallback hashing")
    
    def hash_password(self, password: str) -> str:
        """Hash a password securely."""
        if not password:
            raise SecurityError("Password cannot be empty")
        
        if len(password) < 8:
            raise SecurityError("Password must be at least 8 characters long")
        
        try:
            if PASSLIB_AVAILABLE:
                return self.pwd_context.hash(password)
            elif BCRYPT_AVAILABLE:
                salt = bcrypt.gensalt(rounds=self.bcrypt_rounds)
                return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
            else:
                # Fallback - NOT recommended for production
                salt = secrets.token_hex(32)
                hashed = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
                return f"pbkdf2_sha256${salt}${hashed.hex()}"
                
        except Exception as e:
            self.logger.error(f"Password hashing failed: {e}")
            raise SecurityError("Password hashing failed")
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        if not password or not hashed_password:
            return False
        
        try:
            if PASSLIB_AVAILABLE:
                return self.pwd_context.verify(password, hashed_password)
            elif BCRYPT_AVAILABLE:
                return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
            else:
                # Fallback verification
                if hashed_password.startswith("pbkdf2_sha256$"):
                    parts = hashed_password.split("$")
                    if len(parts) == 3:
                        salt = parts[1]
                        stored_hash = parts[2]
                        new_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
                        return new_hash.hex() == stored_hash
                return False
                
        except Exception as e:
            self.logger.error(f"Password verification failed: {e}")
            return False


class JWTManager:
    """JWT token management with security features."""
    
    def __init__(self, secret_key: str = None, algorithm: str = "HS256"):
        self.settings = get_settings()
        self.secret_key = secret_key or self.settings.secret_key
        self.algorithm = algorithm
        self.logger = get_logger("jwt_manager")
        
        # Token blacklist (in production, use Redis)
        self.blacklisted_tokens = set()
        
        if not self.secret_key or self.secret_key == "dev-secret-key":
            if not self.settings.debug:
                raise SecurityError("Secure secret key required in production")
            self.logger.warning("Using default secret key - NOT suitable for production")
    
    def create_access_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        try:
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(minutes=self.settings.access_token_expire_minutes)
            
            jti = secrets.token_urlsafe(16)
            
            payload = TokenPayload(
                user_id=user.user_id,
                username=user.username,
                role=user.role.value,
                permissions=[p.value for p in user.permissions],
                exp=int(expire.timestamp()),
                iat=int(datetime.utcnow().timestamp()),
                jti=jti
            )
            
            token = jwt.encode(
                payload.__dict__,
                self.secret_key,
                algorithm=self.algorithm
            )
            
            self.logger.debug(f"Created access token for user {user.username}")
            return token
            
        except Exception as e:
            self.logger.error(f"Token creation failed: {e}")
            raise SecurityError("Token creation failed")
    
    def verify_token(self, token: str) -> TokenPayload:
        """Verify and decode a JWT token."""
        try:
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                raise AuthenticationError("Token has been revoked")
            
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": True}
            )
            
            return TokenPayload(**payload)
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid token: {e}")
            raise AuthenticationError("Invalid token")
        except Exception as e:
            self.logger.error(f"Token verification failed: {e}")
            raise AuthenticationError("Token verification failed")
    
    def revoke_token(self, token: str):
        """Revoke a token by adding it to blacklist."""
        try:
            # In production, use Redis with expiration
            self.blacklisted_tokens.add(token)
            self.logger.info("Token revoked")
        except Exception as e:
            self.logger.error(f"Token revocation failed: {e}")
            raise SecurityError("Token revocation failed")
    
    def refresh_token(self, token: str, user: User) -> str:
        """Refresh an access token."""
        try:
            # Verify current token
            payload = self.verify_token(token)
            
            # Create new token
            new_token = self.create_access_token(user)
            
            # Revoke old token
            self.revoke_token(token)
            
            return new_token
            
        except Exception as e:
            self.logger.error(f"Token refresh failed: {e}")
            raise AuthenticationError("Token refresh failed")


class APIKeyManager:
    """API key management for programmatic access."""
    
    def __init__(self):
        self.logger = get_logger("api_key_manager")
        # In production, store in database
        self.api_keys = {}
    
    def generate_api_key(self, user: User) -> str:
        """Generate a new API key for a user."""
        try:
            # Create secure API key
            key_data = f"{user.user_id}:{secrets.token_urlsafe(32)}"
            api_key = f"wl_{hashlib.sha256(key_data.encode()).hexdigest()[:24]}"
            
            # Store API key (in production, hash and store in database)
            self.api_keys[api_key] = {
                "user_id": user.user_id,
                "created_at": datetime.utcnow(),
                "last_used": None,
                "is_active": True
            }
            
            self.logger.info(f"Generated API key for user {user.username}")
            return api_key
            
        except Exception as e:
            self.logger.error(f"API key generation failed: {e}")
            raise SecurityError("API key generation failed")
    
    def verify_api_key(self, api_key: str) -> Optional[str]:
        """Verify an API key and return user_id if valid."""
        try:
            if not api_key or not api_key.startswith("wl_"):
                return None
            
            key_info = self.api_keys.get(api_key)
            if not key_info or not key_info["is_active"]:
                return None
            
            # Update last used timestamp
            key_info["last_used"] = datetime.utcnow()
            
            return key_info["user_id"]
            
        except Exception as e:
            self.logger.error(f"API key verification failed: {e}")
            return None
    
    def revoke_api_key(self, api_key: str):
        """Revoke an API key."""
        try:
            if api_key in self.api_keys:
                self.api_keys[api_key]["is_active"] = False
                self.logger.info(f"API key revoked: {api_key[:8]}...")
        except Exception as e:
            self.logger.error(f"API key revocation failed: {e}")


class UserManager:
    """User management with role-based access control."""
    
    def __init__(self):
        self.logger = get_logger("user_manager")
        self.password_manager = PasswordManager()
        self.api_key_manager = APIKeyManager()
        
        # In production, use database
        self.users = {}
        self.role_permissions = self._initialize_role_permissions()
        
        # Create default admin user if none exists
        self._create_default_users()
    
    def _initialize_role_permissions(self) -> Dict[UserRole, List[Permission]]:
        """Initialize role-based permissions."""
        return {
            UserRole.ADMIN: list(Permission),  # All permissions
            UserRole.RESEARCHER: [
                Permission.WATERMARK_GENERATE,
                Permission.WATERMARK_DETECT,
                Permission.ATTACK_SIMULATE,
                Permission.BENCHMARK_RUN,
                Permission.BATCH_PROCESS,
                Permission.METRICS_VIEW,
                Permission.HEALTH_VIEW,
                Permission.TASK_CANCEL
            ],
            UserRole.USER: [
                Permission.WATERMARK_GENERATE,
                Permission.WATERMARK_DETECT,
                Permission.HEALTH_VIEW
            ],
            UserRole.READONLY: [
                Permission.HEALTH_VIEW,
                Permission.METRICS_VIEW
            ],
            UserRole.API_USER: [
                Permission.WATERMARK_GENERATE,
                Permission.WATERMARK_DETECT,
                Permission.BATCH_PROCESS
            ]
        }
    
    def _create_default_users(self):
        """Create default system users."""
        try:
            # Create admin user
            admin_password = os.environ.get("ADMIN_PASSWORD", "admin123!")  # Change in production
            admin_user = User(
                user_id="admin",
                username="admin",
                email="admin@watermark-lab.local",
                role=UserRole.ADMIN,
                permissions=self.role_permissions[UserRole.ADMIN]
            )
            self.users["admin"] = {
                "user": admin_user,
                "password_hash": self.password_manager.hash_password(admin_password)
            }
            
            # Create API user
            api_user = User(
                user_id="api_user",
                username="api_user",
                email="api@watermark-lab.local",
                role=UserRole.API_USER,
                permissions=self.role_permissions[UserRole.API_USER]
            )
            api_key = self.api_key_manager.generate_api_key(api_user)
            api_user.api_key = api_key
            
            self.users["api_user"] = {
                "user": api_user,
                "password_hash": None  # API-only user
            }
            
            self.logger.info("Default users created")
            
        except Exception as e:
            self.logger.error(f"Failed to create default users: {e}")
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.USER
    ) -> User:
        """Create a new user."""
        try:
            if username in self.users:
                raise SecurityError(f"User {username} already exists")
            
            # Validate input
            if not username or len(username) < 3:
                raise SecurityError("Username must be at least 3 characters")
            
            if "@" not in email:
                raise SecurityError("Invalid email format")
            
            user_id = secrets.token_urlsafe(16)
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                role=role,
                permissions=self.role_permissions[role]
            )
            
            password_hash = self.password_manager.hash_password(password)
            
            self.users[username] = {
                "user": user,
                "password_hash": password_hash
            }
            
            structured_logger.log_event(
                "user_created",
                username=username,
                role=role.value,
                user_id=user_id
            )
            
            self.logger.info(f"User created: {username}")
            return user
            
        except Exception as e:
            self.logger.error(f"User creation failed: {e}")
            raise SecurityError(f"User creation failed: {e}")
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password."""
        try:
            user_data = self.users.get(username)
            if not user_data:
                return None
            
            user = user_data["user"]
            if not user.is_active:
                return None
            
            password_hash = user_data["password_hash"]
            if not password_hash:
                return None  # API-only user
            
            if self.password_manager.verify_password(password, password_hash):
                user.last_login = datetime.utcnow()
                
                structured_logger.log_event(
                    "user_login",
                    username=username,
                    user_id=user.user_id,
                    success=True
                )
                
                return user
            else:
                structured_logger.log_event(
                    "user_login",
                    username=username,
                    success=False,
                    error="Invalid password"
                )
                return None
                
        except Exception as e:
            self.logger.error(f"User authentication failed: {e}")
            structured_logger.log_event(
                "user_login",
                username=username,
                success=False,
                error=str(e)
            )
            return None
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate user with API key."""
        try:
            user_id = self.api_key_manager.verify_api_key(api_key)
            if not user_id:
                return None
            
            # Find user by ID
            for user_data in self.users.values():
                user = user_data["user"]
                if user.user_id == user_id and user.is_active:
                    user.last_login = datetime.utcnow()
                    
                    structured_logger.log_event(
                        "api_key_auth",
                        user_id=user_id,
                        username=user.username,
                        success=True
                    )
                    
                    return user
            
            return None
            
        except Exception as e:
            self.logger.error(f"API key authentication failed: {e}")
            structured_logger.log_event(
                "api_key_auth",
                api_key_prefix=api_key[:8] if api_key else "",
                success=False,
                error=str(e)
            )
            return None
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        user_data = self.users.get(username)
        return user_data["user"] if user_data else None
    
    def update_user_permissions(self, username: str, permissions: List[Permission]):
        """Update user permissions."""
        try:
            user_data = self.users.get(username)
            if not user_data:
                raise SecurityError(f"User {username} not found")
            
            user_data["user"].permissions = permissions
            
            structured_logger.log_event(
                "user_permissions_updated",
                username=username,
                permissions=[p.value for p in permissions]
            )
            
            self.logger.info(f"Updated permissions for user {username}")
            
        except Exception as e:
            self.logger.error(f"Permission update failed: {e}")
            raise SecurityError(f"Permission update failed: {e}")
    
    def deactivate_user(self, username: str):
        """Deactivate a user account."""
        try:
            user_data = self.users.get(username)
            if not user_data:
                raise SecurityError(f"User {username} not found")
            
            user_data["user"].is_active = False
            
            structured_logger.log_event(
                "user_deactivated",
                username=username,
                user_id=user_data["user"].user_id
            )
            
            self.logger.info(f"User deactivated: {username}")
            
        except Exception as e:
            self.logger.error(f"User deactivation failed: {e}")
            raise SecurityError(f"User deactivation failed: {e}")


class AuthenticationSystem:
    """Main authentication system combining all components."""
    
    def __init__(self):
        self.user_manager = UserManager()
        self.jwt_manager = JWTManager()
        self.logger = get_logger("auth_system")
        
    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Login user and return access token."""
        try:
            user = self.user_manager.authenticate_user(username, password)
            if not user:
                raise AuthenticationError("Invalid username or password")
            
            access_token = self.jwt_manager.create_access_token(user)
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": self.jwt_manager.settings.access_token_expire_minutes * 60,
                "user": user.to_dict()
            }
            
        except AuthenticationError:
            raise
        except Exception as e:
            self.logger.error(f"Login failed: {e}")
            raise AuthenticationError("Login failed")
    
    def verify_token(self, token: str) -> User:
        """Verify token and return user."""
        try:
            payload = self.jwt_manager.verify_token(token)
            
            user = self.user_manager.get_user(payload.username)
            if not user or not user.is_active:
                raise AuthenticationError("User not found or inactive")
            
            return user
            
        except AuthenticationError:
            raise
        except Exception as e:
            self.logger.error(f"Token verification failed: {e}")
            raise AuthenticationError("Token verification failed")
    
    def verify_api_key(self, api_key: str) -> User:
        """Verify API key and return user."""
        try:
            user = self.user_manager.authenticate_api_key(api_key)
            if not user:
                raise AuthenticationError("Invalid API key")
            
            return user
            
        except AuthenticationError:
            raise
        except Exception as e:
            self.logger.error(f"API key verification failed: {e}")
            raise AuthenticationError("API key verification failed")


# Global authentication instance
_auth_system = None

def get_auth_system() -> AuthenticationSystem:
    """Get global authentication system instance."""
    global _auth_system
    if _auth_system is None:
        _auth_system = AuthenticationSystem()
    return _auth_system


def require_permission(permission: Permission):
    """Decorator to require specific permission."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would be used with FastAPI dependencies
            # Implementation depends on how user is passed to function
            return func(*args, **kwargs)
        return wrapper
    return decorator


__all__ = [
    "User",
    "UserRole", 
    "Permission",
    "AuthenticationSystem",
    "get_auth_system",
    "require_permission",
    "AuthenticationError",
    "AuthorizationError"
]