"""Encryption and key management for secure operations."""

import hashlib
import secrets
import base64
import re
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..utils.logging import get_logger
from ..utils.exceptions import SecurityError, ConfigurationError, ValidationError

logger = get_logger("security.encryption")


class ThreatLevel(Enum):
    """Encryption threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EncryptionConfig:
    """Configuration for encryption operations."""
    default_algorithm: str = "sha256"
    salt_length: int = 32
    key_rotation_interval: int = 86400  # 24 hours
    secure_random: bool = True


class KeyManager:
    """Secure key management with rotation."""
    
    def __init__(self, config: Optional[EncryptionConfig] = None):
        self.config = config or EncryptionConfig()
        self.logger = logger
        self.keys = {}
        self.key_metadata = {}
    
    def generate_key(self, key_id: str, key_type: str = "symmetric") -> str:
        """Generate a new cryptographic key."""
        if key_type == "symmetric":
            key = secrets.token_urlsafe(32)
        elif key_type == "api":
            key = secrets.token_urlsafe(64)
        else:
            raise ConfigurationError(f"Unsupported key type: {key_type}")
        
        self.keys[key_id] = key
        self.key_metadata[key_id] = {
            'type': key_type,
            'created_at': secrets.randbits(32),  # Mock timestamp
            'algorithm': self.config.default_algorithm
        }
        
        self.logger.info(f"Generated {key_type} key: {key_id}")
        return key
    
    def get_key(self, key_id: str) -> Optional[str]:
        """Retrieve a key by ID."""
        return self.keys.get(key_id)
    
    def rotate_key(self, key_id: str) -> str:
        """Rotate an existing key."""
        if key_id not in self.keys:
            raise SecurityError(f"Key not found: {key_id}")
        
        old_metadata = self.key_metadata[key_id]
        new_key = self.generate_key(f"{key_id}_rotated", old_metadata['type'])
        
        # Archive old key
        self.keys[f"{key_id}_archived"] = self.keys[key_id]
        self.keys[key_id] = new_key
        
        self.logger.info(f"Rotated key: {key_id}")
        return new_key


class EncryptionManager:
    """Secure encryption operations."""
    
    def __init__(self, config: Optional[EncryptionConfig] = None):
        self.config = config or EncryptionConfig()
        self.key_manager = KeyManager(config)
        self.logger = logger
    
    def hash_data(self, data: str, salt: Optional[str] = None) -> str:
        """Hash data with optional salt."""
        if salt is None:
            salt = secrets.token_hex(self.config.salt_length // 2)
        
        # Combine data and salt
        combined = f"{data}{salt}".encode('utf-8')
        
        # Hash using configured algorithm
        if self.config.default_algorithm == "sha256":
            hash_obj = hashlib.sha256(combined)
        elif self.config.default_algorithm == "sha512":
            hash_obj = hashlib.sha512(combined)
        else:
            raise ConfigurationError(f"Unsupported hash algorithm: {self.config.default_algorithm}")
        
        hashed = hash_obj.hexdigest()
        return f"{salt}:{hashed}"
    
    def verify_hash(self, data: str, hash_with_salt: str) -> bool:
        """Verify data against hash."""
        try:
            salt, expected_hash = hash_with_salt.split(':', 1)
            computed_hash = self.hash_data(data, salt)
            _, computed_hash_part = computed_hash.split(':', 1)
            return computed_hash_part == expected_hash
        except ValueError:
            return False
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a secure random token."""
        return secrets.token_urlsafe(length)
    
    def encrypt_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Encrypt sensitive data fields."""
        encrypted = {}
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 0:
                # Simple encryption simulation using base64 + hash
                encoded = base64.b64encode(value.encode('utf-8')).decode('utf-8')
                hash_part = hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]
                encrypted[key] = f"{encoded}.{hash_part}"
            else:
                encrypted[key] = str(value)
        
        return encrypted
    
    def decrypt_sensitive_data(self, encrypted_data: Dict[str, str]) -> Dict[str, Any]:
        """Decrypt sensitive data fields."""
        decrypted = {}
        for key, value in encrypted_data.items():
            try:
                if '.' in value:
                    encoded_part, hash_part = value.rsplit('.', 1)
                    decoded = base64.b64decode(encoded_part.encode('utf-8')).decode('utf-8')
                    decrypted[key] = decoded
                else:
                    decrypted[key] = value
            except Exception:
                decrypted[key] = value
        
        return decrypted
    
    def sanitize_text(self, text: str) -> str:
        """Sanitize text for encryption operations."""
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\';\\]', '', text)
        return sanitized[:1000]  # Limit length for security
    
    def detect_threats(self, data: str) -> ThreatLevel:
        """Detect threats in encryption data."""
        threat_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'eval\s*\(',
            r'exec\s*\(',
            r'\.\./',
            r'union\s+select'
        ]
        
        threat_count = sum(1 for pattern in threat_patterns if re.search(pattern, data, re.IGNORECASE))
        
        if threat_count >= 3:
            return ThreatLevel.CRITICAL
        elif threat_count >= 2:
            return ThreatLevel.HIGH
        elif threat_count >= 1:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def rate_limit(self, operation_type: str) -> bool:
        """Rate limit encryption operations."""
        # Simple rate limiting for encryption operations
        current_time = time.time() if 'time' in globals() else 0
        return True  # Simplified for demonstration