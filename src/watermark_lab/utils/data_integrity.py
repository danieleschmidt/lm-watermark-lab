"""Data integrity and validation utilities with checksums and verification."""

import hashlib
import hmac
import secrets
import json
import struct
import time
import zlib
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from .logging import get_logger, StructuredLogger
from .exceptions import ValidationError, SecurityError

logger = get_logger("data_integrity")
structured_logger = StructuredLogger("data_integrity")


class HashAlgorithm(Enum):
    """Supported hash algorithms for integrity checking."""
    SHA256 = "sha256"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"
    MD5 = "md5"  # For compatibility only, not recommended


@dataclass
class IntegrityData:
    """Data structure for integrity verification."""
    checksum: str
    algorithm: HashAlgorithm
    timestamp: float
    size: int
    metadata: Optional[Dict[str, Any]] = None
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "checksum": self.checksum,
            "algorithm": self.algorithm.value,
            "timestamp": self.timestamp,
            "size": self.size,
            "metadata": self.metadata or {},
            "signature": self.signature
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntegrityData':
        """Create from dictionary."""
        return cls(
            checksum=data["checksum"],
            algorithm=HashAlgorithm(data["algorithm"]),
            timestamp=data["timestamp"],
            size=data["size"],
            metadata=data.get("metadata"),
            signature=data.get("signature")
        )


@dataclass
class WatermarkIntegrity:
    """Watermark-specific integrity data."""
    prompt_hash: str
    output_hash: str
    config_hash: str
    method: str
    generation_time: float
    model_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DataIntegrityManager:
    """Comprehensive data integrity management system."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.logger = get_logger("data_integrity_manager")
        self.secret_key = secret_key or self._generate_key()
        
        # Supported algorithms with their implementations
        self.hash_functions = {
            HashAlgorithm.SHA256: hashlib.sha256,
            HashAlgorithm.SHA512: hashlib.sha512,
            HashAlgorithm.BLAKE2B: hashlib.blake2b,
            HashAlgorithm.MD5: hashlib.md5
        }
        
        # Algorithm preferences (most secure first)
        self.preferred_algorithms = [
            HashAlgorithm.BLAKE2B,
            HashAlgorithm.SHA512,
            HashAlgorithm.SHA256
        ]
    
    def _generate_key(self) -> str:
        """Generate a secure key for HMAC operations."""
        return secrets.token_hex(32)
    
    def compute_hash(
        self,
        data: Union[str, bytes, Dict[str, Any]],
        algorithm: HashAlgorithm = HashAlgorithm.SHA256
    ) -> str:
        """Compute hash of data using specified algorithm."""
        try:
            if algorithm not in self.hash_functions:
                raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
            # Normalize data to bytes
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, dict):
                # Deterministic JSON serialization
                json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
                data_bytes = json_str.encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                # Try to serialize other types
                data_bytes = str(data).encode('utf-8')
            
            # Compute hash
            hash_func = self.hash_functions[algorithm]()
            hash_func.update(data_bytes)
            
            return hash_func.hexdigest()
            
        except Exception as e:
            self.logger.error(f"Hash computation failed: {e}")
            raise ValidationError(f"Hash computation failed: {e}")
    
    def compute_hmac(
        self,
        data: Union[str, bytes],
        algorithm: HashAlgorithm = HashAlgorithm.SHA256
    ) -> str:
        """Compute HMAC signature for data authentication."""
        try:
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            key_bytes = self.secret_key.encode('utf-8')
            
            if algorithm == HashAlgorithm.SHA256:
                signature = hmac.new(key_bytes, data_bytes, hashlib.sha256)
            elif algorithm == HashAlgorithm.SHA512:
                signature = hmac.new(key_bytes, data_bytes, hashlib.sha512)
            elif algorithm == HashAlgorithm.BLAKE2B:
                signature = hmac.new(key_bytes, data_bytes, hashlib.blake2b)
            else:
                signature = hmac.new(key_bytes, data_bytes, hashlib.md5)
            
            return signature.hexdigest()
            
        except Exception as e:
            self.logger.error(f"HMAC computation failed: {e}")
            raise SecurityError(f"HMAC computation failed: {e}")
    
    def verify_hmac(
        self,
        data: Union[str, bytes],
        signature: str,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256
    ) -> bool:
        """Verify HMAC signature."""
        try:
            expected_signature = self.compute_hmac(data, algorithm)
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            self.logger.error(f"HMAC verification failed: {e}")
            return False
    
    def create_integrity_data(
        self,
        data: Union[str, bytes, Dict[str, Any]],
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        metadata: Optional[Dict[str, Any]] = None,
        sign: bool = True
    ) -> IntegrityData:
        """Create comprehensive integrity data for input."""
        try:
            # Normalize data for consistent processing
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
                size = len(data_bytes)
            elif isinstance(data, dict):
                json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
                data_bytes = json_str.encode('utf-8')
                size = len(data_bytes)
            elif isinstance(data, bytes):
                data_bytes = data
                size = len(data_bytes)
            else:
                str_data = str(data)
                data_bytes = str_data.encode('utf-8')
                size = len(data_bytes)
            
            # Compute checksum
            checksum = self.compute_hash(data_bytes, algorithm)
            
            # Create integrity data
            integrity_data = IntegrityData(
                checksum=checksum,
                algorithm=algorithm,
                timestamp=time.time(),
                size=size,
                metadata=metadata or {}
            )
            
            # Add signature if requested
            if sign:
                serialized = json.dumps(integrity_data.to_dict(), sort_keys=True)
                integrity_data.signature = self.compute_hmac(serialized, algorithm)
            
            structured_logger.log_event(
                "integrity_data_created",
                algorithm=algorithm.value,
                size=size,
                signed=sign
            )
            
            return integrity_data
            
        except Exception as e:
            self.logger.error(f"Integrity data creation failed: {e}")
            raise ValidationError(f"Integrity data creation failed: {e}")
    
    def verify_integrity(
        self,
        data: Union[str, bytes, Dict[str, Any]],
        integrity_data: IntegrityData,
        verify_signature: bool = True
    ) -> bool:
        """Verify data integrity against integrity data."""
        try:
            # Verify signature first if present
            if verify_signature and integrity_data.signature:
                # Create copy without signature for verification
                temp_data = IntegrityData(
                    checksum=integrity_data.checksum,
                    algorithm=integrity_data.algorithm,
                    timestamp=integrity_data.timestamp,
                    size=integrity_data.size,
                    metadata=integrity_data.metadata
                )
                
                serialized = json.dumps(temp_data.to_dict(), sort_keys=True)
                if not self.verify_hmac(serialized, integrity_data.signature, integrity_data.algorithm):
                    self.logger.warning("Integrity signature verification failed")
                    return False
            
            # Verify checksum
            current_checksum = self.compute_hash(data, integrity_data.algorithm)
            checksum_valid = hmac.compare_digest(current_checksum, integrity_data.checksum)
            
            if not checksum_valid:
                self.logger.warning("Data checksum verification failed")
                structured_logger.log_event(
                    "integrity_verification_failed",
                    expected_checksum=integrity_data.checksum,
                    actual_checksum=current_checksum,
                    algorithm=integrity_data.algorithm.value
                )
                return False
            
            # Verify size if available
            if isinstance(data, str):
                current_size = len(data.encode('utf-8'))
            elif isinstance(data, bytes):
                current_size = len(data)
            elif isinstance(data, dict):
                json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
                current_size = len(json_str.encode('utf-8'))
            else:
                current_size = len(str(data).encode('utf-8'))
            
            if current_size != integrity_data.size:
                self.logger.warning(f"Size mismatch: expected {integrity_data.size}, got {current_size}")
                return False
            
            structured_logger.log_event(
                "integrity_verification_success",
                algorithm=integrity_data.algorithm.value,
                size=integrity_data.size
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Integrity verification failed: {e}")
            return False
    
    def create_watermark_integrity(
        self,
        prompt: str,
        output: str,
        config: Dict[str, Any],
        method: str,
        generation_time: float,
        model_version: Optional[str] = None
    ) -> WatermarkIntegrity:
        """Create watermark-specific integrity data."""
        try:
            prompt_hash = self.compute_hash(prompt)
            output_hash = self.compute_hash(output)
            config_hash = self.compute_hash(config)
            
            watermark_integrity = WatermarkIntegrity(
                prompt_hash=prompt_hash,
                output_hash=output_hash,
                config_hash=config_hash,
                method=method,
                generation_time=generation_time,
                model_version=model_version
            )
            
            structured_logger.log_event(
                "watermark_integrity_created",
                method=method,
                prompt_length=len(prompt),
                output_length=len(output),
                generation_time=generation_time
            )
            
            return watermark_integrity
            
        except Exception as e:
            self.logger.error(f"Watermark integrity creation failed: {e}")
            raise ValidationError(f"Watermark integrity creation failed: {e}")
    
    def verify_watermark_integrity(
        self,
        prompt: str,
        output: str,
        config: Dict[str, Any],
        watermark_integrity: WatermarkIntegrity
    ) -> bool:
        """Verify watermark operation integrity."""
        try:
            # Verify each component
            prompt_hash = self.compute_hash(prompt)
            if not hmac.compare_digest(prompt_hash, watermark_integrity.prompt_hash):
                self.logger.warning("Prompt integrity verification failed")
                return False
            
            output_hash = self.compute_hash(output)
            if not hmac.compare_digest(output_hash, watermark_integrity.output_hash):
                self.logger.warning("Output integrity verification failed")
                return False
            
            config_hash = self.compute_hash(config)
            if not hmac.compare_digest(config_hash, watermark_integrity.config_hash):
                self.logger.warning("Configuration integrity verification failed")
                return False
            
            structured_logger.log_event(
                "watermark_integrity_verified",
                method=watermark_integrity.method,
                success=True
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Watermark integrity verification failed: {e}")
            return False
    
    def compress_and_hash(self, data: Union[str, bytes]) -> Tuple[bytes, str]:
        """Compress data and compute hash for storage efficiency."""
        try:
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Compress data
            compressed = zlib.compress(data_bytes, level=6)
            
            # Compute hash of original data
            data_hash = self.compute_hash(data_bytes)
            
            self.logger.debug(f"Compressed data: {len(data_bytes)} -> {len(compressed)} bytes")
            
            return compressed, data_hash
            
        except Exception as e:
            self.logger.error(f"Data compression failed: {e}")
            raise ValidationError(f"Data compression failed: {e}")
    
    def decompress_and_verify(
        self,
        compressed_data: bytes,
        expected_hash: str,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256
    ) -> bytes:
        """Decompress data and verify integrity."""
        try:
            # Decompress data
            decompressed = zlib.decompress(compressed_data)
            
            # Verify hash
            actual_hash = self.compute_hash(decompressed, algorithm)
            if not hmac.compare_digest(actual_hash, expected_hash):
                raise SecurityError("Decompressed data integrity verification failed")
            
            return decompressed
            
        except zlib.error as e:
            self.logger.error(f"Data decompression failed: {e}")
            raise ValidationError(f"Data decompression failed: {e}")
        except Exception as e:
            self.logger.error(f"Decompress and verify failed: {e}")
            raise SecurityError(f"Decompress and verify failed: {e}")
    
    def create_file_integrity(self, file_path: Union[str, Path]) -> IntegrityData:
        """Create integrity data for a file."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise ValidationError(f"File does not exist: {file_path}")
            
            # Read file and compute hash
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            file_stats = file_path.stat()
            
            integrity_data = self.create_integrity_data(
                file_data,
                algorithm=HashAlgorithm.SHA256,
                metadata={
                    "filename": file_path.name,
                    "file_size": file_stats.st_size,
                    "modified_time": file_stats.st_mtime,
                    "created_time": file_stats.st_ctime if hasattr(file_stats, 'st_ctime') else None
                }
            )
            
            self.logger.info(f"Created integrity data for file: {file_path}")
            return integrity_data
            
        except Exception as e:
            self.logger.error(f"File integrity creation failed: {e}")
            raise ValidationError(f"File integrity creation failed: {e}")
    
    def verify_file_integrity(
        self,
        file_path: Union[str, Path],
        integrity_data: IntegrityData
    ) -> bool:
        """Verify file integrity against stored data."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                self.logger.error(f"File does not exist: {file_path}")
                return False
            
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            return self.verify_integrity(file_data, integrity_data)
            
        except Exception as e:
            self.logger.error(f"File integrity verification failed: {e}")
            return False
    
    def batch_create_integrity(
        self,
        data_items: List[Tuple[str, Union[str, bytes, Dict[str, Any]]]],
        algorithm: HashAlgorithm = HashAlgorithm.SHA256
    ) -> Dict[str, IntegrityData]:
        """Create integrity data for multiple items efficiently."""
        try:
            results = {}
            
            for identifier, data in data_items:
                try:
                    integrity_data = self.create_integrity_data(
                        data,
                        algorithm=algorithm,
                        metadata={"batch_id": identifier}
                    )
                    results[identifier] = integrity_data
                except Exception as e:
                    self.logger.error(f"Failed to create integrity for {identifier}: {e}")
                    continue
            
            structured_logger.log_event(
                "batch_integrity_created",
                total_items=len(data_items),
                successful_items=len(results),
                algorithm=algorithm.value
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch integrity creation failed: {e}")
            raise ValidationError(f"Batch integrity creation failed: {e}")
    
    def batch_verify_integrity(
        self,
        data_items: List[Tuple[str, Union[str, bytes, Dict[str, Any]]]],
        integrity_map: Dict[str, IntegrityData]
    ) -> Dict[str, bool]:
        """Verify integrity for multiple items efficiently."""
        try:
            results = {}
            
            for identifier, data in data_items:
                if identifier not in integrity_map:
                    results[identifier] = False
                    continue
                
                try:
                    results[identifier] = self.verify_integrity(
                        data,
                        integrity_map[identifier]
                    )
                except Exception as e:
                    self.logger.error(f"Failed to verify integrity for {identifier}: {e}")
                    results[identifier] = False
            
            successful_verifications = sum(1 for v in results.values() if v)
            
            structured_logger.log_event(
                "batch_integrity_verified",
                total_items=len(data_items),
                successful_verifications=successful_verifications,
                verification_rate=successful_verifications / len(data_items) if data_items else 0
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch integrity verification failed: {e}")
            raise ValidationError(f"Batch integrity verification failed: {e}")


# Global integrity manager
_integrity_manager = None

def get_integrity_manager(secret_key: Optional[str] = None) -> DataIntegrityManager:
    """Get global integrity manager instance."""
    global _integrity_manager
    if _integrity_manager is None:
        _integrity_manager = DataIntegrityManager(secret_key)
    return _integrity_manager


# Utility functions for common operations
def quick_hash(data: Union[str, bytes, Dict[str, Any]], algorithm: str = "sha256") -> str:
    """Quick hash computation for common use cases."""
    manager = get_integrity_manager()
    hash_alg = HashAlgorithm(algorithm)
    return manager.compute_hash(data, hash_alg)


def verify_data_integrity(
    data: Union[str, bytes, Dict[str, Any]],
    expected_checksum: str,
    algorithm: str = "sha256"
) -> bool:
    """Quick integrity verification."""
    try:
        actual_checksum = quick_hash(data, algorithm)
        return hmac.compare_digest(actual_checksum, expected_checksum)
    except Exception:
        return False


__all__ = [
    "DataIntegrityManager",
    "IntegrityData", 
    "WatermarkIntegrity",
    "HashAlgorithm",
    "get_integrity_manager",
    "quick_hash",
    "verify_data_integrity"
]