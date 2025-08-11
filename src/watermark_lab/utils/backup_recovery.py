"""Comprehensive backup and recovery system for critical data and configurations."""

import os
import json
import shutil
import tarfile
import gzip
import hashlib
import threading
import time
import schedule
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

try:
    import paramiko
    SFTP_AVAILABLE = True
except ImportError:
    SFTP_AVAILABLE = False

from .logging import get_logger, StructuredLogger
from .exceptions import ValidationError, SecurityError, ResourceError
from .data_integrity import get_integrity_manager, IntegrityData
from .encryption import EncryptionManager
from ..config.settings import get_settings

logger = get_logger("backup_recovery")
structured_logger = StructuredLogger("backup")
settings = get_settings()


class BackupType(Enum):
    """Types of backups."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    CONFIGURATION = "configuration"
    MODELS = "models"
    DATA = "data"
    LOGS = "logs"


class BackupStatus(Enum):
    """Backup operation status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StorageBackend(Enum):
    """Storage backend types."""
    LOCAL = "local"
    S3 = "s3"
    SFTP = "sftp"
    NFS = "nfs"


@dataclass
class BackupConfig:
    """Configuration for backup operations."""
    name: str
    backup_type: BackupType
    source_paths: List[str]
    destination: str
    storage_backend: StorageBackend = StorageBackend.LOCAL
    
    # Retention settings
    max_backups: int = 7
    max_age_days: int = 30
    
    # Compression and encryption
    compress: bool = True
    encrypt: bool = True
    encryption_key: Optional[str] = None
    
    # Scheduling
    schedule_enabled: bool = False
    schedule_cron: Optional[str] = None
    schedule_interval_hours: Optional[int] = None
    
    # Verification
    verify_integrity: bool = True
    verify_after_backup: bool = True
    
    # Storage backend specific settings
    s3_bucket: Optional[str] = None
    s3_region: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    
    sftp_host: Optional[str] = None
    sftp_username: Optional[str] = None
    sftp_password: Optional[str] = None
    sftp_private_key_path: Optional[str] = None
    
    # Exclusion patterns
    exclude_patterns: List[str] = None
    
    def __post_init__(self):
        if self.exclude_patterns is None:
            self.exclude_patterns = [
                "*.tmp",
                "*.log",
                "*.pyc",
                "__pycache__",
                ".git",
                ".pytest_cache"
            ]


@dataclass
class BackupRecord:
    """Record of a backup operation."""
    id: str
    config_name: str
    backup_type: BackupType
    status: BackupStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # File information
    source_size_bytes: int = 0
    compressed_size_bytes: int = 0
    file_count: int = 0
    backup_path: Optional[str] = None
    
    # Verification
    checksum: Optional[str] = None
    integrity_verified: bool = False
    
    # Error information
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    # Storage information
    storage_backend: StorageBackend = StorageBackend.LOCAL
    remote_path: Optional[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['backup_type'] = self.backup_type.value
        result['status'] = self.status.value
        result['storage_backend'] = self.storage_backend.value
        result['start_time'] = self.start_time.isoformat() if self.start_time else None
        result['end_time'] = self.end_time.isoformat() if self.end_time else None
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackupRecord':
        """Create from dictionary."""
        # Convert enum strings back to enums
        data['backup_type'] = BackupType(data['backup_type'])
        data['status'] = BackupStatus(data['status'])
        data['storage_backend'] = StorageBackend(data['storage_backend'])
        
        # Convert datetime strings back to datetime objects
        if data.get('start_time'):
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        
        return cls(**data)


class StorageBackendInterface:
    """Interface for storage backends."""
    
    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload file to remote storage."""
        raise NotImplementedError
    
    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download file from remote storage."""
        raise NotImplementedError
    
    def delete_file(self, remote_path: str) -> bool:
        """Delete file from remote storage."""
        raise NotImplementedError
    
    def list_files(self, remote_path: str) -> List[str]:
        """List files in remote directory."""
        raise NotImplementedError
    
    def file_exists(self, remote_path: str) -> bool:
        """Check if file exists in remote storage."""
        raise NotImplementedError


class LocalStorageBackend(StorageBackendInterface):
    """Local filesystem storage backend."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("local_storage")
    
    def upload_file(self, local_path: str, remote_path: str) -> bool:
        try:
            full_remote_path = self.base_path / remote_path
            full_remote_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, full_remote_path)
            return True
        except Exception as e:
            self.logger.error(f"Local upload failed: {e}")
            return False
    
    def download_file(self, remote_path: str, local_path: str) -> bool:
        try:
            full_remote_path = self.base_path / remote_path
            shutil.copy2(full_remote_path, local_path)
            return True
        except Exception as e:
            self.logger.error(f"Local download failed: {e}")
            return False
    
    def delete_file(self, remote_path: str) -> bool:
        try:
            full_remote_path = self.base_path / remote_path
            full_remote_path.unlink(missing_ok=True)
            return True
        except Exception as e:
            self.logger.error(f"Local delete failed: {e}")
            return False
    
    def list_files(self, remote_path: str) -> List[str]:
        try:
            full_remote_path = self.base_path / remote_path
            if full_remote_path.is_dir():
                return [str(p.relative_to(self.base_path)) for p in full_remote_path.rglob('*') if p.is_file()]
            return []
        except Exception as e:
            self.logger.error(f"Local list failed: {e}")
            return []
    
    def file_exists(self, remote_path: str) -> bool:
        full_remote_path = self.base_path / remote_path
        return full_remote_path.exists()


class S3StorageBackend(StorageBackendInterface):
    """Amazon S3 storage backend."""
    
    def __init__(self, bucket: str, region: str = "us-east-1", 
                 access_key: Optional[str] = None, secret_key: Optional[str] = None):
        if not S3_AVAILABLE:
            raise ResourceError("boto3 library not available for S3 storage")
        
        self.bucket = bucket
        self.region = region
        self.logger = get_logger("s3_storage")
        
        # Initialize S3 client
        session_kwargs = {'region_name': region}
        if access_key and secret_key:
            session_kwargs.update({
                'aws_access_key_id': access_key,
                'aws_secret_access_key': secret_key
            })
        
        self.s3_client = boto3.client('s3', **session_kwargs)
        
        # Verify bucket access
        try:
            self.s3_client.head_bucket(Bucket=bucket)
        except ClientError as e:
            raise ResourceError(f"Cannot access S3 bucket {bucket}: {e}")
    
    def upload_file(self, local_path: str, remote_path: str) -> bool:
        try:
            self.s3_client.upload_file(local_path, self.bucket, remote_path)
            self.logger.info(f"Uploaded to S3: {remote_path}")
            return True
        except ClientError as e:
            self.logger.error(f"S3 upload failed: {e}")
            return False
    
    def download_file(self, remote_path: str, local_path: str) -> bool:
        try:
            self.s3_client.download_file(self.bucket, remote_path, local_path)
            self.logger.info(f"Downloaded from S3: {remote_path}")
            return True
        except ClientError as e:
            self.logger.error(f"S3 download failed: {e}")
            return False
    
    def delete_file(self, remote_path: str) -> bool:
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=remote_path)
            self.logger.info(f"Deleted from S3: {remote_path}")
            return True
        except ClientError as e:
            self.logger.error(f"S3 delete failed: {e}")
            return False
    
    def list_files(self, remote_path: str) -> List[str]:
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=remote_path
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except ClientError as e:
            self.logger.error(f"S3 list failed: {e}")
            return []
    
    def file_exists(self, remote_path: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=remote_path)
            return True
        except ClientError:
            return False


class SFTPStorageBackend(StorageBackendInterface):
    """SFTP storage backend."""
    
    def __init__(self, host: str, username: str, 
                 password: Optional[str] = None, private_key_path: Optional[str] = None):
        if not SFTP_AVAILABLE:
            raise ResourceError("paramiko library not available for SFTP storage")
        
        self.host = host
        self.username = username
        self.password = password
        self.private_key_path = private_key_path
        self.logger = get_logger("sftp_storage")
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test SFTP connection."""
        try:
            with self._get_sftp_client() as sftp:
                sftp.listdir('.')
        except Exception as e:
            raise ResourceError(f"SFTP connection test failed: {e}")
    
    def _get_sftp_client(self):
        """Get SFTP client with proper authentication."""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        if self.private_key_path:
            ssh.connect(self.host, username=self.username, key_filename=self.private_key_path)
        else:
            ssh.connect(self.host, username=self.username, password=self.password)
        
        return ssh.open_sftp()
    
    def upload_file(self, local_path: str, remote_path: str) -> bool:
        try:
            with self._get_sftp_client() as sftp:
                # Create remote directory if needed
                remote_dir = os.path.dirname(remote_path)
                if remote_dir:
                    self._create_remote_directory(sftp, remote_dir)
                
                sftp.put(local_path, remote_path)
                self.logger.info(f"Uploaded via SFTP: {remote_path}")
                return True
        except Exception as e:
            self.logger.error(f"SFTP upload failed: {e}")
            return False
    
    def download_file(self, remote_path: str, local_path: str) -> bool:
        try:
            with self._get_sftp_client() as sftp:
                # Create local directory if needed
                local_dir = os.path.dirname(local_path)
                if local_dir:
                    os.makedirs(local_dir, exist_ok=True)
                
                sftp.get(remote_path, local_path)
                self.logger.info(f"Downloaded via SFTP: {remote_path}")
                return True
        except Exception as e:
            self.logger.error(f"SFTP download failed: {e}")
            return False
    
    def delete_file(self, remote_path: str) -> bool:
        try:
            with self._get_sftp_client() as sftp:
                sftp.remove(remote_path)
                self.logger.info(f"Deleted via SFTP: {remote_path}")
                return True
        except Exception as e:
            self.logger.error(f"SFTP delete failed: {e}")
            return False
    
    def list_files(self, remote_path: str) -> List[str]:
        try:
            with self._get_sftp_client() as sftp:
                return self._list_recursive(sftp, remote_path)
        except Exception as e:
            self.logger.error(f"SFTP list failed: {e}")
            return []
    
    def file_exists(self, remote_path: str) -> bool:
        try:
            with self._get_sftp_client() as sftp:
                sftp.stat(remote_path)
                return True
        except:
            return False
    
    def _create_remote_directory(self, sftp, path: str):
        """Create remote directory recursively."""
        parts = path.split('/')
        current_path = ""
        
        for part in parts:
            if not part:
                continue
            
            current_path = f"{current_path}/{part}" if current_path else part
            
            try:
                sftp.mkdir(current_path)
            except:
                pass  # Directory might already exist
    
    def _list_recursive(self, sftp, path: str) -> List[str]:
        """List files recursively in SFTP directory."""
        files = []
        
        try:
            for item in sftp.listdir_attr(path):
                item_path = f"{path}/{item.filename}"
                
                if item.st_mode is not None and item.st_mode & 0o170000 == 0o040000:  # Directory
                    files.extend(self._list_recursive(sftp, item_path))
                else:  # File
                    files.append(item_path)
        except:
            pass
        
        return files


class BackupManager:
    """Comprehensive backup and recovery manager."""
    
    def __init__(self, backup_dir: Optional[str] = None):
        self.backup_dir = Path(backup_dir or settings.data_dir) / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("backup_manager")
        self.integrity_manager = get_integrity_manager()
        self.encryption_manager = EncryptionManager()
        
        # Backup configurations
        self.configs: Dict[str, BackupConfig] = {}
        self.records: Dict[str, BackupRecord] = {}
        
        # Storage backends
        self.storage_backends: Dict[str, StorageBackendInterface] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Scheduler
        self.scheduler_running = False
        self.scheduler_thread = None
        
        # Load configurations and records
        self._load_configurations()
        self._load_backup_records()
        
        # Initialize default storage backends
        self._initialize_storage_backends()
        
        # Initialize default configurations
        self._initialize_default_configs()
    
    def _load_configurations(self):
        """Load backup configurations from disk."""
        config_file = self.backup_dir / "backup_configs.json"
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    configs_data = json.load(f)
                
                for name, config_dict in configs_data.items():
                    # Convert strings back to enums
                    config_dict['backup_type'] = BackupType(config_dict['backup_type'])
                    config_dict['storage_backend'] = StorageBackend(config_dict['storage_backend'])
                    
                    self.configs[name] = BackupConfig(**config_dict)
                
                self.logger.info(f"Loaded {len(self.configs)} backup configurations")
                
        except Exception as e:
            self.logger.error(f"Failed to load backup configurations: {e}")
    
    def _save_configurations(self):
        """Save backup configurations to disk."""
        config_file = self.backup_dir / "backup_configs.json"
        
        try:
            with self.lock:
                configs_data = {}
                for name, config in self.configs.items():
                    config_dict = asdict(config)
                    config_dict['backup_type'] = config.backup_type.value
                    config_dict['storage_backend'] = config.storage_backend.value
                    configs_data[name] = config_dict
                
                with open(config_file, 'w') as f:
                    json.dump(configs_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save backup configurations: {e}")
    
    def _load_backup_records(self):
        """Load backup records from disk."""
        records_file = self.backup_dir / "backup_records.json"
        
        try:
            if records_file.exists():
                with open(records_file, 'r') as f:
                    records_data = json.load(f)
                
                for record_id, record_dict in records_data.items():
                    self.records[record_id] = BackupRecord.from_dict(record_dict)
                
                self.logger.info(f"Loaded {len(self.records)} backup records")
                
        except Exception as e:
            self.logger.error(f"Failed to load backup records: {e}")
    
    def _save_backup_records(self):
        """Save backup records to disk."""
        records_file = self.backup_dir / "backup_records.json"
        
        try:
            with self.lock:
                records_data = {}
                for record_id, record in self.records.items():
                    records_data[record_id] = record.to_dict()
                
                with open(records_file, 'w') as f:
                    json.dump(records_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save backup records: {e}")
    
    def _initialize_storage_backends(self):
        """Initialize storage backends."""
        # Local storage
        self.storage_backends['local'] = LocalStorageBackend(str(self.backup_dir / "storage"))
        
        # S3 storage (if configured)
        if S3_AVAILABLE and hasattr(settings, 's3_backup_bucket') and settings.s3_backup_bucket:
            try:
                self.storage_backends['s3'] = S3StorageBackend(
                    bucket=settings.s3_backup_bucket,
                    region=getattr(settings, 's3_backup_region', 'us-east-1'),
                    access_key=getattr(settings, 's3_access_key', None),
                    secret_key=getattr(settings, 's3_secret_key', None)
                )
                self.logger.info("S3 storage backend initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize S3 storage: {e}")
    
    def _initialize_default_configs(self):
        """Initialize default backup configurations."""
        if not self.configs:
            # Configuration backup
            self.add_backup_config(BackupConfig(
                name="configurations",
                backup_type=BackupType.CONFIGURATION,
                source_paths=[
                    str(settings.data_dir),
                    str(Path.cwd() / "*.yml"),
                    str(Path.cwd() / "*.yaml"),
                    str(Path.cwd() / "*.json"),
                    str(Path.cwd() / "*.toml")
                ],
                destination="configs",
                schedule_enabled=True,
                schedule_interval_hours=24,
                max_backups=7
            ))
            
            # Model backup
            if hasattr(settings, 'default_model_cache'):
                self.add_backup_config(BackupConfig(
                    name="models",
                    backup_type=BackupType.MODELS,
                    source_paths=[settings.default_model_cache],
                    destination="models",
                    schedule_enabled=True,
                    schedule_interval_hours=168,  # Weekly
                    max_backups=3,
                    compress=True
                ))
            
            # Data backup
            self.add_backup_config(BackupConfig(
                name="data",
                backup_type=BackupType.DATA,
                source_paths=[settings.data_dir],
                destination="data",
                schedule_enabled=True,
                schedule_interval_hours=24,
                max_backups=7
            ))
            
            self.logger.info("Initialized default backup configurations")
    
    def add_backup_config(self, config: BackupConfig) -> bool:
        """Add a backup configuration."""
        try:
            with self.lock:
                self.configs[config.name] = config
                self._save_configurations()
                
                structured_logger.log_event(
                    "backup_config_added",
                    config_name=config.name,
                    backup_type=config.backup_type.value,
                    storage_backend=config.storage_backend.value
                )
                
                self.logger.info(f"Added backup configuration: {config.name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to add backup configuration: {e}")
            return False
    
    def remove_backup_config(self, name: str) -> bool:
        """Remove a backup configuration."""
        try:
            with self.lock:
                if name in self.configs:
                    del self.configs[name]
                    self._save_configurations()
                    
                    structured_logger.log_event(
                        "backup_config_removed",
                        config_name=name
                    )
                    
                    self.logger.info(f"Removed backup configuration: {name}")
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to remove backup configuration: {e}")
            return False
    
    def create_backup(self, config_name: str, backup_type: Optional[BackupType] = None) -> Optional[BackupRecord]:
        """Create a backup using the specified configuration."""
        try:
            if config_name not in self.configs:
                raise ValidationError(f"Backup configuration not found: {config_name}")
            
            config = self.configs[config_name]
            
            # Override backup type if specified
            if backup_type:
                config.backup_type = backup_type
            
            # Generate backup ID
            backup_id = f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
            
            # Create backup record
            record = BackupRecord(
                id=backup_id,
                config_name=config_name,
                backup_type=config.backup_type,
                status=BackupStatus.RUNNING,
                start_time=datetime.now(),
                storage_backend=config.storage_backend
            )\n            \n            # Store record\n            with self.lock:\n                self.records[backup_id] = record\n                self._save_backup_records()\n            \n            structured_logger.log_event(\n                "backup_started",\n                backup_id=backup_id,\n                config_name=config_name,\n                backup_type=config.backup_type.value\n            )\n            \n            self.logger.info(f"Starting backup: {backup_id}")\n            \n            try:\n                # Perform backup\n                success = self._perform_backup(config, record)\n                \n                if success:\n                    record.status = BackupStatus.COMPLETED\n                    record.end_time = datetime.now()\n                    \n                    # Verify backup if configured\n                    if config.verify_after_backup:\n                        record.integrity_verified = self._verify_backup(record)\n                    \n                    structured_logger.log_event(\n                        "backup_completed",\n                        backup_id=backup_id,\n                        duration=(record.end_time - record.start_time).total_seconds(),\n                        source_size=record.source_size_bytes,\n                        compressed_size=record.compressed_size_bytes,\n                        file_count=record.file_count\n                    )\n                    \n                    self.logger.info(f"Backup completed successfully: {backup_id}")\n                    \n                else:\n                    record.status = BackupStatus.FAILED\n                    record.end_time = datetime.now()\n                    \n                    structured_logger.log_event(\n                        "backup_failed",\n                        backup_id=backup_id,\n                        error=record.error_message\n                    )\n                    \n                    self.logger.error(f"Backup failed: {backup_id}")\n                \n            except Exception as e:\n                record.status = BackupStatus.FAILED\n                record.end_time = datetime.now()\n                record.error_message = str(e)\n                \n                structured_logger.log_event(\n                    "backup_failed",\n                    backup_id=backup_id,\n                    error=str(e)\n                )\n                \n                self.logger.error(f"Backup failed with exception: {backup_id} - {e}")\n            \n            finally:\n                # Update record\n                with self.lock:\n                    self.records[backup_id] = record\n                    self._save_backup_records()\n            \n            # Clean up old backups\n            self._cleanup_old_backups(config)\n            \n            return record\n            \n        except Exception as e:\n            self.logger.error(f"Failed to create backup {config_name}: {e}")\n            return None\n    \n    def _perform_backup(self, config: BackupConfig, record: BackupRecord) -> bool:\n        """Perform the actual backup operation."""\n        try:\n            # Create temporary directory for backup preparation\n            temp_dir = self.backup_dir / "temp" / record.id\n            temp_dir.mkdir(parents=True, exist_ok=True)\n            \n            try:\n                # Collect files to backup\n                files_to_backup = self._collect_backup_files(config)\n                record.file_count = len(files_to_backup)\n                \n                if not files_to_backup:\n                    record.error_message = "No files found to backup"\n                    return False\n                \n                # Calculate source size\n                record.source_size_bytes = sum(os.path.getsize(f) for f in files_to_backup if os.path.exists(f))\n                \n                # Create backup archive\n                backup_filename = f"{record.id}.tar.gz" if config.compress else f"{record.id}.tar"\n                backup_path = temp_dir / backup_filename\n                \n                if config.compress:\n                    tar_mode = 'w:gz'\n                else:\n                    tar_mode = 'w'\n                \n                with tarfile.open(backup_path, tar_mode) as tar:\n                    for file_path in files_to_backup:\n                        if os.path.exists(file_path):\n                            # Add file with relative path\n                            arcname = os.path.relpath(file_path)\n                            tar.add(file_path, arcname=arcname)\n                \n                record.compressed_size_bytes = os.path.getsize(backup_path)\n                \n                # Encrypt if configured\n                if config.encrypt:\n                    encrypted_path = backup_path.with_suffix(backup_path.suffix + '.enc')\n                    \n                    encryption_key = config.encryption_key or self.encryption_manager.generate_key()\n                    \n                    if self.encryption_manager.encrypt_file(str(backup_path), str(encrypted_path), encryption_key):\n                        backup_path.unlink()  # Remove unencrypted file\n                        backup_path = encrypted_path\n                        \n                        # Store encryption key securely (in production, use proper key management)\n                        key_file = temp_dir / f"{record.id}.key"\n                        with open(key_file, 'w') as f:\n                            f.write(encryption_key)\n                \n                # Generate checksum\n                record.checksum = self.integrity_manager.compute_hash(\n                    backup_path.read_bytes()\n                )\n                \n                # Upload to storage backend\n                storage_backend = self.storage_backends.get(config.storage_backend.value)\n                if storage_backend:\n                    remote_path = f"{config.destination}/{record.id}/{backup_filename}"\n                    \n                    if storage_backend.upload_file(str(backup_path), remote_path):\n                        record.backup_path = str(backup_path)\n                        record.remote_path = remote_path\n                        return True\n                    else:\n                        record.error_message = "Failed to upload backup to storage backend"\n                        return False\n                else:\n                    record.error_message = f"Storage backend not available: {config.storage_backend.value}"\n                    return False\n                \n            finally:\n                # Cleanup temporary directory\n                shutil.rmtree(temp_dir, ignore_errors=True)\n                \n        except Exception as e:\n            record.error_message = str(e)\n            return False\n    \n    def _collect_backup_files(self, config: BackupConfig) -> List[str]:\n        """Collect files to be included in backup."""\n        files = []\n        \n        try:\n            for source_path in config.source_paths:\n                path = Path(source_path)\n                \n                if path.is_file():\n                    if not self._is_excluded(str(path), config.exclude_patterns):\n                        files.append(str(path))\n                        \n                elif path.is_dir():\n                    for file_path in path.rglob('*'):\n                        if (file_path.is_file() and \n                            not self._is_excluded(str(file_path), config.exclude_patterns)):\n                            files.append(str(file_path))\n                            \n                else:\n                    # Handle glob patterns\n                    import glob\n                    glob_files = glob.glob(source_path, recursive=True)\n                    for file_path in glob_files:\n                        if (os.path.isfile(file_path) and \n                            not self._is_excluded(file_path, config.exclude_patterns)):\n                            files.append(file_path)\n            \n            return files\n            \n        except Exception as e:\n            self.logger.error(f"Failed to collect backup files: {e}")\n            return []\n    \n    def _is_excluded(self, file_path: str, exclude_patterns: List[str]) -> bool:\n        """Check if file should be excluded from backup."""\n        import fnmatch\n        \n        for pattern in exclude_patterns:\n            if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(os.path.basename(file_path), pattern):\n                return True\n        \n        return False\n    \n    def _verify_backup(self, record: BackupRecord) -> bool:\n        """Verify backup integrity."""\n        try:\n            if not record.remote_path or not record.checksum:\n                return False\n            \n            config = self.configs.get(record.config_name)\n            if not config:\n                return False\n            \n            storage_backend = self.storage_backends.get(config.storage_backend.value)\n            if not storage_backend:\n                return False\n            \n            # Download backup for verification\n            temp_file = self.backup_dir / "temp" / f"verify_{record.id}"\n            temp_file.parent.mkdir(parents=True, exist_ok=True)\n            \n            try:\n                if storage_backend.download_file(record.remote_path, str(temp_file)):\n                    # Verify checksum\n                    actual_checksum = self.integrity_manager.compute_hash(\n                        temp_file.read_bytes()\n                    )\n                    \n                    verified = actual_checksum == record.checksum\n                    \n                    if verified:\n                        self.logger.info(f"Backup verification successful: {record.id}")\n                    else:\n                        self.logger.error(f"Backup verification failed: {record.id} - checksum mismatch")\n                    \n                    return verified\n                    \n            finally:\n                temp_file.unlink(missing_ok=True)\n            \n            return False\n            \n        except Exception as e:\n            self.logger.error(f"Backup verification failed: {e}")\n            return False\n    \n    def _cleanup_old_backups(self, config: BackupConfig):\n        \"\"\"Clean up old backups based on retention policy.\"\"\"\n        try:\n            # Get all backup records for this config\n            config_records = [\n                record for record in self.records.values() \n                if record.config_name == config.name and record.status == BackupStatus.COMPLETED\n            ]\n            \n            # Sort by start time (newest first)\n            config_records.sort(key=lambda r: r.start_time, reverse=True)\n            \n            # Apply retention policies\n            records_to_delete = []\n            \n            # Max backups limit\n            if len(config_records) > config.max_backups:\n                records_to_delete.extend(config_records[config.max_backups:])\n            \n            # Max age limit\n            cutoff_date = datetime.now() - timedelta(days=config.max_age_days)\n            for record in config_records:\n                if record.start_time < cutoff_date and record not in records_to_delete:\n                    records_to_delete.append(record)\n            \n            # Delete old backups\n            storage_backend = self.storage_backends.get(config.storage_backend.value)\n            \n            for record in records_to_delete:\n                try:\n                    if storage_backend and record.remote_path:\n                        storage_backend.delete_file(record.remote_path)\n                    \n                    # Remove from records\n                    with self.lock:\n                        if record.id in self.records:\n                            del self.records[record.id]\n                    \n                    self.logger.info(f"Deleted old backup: {record.id}")\n                    \n                except Exception as e:\n                    self.logger.error(f"Failed to delete old backup {record.id}: {e}")\n            \n            if records_to_delete:\n                self._save_backup_records()\n                \n                structured_logger.log_event(\n                    "backup_cleanup",\n                    config_name=config.name,\n                    deleted_count=len(records_to_delete)\n                )\n                \n        except Exception as e:\n            self.logger.error(f"Backup cleanup failed: {e}")\n    \n    def restore_backup(self, backup_id: str, restore_path: Optional[str] = None) -> bool:\n        \"\"\"Restore from a backup.\"\"\"\n        try:\n            if backup_id not in self.records:\n                raise ValidationError(f"Backup record not found: {backup_id}")\n            \n            record = self.records[backup_id]\n            \n            if record.status != BackupStatus.COMPLETED:\n                raise ValidationError(f"Cannot restore from incomplete backup: {backup_id}")\n            \n            config = self.configs.get(record.config_name)\n            if not config:\n                raise ValidationError(f"Backup configuration not found: {record.config_name}")\n            \n            storage_backend = self.storage_backends.get(config.storage_backend.value)\n            if not storage_backend:\n                raise ResourceError(f"Storage backend not available: {config.storage_backend.value}")\n            \n            # Create restore directory\n            if restore_path:\n                restore_dir = Path(restore_path)\n            else:\n                restore_dir = self.backup_dir / "restore" / backup_id\n            \n            restore_dir.mkdir(parents=True, exist_ok=True)\n            \n            # Download backup\n            temp_backup = restore_dir / "backup.tar.gz"\n            \n            if not storage_backend.download_file(record.remote_path, str(temp_backup)):\n                raise ResourceError("Failed to download backup from storage")\n            \n            try:\n                # Verify integrity\n                if record.checksum:\n                    actual_checksum = self.integrity_manager.compute_hash(\n                        temp_backup.read_bytes()\n                    )\n                    \n                    if actual_checksum != record.checksum:\n                        raise SecurityError("Backup integrity verification failed")\n                \n                # Decrypt if needed\n                backup_file = temp_backup\n                if config.encrypt:\n                    # In production, retrieve encryption key from secure storage\n                    decrypted_file = restore_dir / "backup_decrypted.tar.gz"\n                    \n                    # For now, assume key is stored with backup (not secure)\n                    key_path = restore_dir / f"{backup_id}.key"\n                    if storage_backend.download_file(\n                        f"{config.destination}/{backup_id}/{backup_id}.key",\n                        str(key_path)\n                    ):\n                        with open(key_path, 'r') as f:\n                            encryption_key = f.read()\n                        \n                        if self.encryption_manager.decrypt_file(\n                            str(backup_file), str(decrypted_file), encryption_key\n                        ):\n                            backup_file = decrypted_file\n                        else:\n                            raise SecurityError("Failed to decrypt backup")\n                    else:\n                        raise SecurityError("Encryption key not found")\n                \n                # Extract backup\n                with tarfile.open(backup_file, 'r:*') as tar:\n                    tar.extractall(restore_dir)\n                \n                structured_logger.log_event(\n                    "backup_restored",\n                    backup_id=backup_id,\n                    restore_path=str(restore_dir)\n                )\n                \n                self.logger.info(f"Backup restored successfully: {backup_id} to {restore_dir}")\n                return True\n                \n            finally:\n                # Cleanup temporary files\n                temp_backup.unlink(missing_ok=True)\n                \n        except Exception as e:\n            self.logger.error(f"Backup restoration failed: {e}")\n            return False\n    \n    def start_scheduler(self):\n        \"\"\"Start backup scheduler.\"\"\"\n        if self.scheduler_running:\n            return\n        \n        self.scheduler_running = True\n        \n        def scheduler_worker():\n            while self.scheduler_running:\n                try:\n                    schedule.run_pending()\n                    time.sleep(60)  # Check every minute\n                except Exception as e:\n                    self.logger.error(f"Scheduler error: {e}")\n        \n        # Schedule backups\n        for config in self.configs.values():\n            if config.schedule_enabled:\n                if config.schedule_interval_hours:\n                    schedule.every(config.schedule_interval_hours).hours.do(\n                        self._scheduled_backup, config.name\n                    )\n                elif config.schedule_cron:\n                    # TODO: Implement cron-style scheduling\n                    pass\n        \n        self.scheduler_thread = threading.Thread(target=scheduler_worker, daemon=True)\n        self.scheduler_thread.start()\n        \n        self.logger.info("Backup scheduler started")\n    \n    def stop_scheduler(self):\n        \"\"\"Stop backup scheduler.\"\"\"\n        self.scheduler_running = False\n        \n        if self.scheduler_thread:\n            self.scheduler_thread.join(timeout=5)\n        \n        schedule.clear()\n        self.logger.info("Backup scheduler stopped")\n    \n    def _scheduled_backup(self, config_name: str):\n        \"\"\"Perform scheduled backup.\"\"\"\n        try:\n            structured_logger.log_event(\n                "scheduled_backup_triggered",\n                config_name=config_name\n            )\n            \n            self.create_backup(config_name)\n            \n        except Exception as e:\n            self.logger.error(f"Scheduled backup failed for {config_name}: {e}")\n    \n    def get_backup_status(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive backup system status.\"\"\"\n        with self.lock:\n            total_backups = len(self.records)\n            completed_backups = sum(1 for r in self.records.values() if r.status == BackupStatus.COMPLETED)\n            failed_backups = sum(1 for r in self.records.values() if r.status == BackupStatus.FAILED)\n            \n            # Calculate total storage used\n            total_storage_bytes = sum(\n                r.compressed_size_bytes for r in self.records.values() \n                if r.status == BackupStatus.COMPLETED and r.compressed_size_bytes\n            )\n            \n            # Recent backups (last 7 days)\n            recent_cutoff = datetime.now() - timedelta(days=7)\n            recent_backups = [\n                r for r in self.records.values() \n                if r.start_time > recent_cutoff\n            ]\n            \n            return {\n                'total_configurations': len(self.configs),\n                'total_backups': total_backups,\n                'completed_backups': completed_backups,\n                'failed_backups': failed_backups,\n                'running_backups': sum(1 for r in self.records.values() if r.status == BackupStatus.RUNNING),\n                'total_storage_bytes': total_storage_bytes,\n                'total_storage_mb': total_storage_bytes / (1024 * 1024),\n                'recent_backups': len(recent_backups),\n                'scheduler_running': self.scheduler_running,\n                'storage_backends': list(self.storage_backends.keys()),\n                'configurations': [{\n                    'name': config.name,\n                    'type': config.backup_type.value,\n                    'storage': config.storage_backend.value,\n                    'scheduled': config.schedule_enabled\n                } for config in self.configs.values()]\n            }\n    \n    def get_backup_records(self, config_name: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:\n        \"\"\"Get backup records with optional filtering.\"\"\"\n        with self.lock:\n            records = list(self.records.values())\n            \n            if config_name:\n                records = [r for r in records if r.config_name == config_name]\n            \n            # Sort by start time (newest first)\n            records.sort(key=lambda r: r.start_time, reverse=True)\n            \n            if limit:\n                records = records[:limit]\n            \n            return [record.to_dict() for record in records]\n\n\n# Global backup manager\n_backup_manager = None\n\ndef get_backup_manager(backup_dir: Optional[str] = None) -> BackupManager:\n    \"\"\"Get global backup manager instance.\"\"\"\n    global _backup_manager\n    if _backup_manager is None:\n        _backup_manager = BackupManager(backup_dir)\n    return _backup_manager\n\n\n__all__ = [\n    \"BackupManager\",\n    \"BackupConfig\",\n    \"BackupRecord\",\n    \"BackupType\",\n    \"BackupStatus\",\n    \"StorageBackend\",\n    \"get_backup_manager\"\n]