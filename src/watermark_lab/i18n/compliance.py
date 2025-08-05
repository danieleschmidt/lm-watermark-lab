"""Privacy compliance implementations for global regulations."""

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from ..utils.logging import get_logger
from ..utils.exceptions import ComplianceError


class DataProcessingPurpose(Enum):
    """Data processing purposes for compliance."""
    WATERMARK_GENERATION = "watermark_generation"
    WATERMARK_DETECTION = "watermark_detection"
    PERFORMANCE_ANALYTICS = "performance_analytics"
    SECURITY_MONITORING = "security_monitoring"
    SERVICE_IMPROVEMENT = "service_improvement"
    RESEARCH = "research"
    LEGAL_COMPLIANCE = "legal_compliance"


class ConsentStatus(Enum):
    """User consent status."""
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"
    EXPIRED = "expired"


@dataclass
class DataSubject:
    """Represents a data subject under privacy regulations."""
    subject_id: str
    email: Optional[str] = None
    jurisdiction: Optional[str] = None
    consent_date: Optional[datetime] = None
    consent_status: ConsentStatus = ConsentStatus.PENDING
    data_retention_period: int = 365  # days
    opt_out_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert datetime objects to ISO strings
        if self.consent_date:
            result["consent_date"] = self.consent_date.isoformat()
        if self.opt_out_date:
            result["opt_out_date"] = self.opt_out_date.isoformat()
        result["consent_status"] = self.consent_status.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataSubject":
        """Create from dictionary."""
        if "consent_date" in data and data["consent_date"]:
            data["consent_date"] = datetime.fromisoformat(data["consent_date"])
        if "opt_out_date" in data and data["opt_out_date"]:
            data["opt_out_date"] = datetime.fromisoformat(data["opt_out_date"])
        if "consent_status" in data:
            data["consent_status"] = ConsentStatus(data["consent_status"])
        return cls(**data)


@dataclass
class ProcessingRecord:
    """Record of data processing activity."""
    record_id: str
    subject_id: str
    purpose: DataProcessingPurpose
    data_types: List[str]
    processing_date: datetime
    legal_basis: str
    retention_period: int  # days
    data_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result["processing_date"] = self.processing_date.isoformat()
        result["purpose"] = self.purpose.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingRecord":
        """Create from dictionary."""
        data["processing_date"] = datetime.fromisoformat(data["processing_date"])
        data["purpose"] = DataProcessingPurpose(data["purpose"])
        return cls(**data)


class ComplianceManagerBase:
    """Base class for privacy compliance management."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize compliance manager."""
        self.logger = get_logger("compliance")
        self.storage_path = storage_path or Path.cwd() / "compliance_data"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory stores (in production, use proper database)
        self.data_subjects: Dict[str, DataSubject] = {}
        self.processing_records: Dict[str, ProcessingRecord] = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load compliance data from storage."""
        subjects_file = self.storage_path / "data_subjects.json"
        records_file = self.storage_path / "processing_records.json"
        
        try:
            if subjects_file.exists():
                with open(subjects_file, 'r') as f:
                    subjects_data = json.load(f)
                    self.data_subjects = {
                        k: DataSubject.from_dict(v) for k, v in subjects_data.items()
                    }
            
            if records_file.exists():
                with open(records_file, 'r') as f:
                    records_data = json.load(f)
                    self.processing_records = {
                        k: ProcessingRecord.from_dict(v) for k, v in records_data.items()
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to load compliance data: {e}")
    
    def _save_data(self):
        """Save compliance data to storage."""
        try:
            subjects_file = self.storage_path / "data_subjects.json"
            records_file = self.storage_path / "processing_records.json"
            
            with open(subjects_file, 'w') as f:
                subjects_data = {k: v.to_dict() for k, v in self.data_subjects.items()}
                json.dump(subjects_data, f, indent=2)
            
            with open(records_file, 'w') as f:
                records_data = {k: v.to_dict() for k, v in self.processing_records.items()}
                json.dump(records_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save compliance data: {e}")
    
    def register_data_subject(self, subject_id: str, **kwargs) -> DataSubject:
        """Register a new data subject."""
        if subject_id in self.data_subjects:
            return self.data_subjects[subject_id]
        
        subject = DataSubject(subject_id=subject_id, **kwargs)
        self.data_subjects[subject_id] = subject
        self._save_data()
        
        self.logger.info(f"Registered data subject: {subject_id}")
        return subject
    
    def record_processing(
        self,
        subject_id: str,
        purpose: DataProcessingPurpose,
        data_types: List[str],
        legal_basis: str,
        data_content: Optional[str] = None,
        retention_period: int = 365
    ) -> ProcessingRecord:
        """Record a data processing activity."""
        record_id = str(uuid.uuid4())
        
        # Hash data content for privacy
        data_hash = None
        if data_content:
            data_hash = hashlib.sha256(data_content.encode()).hexdigest()
        
        record = ProcessingRecord(
            record_id=record_id,
            subject_id=subject_id,
            purpose=purpose,
            data_types=data_types,
            processing_date=datetime.now(),
            legal_basis=legal_basis,
            retention_period=retention_period,
            data_hash=data_hash
        )
        
        self.processing_records[record_id] = record
        self._save_data()
        
        self.logger.info(f"Recorded processing activity: {record_id}")
        return record
    
    def get_subject_data(self, subject_id: str) -> Dict[str, Any]:
        """Get all data for a subject (for data export requests)."""
        if subject_id not in self.data_subjects:
            raise ComplianceError(f"Data subject not found: {subject_id}")
        
        subject = self.data_subjects[subject_id]
        subject_records = [
            record for record in self.processing_records.values()
            if record.subject_id == subject_id
        ]
        
        return {
            "subject": subject.to_dict(),
            "processing_records": [record.to_dict() for record in subject_records],
            "export_date": datetime.now().isoformat(),
            "total_records": len(subject_records)
        }
    
    def delete_subject_data(self, subject_id: str) -> bool:
        """Delete all data for a subject (right to erasure)."""
        if subject_id not in self.data_subjects:
            return False
        
        # Remove subject
        del self.data_subjects[subject_id]
        
        # Remove processing records
        records_to_remove = [
            record_id for record_id, record in self.processing_records.items()
            if record.subject_id == subject_id
        ]
        
        for record_id in records_to_remove:
            del self.processing_records[record_id]
        
        self._save_data()
        
        self.logger.info(f"Deleted all data for subject: {subject_id}")
        return True
    
    def cleanup_expired_data(self) -> int:
        """Clean up expired data based on retention periods."""
        current_time = datetime.now()
        expired_records = []
        
        for record_id, record in self.processing_records.items():
            expiry_date = record.processing_date + timedelta(days=record.retention_period)
            if current_time > expiry_date:
                expired_records.append(record_id)
        
        for record_id in expired_records:
            del self.processing_records[record_id]
        
        if expired_records:
            self._save_data()
            self.logger.info(f"Cleaned up {len(expired_records)} expired records")
        
        return len(expired_records)
    
    def check_compliance(self) -> Dict[str, Any]:
        """Check compliance status."""
        current_time = datetime.now()
        total_subjects = len(self.data_subjects)
        total_records = len(self.processing_records)
        
        # Check consent status
        consent_stats = {status.value: 0 for status in ConsentStatus}
        expired_consents = 0
        
        for subject in self.data_subjects.values():
            consent_stats[subject.consent_status.value] += 1
            
            if (subject.consent_date and 
                current_time > subject.consent_date + timedelta(days=365)):
                expired_consents += 1
        
        # Check data retention
        upcoming_expiry = 0
        for record in self.processing_records.values():
            expiry_date = record.processing_date + timedelta(days=record.retention_period)
            if current_time + timedelta(days=30) > expiry_date:
                upcoming_expiry += 1
        
        return {
            "total_subjects": total_subjects,
            "total_records": total_records,
            "consent_stats": consent_stats,
            "expired_consents": expired_consents,
            "upcoming_data_expiry": upcoming_expiry,
            "last_check": current_time.isoformat()
        }


class GDPRCompliance(ComplianceManagerBase):
    """GDPR-specific compliance implementation."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize GDPR compliance manager."""
        super().__init__(storage_path)
        self.regulation = "GDPR"
        self.logger = get_logger("gdpr_compliance")
    
    def request_consent(
        self,
        subject_id: str,
        purposes: List[DataProcessingPurpose],
        email: Optional[str] = None
    ) -> bool:
        """Request GDPR consent from data subject."""
        subject = self.register_data_subject(
            subject_id=subject_id,
            email=email,
            jurisdiction="EU",
            consent_status=ConsentStatus.PENDING
        )
        
        # In a real implementation, this would send consent request
        self.logger.info(f"GDPR consent requested for subject: {subject_id}")
        return True
    
    def grant_consent(self, subject_id: str, purposes: List[DataProcessingPurpose]) -> bool:
        """Grant GDPR consent."""
        if subject_id not in self.data_subjects:
            raise ComplianceError(f"Subject not found: {subject_id}")
        
        subject = self.data_subjects[subject_id]
        subject.consent_status = ConsentStatus.GRANTED
        subject.consent_date = datetime.now()
        
        self._save_data()
        self.logger.info(f"GDPR consent granted for subject: {subject_id}")
        return True
    
    def withdraw_consent(self, subject_id: str) -> bool:
        """Withdraw GDPR consent and delete data."""
        if subject_id not in self.data_subjects:
            return False
        
        subject = self.data_subjects[subject_id]
        subject.consent_status = ConsentStatus.WITHDRAWN
        
        # Under GDPR, withdrawn consent often means data deletion
        self.delete_subject_data(subject_id)
        return True
        
    def process_data_subject_request(self, subject_id: str, request_type: str) -> Dict[str, Any]:
        """Process GDPR data subject requests."""
        if request_type == "access":
            return self.get_subject_data(subject_id)
        elif request_type == "portability":
            return self.get_subject_data(subject_id)
        elif request_type == "erasure":
            success = self.delete_subject_data(subject_id)
            return {"deleted": success, "subject_id": subject_id}
        else:
            raise ComplianceError(f"Unknown request type: {request_type}")


class CCPACompliance(ComplianceManagerBase):
    """CCPA-specific compliance implementation."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize CCPA compliance manager."""
        super().__init__(storage_path)
        self.regulation = "CCPA"
        self.logger = get_logger("ccpa_compliance")
    
    def opt_out_sale(self, subject_id: str) -> bool:
        """Process CCPA opt-out of sale request."""
        if subject_id not in self.data_subjects:
            self.register_data_subject(subject_id, jurisdiction="CA")
        
        subject = self.data_subjects[subject_id]
        subject.opt_out_date = datetime.now()
        
        self._save_data()
        self.logger.info(f"CCPA opt-out processed for subject: {subject_id}")
        return True
    
    def provide_privacy_notice(self, categories: List[str]) -> Dict[str, Any]:
        """Provide CCPA privacy notice."""
        return {
            "regulation": "CCPA",
            "data_categories": categories,
            "purposes": [purpose.value for purpose in DataProcessingPurpose],
            "retention_periods": "Varies by purpose, typically 12-36 months",
            "rights": [
                "Right to know about personal information collected",
                "Right to delete personal information", 
                "Right to opt-out of sale of personal information",
                "Right to non-discrimination"
            ],
            "contact": "privacy@example.com",
            "notice_date": datetime.now().isoformat()
        }


class PDPACompliance(ComplianceManagerBase):
    """Singapore PDPA-specific compliance implementation."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize PDPA compliance manager."""
        super().__init__(storage_path)
        self.regulation = "PDPA"
        self.logger = get_logger("pdpa_compliance")
    
    def notify_data_breach(self, breach_details: Dict[str, Any]) -> str:
        """Notify PDPA data breach."""
        breach_id = str(uuid.uuid4())
        breach_record = {
            "breach_id": breach_id,
            "notification_date": datetime.now().isoformat(),
            "regulation": "PDPA",
            **breach_details
        }
        
        # In real implementation, notify authorities within 72 hours
        breach_file = self.storage_path / f"breach_{breach_id}.json"
        with open(breach_file, 'w') as f:
            json.dump(breach_record, f, indent=2)
        
        self.logger.critical(f"PDPA data breach reported: {breach_id}")
        return breach_id
    
    def do_not_call_register(self, subject_id: str, phone_number: str) -> bool:
        """Register phone number in Do Not Call registry."""
        if subject_id not in self.data_subjects:
            self.register_data_subject(subject_id, jurisdiction="SG")
        
        # Record the DNC registration
        self.record_processing(
            subject_id=subject_id,
            purpose=DataProcessingPurpose.LEGAL_COMPLIANCE,
            data_types=["phone_number"],
            legal_basis="DNC Registry compliance",
            data_content=phone_number
        )
        
        self.logger.info(f"PDPA DNC registration for subject: {subject_id}")
        return True


class ComplianceManager:
    """Main compliance manager that handles multiple regulations."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize multi-regulation compliance manager."""
        self.logger = get_logger("compliance_manager")
        self.storage_path = storage_path or Path.cwd() / "compliance_data"
        
        # Initialize regulation-specific managers
        self.gdpr = GDPRCompliance(self.storage_path / "gdpr")
        self.ccpa = CCPACompliance(self.storage_path / "ccpa") 
        self.pdpa = PDPACompliance(self.storage_path / "pdpa")
        
        self.managers = {
            "GDPR": self.gdpr,
            "CCPA": self.ccpa,
            "PDPA": self.pdpa
        }
    
    def get_applicable_regulations(self, jurisdiction: Optional[str] = None) -> List[str]:
        """Get applicable regulations based on jurisdiction."""
        if not jurisdiction:
            return list(self.managers.keys())
        
        jurisdiction = jurisdiction.upper()
        applicable = []
        
        if jurisdiction in ["EU", "EEA"] or jurisdiction.startswith("EU-"):
            applicable.append("GDPR")
        if jurisdiction in ["CA", "US-CA"]:
            applicable.append("CCPA")
        if jurisdiction in ["SG", "SINGAPORE"]:
            applicable.append("PDPA")
        
        return applicable or ["GDPR"]  # Default to GDPR as most restrictive
    
    def process_with_compliance(
        self,
        subject_id: str,
        data_content: str,
        purpose: DataProcessingPurpose,
        jurisdiction: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process data with compliance checks."""
        applicable_regs = self.get_applicable_regulations(jurisdiction)
        results = {}
        
        for regulation in applicable_regs:
            manager = self.managers[regulation]
            
            # Check if subject exists and has valid consent
            if subject_id not in manager.data_subjects:
                manager.register_data_subject(subject_id, jurisdiction=jurisdiction)
            
            subject = manager.data_subjects[subject_id]
            
            # Check consent for GDPR
            if regulation == "GDPR" and subject.consent_status != ConsentStatus.GRANTED:
                raise ComplianceError(f"GDPR consent required for subject: {subject_id}")
            
            # Record processing activity
            record = manager.record_processing(
                subject_id=subject_id,
                purpose=purpose,
                data_types=["text_content"],
                legal_basis=f"{regulation} compliant processing",
                data_content=data_content
            )
            
            results[regulation] = {
                "compliant": True,
                "record_id": record.record_id,
                "processing_date": record.processing_date.isoformat()
            }
        
        return results
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Get comprehensive compliance report."""
        return {
            regulation: manager.check_compliance()
            for regulation, manager in self.managers.items()
        }
    
    def cleanup_all_expired_data(self) -> Dict[str, int]:
        """Clean up expired data across all regulations."""
        return {
            regulation: manager.cleanup_expired_data()
            for regulation, manager in self.managers.items()
        }