"""Security audit logging and monitoring."""

import time
import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from ..utils.logging import get_logger
from ..utils.exceptions import SecurityError, ValidationError

logger = get_logger("security.audit")


class AuditEventType(Enum):
    """Types of security audit events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_EVENT = "system_event"
    USER_ACTION = "user_action"


@dataclass
class AuditConfig:
    """Configuration for audit logging."""
    enable_audit: bool = True
    log_level: str = "INFO"
    max_log_entries: int = 10000
    retention_days: int = 90
    enable_real_time_alerts: bool = True
    sensitive_fields: List[str] = None
    
    def __post_init__(self):
        if self.sensitive_fields is None:
            self.sensitive_fields = ['password', 'token', 'key', 'secret']


@dataclass
class AuditEvent:
    """Security audit event."""
    timestamp: float
    event_type: AuditEventType
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    action: str
    resource: str
    outcome: str  # success, failure, error
    details: Dict[str, Any]
    severity: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        return data


class SecurityAuditor:
    """Comprehensive security audit logging."""
    
    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
        self.logger = logger
        self.audit_log: List[AuditEvent] = []
        self.security_alerts = []
    
    def log_event(self, 
                  event_type: AuditEventType,
                  action: str,
                  resource: str,
                  outcome: str,
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  ip_address: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None,
                  severity: str = "INFO") -> None:
        """Log a security audit event."""
        
        if not self.config.enable_audit:
            return
        
        # Sanitize details to remove sensitive information
        sanitized_details = self._sanitize_details(details or {})
        
        event = AuditEvent(
            timestamp=time.time(),
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            action=action,
            resource=resource,
            outcome=outcome,
            details=sanitized_details,
            severity=severity
        )
        
        # Add to audit log
        self.audit_log.append(event)
        
        # Maintain log size
        if len(self.audit_log) > self.config.max_log_entries:
            self.audit_log = self.audit_log[-self.config.max_log_entries:]
        
        # Log to system logger
        log_message = f"AUDIT: {action} on {resource} by {user_id or 'anonymous'} - {outcome}"
        
        if severity == "CRITICAL":
            self.logger.critical(log_message, extra=event.to_dict())
        elif severity == "ERROR":
            self.logger.error(log_message, extra=event.to_dict())
        elif severity == "WARNING":
            self.logger.warning(log_message, extra=event.to_dict())
        else:
            self.logger.info(log_message, extra=event.to_dict())
        
        # Check for security alerts
        if self.config.enable_real_time_alerts:
            self._check_security_alerts(event)
    
    def log_authentication(self, user_id: str, outcome: str, ip_address: Optional[str] = None, details: Optional[Dict] = None):
        """Log authentication event."""
        self.log_event(
            AuditEventType.AUTHENTICATION,
            "user_login",
            f"user:{user_id}",
            outcome,
            user_id=user_id,
            ip_address=ip_address,
            details=details,
            severity="WARNING" if outcome == "failure" else "INFO"
        )
    
    def log_authorization(self, user_id: str, resource: str, action: str, outcome: str, details: Optional[Dict] = None):
        """Log authorization event."""
        self.log_event(
            AuditEventType.AUTHORIZATION,
            f"access_{action}",
            resource,
            outcome,
            user_id=user_id,
            details=details,
            severity="WARNING" if outcome == "denied" else "INFO"
        )
    
    def log_security_violation(self, violation_type: str, details: Dict[str, Any], user_id: Optional[str] = None):
        """Log security violation."""
        self.log_event(
            AuditEventType.SECURITY_VIOLATION,
            violation_type,
            "system",
            "detected",
            user_id=user_id,
            details=details,
            severity="ERROR"
        )
    
    def _sanitize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from audit details."""
        sanitized = {}
        for key, value in details.items():
            if any(sensitive in key.lower() for sensitive in self.config.sensitive_fields):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        return sanitized
    
    def _check_security_alerts(self, event: AuditEvent):
        """Check for patterns that require security alerts."""
        # Multiple failed logins
        if (event.event_type == AuditEventType.AUTHENTICATION and 
            event.outcome == "failure" and event.user_id):
            
            recent_failures = [
                e for e in self.audit_log[-10:] 
                if (e.event_type == AuditEventType.AUTHENTICATION and 
                    e.outcome == "failure" and 
                    e.user_id == event.user_id and
                    time.time() - e.timestamp < 300)  # Last 5 minutes
            ]
            
            if len(recent_failures) >= 3:
                alert = {
                    'type': 'multiple_failed_logins',
                    'user_id': event.user_id,
                    'count': len(recent_failures),
                    'timestamp': time.time(),
                    'severity': 'HIGH'
                }
                self.security_alerts.append(alert)
                self.logger.critical(f"SECURITY ALERT: Multiple failed logins for user {event.user_id}")
    
    def get_audit_log(self, 
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None,
                      event_type: Optional[AuditEventType] = None,
                      user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve audit log entries with filters."""
        filtered_events = self.audit_log
        
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        return [event.to_dict() for event in filtered_events]
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary statistics."""
        current_time = time.time()
        last_24h = current_time - 86400
        
        recent_events = [e for e in self.audit_log if e.timestamp >= last_24h]
        
        return {
            'total_events_24h': len(recent_events),
            'authentication_failures_24h': len([e for e in recent_events 
                                               if e.event_type == AuditEventType.AUTHENTICATION and e.outcome == "failure"]),
            'security_violations_24h': len([e for e in recent_events 
                                           if e.event_type == AuditEventType.SECURITY_VIOLATION]),
            'active_alerts': len(self.security_alerts),
            'last_event_timestamp': self.audit_log[-1].timestamp if self.audit_log else None
        }
    
    def sanitize_text(self, text: str) -> str:
        """Sanitize text for audit logging."""
        # Remove potentially dangerous characters and limit length
        sanitized = re.sub(r'[<>"\';\\]', '', text)
        return sanitized[:500]  # Limit length for audit logs
    
    def detect_threats(self, log_entry: str) -> AuditEventType:
        """Detect threat patterns in audit logs."""
        threat_patterns = {
            r'multiple.*fail.*login': AuditEventType.SECURITY_VIOLATION,
            r'brute.*force': AuditEventType.SECURITY_VIOLATION,
            r'injection.*attempt': AuditEventType.SECURITY_VIOLATION,
            r'unauthorized.*access': AuditEventType.AUTHORIZATION
        }
        
        for pattern, event_type in threat_patterns.items():
            if re.search(pattern, log_entry, re.IGNORECASE):
                return event_type
        
        return AuditEventType.SYSTEM_EVENT
    
    def rate_limit(self, client_id: str) -> bool:
        """Rate limit audit log operations."""
        # Count recent log entries for this client
        current_time = time.time()
        recent_logs = [
            event for event in self.audit_log[-100:]  # Check last 100 events
            if event.user_id == client_id and current_time - event.timestamp < 60
        ]
        
        # Allow max 50 audit events per minute per client
        return len(recent_logs) < 50