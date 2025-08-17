"""Advanced threat detection and response system."""

import time
import hashlib
import threading
import json
import re
from typing import Dict, Any, List, Optional, Set, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
from enum import Enum
import logging
from pathlib import Path
import ipaddress

logger = logging.getLogger(__name__)


class ThreatType(Enum):
    """Types of security threats."""
    BRUTE_FORCE = "brute_force"
    DDoS = "ddos"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    RATE_LIMIT_VIOLATION = "rate_limit_violation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


class ThreatSeverity(Enum):
    """Threat severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ThreatEvent:
    """Individual threat event."""
    
    threat_type: ThreatType
    severity: ThreatSeverity
    source_ip: str
    user_agent: str = ""
    request_path: str = ""
    payload: str = ""
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'threat_type': self.threat_type.value,
            'severity': self.severity.value,
            'source_ip': self.source_ip,
            'user_agent': self.user_agent,
            'request_path': self.request_path,
            'payload': self.payload,
            'timestamp': self.timestamp,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'additional_data': self.additional_data
        }
    
    def get_threat_id(self) -> str:
        """Generate unique threat ID."""
        data = f"{self.threat_type.value}:{self.source_ip}:{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class ThreatResponse:
    """Response to a threat."""
    
    action: str  # block, monitor, alert, throttle
    duration: Optional[float] = None  # Duration in seconds
    reason: str = ""
    auto_generated: bool = True
    escalation_level: int = 0


class PatternMatcher:
    """Advanced pattern matching for threat detection."""
    
    def __init__(self):
        self.sql_injection_patterns = [
            r"(\bunion\b.*\bselect\b)",
            r"(\bdrop\b.*\btable\b)",
            r"(\binsert\b.*\bvalues\b)",
            r"(\bdelete\b.*\bfrom\b)",
            r"(\bupdate\b.*\bset\b)",
            r"(\bor\b.*\b1\s*=\s*1\b)",
            r"(\band\b.*\b1\s*=\s*1\b)",
            r"(\b'\s*or\s*'1'\s*=\s*'1\b)",
            r"(\bexec\s*\()",
            r"(\bsp_executesql\b)"
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
            r"eval\s*\(",
            r"document\.cookie",
            r"document\.write"
        ]
        
        self.suspicious_paths = [
            r"/admin",
            r"/wp-admin",
            r"/phpmyadmin",
            r"/.env",
            r"/config",
            r"/.git",
            r"/backup",
            r"/test",
            r"/debug"
        ]
        
        # Compile patterns for efficiency
        self.compiled_sql_patterns = [re.compile(p, re.IGNORECASE) for p in self.sql_injection_patterns]
        self.compiled_xss_patterns = [re.compile(p, re.IGNORECASE) for p in self.xss_patterns]
        self.compiled_path_patterns = [re.compile(p, re.IGNORECASE) for p in self.suspicious_paths]
    
    def detect_sql_injection(self, text: str) -> bool:
        """Detect SQL injection patterns."""
        for pattern in self.compiled_sql_patterns:
            if pattern.search(text):
                return True
        return False
    
    def detect_xss(self, text: str) -> bool:
        """Detect XSS patterns."""
        for pattern in self.compiled_xss_patterns:
            if pattern.search(text):
                return True
        return False
    
    def detect_suspicious_path(self, path: str) -> bool:
        """Detect suspicious request paths."""
        for pattern in self.compiled_path_patterns:
            if pattern.search(path):
                return True
        return False


class AnomalyDetector:
    """Behavioral anomaly detection."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.user_behavior: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.global_behavior = deque(maxlen=window_size * 10)
        self._lock = threading.Lock()
    
    def record_behavior(self, user_id: str, action: str, context: Dict[str, Any]):
        """Record user behavior for analysis."""
        behavior_point = {
            'action': action,
            'timestamp': time.time(),
            'context': context
        }
        
        with self._lock:
            self.user_behavior[user_id].append(behavior_point)
            self.global_behavior.append(behavior_point)
    
    def detect_anomaly(self, user_id: str, action: str, context: Dict[str, Any]) -> float:
        """Detect anomalous behavior. Returns anomaly score (0-1)."""
        with self._lock:
            user_history = list(self.user_behavior.get(user_id, []))
            global_history = list(self.global_behavior)
        
        if len(user_history) < 10:  # Not enough data
            return 0.0
        
        # Analyze request frequency
        current_time = time.time()
        recent_requests = [
            b for b in user_history 
            if current_time - b['timestamp'] < 300  # Last 5 minutes
        ]
        
        frequency_score = min(len(recent_requests) / 50.0, 1.0)  # Anomalous if >50 requests/5min
        
        # Analyze action patterns
        user_actions = [b['action'] for b in user_history[-20:]]  # Last 20 actions
        action_diversity = len(set(user_actions)) / len(user_actions) if user_actions else 0
        pattern_score = 1.0 - action_diversity  # Lower diversity = higher anomaly
        
        # Time-based analysis
        timestamps = [b['timestamp'] for b in user_history[-10:]]
        if len(timestamps) > 1:
            intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            avg_interval = sum(intervals) / len(intervals)
            std_interval = (sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)) ** 0.5
            
            # Regular intervals might indicate automation
            regularity_score = 1.0 / (1.0 + std_interval) if std_interval > 0 else 1.0
        else:
            regularity_score = 0.0
        
        # Combine scores
        anomaly_score = (frequency_score * 0.4 + pattern_score * 0.3 + regularity_score * 0.3)
        return min(anomaly_score, 1.0)


class ThreatIntelligence:
    """Threat intelligence and IP reputation."""
    
    def __init__(self):
        self.malicious_ips: Set[str] = set()
        self.suspicious_ips: Set[str] = set()
        self.blocked_countries: Set[str] = set()
        self.ip_reputation: Dict[str, float] = {}  # 0.0 = good, 1.0 = bad
        self._lock = threading.Lock()
    
    def add_malicious_ip(self, ip: str):
        """Add IP to malicious list."""
        with self._lock:
            self.malicious_ips.add(ip)
            self.ip_reputation[ip] = 1.0
    
    def add_suspicious_ip(self, ip: str, reputation: float = 0.5):
        """Add IP to suspicious list."""
        with self._lock:
            self.suspicious_ips.add(ip)
            self.ip_reputation[ip] = max(self.ip_reputation.get(ip, 0.0), reputation)
    
    def get_ip_reputation(self, ip: str) -> float:
        """Get IP reputation score."""
        with self._lock:
            if ip in self.malicious_ips:
                return 1.0
            return self.ip_reputation.get(ip, 0.0)
    
    def is_private_ip(self, ip: str) -> bool:
        """Check if IP is private."""
        try:
            ip_obj = ipaddress.ip_address(ip)
            return ip_obj.is_private
        except ValueError:
            return False
    
    def analyze_ip_geolocation(self, ip: str) -> Dict[str, Any]:
        """Analyze IP geolocation (placeholder for external service)."""
        # In real implementation, would integrate with GeoIP service
        return {
            'country': 'Unknown',
            'region': 'Unknown',
            'city': 'Unknown',
            'is_tor': False,
            'is_proxy': False,
            'is_hosting': False
        }


class ThreatDetectionEngine:
    """Main threat detection engine."""
    
    def __init__(self):
        self.pattern_matcher = PatternMatcher()
        self.anomaly_detector = AnomalyDetector()
        self.threat_intelligence = ThreatIntelligence()
        self.events: deque = deque(maxlen=10000)
        self.active_threats: Dict[str, ThreatEvent] = {}
        self.response_handlers: Dict[ThreatType, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Rate limiting tracking
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
    def analyze_request(
        self,
        source_ip: str,
        request_path: str,
        payload: str = "",
        user_agent: str = "",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[ThreatEvent]:
        """Analyze a request for threats."""
        threats = []
        
        # IP reputation check
        ip_reputation = self.threat_intelligence.get_ip_reputation(source_ip)
        if ip_reputation > 0.7:
            threats.append(ThreatEvent(
                threat_type=ThreatType.SUSPICIOUS_PATTERN,
                severity=ThreatSeverity.HIGH,
                source_ip=source_ip,
                user_agent=user_agent,
                request_path=request_path,
                payload=payload,
                session_id=session_id,
                user_id=user_id,
                additional_data={'ip_reputation': ip_reputation}
            ))
        
        # Pattern-based detection
        full_request = f"{request_path} {payload}"
        
        if self.pattern_matcher.detect_sql_injection(full_request):
            threats.append(ThreatEvent(
                threat_type=ThreatType.SQL_INJECTION,
                severity=ThreatSeverity.HIGH,
                source_ip=source_ip,
                user_agent=user_agent,
                request_path=request_path,
                payload=payload,
                session_id=session_id,
                user_id=user_id
            ))
        
        if self.pattern_matcher.detect_xss(full_request):
            threats.append(ThreatEvent(
                threat_type=ThreatType.XSS,
                severity=ThreatSeverity.MEDIUM,
                source_ip=source_ip,
                user_agent=user_agent,
                request_path=request_path,
                payload=payload,
                session_id=session_id,
                user_id=user_id
            ))
        
        if self.pattern_matcher.detect_suspicious_path(request_path):
            threats.append(ThreatEvent(
                threat_type=ThreatType.UNAUTHORIZED_ACCESS,
                severity=ThreatSeverity.MEDIUM,
                source_ip=source_ip,
                user_agent=user_agent,
                request_path=request_path,
                payload=payload,
                session_id=session_id,
                user_id=user_id
            ))
        
        # Rate limiting check
        current_time = time.time()
        with self._lock:
            self.request_counts[source_ip].append(current_time)
            recent_requests = [
                t for t in self.request_counts[source_ip]
                if current_time - t < 60  # Last minute
            ]
            
            if len(recent_requests) > 100:  # More than 100 requests per minute
                threats.append(ThreatEvent(
                    threat_type=ThreatType.RATE_LIMIT_VIOLATION,
                    severity=ThreatSeverity.MEDIUM,
                    source_ip=source_ip,
                    user_agent=user_agent,
                    request_path=request_path,
                    payload=payload,
                    session_id=session_id,
                    user_id=user_id,
                    additional_data={'request_count': len(recent_requests)}
                ))
        
        # Behavioral anomaly detection
        if user_id:
            self.anomaly_detector.record_behavior(
                user_id, 'request', 
                {'path': request_path, 'ip': source_ip, 'user_agent': user_agent}
            )
            
            anomaly_score = self.anomaly_detector.detect_anomaly(
                user_id, 'request',
                {'path': request_path, 'ip': source_ip, 'user_agent': user_agent}
            )
            
            if anomaly_score > 0.7:
                threats.append(ThreatEvent(
                    threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                    severity=ThreatSeverity.MEDIUM if anomaly_score < 0.9 else ThreatSeverity.HIGH,
                    source_ip=source_ip,
                    user_agent=user_agent,
                    request_path=request_path,
                    payload=payload,
                    session_id=session_id,
                    user_id=user_id,
                    additional_data={'anomaly_score': anomaly_score}
                ))
        
        # Record and respond to threats
        for threat in threats:
            self._record_threat(threat)
            response = self._generate_response(threat)
            self._execute_response(threat, response)
        
        return threats
    
    def _record_threat(self, threat: ThreatEvent):
        """Record threat event."""
        with self._lock:
            self.events.append(threat)
            threat_id = threat.get_threat_id()
            self.active_threats[threat_id] = threat
            
            # Update IP reputation
            if threat.severity.value >= ThreatSeverity.HIGH.value:
                current_reputation = self.threat_intelligence.get_ip_reputation(threat.source_ip)
                new_reputation = min(current_reputation + 0.2, 1.0)
                self.threat_intelligence.add_suspicious_ip(threat.source_ip, new_reputation)
        
        logger.warning(f"Threat detected: {threat.threat_type.value} from {threat.source_ip}")
    
    def _generate_response(self, threat: ThreatEvent) -> ThreatResponse:
        """Generate response to threat."""
        if threat.severity == ThreatSeverity.CRITICAL:
            return ThreatResponse(
                action="block",
                duration=3600,  # 1 hour
                reason=f"Critical threat: {threat.threat_type.value}",
                escalation_level=3
            )
        elif threat.severity == ThreatSeverity.HIGH:
            return ThreatResponse(
                action="throttle",
                duration=1800,  # 30 minutes
                reason=f"High severity threat: {threat.threat_type.value}",
                escalation_level=2
            )
        elif threat.severity == ThreatSeverity.MEDIUM:
            return ThreatResponse(
                action="monitor",
                duration=600,  # 10 minutes
                reason=f"Medium severity threat: {threat.threat_type.value}",
                escalation_level=1
            )
        else:
            return ThreatResponse(
                action="alert",
                reason=f"Low severity threat: {threat.threat_type.value}",
                escalation_level=0
            )
    
    def _execute_response(self, threat: ThreatEvent, response: ThreatResponse):
        """Execute threat response."""
        # Call registered handlers
        for handler in self.response_handlers[threat.threat_type]:
            try:
                handler(threat, response)
            except Exception as e:
                logger.error(f"Error executing threat response handler: {e}")
        
        logger.info(
            f"Threat response: {response.action} for {threat.threat_type.value} "
            f"from {threat.source_ip} (duration: {response.duration}s)"
        )
    
    def register_response_handler(self, threat_type: ThreatType, handler: Callable):
        """Register threat response handler."""
        self.response_handlers[threat_type].append(handler)
    
    def get_threat_summary(self, time_window: float = 3600) -> Dict[str, Any]:
        """Get threat summary for time window."""
        current_time = time.time()
        recent_threats = [
            event for event in self.events
            if current_time - event.timestamp <= time_window
        ]
        
        threat_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        source_ips = defaultdict(int)
        
        for threat in recent_threats:
            threat_counts[threat.threat_type.value] += 1
            severity_counts[threat.severity.value] += 1
            source_ips[threat.source_ip] += 1
        
        return {
            'time_window': time_window,
            'total_threats': len(recent_threats),
            'threat_types': dict(threat_counts),
            'severity_levels': dict(severity_counts),
            'top_source_ips': dict(sorted(source_ips.items(), key=lambda x: x[1], reverse=True)[:10]),
            'active_threats': len(self.active_threats)
        }
    
    def export_threats(self, filename: str, time_window: float = 3600):
        """Export threat events to file."""
        current_time = time.time()
        recent_threats = [
            event.to_dict() for event in self.events
            if current_time - event.timestamp <= time_window
        ]
        
        with open(filename, 'w') as f:
            json.dump({
                'export_time': current_time,
                'time_window': time_window,
                'threats': recent_threats
            }, f, indent=2)


# Global threat detection engine
_global_engine = ThreatDetectionEngine()

def get_global_threat_engine() -> ThreatDetectionEngine:
    """Get global threat detection engine."""
    return _global_engine


def analyze_request_for_threats(
    source_ip: str,
    request_path: str,
    payload: str = "",
    user_agent: str = "",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> List[ThreatEvent]:
    """Analyze request for threats using global engine."""
    return _global_engine.analyze_request(
        source_ip, request_path, payload, user_agent, user_id, session_id
    )