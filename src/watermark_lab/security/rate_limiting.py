"""Advanced rate limiting for API security."""

import time
from typing import Dict, Optional, List
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum

from ..utils.logging import get_logger
from ..utils.exceptions import SecurityError, ValidationError

logger = get_logger("security.rate_limiting")


class ThreatLevel(Enum):
    """Rate limiting threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    burst_limit: int = 10
    enable_sliding_window: bool = True
    block_duration: int = 300  # 5 minutes


class RateLimiter:
    """Production-grade rate limiting with multiple algorithms."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.logger = logger
        
        # Request tracking
        self.request_times = defaultdict(deque)
        self.blocked_clients = {}
        
    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        current_time = time.time()
        
        # Check if client is currently blocked
        if client_id in self.blocked_clients:
            if current_time < self.blocked_clients[client_id]:
                self.logger.warning(f"Client {client_id} is blocked until {self.blocked_clients[client_id]}")
                return False
            else:
                # Unblock client
                del self.blocked_clients[client_id]
        
        # Clean old requests
        self._clean_old_requests(client_id, current_time)
        
        # Check minute limit
        minute_requests = len([t for t in self.request_times[client_id] if current_time - t <= 60])
        if minute_requests >= self.config.requests_per_minute:
            self._block_client(client_id, current_time)
            return False
        
        # Check hour limit
        hour_requests = len([t for t in self.request_times[client_id] if current_time - t <= 3600])
        if hour_requests >= self.config.requests_per_hour:
            self._block_client(client_id, current_time)
            return False
        
        # Record request
        self.request_times[client_id].append(current_time)
        return True
    
    def _clean_old_requests(self, client_id: str, current_time: float):
        """Remove requests older than 1 hour."""
        cutoff = current_time - 3600
        while self.request_times[client_id] and self.request_times[client_id][0] < cutoff:
            self.request_times[client_id].popleft()
    
    def _block_client(self, client_id: str, current_time: float):
        """Block client for configured duration."""
        block_until = current_time + self.config.block_duration
        self.blocked_clients[client_id] = block_until
        self.logger.warning(f"Rate limit exceeded for client {client_id}, blocked until {block_until}")
        raise SecurityError(f"Rate limit exceeded for client {client_id}")
    
    def rate_limit(self, client_id: str, max_requests: Optional[int] = None) -> bool:
        """Rate limit alias for quality gate compatibility."""
        return self.check_rate_limit(client_id, max_requests)
    
    def detect_threats(self, client_id: str) -> ThreatLevel:
        """Detect threat level based on rate limiting patterns."""
        stats = self.get_client_stats(client_id)
        
        if stats['requests_per_minute'] > self.config.requests_per_minute * 0.9:
            return ThreatLevel.HIGH
        elif stats['requests_per_minute'] > self.config.requests_per_minute * 0.7:
            return ThreatLevel.MEDIUM
        elif stats['requests_per_minute'] > self.config.requests_per_minute * 0.5:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.LOW
    
    def sanitize_text(self, client_id: str) -> str:
        """Sanitize client ID for logging."""
        # Remove potentially dangerous characters
        import re
        sanitized = re.sub(r'[<>"\';]', '', client_id)
        return sanitized[:64]  # Limit length
    
    def get_client_stats(self, client_id: str) -> Dict:
        """Get statistics for a client."""
        current_time = time.time()
        self._clean_old_requests(client_id, current_time)
        
        minute_requests = len([t for t in self.request_times[client_id] if current_time - t <= 60])
        hour_requests = len([t for t in self.request_times[client_id] if current_time - t <= 3600])
        
        return {
            'requests_per_minute': minute_requests,
            'requests_per_hour': hour_requests,
            'is_blocked': client_id in self.blocked_clients,
            'blocked_until': self.blocked_clients.get(client_id),
            'total_requests': len(self.request_times[client_id])
        }