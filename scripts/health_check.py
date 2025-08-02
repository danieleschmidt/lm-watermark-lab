#!/usr/bin/env python3
"""
Health check script for LM Watermark Lab
"""

import asyncio
import json
import sys
import time
from typing import Dict, List, Any
import httpx
import redis
import psycopg2
from pathlib import Path

# Health check configuration
HEALTH_CHECKS = {
    "api": {
        "url": "http://localhost:8080/health",
        "timeout": 10,
        "critical": True
    },
    "api_status": {
        "url": "http://localhost:8080/api/v1/status",
        "timeout": 10,
        "critical": False
    },
    "redis": {
        "host": "localhost",
        "port": 6379,
        "timeout": 5,
        "critical": True
    },
    "postgres": {
        "host": "localhost",
        "port": 5432,
        "dbname": "watermark_lab",
        "user": "postgres",
        "password": "password",
        "timeout": 5,
        "critical": True
    },
    "disk_space": {
        "paths": ["/app/data", "/app/logs"],
        "min_free_gb": 1.0,
        "critical": True
    },
    "model_cache": {
        "path": "/app/data/models",
        "critical": False
    }
}


class HealthChecker:
    """Comprehensive health checking for Watermark Lab"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or HEALTH_CHECKS
        self.results: Dict[str, Dict[str, Any]] = {}
    
    async def check_api_health(self, check_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check API endpoint health"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    check_config["url"],
                    timeout=check_config["timeout"]
                )
                
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "response_time_ms": response.elapsed.total_seconds() * 1000,
                        "status_code": response.status_code,
                        "details": response.json() if response.headers.get("content-type", "").startswith("application/json") else None
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status_code}",
                        "status_code": response.status_code
                    }
        except httpx.TimeoutException:
            return {
                "status": "unhealthy",
                "error": "Request timeout"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def check_redis_health(self, check_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check Redis connection and performance"""
        try:
            r = redis.Redis(
                host=check_config["host"],
                port=check_config["port"],
                socket_timeout=check_config["timeout"]
            )
            
            # Test basic operations
            start_time = time.time()
            r.ping()
            ping_time = (time.time() - start_time) * 1000
            
            # Get Redis info
            info = r.info()
            
            return {
                "status": "healthy",
                "ping_time_ms": ping_time,
                "memory_usage_mb": info.get("used_memory", 0) / 1024 / 1024,
                "connected_clients": info.get("connected_clients", 0),
                "uptime_seconds": info.get("uptime_in_seconds", 0)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def check_postgres_health(self, check_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check PostgreSQL connection and performance"""
        try:
            conn = psycopg2.connect(
                host=check_config["host"],
                port=check_config["port"],
                dbname=check_config["dbname"],
                user=check_config["user"],
                password=check_config["password"],
                connect_timeout=check_config["timeout"]
            )
            
            cursor = conn.cursor()
            
            # Test query performance
            start_time = time.time()
            cursor.execute("SELECT 1")
            query_time = (time.time() - start_time) * 1000
            
            # Get database stats
            cursor.execute("SELECT pg_database_size(current_database())")
            db_size = cursor.fetchone()[0]
            
            cursor.execute("SELECT count(*) FROM pg_stat_activity")
            active_connections = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return {
                "status": "healthy",
                "query_time_ms": query_time,
                "database_size_mb": db_size / 1024 / 1024,
                "active_connections": active_connections
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def check_disk_space(self, check_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check disk space for critical paths"""
        import shutil
        
        results = {}
        overall_status = "healthy"
        
        for path in check_config["paths"]:
            try:
                path_obj = Path(path)
                if path_obj.exists():
                    total, used, free = shutil.disk_usage(path)
                    free_gb = free / (1024 ** 3)
                    
                    status = "healthy" if free_gb >= check_config["min_free_gb"] else "unhealthy"
                    if status == "unhealthy":
                        overall_status = "unhealthy"
                    
                    results[path] = {
                        "status": status,
                        "free_gb": free_gb,
                        "total_gb": total / (1024 ** 3),
                        "used_gb": used / (1024 ** 3),
                        "usage_percent": (used / total) * 100
                    }
                else:
                    results[path] = {
                        "status": "unhealthy",
                        "error": "Path does not exist"
                    }
                    overall_status = "unhealthy"
            except Exception as e:
                results[path] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "paths": results
        }
    
    def check_model_cache(self, check_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check model cache status"""
        try:
            cache_path = Path(check_config["path"])
            
            if not cache_path.exists():
                return {
                    "status": "warning",
                    "error": "Model cache directory does not exist",
                    "models_cached": 0
                }
            
            # Count cached models
            model_files = list(cache_path.glob("**/*.bin")) + list(cache_path.glob("**/*.safetensors"))
            
            # Calculate cache size
            total_size = sum(f.stat().st_size for f in model_files if f.is_file())
            
            return {
                "status": "healthy",
                "models_cached": len(model_files),
                "cache_size_gb": total_size / (1024 ** 3),
                "cache_path": str(cache_path)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all configured health checks"""
        self.results = {}
        
        for check_name, check_config in self.config.items():
            print(f"Running health check: {check_name}")
            
            try:
                if check_name in ["api", "api_status"]:
                    result = await self.check_api_health(check_config)
                elif check_name == "redis":
                    result = self.check_redis_health(check_config)
                elif check_name == "postgres":
                    result = self.check_postgres_health(check_config)
                elif check_name == "disk_space":
                    result = self.check_disk_space(check_config)
                elif check_name == "model_cache":
                    result = self.check_model_cache(check_config)
                else:
                    result = {"status": "unknown", "error": "Unknown check type"}
                
                result["critical"] = check_config.get("critical", False)
                self.results[check_name] = result
                
            except Exception as e:
                self.results[check_name] = {
                    "status": "error",
                    "error": f"Check failed: {str(e)}",
                    "critical": check_config.get("critical", False)
                }
        
        return self.results
    
    def get_overall_status(self) -> str:
        """Determine overall system health status"""
        if not self.results:
            return "unknown"
        
        # Check for critical failures
        for check_name, result in self.results.items():
            if result.get("critical", False) and result.get("status") != "healthy":
                return "unhealthy"
        
        # Check for any unhealthy services
        for check_name, result in self.results.items():
            if result.get("status") == "unhealthy":
                return "degraded"
        
        # Check for warnings
        for check_name, result in self.results.items():
            if result.get("status") == "warning":
                return "warning"
        
        return "healthy"
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        overall_status = self.get_overall_status()
        
        return {
            "timestamp": time.time(),
            "overall_status": overall_status,
            "checks": self.results,
            "summary": {
                "total_checks": len(self.results),
                "healthy": len([r for r in self.results.values() if r.get("status") == "healthy"]),
                "unhealthy": len([r for r in self.results.values() if r.get("status") == "unhealthy"]),
                "warnings": len([r for r in self.results.values() if r.get("status") == "warning"]),
                "errors": len([r for r in self.results.values() if r.get("status") == "error"])
            }
        }


async def main():
    """Main health check execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Watermark Lab Health Checker")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--critical-only", action="store_true", help="Only check critical services")
    parser.add_argument("--timeout", type=int, default=30, help="Overall timeout in seconds")
    parser.add_argument("--config", type=str, help="Path to custom health check configuration")
    
    args = parser.parse_args()
    
    # Load custom config if provided
    config = HEALTH_CHECKS
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Filter to critical checks only if requested
    if args.critical_only:
        config = {k: v for k, v in config.items() if v.get("critical", False)}
    
    # Run health checks with timeout
    checker = HealthChecker(config)
    
    try:
        results = await asyncio.wait_for(
            checker.run_all_checks(),
            timeout=args.timeout
        )
        
        report = checker.generate_report()
        
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            # Human-readable output
            print(f"\n🏥 Watermark Lab Health Check Report")
            print(f"📊 Overall Status: {report['overall_status'].upper()}")
            print(f"⏰ Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report['timestamp']))}")
            print(f"\n📈 Summary:")
            print(f"  ✅ Healthy: {report['summary']['healthy']}")
            print(f"  ❌ Unhealthy: {report['summary']['unhealthy']}")
            print(f"  ⚠️  Warnings: {report['summary']['warnings']}")
            print(f"  🔥 Errors: {report['summary']['errors']}")
            
            print(f"\n🔍 Detailed Results:")
            for check_name, result in report['checks'].items():
                status_emoji = {
                    "healthy": "✅",
                    "unhealthy": "❌",
                    "warning": "⚠️",
                    "error": "🔥",
                    "unknown": "❓"
                }.get(result.get('status'), "❓")
                
                critical = " (CRITICAL)" if result.get('critical') else ""
                print(f"  {status_emoji} {check_name}{critical}: {result.get('status', 'unknown')}")
                
                if result.get('error'):
                    print(f"    Error: {result['error']}")
                
                # Show additional details for healthy checks
                if result.get('status') == 'healthy':
                    if 'response_time_ms' in result:
                        print(f"    Response time: {result['response_time_ms']:.2f}ms")
                    if 'ping_time_ms' in result:
                        print(f"    Ping time: {result['ping_time_ms']:.2f}ms")
                    if 'query_time_ms' in result:
                        print(f"    Query time: {result['query_time_ms']:.2f}ms")
        
        # Exit with appropriate code
        if report['overall_status'] in ['unhealthy', 'error']:
            sys.exit(1)
        elif report['overall_status'] in ['degraded', 'warning']:
            sys.exit(2)
        else:
            sys.exit(0)
            
    except asyncio.TimeoutError:
        print("❌ Health check timed out")
        sys.exit(3)
    except Exception as e:
        print(f"🔥 Health check failed: {str(e)}")
        sys.exit(4)


if __name__ == "__main__":
    asyncio.run(main())