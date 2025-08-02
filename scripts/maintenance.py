#!/usr/bin/env python3
"""
Automated maintenance tasks for LM Watermark Lab
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MaintenanceTask:
    """Base class for maintenance tasks"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.last_run = None
        self.duration = None
        self.status = "pending"
        self.error = None
    
    async def run(self) -> bool:
        """Run the maintenance task"""
        start_time = time.time()
        self.status = "running"
        self.error = None
        
        try:
            logger.info(f"Starting maintenance task: {self.name}")
            success = await self.execute()
            self.status = "completed" if success else "failed"
            self.duration = time.time() - start_time
            self.last_run = datetime.now()
            
            if success:
                logger.info(f"Completed {self.name} in {self.duration:.2f}s")
            else:
                logger.error(f"Failed {self.name} after {self.duration:.2f}s")
            
            return success
        except Exception as e:
            self.status = "error"
            self.error = str(e)
            self.duration = time.time() - start_time
            self.last_run = datetime.now()
            logger.error(f"Error in {self.name}: {e}")
            return False
    
    async def execute(self) -> bool:
        """Override this method in subclasses"""
        raise NotImplementedError


class CleanupCacheTask(MaintenanceTask):
    """Clean up old cache files"""
    
    def __init__(self, cache_dirs: List[str], max_age_days: int = 7):
        super().__init__("cleanup_cache", "Clean up old cache files")
        self.cache_dirs = cache_dirs
        self.max_age_days = max_age_days
    
    async def execute(self) -> bool:
        """Clean up old cache files"""
        total_freed = 0
        cutoff_time = time.time() - (self.max_age_days * 24 * 3600)
        
        for cache_dir in self.cache_dirs:
            cache_path = Path(cache_dir)
            if not cache_path.exists():
                continue
            
            for file_path in cache_path.rglob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        size = file_path.stat().st_size
                        file_path.unlink()
                        total_freed += size
                        logger.debug(f"Deleted: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
        
        logger.info(f"Freed {total_freed / 1024 / 1024:.2f} MB from cache cleanup")
        return True


class UpdateDependenciesTask(MaintenanceTask):
    """Update Python dependencies"""
    
    def __init__(self, check_only: bool = False):
        super().__init__("update_dependencies", "Update Python dependencies")
        self.check_only = check_only
    
    async def execute(self) -> bool:
        """Check for and optionally update dependencies"""
        try:
            # Check for outdated packages
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            outdated = json.loads(result.stdout)
            if not outdated:
                logger.info("All dependencies are up to date")
                return True
            
            logger.info(f"Found {len(outdated)} outdated packages")
            for package in outdated:
                logger.info(f"  {package['name']}: {package['version']} -> {package['latest_version']}")
            
            if self.check_only:
                return True
            
            # Update packages (be careful with this in production)
            for package in outdated:
                if package['name'] not in ['pip', 'setuptools']:  # Skip critical packages
                    try:
                        subprocess.run(
                            ["pip", "install", "--upgrade", package['name']],
                            check=True,
                            capture_output=True
                        )
                        logger.info(f"Updated {package['name']}")
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"Failed to update {package['name']}: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Dependency update failed: {e}")
            return False


class SecurityScanTask(MaintenanceTask):
    """Run security scans"""
    
    def __init__(self):
        super().__init__("security_scan", "Run security vulnerability scans")
    
    async def execute(self) -> bool:
        """Run security scans"""
        results = []
        
        # Safety scan
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info("Safety scan: No vulnerabilities found")
                results.append(True)
            else:
                safety_results = json.loads(result.stdout) if result.stdout else []
                logger.warning(f"Safety scan found {len(safety_results)} vulnerabilities")
                results.append(False)
        except Exception as e:
            logger.error(f"Safety scan failed: {e}")
            results.append(False)
        
        # Bandit scan
        try:
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                capture_output=True,
                text=True
            )
            bandit_results = json.loads(result.stdout) if result.stdout else {}
            issues = bandit_results.get("results", [])
            
            if not issues:
                logger.info("Bandit scan: No security issues found")
                results.append(True)
            else:
                logger.warning(f"Bandit scan found {len(issues)} security issues")
                results.append(False)
        except Exception as e:
            logger.error(f"Bandit scan failed: {e}")
            results.append(False)
        
        return all(results)


class ModelCleanupTask(MaintenanceTask):
    """Clean up old model files"""
    
    def __init__(self, model_cache_dir: str, keep_recent: int = 5):
        super().__init__("model_cleanup", "Clean up old model cache files")
        self.model_cache_dir = model_cache_dir
        self.keep_recent = keep_recent
    
    async def execute(self) -> bool:
        """Clean up old model files, keeping only recent ones"""
        cache_path = Path(self.model_cache_dir)
        if not cache_path.exists():
            return True
        
        # Group models by name and keep only recent versions
        model_dirs = [d for d in cache_path.iterdir() if d.is_dir()]
        total_freed = 0
        
        for model_dir in model_dirs:
            # Get all snapshots/versions for this model
            snapshots = [d for d in model_dir.iterdir() if d.is_dir()]
            if len(snapshots) <= self.keep_recent:
                continue
            
            # Sort by modification time, keep newest
            snapshots.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            to_delete = snapshots[self.keep_recent:]
            
            for snapshot in to_delete:
                try:
                    size = sum(f.stat().st_size for f in snapshot.rglob("*") if f.is_file())
                    shutil.rmtree(snapshot)
                    total_freed += size
                    logger.info(f"Deleted old model snapshot: {snapshot}")
                except Exception as e:
                    logger.warning(f"Failed to delete {snapshot}: {e}")
        
        logger.info(f"Model cleanup freed {total_freed / 1024 / 1024:.2f} MB")
        return True


class DatabaseMaintenanceTask(MaintenanceTask):
    """Database maintenance operations"""
    
    def __init__(self, database_url: Optional[str] = None):
        super().__init__("database_maintenance", "Database maintenance operations")
        self.database_url = database_url or os.getenv("DATABASE_URL")
    
    async def execute(self) -> bool:
        """Run database maintenance"""
        if not self.database_url or "sqlite" in self.database_url:
            logger.info("Skipping database maintenance for SQLite")
            return True
        
        try:
            import psycopg2
            
            # Basic connection test
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            # Get database size
            cursor.execute("SELECT pg_size_pretty(pg_database_size(current_database()));")
            db_size = cursor.fetchone()[0]
            logger.info(f"Database size: {db_size}")
            
            # Vacuum analyze (if needed)
            conn.autocommit = True
            cursor.execute("VACUUM ANALYZE;")
            logger.info("Database vacuum analyze completed")
            
            cursor.close()
            conn.close()
            
            return True
        except ImportError:
            logger.info("psycopg2 not available, skipping database maintenance")
            return True
        except Exception as e:
            logger.error(f"Database maintenance failed: {e}")
            return False


class LogRotationTask(MaintenanceTask):
    """Rotate and compress old log files"""
    
    def __init__(self, log_dirs: List[str], max_age_days: int = 30):
        super().__init__("log_rotation", "Rotate and clean old log files")
        self.log_dirs = log_dirs
        self.max_age_days = max_age_days
    
    async def execute(self) -> bool:
        """Rotate log files"""
        total_freed = 0
        cutoff_time = time.time() - (self.max_age_days * 24 * 3600)
        
        for log_dir in self.log_dirs:
            log_path = Path(log_dir)
            if not log_path.exists():
                continue
            
            for log_file in log_path.glob("*.log"):
                if log_file.stat().st_mtime < cutoff_time:
                    try:
                        size = log_file.stat().st_size
                        log_file.unlink()
                        total_freed += size
                        logger.debug(f"Deleted old log: {log_file}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {log_file}: {e}")
        
        logger.info(f"Log rotation freed {total_freed / 1024 / 1024:.2f} MB")
        return True


class MaintenanceManager:
    """Manages and coordinates maintenance tasks"""
    
    def __init__(self):
        self.tasks: Dict[str, MaintenanceTask] = {}
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load maintenance configuration"""
        config_file = Path("maintenance_config.json")
        if config_file.exists():
            with open(config_file) as f:
                return json.load(f)
        
        # Default configuration
        return {
            "cache_dirs": ["./data/cache", "./data/hf_cache", "./.mypy_cache"],
            "model_cache_dir": "./data/models",
            "log_dirs": ["./logs"],
            "max_cache_age_days": 7,
            "max_log_age_days": 30,
            "keep_recent_models": 5,
            "security_scan_enabled": True,
            "dependency_check_only": True
        }
    
    def register_task(self, task: MaintenanceTask):
        """Register a maintenance task"""
        self.tasks[task.name] = task
    
    def register_default_tasks(self):
        """Register default maintenance tasks"""
        self.register_task(CleanupCacheTask(
            self.config["cache_dirs"],
            self.config["max_cache_age_days"]
        ))
        
        self.register_task(UpdateDependenciesTask(
            self.config["dependency_check_only"]
        ))
        
        if self.config["security_scan_enabled"]:
            self.register_task(SecurityScanTask())
        
        self.register_task(ModelCleanupTask(
            self.config["model_cache_dir"],
            self.config["keep_recent_models"]
        ))
        
        self.register_task(DatabaseMaintenanceTask())
        
        self.register_task(LogRotationTask(
            self.config["log_dirs"],
            self.config["max_log_age_days"]
        ))
    
    async def run_task(self, task_name: str) -> bool:
        """Run a specific task"""
        if task_name not in self.tasks:
            logger.error(f"Task not found: {task_name}")
            return False
        
        return await self.tasks[task_name].run()
    
    async def run_all_tasks(self, parallel: bool = False) -> Dict[str, bool]:
        """Run all registered tasks"""
        results = {}
        
        if parallel:
            # Run tasks in parallel
            tasks = [self.tasks[name].run() for name in self.tasks]
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, (name, result) in enumerate(zip(self.tasks.keys(), task_results)):
                if isinstance(result, Exception):
                    logger.error(f"Task {name} failed with exception: {result}")
                    results[name] = False
                else:
                    results[name] = result
        else:
            # Run tasks sequentially
            for name, task in self.tasks.items():
                results[name] = await task.run()
        
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate maintenance report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tasks": len(self.tasks),
            "tasks": {}
        }
        
        for name, task in self.tasks.items():
            report["tasks"][name] = {
                "description": task.description,
                "status": task.status,
                "last_run": task.last_run.isoformat() if task.last_run else None,
                "duration_seconds": task.duration,
                "error": task.error
            }
        
        return report


async def main():
    """Main maintenance script"""
    parser = argparse.ArgumentParser(description="LM Watermark Lab Maintenance")
    parser.add_argument("--task", help="Run specific task")
    parser.add_argument("--list", action="store_true", help="List available tasks")
    parser.add_argument("--parallel", action="store_true", help="Run tasks in parallel")
    parser.add_argument("--config", help="Path to maintenance config file")
    parser.add_argument("--report", help="Output file for maintenance report")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    
    args = parser.parse_args()
    
    # Initialize maintenance manager
    manager = MaintenanceManager()
    manager.register_default_tasks()
    
    if args.list:
        print("Available maintenance tasks:")
        for name, task in manager.tasks.items():
            print(f"  {name}: {task.description}")
        return
    
    if args.dry_run:
        print("Dry run mode - showing registered tasks:")
        for name, task in manager.tasks.items():
            print(f"  Would run: {name} - {task.description}")
        return
    
    # Run maintenance tasks
    if args.task:
        success = await manager.run_task(args.task)
        sys.exit(0 if success else 1)
    else:
        results = await manager.run_all_tasks(args.parallel)
        
        # Generate report
        report = manager.generate_report()
        
        if args.report:
            with open(args.report, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Maintenance report written to {args.report}")
        
        # Print summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        print(f"\nMaintenance Summary:")
        print(f"  Successful: {successful}/{total}")
        print(f"  Failed: {total - successful}/{total}")
        
        if successful < total:
            print("\nFailed tasks:")
            for name, success in results.items():
                if not success:
                    task = manager.tasks[name]
                    print(f"  {name}: {task.error or 'Unknown error'}")
        
        sys.exit(0 if successful == total else 1)


if __name__ == "__main__":
    asyncio.run(main())