#!/usr/bin/env python3
"""
Repository maintenance automation script for LM Watermark Lab.
Handles routine maintenance tasks, cleanup, and health monitoring.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import re


class RepositoryMaintainer:
    """Handles automated repository maintenance tasks."""
    
    def __init__(self, repo_path: Path = None):
        self.repo_path = repo_path or Path.cwd()
        self.maintenance_log = self.repo_path / "maintenance.log"
        self.config_file = self.repo_path / ".maintenance_config.json"
        self.load_config()
    
    def load_config(self):
        """Load maintenance configuration."""
        default_config = {
            "cleanup": {
                "max_log_age_days": 30,
                "max_cache_size_mb": 1000,
                "clean_temp_files": True,
                "clean_build_artifacts": True
            },
            "monitoring": {
                "check_disk_usage": True,
                "disk_usage_threshold": 90,
                "check_large_files": True,
                "large_file_threshold_mb": 100
            },
            "git": {
                "prune_branches": True,
                "keep_recent_branches_days": 30,
                "gc_aggressive": False
            },
            "alerts": {
                "disk_usage_critical": 95,
                "large_repo_size_gb": 5,
                "stale_branch_days": 60
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    self.config = {**default_config, **user_config}
            except (json.JSONDecodeError, FileNotFoundError):
                self.config = default_config
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """Save maintenance configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def log_maintenance(self, message: str, level: str = "INFO"):
        """Log maintenance activity."""
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        with open(self.maintenance_log, "a") as f:
            f.write(log_entry)
        
        print(f"📝 [{level}] {message}")
    
    def run_command(self, cmd: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run shell command and return result."""
        try:
            return subprocess.run(
                cmd, 
                capture_output=capture_output, 
                text=True, 
                cwd=self.repo_path,
                timeout=300
            )
        except subprocess.TimeoutExpired:
            self.log_maintenance(f"Command timed out: {' '.join(cmd)}", "ERROR")
            return subprocess.CompletedProcess(cmd, 1, "", "Timeout")
        except Exception as e:
            self.log_maintenance(f"Command failed: {' '.join(cmd)}, Error: {e}", "ERROR")
            return subprocess.CompletedProcess(cmd, 1, "", str(e))
    
    def get_directory_size(self, path: Path) -> int:
        """Get directory size in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath) and not os.path.islink(filepath):
                        total_size += os.path.getsize(filepath)
        except OSError:
            pass
        return total_size
    
    def clean_temporary_files(self) -> Dict[str, Any]:
        """Clean temporary files and caches."""
        self.log_maintenance("🧹 Cleaning temporary files and caches")
        
        results = {
            "files_removed": 0,
            "space_freed_mb": 0,
            "directories_cleaned": []
        }
        
        # Directories and patterns to clean
        cleanup_patterns = [
            (".pytest_cache", "**/*"),
            ("__pycache__", "**/*"),
            (".mypy_cache", "**/*"),
            (".ruff_cache", "**/*"),
            ("build", "**/*"),
            ("dist", "**/*"),
            (".tox", "**/*"),
            ("htmlcov", "**/*"),
            ("node_modules", "**/*"),
            (".coverage*", ""),
            ("*.pyc", ""),
            ("*.pyo", ""),
            ("*.egg-info", "**/*"),
            ("*~", ""),
            ("*.tmp", ""),
            ("*.temp", "")
        ]
        
        initial_size = self.get_directory_size(self.repo_path)
        
        for pattern, subpattern in cleanup_patterns:
            if subpattern:
                # Directory cleanup
                for item in self.repo_path.rglob(pattern):
                    if item.is_dir():
                        try:
                            size_before = self.get_directory_size(item)
                            shutil.rmtree(item)
                            results["files_removed"] += 1
                            results["space_freed_mb"] += size_before / (1024 * 1024)
                            results["directories_cleaned"].append(str(item))
                        except OSError:
                            pass
            else:
                # File cleanup
                for item in self.repo_path.rglob(pattern):
                    if item.is_file():
                        try:
                            size_before = item.stat().st_size
                            item.unlink()
                            results["files_removed"] += 1
                            results["space_freed_mb"] += size_before / (1024 * 1024)
                        except OSError:
                            pass
        
        results["space_freed_mb"] = round(results["space_freed_mb"], 2)
        
        self.log_maintenance(
            f"Cleaned {results['files_removed']} items, freed {results['space_freed_mb']} MB"
        )
        
        return results
    
    def clean_old_logs(self) -> Dict[str, Any]:
        """Clean old log files."""
        self.log_maintenance("📋 Cleaning old log files")
        
        results = {"files_removed": 0, "space_freed_mb": 0}
        max_age = self.config["cleanup"]["max_log_age_days"]
        cutoff_date = datetime.now() - timedelta(days=max_age)
        
        log_patterns = ["*.log", "*.log.*", "logs/**/*.log"]
        
        for pattern in log_patterns:
            for log_file in self.repo_path.rglob(pattern):
                if log_file.is_file():
                    try:
                        file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                        if file_time < cutoff_date:
                            size = log_file.stat().st_size
                            log_file.unlink()
                            results["files_removed"] += 1
                            results["space_freed_mb"] += size / (1024 * 1024)
                    except OSError:
                        pass
        
        results["space_freed_mb"] = round(results["space_freed_mb"], 2)
        
        self.log_maintenance(
            f"Removed {results['files_removed']} old log files, freed {results['space_freed_mb']} MB"
        )
        
        return results
    
    def check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage and alert if necessary."""
        self.log_maintenance("💾 Checking disk usage")
        
        try:
            # Get disk usage for repository directory
            statvfs = os.statvfs(self.repo_path)
            total_space = statvfs.f_frsize * statvfs.f_blocks
            free_space = statvfs.f_frsize * statvfs.f_available
            used_space = total_space - free_space
            usage_percent = (used_space / total_space) * 100
            
            results = {
                "total_gb": round(total_space / (1024**3), 2),
                "used_gb": round(used_space / (1024**3), 2),
                "free_gb": round(free_space / (1024**3), 2),
                "usage_percent": round(usage_percent, 2),
                "alert_level": "normal"
            }
            
            # Determine alert level
            if usage_percent >= self.config["alerts"]["disk_usage_critical"]:
                results["alert_level"] = "critical"
                self.log_maintenance(f"⚠️ CRITICAL: Disk usage at {usage_percent:.1f}%", "CRITICAL")
            elif usage_percent >= self.config["monitoring"]["disk_usage_threshold"]:
                results["alert_level"] = "warning"
                self.log_maintenance(f"⚠️ WARNING: Disk usage at {usage_percent:.1f}%", "WARNING")
            else:
                self.log_maintenance(f"✅ Disk usage normal: {usage_percent:.1f}%")
            
            return results
            
        except OSError as e:
            self.log_maintenance(f"Failed to check disk usage: {e}", "ERROR")
            return {"error": str(e)}
    
    def find_large_files(self) -> List[Dict[str, Any]]:
        """Find large files that might need attention."""
        self.log_maintenance("🔍 Scanning for large files")
        
        large_files = []
        threshold_bytes = self.config["monitoring"]["large_file_threshold_mb"] * 1024 * 1024
        
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and not file_path.is_symlink():
                try:
                    size = file_path.stat().st_size
                    if size > threshold_bytes:
                        large_files.append({
                            "path": str(file_path.relative_to(self.repo_path)),
                            "size_mb": round(size / (1024 * 1024), 2),
                            "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        })
                except OSError:
                    continue
        
        # Sort by size, largest first
        large_files.sort(key=lambda x: x["size_mb"], reverse=True)
        
        if large_files:
            self.log_maintenance(f"Found {len(large_files)} large files")
            for file_info in large_files[:5]:  # Log top 5
                self.log_maintenance(f"  {file_info['path']}: {file_info['size_mb']} MB")
        else:
            self.log_maintenance("No unusually large files found")
        
        return large_files
    
    def git_maintenance(self) -> Dict[str, Any]:
        """Perform Git maintenance tasks."""
        self.log_maintenance("🔄 Running Git maintenance")
        
        results = {
            "gc_run": False,
            "branches_pruned": 0,
            "remotes_pruned": False,
            "size_before_mb": 0,
            "size_after_mb": 0
        }
        
        # Get .git directory size before
        git_dir = self.repo_path / ".git"
        if git_dir.exists():
            results["size_before_mb"] = round(self.get_directory_size(git_dir) / (1024 * 1024), 2)
        
        # Git garbage collection
        gc_cmd = ["git", "gc"]
        if self.config["git"]["gc_aggressive"]:
            gc_cmd.append("--aggressive")
        
        gc_result = self.run_command(gc_cmd)
        if gc_result.returncode == 0:
            results["gc_run"] = True
            self.log_maintenance("✅ Git garbage collection completed")
        else:
            self.log_maintenance("❌ Git garbage collection failed", "ERROR")
        
        # Prune remote tracking branches
        prune_result = self.run_command(["git", "remote", "prune", "origin"])
        if prune_result.returncode == 0:
            results["remotes_pruned"] = True
            self.log_maintenance("✅ Remote branches pruned")
        
        # Clean up old local branches (if configured)
        if self.config["git"]["prune_branches"]:
            cutoff_days = self.config["git"]["keep_recent_branches_days"]
            cutoff_date = datetime.now() - timedelta(days=cutoff_days)
            
            # Get merged branches
            merged_result = self.run_command(["git", "branch", "--merged", "main"])
            if merged_result.returncode == 0:
                branches = [
                    line.strip().lstrip("* ") 
                    for line in merged_result.stdout.strip().split('\n')
                    if line.strip() and not line.strip().startswith("* main")
                ]
                
                for branch in branches:
                    if branch and branch != "main":
                        # Check branch age
                        branch_date_result = self.run_command([
                            "git", "log", "-1", "--format=%at", branch
                        ])
                        
                        if branch_date_result.returncode == 0:
                            try:
                                branch_timestamp = int(branch_date_result.stdout.strip())
                                branch_date = datetime.fromtimestamp(branch_timestamp)
                                
                                if branch_date < cutoff_date:
                                    # Delete old merged branch
                                    delete_result = self.run_command(["git", "branch", "-d", branch])
                                    if delete_result.returncode == 0:
                                        results["branches_pruned"] += 1
                                        self.log_maintenance(f"Deleted old branch: {branch}")
                            except (ValueError, OSError):
                                pass
        
        # Get .git directory size after
        if git_dir.exists():
            results["size_after_mb"] = round(self.get_directory_size(git_dir) / (1024 * 1024), 2)
            space_saved = results["size_before_mb"] - results["size_after_mb"]
            if space_saved > 0:
                self.log_maintenance(f"Git maintenance freed {space_saved:.2f} MB")
        
        return results
    
    def check_repository_health(self) -> Dict[str, Any]:
        """Check overall repository health."""
        self.log_maintenance("🏥 Checking repository health")
        
        health = {
            "status": "healthy",
            "issues": [],
            "warnings": [],
            "repo_size_gb": 0,
            "commit_count": 0,
            "branch_count": 0,
            "file_count": 0
        }
        
        # Repository size
        repo_size = self.get_directory_size(self.repo_path)
        health["repo_size_gb"] = round(repo_size / (1024**3), 2)
        
        if health["repo_size_gb"] > self.config["alerts"]["large_repo_size_gb"]:
            health["warnings"].append(f"Repository size is large: {health['repo_size_gb']} GB")
        
        # Git statistics
        commit_result = self.run_command(["git", "rev-list", "--all", "--count"])
        if commit_result.returncode == 0:
            try:
                health["commit_count"] = int(commit_result.stdout.strip())
            except ValueError:
                pass
        
        branch_result = self.run_command(["git", "branch", "-a"])
        if branch_result.returncode == 0:
            health["branch_count"] = len([
                line for line in branch_result.stdout.split('\n') 
                if line.strip() and not line.strip().startswith('*')
            ])
        
        # File count
        file_count = 0
        for item in self.repo_path.rglob("*"):
            if item.is_file():
                file_count += 1
        health["file_count"] = file_count
        
        # Check for stale branches
        stale_days = self.config["alerts"]["stale_branch_days"]
        stale_cutoff = datetime.now() - timedelta(days=stale_days)
        
        branch_list_result = self.run_command(["git", "for-each-ref", "--format=%(refname:short) %(committerdate)", "refs/heads/"])
        if branch_list_result.returncode == 0:
            stale_branches = []
            for line in branch_list_result.stdout.strip().split('\n'):
                if line:
                    parts = line.rsplit(' ', 5)  # Split on last 5 spaces for date
                    if len(parts) >= 6:
                        branch_name = parts[0]
                        try:
                            # Parse date (assuming ISO format or similar)
                            date_str = ' '.join(parts[1:])
                            # This is a simplified check - you might need more sophisticated date parsing
                            if 'main' not in branch_name and 'master' not in branch_name:
                                stale_branches.append(branch_name)
                        except Exception:
                            pass
            
            if stale_branches:
                health["warnings"].append(f"Found {len(stale_branches)} potentially stale branches")
        
        # Overall health status
        if health["issues"]:
            health["status"] = "unhealthy"
        elif health["warnings"]:
            health["status"] = "warning"
        
        self.log_maintenance(f"Repository health: {health['status']}")
        
        return health
    
    def generate_maintenance_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive maintenance report."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        
        report = f"""# 🛠️ Repository Maintenance Report
Generated: {timestamp}

## 📊 Summary
- **Overall Status**: {results.get('health', {}).get('status', 'unknown').title()}
- **Files Cleaned**: {results.get('cleanup', {}).get('files_removed', 0)}
- **Space Freed**: {results.get('cleanup', {}).get('space_freed_mb', 0):.2f} MB
- **Git Branches Pruned**: {results.get('git', {}).get('branches_pruned', 0)}

## 💾 Disk Usage
"""
        
        disk_info = results.get('disk_usage', {})
        if disk_info and 'usage_percent' in disk_info:
            report += f"""- **Total Space**: {disk_info['total_gb']} GB
- **Used Space**: {disk_info['used_gb']} GB ({disk_info['usage_percent']:.1f}%)
- **Free Space**: {disk_info['free_gb']} GB
- **Alert Level**: {disk_info['alert_level'].title()}

"""
        
        # Large files
        large_files = results.get('large_files', [])
        if large_files:
            report += f"## 📁 Large Files ({len(large_files)} found)\n\n"
            report += "| File | Size (MB) | Last Modified |\n"
            report += "|------|-----------|---------------|\n"
            
            for file_info in large_files[:10]:  # Show top 10
                report += f"| {file_info['path']} | {file_info['size_mb']:.1f} | {file_info['modified'][:10]} |\n"
        
        # Repository health
        health = results.get('health', {})
        if health:
            report += f"""
## 🏥 Repository Health
- **Repository Size**: {health.get('repo_size_gb', 0):.2f} GB
- **Total Commits**: {health.get('commit_count', 0):,}
- **Active Branches**: {health.get('branch_count', 0)}
- **Total Files**: {health.get('file_count', 0):,}

"""
            
            if health.get('warnings'):
                report += "### ⚠️ Warnings\n"
                for warning in health['warnings']:
                    report += f"- {warning}\n"
                report += "\n"
            
            if health.get('issues'):
                report += "### ❌ Issues\n"
                for issue in health['issues']:
                    report += f"- {issue}\n"
                report += "\n"
        
        report += """## 🔄 Next Steps
1. Review large files and consider archiving or removing unused assets  
2. Monitor disk usage trends and plan for capacity expansion if needed
3. Review stale branches and merge or delete as appropriate
4. Schedule regular maintenance runs (weekly recommended)

---
*Report generated automatically by repository maintenance script*
"""
        
        return report
    
    def run_full_maintenance(self, skip_cleanup: bool = False) -> Dict[str, Any]:
        """Run complete maintenance routine."""
        self.log_maintenance("🚀 Starting full repository maintenance")
        
        results = {}
        
        try:
            # Repository health check
            results['health'] = self.check_repository_health()
            
            # Disk usage check
            results['disk_usage'] = self.check_disk_usage()
            
            # Find large files
            results['large_files'] = self.find_large_files()
            
            if not skip_cleanup:
                # Cleanup tasks
                cleanup_results = self.clean_temporary_files()
                log_cleanup_results = self.clean_old_logs()
                
                results['cleanup'] = {
                    'files_removed': cleanup_results['files_removed'] + log_cleanup_results['files_removed'],
                    'space_freed_mb': cleanup_results['space_freed_mb'] + log_cleanup_results['space_freed_mb'],
                    'directories_cleaned': cleanup_results['directories_cleaned']
                }
                
                # Git maintenance
                results['git'] = self.git_maintenance()
            
            # Generate report
            report = self.generate_maintenance_report(results)
            report_file = self.repo_path / "maintenance_report.md"
            report_file.write_text(report)
            
            self.log_maintenance(f"✅ Maintenance completed. Report saved to {report_file}")
            
        except Exception as e:
            self.log_maintenance(f"❌ Maintenance failed: {e}", "ERROR")
            results['error'] = str(e)
        
        return results


def main():
    """Main function for repository maintenance."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("""
LM Watermark Lab Repository Maintenance

Usage:
  python scripts/repository_maintenance.py [options]

Options:
  --help            Show this help message
  --check-only      Run health checks without cleanup
  --cleanup-only    Run cleanup tasks only
  --git-only        Run Git maintenance only
  --report          Generate maintenance report
  --config          Show current configuration

Examples:
  python scripts/repository_maintenance.py
  python scripts/repository_maintenance.py --check-only
  python scripts/repository_maintenance.py --cleanup-only
        """)
        return
    
    maintainer = RepositoryMaintainer()
    
    # Parse command line arguments
    check_only = "--check-only" in sys.argv
    cleanup_only = "--cleanup-only" in sys.argv
    git_only = "--git-only" in sys.argv
    report_only = "--report" in sys.argv
    show_config = "--config" in sys.argv
    
    if show_config:
        print("📋 Current Maintenance Configuration:")
        print(json.dumps(maintainer.config, indent=2))
        return
    
    try:
        if cleanup_only:
            # Run cleanup tasks only
            cleanup_results = maintainer.clean_temporary_files()
            log_results = maintainer.clean_old_logs()
            print(f"✅ Cleanup completed: {cleanup_results['files_removed'] + log_results['files_removed']} files removed")
            
        elif git_only:
            # Run Git maintenance only
            git_results = maintainer.git_maintenance()
            print(f"✅ Git maintenance completed: {git_results['branches_pruned']} branches pruned")
            
        elif check_only or report_only:
            # Health check and report only
            health = maintainer.check_repository_health()
            disk_usage = maintainer.check_disk_usage()
            large_files = maintainer.find_large_files()
            
            results = {
                'health': health,
                'disk_usage': disk_usage,
                'large_files': large_files
            }
            
            report = maintainer.generate_maintenance_report(results)
            print(report)
            
            if report_only:
                report_file = Path("maintenance_report.md")
                report_file.write_text(report)
                print(f"\n📄 Report saved to {report_file}")
        else:
            # Run full maintenance
            results = maintainer.run_full_maintenance()
            
            health_status = results.get('health', {}).get('status', 'unknown')
            files_cleaned = results.get('cleanup', {}).get('files_removed', 0)
            space_freed = results.get('cleanup', {}).get('space_freed_mb', 0)
            
            print(f"""
📊 **Maintenance Summary**
• Repository Status: {health_status.title()}
• Files Cleaned: {files_cleaned:,}
• Space Freed: {space_freed:.2f} MB
• Report: maintenance_report.md
            """)
            
            if health_status == "unhealthy":
                print("⚠️ Repository has issues that need attention")
                sys.exit(1)
            elif health_status == "warning":
                print("⚠️ Repository has warnings - review recommended")
            else:
                print("✅ Repository maintenance completed successfully")
    
    except Exception as e:
        print(f"❌ Maintenance failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()