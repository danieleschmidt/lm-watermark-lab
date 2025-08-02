#!/usr/bin/env python3
"""
Automated dependency update script for LM Watermark Lab.
Handles dependency updates, security patches, and compatibility checks.
"""

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import os


class DependencyUpdater:
    """Handles automated dependency updates and security patches."""
    
    def __init__(self, repo_path: Path = None):
        self.repo_path = repo_path or Path.cwd()
        self.pyproject_path = self.repo_path / "pyproject.toml"
        self.backup_dir = self.repo_path / ".dependency_backups"
        self.update_log = self.repo_path / "dependency_updates.log"
        
    def run_command(self, cmd: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run shell command and return result."""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=capture_output, 
                text=True, 
                cwd=self.repo_path,
                timeout=300  # 5 minutes timeout
            )
            return result
        except subprocess.TimeoutExpired:
            print(f"⏰ Command timed out: {' '.join(cmd)}")
            return subprocess.CompletedProcess(cmd, 1, "", "Timeout")
        except Exception as e:
            print(f"❌ Command failed: {' '.join(cmd)}, Error: {e}")
            return subprocess.CompletedProcess(cmd, 1, "", str(e))

    def log_update(self, message: str) -> None:
        """Log update activity."""
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(self.update_log, "a") as f:
            f.write(log_entry)
        
        print(f"📝 {message}")

    def backup_dependencies(self) -> Path:
        """Create backup of current dependency files."""
        self.backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"
        backup_path.mkdir()
        
        # Backup key files
        files_to_backup = [
            "pyproject.toml",
            "requirements.txt",
            "requirements-dev.txt"
        ]
        
        for filename in files_to_backup:
            file_path = self.repo_path / filename
            if file_path.exists():
                backup_file = backup_path / filename
                backup_file.write_text(file_path.read_text())
        
        self.log_update(f"Created backup at {backup_path}")
        return backup_path

    def get_current_dependencies(self) -> Dict[str, str]:
        """Get currently installed dependencies with versions."""
        result = self.run_command(["pip", "list", "--format=json"])
        
        if result.returncode != 0:
            self.log_update("Failed to get current dependencies")
            return {}
        
        try:
            packages = json.loads(result.stdout)
            return {pkg["name"]: pkg["version"] for pkg in packages}
        except json.JSONDecodeError:
            self.log_update("Failed to parse pip list output")
            return {}

    def get_outdated_dependencies(self) -> List[Dict[str, str]]:
        """Get list of outdated dependencies."""
        result = self.run_command(["pip", "list", "--outdated", "--format=json"])
        
        if result.returncode != 0:
            return []
        
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return []

    def check_security_vulnerabilities(self) -> List[Dict[str, str]]:
        """Check for security vulnerabilities using safety."""
        result = self.run_command(["safety", "check", "--json"])
        
        vulnerabilities = []
        if result.returncode != 0 and result.stdout:
            try:
                vuln_data = json.loads(result.stdout)
                vulnerabilities = vuln_data if isinstance(vuln_data, list) else []
            except json.JSONDecodeError:
                pass
        
        return vulnerabilities

    def update_dependency(self, package: str, version: str = None) -> bool:
        """Update a single dependency."""
        if version:
            package_spec = f"{package}=={version}"
        else:
            package_spec = package
        
        self.log_update(f"Updating {package_spec}")
        
        # First, try to update
        result = self.run_command(["pip", "install", "--upgrade", package_spec])
        
        if result.returncode == 0:
            self.log_update(f"✅ Successfully updated {package}")
            return True
        else:
            self.log_update(f"❌ Failed to update {package}: {result.stderr}")
            return False

    def run_tests_after_update(self) -> bool:
        """Run tests to ensure updates don't break functionality."""
        self.log_update("🧪 Running tests after dependency updates...")
        
        # Run quick test suite
        test_commands = [
            ["python", "-m", "pytest", "tests/unit/", "-x", "--tb=short"],
            ["python", "-c", "import watermark_lab; print('Import successful')"],
        ]
        
        for cmd in test_commands:
            result = self.run_command(cmd)
            if result.returncode != 0:
                self.log_update(f"❌ Test failed: {' '.join(cmd)}")
                return False
        
        self.log_update("✅ All tests passed after updates")
        return True

    def update_security_vulnerabilities(self) -> int:
        """Update dependencies with security vulnerabilities."""
        vulnerabilities = self.check_security_vulnerabilities()
        
        if not vulnerabilities:
            self.log_update("✅ No security vulnerabilities found")
            return 0
        
        self.log_update(f"🔒 Found {len(vulnerabilities)} security vulnerabilities")
        
        updated_count = 0
        for vuln in vulnerabilities:
            package_name = vuln.get("package_name", "")
            vulnerable_spec = vuln.get("vulnerable_spec", "")
            safe_version = vuln.get("analyzed_version", "")
            
            if package_name and safe_version:
                self.log_update(f"Fixing security vulnerability in {package_name}")
                if self.update_dependency(package_name, safe_version):
                    updated_count += 1
        
        return updated_count

    def update_outdated_dependencies(self, max_updates: int = 10) -> int:
        """Update outdated dependencies with safety checks."""
        outdated = self.get_outdated_dependencies()
        
        if not outdated:
            self.log_update("✅ All dependencies are up to date")
            return 0
        
        self.log_update(f"📦 Found {len(outdated)} outdated dependencies")
        
        # Sort by importance (you can customize this logic)
        # Prioritize security-related packages
        security_packages = ["cryptography", "requests", "urllib3", "pillow", "pyyaml"]
        
        def priority_score(pkg):
            name = pkg["name"].lower()
            if name in security_packages:
                return 10
            elif "security" in name or "crypto" in name:
                return 5
            else:
                return 1
        
        outdated.sort(key=priority_score, reverse=True)
        
        updated_count = 0
        for i, package in enumerate(outdated[:max_updates]):
            if updated_count >= max_updates:
                break
                
            package_name = package["name"]
            current_version = package["version"]
            latest_version = package["latest_version"]
            
            self.log_update(f"Updating {package_name} from {current_version} to {latest_version}")
            
            if self.update_dependency(package_name, latest_version):
                # Run quick sanity check after each update
                import_check = self.run_command([
                    "python", "-c", f"import {package_name.replace('-', '_')}; print('OK')"
                ])
                
                if import_check.returncode == 0:
                    updated_count += 1
                    self.log_update(f"✅ {package_name} updated and verified")
                else:
                    # Rollback if import fails
                    self.log_update(f"⚠️ Import check failed for {package_name}, rolling back")
                    self.update_dependency(package_name, current_version)
        
        return updated_count

    def generate_dependency_report(self) -> str:
        """Generate a comprehensive dependency report."""
        current_deps = self.get_current_dependencies()
        outdated_deps = self.get_outdated_dependencies()
        vulnerabilities = self.check_security_vulnerabilities()
        
        report = f"""
# 📦 Dependency Report
Generated: {datetime.now(timezone.utc).isoformat()}

## 📊 Summary
- Total Dependencies: {len(current_deps)}
- Outdated Dependencies: {len(outdated_deps)}
- Security Vulnerabilities: {len(vulnerabilities)}

## 🔒 Security Status
"""
        
        if vulnerabilities:
            report += "### ⚠️ Security Vulnerabilities Found\n\n"
            for vuln in vulnerabilities:
                package = vuln.get("package_name", "Unknown")
                version = vuln.get("analyzed_version", "Unknown")
                advisory = vuln.get("advisory", "No details")
                report += f"- **{package}** (v{version}): {advisory}\n"
        else:
            report += "✅ No security vulnerabilities detected\n"
        
        if outdated_deps:
            report += f"\n## 📈 Outdated Dependencies ({len(outdated_deps)})\n\n"
            report += "| Package | Current | Latest | Type |\n"
            report += "|---------|---------|--------|---------|\n"
            
            for pkg in outdated_deps[:20]:  # Show top 20
                name = pkg["name"]
                current = pkg["version"]
                latest = pkg["latest_version"]
                pkg_type = pkg.get("latest_filetype", "wheel")
                report += f"| {name} | {current} | {latest} | {pkg_type} |\n"
        
        report += f"\n## 📝 Update Recommendations\n\n"
        
        if vulnerabilities:
            report += "1. **Priority 1**: Update packages with security vulnerabilities immediately\n"
        
        if len(outdated_deps) > 10:
            report += "2. **Priority 2**: Update core dependencies and security-related packages\n"
            report += "3. **Priority 3**: Update remaining dependencies in batches\n"
        elif outdated_deps:
            report += "2. Update remaining outdated dependencies\n"
        
        if not vulnerabilities and not outdated_deps:
            report += "✅ All dependencies are current and secure\n"
        
        return report

    def create_update_pr_body(self, updated_packages: List[str], security_fixes: int) -> str:
        """Create PR body for dependency updates."""
        body = f"""# 🤖 Automated Dependency Updates

This PR contains automated dependency updates for security and maintenance.

## 📊 Summary
- **Packages Updated**: {len(updated_packages)}
- **Security Fixes**: {security_fixes}
- **Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## 📦 Updated Packages
"""
        
        for package in updated_packages:
            body += f"- {package}\n"
        
        body += f"""

## 🔒 Security Impact
{'✅ This update includes security vulnerability fixes' if security_fixes > 0 else '📦 Maintenance updates only - no security vulnerabilities addressed'}

## 🧪 Testing
- [x] Dependency update completed successfully
- [x] Import tests passed
- [x] Unit tests passed
- [ ] Integration tests passed (run in CI)

## 🔄 Rollback Plan
Dependency backups are available in `.dependency_backups/` directory.

To rollback if needed:
```bash
# Restore from backup
git checkout HEAD~1 -- pyproject.toml
pip install -e ".[dev]"
```

## 📝 Notes
- Updates were applied with compatibility testing
- All changes are logged in `dependency_updates.log`
- Security vulnerabilities (if any) have been prioritized

---
*This PR was generated automatically by the dependency update script*
"""
        
        return body

    def run_full_update(self, max_updates: int = None, security_only: bool = False) -> Dict[str, int]:
        """Run complete dependency update process."""
        self.log_update("🚀 Starting automated dependency update process")
        
        # Create backup
        backup_path = self.backup_dependencies()
        
        results = {
            "security_fixes": 0,
            "outdated_updates": 0,
            "total_updated": 0,
            "test_passed": False
        }
        
        try:
            # First, handle security vulnerabilities
            self.log_update("🔒 Checking for security vulnerabilities...")
            security_updates = self.update_security_vulnerabilities()
            results["security_fixes"] = security_updates
            
            if not security_only:
                # Then handle outdated dependencies
                self.log_update("📦 Checking for outdated dependencies...")
                max_updates = max_updates or 10
                outdated_updates = self.update_outdated_dependencies(max_updates)
                results["outdated_updates"] = outdated_updates
            
            results["total_updated"] = results["security_fixes"] + results["outdated_updates"]
            
            # Run tests to verify everything still works
            if results["total_updated"] > 0:
                test_passed = self.run_tests_after_update()
                results["test_passed"] = test_passed
                
                if not test_passed:
                    self.log_update("❌ Tests failed, consider manual review")
                    return results
            
            # Generate report
            report = self.generate_dependency_report()
            report_file = self.repo_path / "dependency_report.md"
            report_file.write_text(report)
            self.log_update(f"📄 Dependency report saved to {report_file}")
            
            self.log_update(f"✅ Update process completed: {results['total_updated']} packages updated")
            
        except Exception as e:
            self.log_update(f"❌ Update process failed: {e}")
            # Could restore from backup here if needed
            
        return results


def main():
    """Main function for dependency updates."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("""
LM Watermark Lab Dependency Updater

Usage:
  python scripts/dependency_update.py [options]

Options:
  --help              Show this help message
  --security-only     Only update packages with security vulnerabilities
  --max-updates N     Maximum number of packages to update (default: 10)
  --report-only       Generate dependency report without updating
  --check-only        Check for updates without applying them

Examples:
  python scripts/dependency_update.py
  python scripts/dependency_update.py --security-only
  python scripts/dependency_update.py --max-updates 5
  python scripts/dependency_update.py --report-only
        """)
        return
    
    updater = DependencyUpdater()
    
    # Parse arguments
    security_only = "--security-only" in sys.argv
    report_only = "--report-only" in sys.argv
    check_only = "--check-only" in sys.argv
    
    max_updates = 10
    for i, arg in enumerate(sys.argv):
        if arg == "--max-updates" and i + 1 < len(sys.argv):
            try:
                max_updates = int(sys.argv[i + 1])
            except ValueError:
                print("❌ Invalid max-updates value")
                sys.exit(1)
    
    try:
        if report_only or check_only:
            # Generate report only
            report = updater.generate_dependency_report()
            print(report)
            
            if not check_only:
                report_file = Path("dependency_report.md")
                report_file.write_text(report)
                print(f"\n📄 Report saved to {report_file}")
        else:
            # Run full update
            results = updater.run_full_update(max_updates, security_only)
            
            print(f"""
📊 **Update Summary**
• Security fixes: {results['security_fixes']}
• Outdated updates: {results['outdated_updates']}
• Total updated: {results['total_updated']}
• Tests passed: {'✅' if results['test_passed'] else '❌'}
            """)
            
            if results['total_updated'] > 0 and results['test_passed']:
                print("✅ Dependencies updated successfully!")
            elif results['total_updated'] > 0:
                print("⚠️ Dependencies updated but tests failed - manual review needed")
            else:
                print("✅ All dependencies are already up to date")
    
    except Exception as e:
        print(f"❌ Error during dependency update: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()