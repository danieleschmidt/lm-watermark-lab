#!/usr/bin/env python3
"""
Automated metrics collection script for LM Watermark Lab.
Collects comprehensive project metrics and updates project-metrics.json.
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import os
import re


class MetricsCollector:
    """Collects various project metrics for SDLC monitoring."""
    
    def __init__(self, repo_path: Path = None):
        self.repo_path = repo_path or Path.cwd()
        self.metrics_file = self.repo_path / ".github" / "project-metrics.json"
        self.src_path = self.repo_path / "src"
        self.test_path = self.repo_path / "tests"
        
    def run_command(self, cmd: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run shell command and return result."""
        try:
            return subprocess.run(
                cmd, 
                capture_output=capture_output, 
                text=True, 
                cwd=self.repo_path,
                timeout=60
            )
        except subprocess.TimeoutExpired:
            print(f"Command timed out: {' '.join(cmd)}")
            return subprocess.CompletedProcess(cmd, 1, "", "Timeout")
        except Exception as e:
            print(f"Command failed: {' '.join(cmd)}, Error: {e}")
            return subprocess.CompletedProcess(cmd, 1, "", str(e))

    def collect_codebase_metrics(self) -> Dict[str, Any]:
        """Collect codebase statistics."""
        metrics = {
            "lines_of_code": {},
            "files": {},
            "complexity": {}
        }
        
        # Count lines of code by file type
        file_extensions = {
            "python": ["*.py"],
            "yaml": ["*.yml", "*.yaml"],
            "markdown": ["*.md"],
            "dockerfile": ["Dockerfile*", "*.dockerfile"],
            "shell": ["*.sh", "*.bash"]
        }
        
        total_lines = 0
        total_files = 0
        
        for file_type, patterns in file_extensions.items():
            lines = 0
            files = 0
            for pattern in patterns:
                # Find files matching pattern
                result = self.run_command(["find", ".", "-name", pattern, "-type", "f"])
                if result.returncode == 0 and result.stdout.strip():
                    file_list = result.stdout.strip().split('\n')
                    files += len(file_list)
                    
                    # Count lines in each file
                    for file_path in file_list:
                        if os.path.exists(file_path):
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    lines += len(f.readlines())
                            except Exception:
                                continue
            
            metrics["lines_of_code"][file_type] = lines
            total_lines += lines
            total_files += files
        
        metrics["lines_of_code"]["total"] = total_lines
        
        # Count different types of files
        metrics["files"] = {
            "total": total_files,
            "source_files": len(list(self.src_path.rglob("*.py"))) if self.src_path.exists() else 0,
            "test_files": len(list(self.test_path.rglob("test_*.py"))) if self.test_path.exists() else 0,
            "config_files": len(list(Path(".").glob("*.yml"))) + len(list(Path(".").glob("*.yaml"))) + len(list(Path(".").glob("*.toml"))),
            "docs_files": len(list(Path(".").rglob("*.md")))
        }
        
        # Complexity metrics (basic implementation)
        if self.src_path.exists():
            # Use radon for complexity if available
            radon_result = self.run_command(["radon", "cc", str(self.src_path), "-j"])
            if radon_result.returncode == 0:
                try:
                    complexity_data = json.loads(radon_result.stdout)
                    total_complexity = 0
                    function_count = 0
                    
                    for file_data in complexity_data.values():
                        for item in file_data:
                            if item.get('type') in ['function', 'method']:
                                total_complexity += item.get('complexity', 0)
                                function_count += 1
                    
                    avg_complexity = total_complexity / function_count if function_count > 0 else 0
                    metrics["complexity"]["cyclomatic_complexity"] = round(avg_complexity, 2)
                except (json.JSONDecodeError, KeyError, ZeroDivisionError):
                    metrics["complexity"]["cyclomatic_complexity"] = 0
            else:
                metrics["complexity"]["cyclomatic_complexity"] = 0
        
        return metrics

    def collect_testing_metrics(self) -> Dict[str, Any]:
        """Collect testing statistics."""
        metrics = {
            "coverage": {},
            "test_count": {},
            "test_performance": {}
        }
        
        # Coverage metrics
        if self.test_path.exists():
            # Run pytest with coverage
            coverage_result = self.run_command([
                "pytest", "--cov=src/watermark_lab", "--cov-report=json", 
                "--quiet", "--tb=no"
            ])
            
            if coverage_result.returncode == 0:
                coverage_file = Path("coverage.json")
                if coverage_file.exists():
                    try:
                        with open(coverage_file) as f:
                            coverage_data = json.load(f)
                            metrics["coverage"]["total"] = round(coverage_data.get("totals", {}).get("percent_covered", 0), 2)
                    except (json.JSONDecodeError, FileNotFoundError):
                        metrics["coverage"]["total"] = 0
                    finally:
                        # Clean up coverage file
                        if coverage_file.exists():
                            coverage_file.unlink()
        
        # Count test files
        if self.test_path.exists():
            test_files = {
                "unit": list(self.test_path.glob("**/test_*.py")),
                "integration": list(self.test_path.glob("**/integration/test_*.py")),
                "performance": list(self.test_path.glob("**/performance/test_*.py")),
                "smoke": list(self.test_path.glob("**/smoke/test_*.py"))
            }
            
            for test_type, files in test_files.items():
                count = 0
                for file_path in files:
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            # Count test functions
                            count += len(re.findall(r'def test_\w+', content))
                    except Exception:
                        continue
                metrics["test_count"][test_type] = count
            
            metrics["test_count"]["total"] = sum(metrics["test_count"].values())
        
        return metrics

    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security-related metrics."""
        metrics = {
            "vulnerabilities": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "dependency_scan": {},
            "code_analysis": {}
        }
        
        # Safety check for vulnerabilities
        safety_result = self.run_command(["safety", "check", "--json"])
        if safety_result.returncode != 0 and safety_result.stdout:
            try:
                safety_data = json.loads(safety_result.stdout)
                for vuln in safety_data:
                    severity = vuln.get("severity", "").lower()
                    if severity in metrics["vulnerabilities"]:
                        metrics["vulnerabilities"][severity] += 1
            except json.JSONDecodeError:
                pass
        
        # Bandit security analysis
        bandit_result = self.run_command(["bandit", "-r", "src/", "-f", "json"])
        if bandit_result.returncode != 0 and bandit_result.stdout:
            try:
                bandit_data = json.loads(bandit_result.stdout)
                metrics["code_analysis"]["bandit_issues"] = len(bandit_data.get("results", []))
            except json.JSONDecodeError:
                metrics["code_analysis"]["bandit_issues"] = 0
        
        # Dependency information
        pip_outdated = self.run_command(["pip", "list", "--outdated", "--format=json"])
        if pip_outdated.returncode == 0:
            try:
                outdated_data = json.loads(pip_outdated.stdout)
                metrics["dependency_scan"]["outdated_dependencies"] = len(outdated_data)
            except json.JSONDecodeError:
                metrics["dependency_scan"]["outdated_dependencies"] = 0
        
        metrics["dependency_scan"]["last_scan"] = datetime.now(timezone.utc).isoformat()
        
        return metrics

    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git-related metrics."""
        metrics = {
            "commits_last_30_days": 0,
            "contributors": 0,
            "active_branches": 0
        }
        
        # Commits in last 30 days
        since_date = datetime.now().strftime("%Y-%m-%d")
        commits_result = self.run_command([
            "git", "log", "--since=30.days.ago", "--oneline"
        ])
        if commits_result.returncode == 0:
            metrics["commits_last_30_days"] = len(commits_result.stdout.strip().split('\n')) if commits_result.stdout.strip() else 0
        
        # Number of contributors
        contributors_result = self.run_command([
            "git", "log", "--format=%ae", "--since=6.months.ago"
        ])
        if contributors_result.returncode == 0:
            unique_contributors = set(contributors_result.stdout.strip().split('\n')) if contributors_result.stdout.strip() else set()
            metrics["contributors"] = len(unique_contributors)
        
        # Active branches
        branches_result = self.run_command(["git", "branch", "-r"])
        if branches_result.returncode == 0:
            metrics["active_branches"] = len(branches_result.stdout.strip().split('\n')) if branches_result.stdout.strip() else 0
        
        return metrics

    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance-related metrics."""
        metrics = {
            "benchmarks": {
                "generation_speed": {"avg_tokens_per_second": 0, "p95_latency_ms": 0},
                "detection_speed": {"avg_detections_per_second": 0, "p95_latency_ms": 0}
            }
        }
        
        # Run performance tests if available
        if (self.test_path / "performance").exists():
            perf_result = self.run_command([
                "pytest", "tests/performance/", "--benchmark-json=benchmark.json", 
                "--quiet", "--tb=no"
            ])
            
            if perf_result.returncode == 0 and Path("benchmark.json").exists():
                try:
                    with open("benchmark.json") as f:
                        benchmark_data = json.load(f)
                        # Process benchmark results
                        for benchmark in benchmark_data.get("benchmarks", []):
                            name = benchmark.get("name", "")
                            stats = benchmark.get("stats", {})
                            if "generation" in name.lower():
                                metrics["benchmarks"]["generation_speed"]["p95_latency_ms"] = round(stats.get("q95", 0) * 1000, 2)
                            elif "detection" in name.lower():
                                metrics["benchmarks"]["detection_speed"]["p95_latency_ms"] = round(stats.get("q95", 0) * 1000, 2)
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
                finally:
                    # Clean up benchmark file
                    if Path("benchmark.json").exists():
                        Path("benchmark.json").unlink()
        
        return metrics

    def load_existing_metrics(self) -> Dict[str, Any]:
        """Load existing metrics from file."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        return {
            "project_name": "lm-watermark-lab",
            "version": "1.0.0",
            "last_updated": "",
            "metrics": {}
        }

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save metrics to file."""
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"✅ Metrics saved to {self.metrics_file}")

    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all metrics and update the metrics file."""
        print("🔍 Collecting project metrics...")
        
        # Load existing metrics
        all_metrics = self.load_existing_metrics()
        all_metrics["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        # Collect different types of metrics
        print("  📊 Collecting codebase metrics...")
        all_metrics["metrics"]["codebase"] = self.collect_codebase_metrics()
        
        print("  🧪 Collecting testing metrics...")
        all_metrics["metrics"]["testing"] = self.collect_testing_metrics()
        
        print("  🔒 Collecting security metrics...")
        all_metrics["metrics"]["security"] = self.collect_security_metrics()
        
        print("  📈 Collecting performance metrics...")
        all_metrics["metrics"]["performance"] = self.collect_performance_metrics()
        
        print("  🔄 Collecting Git metrics...")
        git_metrics = self.collect_git_metrics()
        all_metrics["metrics"]["team"] = {
            "contributors": {"total": git_metrics["contributors"]},
            "collaboration": {"commits_last_30_days": git_metrics["commits_last_30_days"]}
        }
        
        # Update trends (simple implementation)
        all_metrics["trends"] = {
            "last_30_days": {
                "commits": git_metrics["commits_last_30_days"],
                "contributors": git_metrics["contributors"]
            },
            "quality_trend": "stable",
            "security_trend": "improving" if all_metrics["metrics"]["security"]["vulnerabilities"]["critical"] == 0 else "needs_attention"
        }
        
        return all_metrics

    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a human-readable metrics report."""
        codebase = metrics["metrics"].get("codebase", {})
        testing = metrics["metrics"].get("testing", {})
        security = metrics["metrics"].get("security", {})
        
        report = f"""
📊 **LM Watermark Lab - Metrics Report**
Generated: {metrics['last_updated']}

🏗️ **Codebase Metrics**
• Total Lines of Code: {codebase.get('lines_of_code', {}).get('total', 0):,}
• Python LoC: {codebase.get('lines_of_code', {}).get('python', 0):,}
• Total Files: {codebase.get('files', {}).get('total', 0)}
• Source Files: {codebase.get('files', {}).get('source_files', 0)}
• Test Files: {codebase.get('files', {}).get('test_files', 0)}

🧪 **Testing Metrics**
• Total Test Coverage: {testing.get('coverage', {}).get('total', 0)}%
• Total Tests: {testing.get('test_count', {}).get('total', 0)}
• Unit Tests: {testing.get('test_count', {}).get('unit', 0)}
• Integration Tests: {testing.get('test_count', {}).get('integration', 0)}

🔒 **Security Metrics**
• Critical Vulnerabilities: {security.get('vulnerabilities', {}).get('critical', 0)}
• High Vulnerabilities: {security.get('vulnerabilities', {}).get('high', 0)}
• Medium Vulnerabilities: {security.get('vulnerabilities', {}).get('medium', 0)}
• Outdated Dependencies: {security.get('dependency_scan', {}).get('outdated_dependencies', 0)}

👥 **Team Metrics**
• Contributors: {metrics['metrics'].get('team', {}).get('contributors', {}).get('total', 0)}
• Commits (30 days): {metrics['metrics'].get('team', {}).get('collaboration', {}).get('commits_last_30_days', 0)}
"""
        return report


def main():
    """Main function to run metrics collection."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("""
LM Watermark Lab Metrics Collector

Usage:
  python scripts/collect_metrics.py [options]

Options:
  --help          Show this help message
  --report-only   Generate report from existing metrics
  --save-report   Save report to file
  
Examples:
  python scripts/collect_metrics.py
  python scripts/collect_metrics.py --report-only
  python scripts/collect_metrics.py --save-report
        """)
        return
    
    collector = MetricsCollector()
    
    if "--report-only" in sys.argv:
        # Generate report from existing metrics
        existing_metrics = collector.load_existing_metrics()
        if existing_metrics.get("metrics"):
            report = collector.generate_report(existing_metrics)
            print(report)
        else:
            print("❌ No existing metrics found. Run without --report-only first.")
        return
    
    # Collect all metrics
    try:
        metrics = collector.collect_all_metrics()
        collector.save_metrics(metrics)
        
        # Generate and display report
        report = collector.generate_report(metrics)
        print(report)
        
        if "--save-report" in sys.argv:
            report_file = Path("metrics-report.md")
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to {report_file}")
        
    except Exception as e:
        print(f"❌ Error collecting metrics: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()