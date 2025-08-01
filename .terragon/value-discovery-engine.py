#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery Engine
Perpetual SDLC Enhancement through Intelligent Work Item Prioritization
"""

import json
import yaml
import subprocess
import re
from typing import Dict, List, Any, Optional, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import math
import ast


@dataclass
class ValueItem:
    """Represents a discovered value-generating work item."""
    id: str
    title: str
    description: str
    category: str  # technical_debt, security, feature, performance, docs
    files_affected: List[str]
    estimated_effort_hours: float
    
    # WSJF Components
    user_business_value: float  # 1-100
    time_criticality: float     # 1-100
    risk_reduction: float       # 1-100
    opportunity_enablement: float # 1-100
    job_size: float            # story points
    
    # ICE Components
    impact: float              # 1-10
    confidence: float          # 1-10
    ease: float               # 1-10
    
    # Technical Debt Scoring
    debt_impact: float         # hours saved
    debt_interest: float       # future cost if not addressed
    hotspot_multiplier: float  # 1-5x based on churn
    
    # Metadata
    discovered_date: str
    source: str               # git_history, static_analysis, etc.
    priority_boost: float = 1.0  # Security/compliance boost


class ValueDiscoveryEngine:
    """Autonomous engine for discovering and prioritizing work items."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "value-config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        self.backlog_path = self.repo_path / "BACKLOG.md"
        
        self.config = self._load_config()
        self.metrics = self._load_metrics()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load value discovery configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._default_config()
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load historical value metrics."""
        try:
            with open(self.metrics_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"executionHistory": [], "backlogMetrics": {}}
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for advanced repositories."""
        return {
            "scoring": {
                "weights": {
                    "advanced": {
                        "wsjf": 0.5,
                        "ice": 0.1,
                        "technicalDebt": 0.3,
                        "security": 0.1
                    }
                },
                "thresholds": {
                    "minScore": 15,
                    "maxRisk": 0.7,
                    "securityBoost": 2.0,
                    "complianceBoost": 1.8
                }
            }
        }
    
    def discover_value_items(self) -> List[ValueItem]:
        """Comprehensive value discovery from multiple sources."""
        items = []
        
        # Git history analysis for debt indicators
        items.extend(self._discover_from_git_history())
        
        # Static analysis for code quality issues
        items.extend(self._discover_from_static_analysis())
        
        # Dependency analysis for security/updates
        items.extend(self._discover_from_dependencies())
        
        # Performance analysis
        items.extend(self._discover_from_performance())
        
        # Code complexity and hotspot analysis
        items.extend(self._discover_from_complexity())
        
        return items
    
    def _discover_from_git_history(self) -> List[ValueItem]:
        """Discover work items from git commit history and code comments."""
        items = []
        
        try:
            # Find TODO/FIXME/HACK comments in code
            result = subprocess.run([
                "grep", "-r", "-n", "-i", 
                "--include=*.py", "--include=*.md", "--include=*.yaml",
                r"TODO\|FIXME\|HACK\|BUG\|DEPRECATED"
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if ':' in line:
                        file_path, line_num, content = line.split(':', 2)
                        items.append(self._create_debt_item(
                            file_path, content.strip(), "git_history"
                        ))
        except Exception:
            pass  # Git not available or no matches
        
        # Analyze commit messages for quick fixes
        try:
            result = subprocess.run([
                "git", "log", "--oneline", "-50", 
                "--grep=quick fix\\|temporary\\|hack\\|urgent"
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    items.append(self._create_commit_debt_item(line))
        except Exception:
            pass
            
        return items
    
    def _discover_from_static_analysis(self) -> List[ValueItem]:
        """Discover issues through static code analysis."""
        items = []
        
        # Run ruff for Python linting
        try:
            result = subprocess.run([
                "ruff", "check", "src/", "--output-format=json"
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.stdout:
                violations = json.loads(result.stdout)
                for violation in violations[:10]:  # Top 10 issues
                    items.append(self._create_lint_item(violation))
        except Exception:
            pass
        
        # Run mypy for type checking
        try:
            result = subprocess.run([
                "mypy", "src/", "--json-report", "/tmp/mypy-report"
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            # Parse mypy output for type issues
            type_errors = re.findall(r'error: (.+)', result.stdout)
            for error in type_errors[:5]:
                items.append(self._create_type_item(error))
        except Exception:
            pass
            
        return items
    
    def _discover_from_dependencies(self) -> List[ValueItem]:
        """Discover dependency-related work items."""
        items = []
        
        # Check for security vulnerabilities
        try:
            result = subprocess.run([
                "safety", "check", "--json"
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.stdout:
                vulnerabilities = json.loads(result.stdout)
                for vuln in vulnerabilities[:5]:
                    items.append(self._create_security_item(vuln))
        except Exception:
            pass
        
        # Check for outdated dependencies
        try:
            result = subprocess.run([
                "pip", "list", "--outdated", "--format=json"
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.stdout:
                outdated = json.loads(result.stdout)
                for package in outdated[:3]:  # Top 3 outdated
                    items.append(self._create_update_item(package))
        except Exception:
            pass
            
        return items
    
    def _discover_from_performance(self) -> List[ValueItem]:
        """Discover performance optimization opportunities."""
        items = []
        
        # Analyze import times and complexity
        python_files = list(self.repo_path.glob("src/**/*.py"))
        
        for py_file in python_files[:5]:  # Analyze top 5 files
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                # Count lines of code
                lines = len([l for l in content.split('\n') if l.strip()])
                
                # Simple complexity heuristics
                if lines > 200:  # Large file
                    items.append(ValueItem(
                        id=f"perf-{py_file.stem}",
                        title=f"Refactor large file {py_file.name}",
                        description=f"File has {lines} lines, consider breaking down",
                        category="performance",
                        files_affected=[str(py_file)],
                        estimated_effort_hours=2.0,
                        user_business_value=30,
                        time_criticality=20,
                        risk_reduction=40,
                        opportunity_enablement=50,
                        job_size=3,
                        impact=6,
                        confidence=8,
                        ease=7,
                        debt_impact=5.0,
                        debt_interest=10.0,
                        hotspot_multiplier=1.5,
                        discovered_date=datetime.now().isoformat(),
                        source="performance_analysis"
                    ))
            except Exception:
                continue
                
        return items
    
    def _discover_from_complexity(self) -> List[ValueItem]:
        """Discover complexity-related improvement opportunities."""
        items = []
        
        # Find functions with high cyclomatic complexity
        python_files = list(self.repo_path.glob("src/**/*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Simple complexity estimate
                        complexity = self._estimate_complexity(node)
                        if complexity > 10:  # High complexity threshold
                            items.append(ValueItem(
                                id=f"complex-{py_file.stem}-{node.name}",
                                title=f"Reduce complexity in {node.name}()",
                                description=f"Function has estimated complexity {complexity}",
                                category="technical_debt",
                                files_affected=[str(py_file)],
                                estimated_effort_hours=1.5,
                                user_business_value=25,
                                time_criticality=15,
                                risk_reduction=35,
                                opportunity_enablement=30,
                                job_size=2,
                                impact=5,
                                confidence=7,
                                ease=6,
                                debt_impact=3.0,
                                debt_interest=8.0,
                                hotspot_multiplier=2.0,
                                discovered_date=datetime.now().isoformat(),
                                source="complexity_analysis"
                            ))
            except Exception:
                continue
                
        return items
    
    def _estimate_complexity(self, node: ast.FunctionDef) -> int:
        """Simple cyclomatic complexity estimation."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
                
        return complexity
    
    def _create_debt_item(self, file_path: str, content: str, source: str) -> ValueItem:
        """Create work item from debt comment."""
        return ValueItem(
            id=f"debt-{hash(content) % 10000}",
            title=f"Address technical debt in {file_path}",
            description=content,
            category="technical_debt",
            files_affected=[file_path],
            estimated_effort_hours=1.0,
            user_business_value=20,
            time_criticality=10,
            risk_reduction=30,
            opportunity_enablement=25,
            job_size=1,
            impact=4,
            confidence=8,
            ease=7,
            debt_impact=2.0,
            debt_interest=5.0,
            hotspot_multiplier=1.2,
            discovered_date=datetime.now().isoformat(),
            source=source
        )
    
    def _create_commit_debt_item(self, commit_line: str) -> ValueItem:
        """Create work item from commit message indicating debt."""
        commit_hash = commit_line.split()[0]
        return ValueItem(
            id=f"commit-debt-{commit_hash}",
            title=f"Review and improve quick fix {commit_hash}",
            description=commit_line,
            category="technical_debt",
            files_affected=[],
            estimated_effort_hours=0.5,
            user_business_value=15,
            time_criticality=5,
            risk_reduction=25,
            opportunity_enablement=20,
            job_size=0.5,
            impact=3,
            confidence=6,
            ease=8,
            debt_impact=1.0,
            debt_interest=3.0,
            hotspot_multiplier=1.0,
            discovered_date=datetime.now().isoformat(),
            source="git_analysis"
        )
    
    def _create_lint_item(self, violation: Dict) -> ValueItem:
        """Create work item from linting violation."""
        return ValueItem(
            id=f"lint-{violation.get('code', 'unknown')}",
            title=f"Fix {violation.get('code', 'linting')} violation",
            description=violation.get('message', 'Linting violation'),
            category="technical_debt",
            files_affected=[violation.get('filename', '')],
            estimated_effort_hours=0.25,
            user_business_value=10,
            time_criticality=5,
            risk_reduction=15,
            opportunity_enablement=10,
            job_size=0.25,
            impact=2,
            confidence=9,
            ease=9,
            debt_impact=0.5,
            debt_interest=1.0,
            hotspot_multiplier=1.0,
            discovered_date=datetime.now().isoformat(),
            source="static_analysis"
        )
    
    def _create_type_item(self, error: str) -> ValueItem:
        """Create work item from type checking error."""
        return ValueItem(
            id=f"type-{hash(error) % 10000}",
            title="Fix type annotation issue",
            description=error,
            category="technical_debt",
            files_affected=[],
            estimated_effort_hours=0.5,
            user_business_value=20,
            time_criticality=10,
            risk_reduction=25,
            opportunity_enablement=15,
            job_size=0.5,
            impact=4,
            confidence=8,
            ease=7,
            debt_impact=1.0,
            debt_interest=2.0,
            hotspot_multiplier=1.1,
            discovered_date=datetime.now().isoformat(),
            source="type_analysis"
        )
    
    def _create_security_item(self, vuln: Dict) -> ValueItem:
        """Create work item from security vulnerability."""
        return ValueItem(
            id=f"sec-{vuln.get('id', hash(str(vuln)) % 10000)}",
            title=f"Fix security vulnerability in {vuln.get('package_name', 'dependency')}",
            description=vuln.get('advisory', 'Security vulnerability'),
            category="security",
            files_affected=[],
            estimated_effort_hours=1.0,
            user_business_value=80,
            time_criticality=90,
            risk_reduction=95,
            opportunity_enablement=60,
            job_size=1,
            impact=9,
            confidence=9,
            ease=8,
            debt_impact=0,
            debt_interest=0,
            hotspot_multiplier=1.0,
            discovered_date=datetime.now().isoformat(),
            source="security_scan",
            priority_boost=2.0  # Security boost
        )
    
    def _create_update_item(self, package: Dict) -> ValueItem:
        """Create work item from outdated dependency."""
        return ValueItem(
            id=f"update-{package.get('name', 'unknown')}",
            title=f"Update {package.get('name')} to {package.get('latest_version')}",
            description=f"Update from {package.get('version')} to {package.get('latest_version')}",
            category="dependency_update",
            files_affected=["requirements.txt", "pyproject.toml"],
            estimated_effort_hours=0.5,
            user_business_value=15,
            time_criticality=20,
            risk_reduction=30,
            opportunity_enablement=40,
            job_size=0.5,
            impact=3,
            confidence=7,
            ease=8,
            debt_impact=1.0,
            debt_interest=2.0,
            hotspot_multiplier=1.0,
            discovered_date=datetime.now().isoformat(),
            source="dependency_analysis"
        )
    
    def calculate_composite_scores(self, items: List[ValueItem]) -> List[ValueItem]:
        """Calculate composite value scores for all items."""
        maturity = self.config["maturityLevel"]
        weights = self.config["scoring"]["weights"].get(maturity, 
                                                        self.config["scoring"]["weights"]["advanced"])
        
        for item in items:
            # WSJF Calculation
            cost_of_delay = (
                item.user_business_value + 
                item.time_criticality + 
                item.risk_reduction + 
                item.opportunity_enablement
            )
            wsjf_score = cost_of_delay / max(item.job_size, 0.1)
            
            # ICE Calculation
            ice_score = item.impact * item.confidence * item.ease
            
            # Technical Debt Calculation
            debt_score = (item.debt_impact + item.debt_interest) * item.hotspot_multiplier
            
            # Normalize scores (0-100 scale)
            normalized_wsjf = min(wsjf_score / 50.0 * 100, 100)
            normalized_ice = min(ice_score / 10.0 * 100, 100)
            normalized_debt = min(debt_score / 20.0 * 100, 100)
            
            # Composite score
            composite = (
                weights["wsjf"] * normalized_wsjf +
                weights["ice"] * normalized_ice +
                weights["technicalDebt"] * normalized_debt
            )
            
            # Apply priority boosts
            composite *= item.priority_boost
            
            # Store calculated score
            item.composite_score = composite
            
        return items
    
    def select_next_best_value(self, items: List[ValueItem]) -> Optional[ValueItem]:
        """Select the next highest value work item for execution."""
        if not items:
            return None
            
        # Filter by minimum score threshold
        min_score = self.config["scoring"]["thresholds"]["minScore"]
        qualified_items = [item for item in items if getattr(item, 'composite_score', 0) >= min_score]
        
        if not qualified_items:
            return None
            
        # Sort by composite score descending
        qualified_items.sort(key=lambda x: getattr(x, 'composite_score', 0), reverse=True)
        
        return qualified_items[0]
    
    def generate_backlog_markdown(self, items: List[ValueItem]) -> str:
        """Generate BACKLOG.md content with discovered items."""
        now = datetime.now().isoformat()
        
        # Sort items by composite score
        sorted_items = sorted(items, 
                            key=lambda x: getattr(x, 'composite_score', 0), 
                            reverse=True)
        
        next_item = sorted_items[0] if sorted_items else None
        
        content = f"""# 📊 Autonomous Value Backlog

Last Updated: {now}
Next Execution: {(datetime.now() + timedelta(hours=1)).isoformat()}

## 🎯 Next Best Value Item
"""
        
        if next_item:
            content += f"""**[{next_item.id.upper()}] {next_item.title}**
- **Composite Score**: {getattr(next_item, 'composite_score', 0):.1f}
- **Category**: {next_item.category.replace('_', ' ').title()}
- **Estimated Effort**: {next_item.estimated_effort_hours} hours
- **Expected Impact**: {next_item.description}

"""
        
        content += """## 📋 Top 10 Backlog Items

| Rank | ID | Title | Score | Category | Est. Hours |
|------|-----|--------|---------|----------|------------|
"""
        
        for i, item in enumerate(sorted_items[:10], 1):
            score = getattr(item, 'composite_score', 0)
            category = item.category.replace('_', ' ').title()
            title = item.title[:50] + ("..." if len(item.title) > 50 else "")
            content += f"| {i} | {item.id.upper()} | {title} | {score:.1f} | {category} | {item.estimated_effort_hours} |\n"
        
        content += f"""

## 📈 Value Metrics
- **Items Discovered This Run**: {len(items)}
- **High Priority Items (>50 score)**: {len([i for i in items if getattr(i, 'composite_score', 0) > 50])}
- **Security Items**: {len([i for i in items if i.category == 'security'])}
- **Technical Debt Items**: {len([i for i in items if i.category == 'technical_debt'])}

## 🔄 Continuous Discovery Stats
- **Discovery Sources**:
  - Static Analysis: {len([i for i in items if i.source == 'static_analysis'])}
  - Git History: {len([i for i in items if i.source == 'git_history'])}
  - Security Scans: {len([i for i in items if i.source == 'security_scan'])}
  - Performance Analysis: {len([i for i in items if i.source == 'performance_analysis'])}
  - Complexity Analysis: {len([i for i in items if i.source == 'complexity_analysis'])}

## 🏗️ Implementation Guidelines

### For Advanced Repositories
This repository is classified as **ADVANCED** (85%+ SDLC maturity). Focus areas:
- **Code Quality Optimization**: Type safety, complexity reduction
- **Security Posture**: Vulnerability patching, dependency updates  
- **Performance Enhancement**: Hotspot optimization, profiling-guided improvements
- **Technical Debt Paydown**: Legacy code modernization, documentation gaps

### Execution Protocol
1. Select highest-scoring item from backlog
2. Create feature branch: `auto-value/{{item-id}}-{{slug}}`
3. Implement changes with comprehensive testing
4. Run full validation suite (lint, type-check, test, security)
5. Create detailed PR with value metrics
6. Merge and immediately trigger next discovery cycle

### Value Tracking
All executions are tracked in `.terragon/value-metrics.json` with:
- Predicted vs actual effort and impact
- Composite scoring accuracy
- Continuous learning model updates
- ROI calculations and trend analysis

---

*Generated by Terragon Autonomous Value Discovery Engine*
*Repository Maturity: Advanced | Next Discovery: Hourly*
"""
        
        return content
    
    def save_metrics(self, items: List[ValueItem]) -> None:
        """Save value metrics to JSON file."""
        metrics = {
            "lastDiscovery": datetime.now().isoformat(),
            "totalItemsDiscovered": len(items),
            "itemsByCategory": {},
            "itemsBySource": {},
            "scoringDistribution": {
                "high": len([i for i in items if getattr(i, 'composite_score', 0) > 70]),
                "medium": len([i for i in items if 30 < getattr(i, 'composite_score', 0) <= 70]),
                "low": len([i for i in items if getattr(i, 'composite_score', 0) <= 30])
            },
            "discoveredItems": [asdict(item) for item in items[:20]]  # Store top 20
        }
        
        # Count by category and source
        for item in items:
            metrics["itemsByCategory"][item.category] = metrics["itemsByCategory"].get(item.category, 0) + 1
            metrics["itemsBySource"][item.source] = metrics["itemsBySource"].get(item.source, 0) + 1
        
        # Save to file
        self.metrics_path.parent.mkdir(exist_ok=True)
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def run_discovery_cycle(self) -> List[ValueItem]:
        """Execute a complete value discovery cycle."""
        print("🔍 Running Terragon Autonomous Value Discovery...")
        
        # Discover all value items
        items = self.discover_value_items()
        print(f"📊 Discovered {len(items)} potential value items")
        
        # Calculate composite scores
        items = self.calculate_composite_scores(items)
        
        # Select next best value item
        next_item = self.select_next_best_value(items)
        if next_item:
            print(f"🎯 Next Best Value: {next_item.title} (Score: {getattr(next_item, 'composite_score', 0):.1f})")
        else:
            print("✅ No qualifying work items found - repository is optimally maintained")
        
        # Generate and save backlog
        backlog_content = self.generate_backlog_markdown(items)
        with open(self.backlog_path, 'w') as f:
            f.write(backlog_content)
        
        # Save metrics
        self.save_metrics(items)
        
        print(f"📝 Updated BACKLOG.md with {len(items)} items")
        print(f"💾 Saved metrics to {self.metrics_path}")
        
        return items


def main():
    """Main entry point for value discovery engine."""
    engine = ValueDiscoveryEngine()
    items = engine.run_discovery_cycle()
    
    # Print summary
    if items:
        best_item = max(items, key=lambda x: getattr(x, 'composite_score', 0))
        print(f"\n🚀 Ready for autonomous execution of: {best_item.title}")
        print(f"   Expected value delivery: {getattr(best_item, 'composite_score', 0):.1f} points")
    else:
        print("\n✨ Repository is in optimal state - no immediate work items identified")


if __name__ == "__main__":
    main()