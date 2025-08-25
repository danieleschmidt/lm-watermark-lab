#!/usr/bin/env python3
"""
Comprehensive Quality Gates Validation
Autonomous SDLC - Final Quality Assurance Check
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

class QualityGateValidator:
    """Comprehensive quality gate validation."""
    
    def __init__(self):
        self.results = {
            'timestamp': time.time(),
            'overall_status': 'pending',
            'gates': {},
            'summary': {}
        }
        self.required_gates = [
            'code_structure',
            'import_validation', 
            'security_checks',
            'performance_tests',
            'documentation_check',
            'deployment_readiness'
        ]
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates."""
        print("🚀 Starting Comprehensive Quality Gates Validation")
        print("=" * 60)
        
        all_passed = True
        
        for gate in self.required_gates:
            print(f"\n🔍 Running {gate.replace('_', ' ').title()}...")
            
            try:
                gate_method = getattr(self, f'validate_{gate}')
                result = gate_method()
                self.results['gates'][gate] = result
                
                status = "✅ PASS" if result['passed'] else "❌ FAIL"
                print(f"{status} - {result.get('summary', 'Completed')}")
                
                if not result['passed']:
                    all_passed = False
                    
            except Exception as e:
                self.results['gates'][gate] = {
                    'passed': False,
                    'error': str(e),
                    'summary': f'Gate execution failed: {e}'
                }
                print(f"❌ FAIL - Gate execution failed: {e}")
                all_passed = False
        
        self.results['overall_status'] = 'passed' if all_passed else 'failed'
        self._generate_summary()
        
        print(f"\n{'='*60}")
        print(f"🎯 OVERALL STATUS: {'✅ ALL GATES PASSED' if all_passed else '❌ SOME GATES FAILED'}")
        print(f"{'='*60}")
        
        return self.results
    
    def validate_code_structure(self) -> Dict[str, Any]:
        """Validate code structure and organization."""
        checks = []
        
        # Check core directories exist
        required_dirs = [
            'src/watermark_lab',
            'src/watermark_lab/core', 
            'src/watermark_lab/methods',
            'src/watermark_lab/utils',
            'src/watermark_lab/security',
            'src/watermark_lab/optimization',
            'tests',
            'docs'
        ]
        
        for dir_path in required_dirs:
            exists = Path(dir_path).exists()
            checks.append({
                'check': f'Directory {dir_path} exists',
                'passed': exists
            })
        
        # Check key files exist
        required_files = [
            'src/watermark_lab/__init__.py',
            'src/watermark_lab/core/factory.py',
            'src/watermark_lab/methods/kirchenbauer.py',
            'src/watermark_lab/utils/enhanced_resilience.py',
            'src/watermark_lab/security/advanced_security.py',
            'src/watermark_lab/optimization/ultra_performance.py',
            'pyproject.toml',
            'README.md'
        ]
        
        for file_path in required_files:
            exists = Path(file_path).exists()
            checks.append({
                'check': f'File {file_path} exists',
                'passed': exists
            })
        
        passed_checks = sum(1 for check in checks if check['passed'])
        total_checks = len(checks)
        
        return {
            'passed': passed_checks == total_checks,
            'checks': checks,
            'summary': f'{passed_checks}/{total_checks} structure checks passed'
        }
    
    def validate_import_validation(self) -> Dict[str, Any]:
        """Validate Python imports and basic functionality."""
        checks = []
        
        # Test basic package import
        try:
            import watermark_lab
            checks.append({
                'check': 'Basic package import',
                'passed': True,
                'details': f'Version: {watermark_lab.__version__}'
            })
        except Exception as e:
            checks.append({
                'check': 'Basic package import',
                'passed': False,
                'error': str(e)
            })
        
        # Test core component imports (graceful handling of missing deps)
        test_imports = [
            ('watermark_lab.utils.exceptions', 'WatermarkLabError'),
            ('watermark_lab.utils.validation', 'validate_text'),
            ('watermark_lab.config.settings', None),
        ]
        
        for module_name, class_name in test_imports:
            try:
                if class_name:
                    module = __import__(module_name, fromlist=[class_name])
                    getattr(module, class_name)
                else:
                    __import__(module_name)
                    
                checks.append({
                    'check': f'Import {module_name}{f".{class_name}" if class_name else ""}',
                    'passed': True
                })
            except Exception as e:
                checks.append({
                    'check': f'Import {module_name}{f".{class_name}" if class_name else ""}',
                    'passed': False,
                    'error': str(e)
                })
        
        # Test utility functions
        try:
            from watermark_lab.utils.validation import validate_text
            result = validate_text("test string", min_length=1)
            checks.append({
                'check': 'Utility function validation',
                'passed': result == "test string"
            })
        except Exception as e:
            checks.append({
                'check': 'Utility function validation', 
                'passed': False,
                'error': str(e)
            })
        
        passed_checks = sum(1 for check in checks if check['passed'])
        return {
            'passed': passed_checks >= len(checks) * 0.8,  # 80% threshold
            'checks': checks,
            'summary': f'{passed_checks}/{len(checks)} import checks passed'
        }
    
    def validate_security_checks(self) -> Dict[str, Any]:
        """Validate security implementations."""
        checks = []
        
        # Test input sanitization
        try:
            from watermark_lab.security.advanced_security import AdvancedInputSanitizer
            sanitizer = AdvancedInputSanitizer()
            
            # Test dangerous pattern detection
            test_inputs = [
                "normal text",
                "<script>alert('xss')</script>", 
                "SELECT * FROM users WHERE id=1",
                "../../../etc/passwd"
            ]
            
            safe_count = 0
            for test_input in test_inputs:
                try:
                    threats = sanitizer.detect_threats(test_input)
                    if test_input == "normal text" and not threats:
                        safe_count += 1
                    elif test_input != "normal text" and threats:
                        safe_count += 1
                except Exception:
                    pass
            
            checks.append({
                'check': 'Input sanitization patterns',
                'passed': safe_count >= 3,
                'details': f'{safe_count}/4 threat detection tests passed'
            })
            
        except Exception as e:
            checks.append({
                'check': 'Input sanitization patterns',
                'passed': False,
                'error': str(e)
            })
        
        # Test validation schemas
        try:
            from watermark_lab.utils.validation import WATERMARK_CONFIG_SCHEMA
            schema_valid = isinstance(WATERMARK_CONFIG_SCHEMA, dict) and len(WATERMARK_CONFIG_SCHEMA) > 0
            checks.append({
                'check': 'Validation schemas defined',
                'passed': schema_valid
            })
        except Exception as e:
            checks.append({
                'check': 'Validation schemas defined',
                'passed': False,
                'error': str(e)
            })
        
        passed_checks = sum(1 for check in checks if check['passed'])
        return {
            'passed': passed_checks == len(checks),
            'checks': checks,
            'summary': f'{passed_checks}/{len(checks)} security checks passed'
        }
    
    def validate_performance_tests(self) -> Dict[str, Any]:
        """Validate performance optimizations."""
        checks = []
        
        # Test ultra-performance manager
        try:
            from watermark_lab.optimization.ultra_performance import UltraPerformanceManager
            manager = UltraPerformanceManager()
            
            # Test basic functionality
            metrics = manager.get_performance_metrics()
            checks.append({
                'check': 'Ultra-performance manager creation',
                'passed': isinstance(metrics, dict)
            })
            
            # Test batch processing (with simple function)
            test_data = list(range(100))
            results = manager.ultra_batch_process(lambda x: x * 2, test_data, chunk_size=10)
            checks.append({
                'check': 'Batch processing functionality',
                'passed': len(results) == len(test_data) and results[0] == 0 and results[1] == 2
            })
            
        except Exception as e:
            checks.append({
                'check': 'Ultra-performance manager',
                'passed': False,
                'error': str(e)
            })
        
        # Test caching system
        try:
            from watermark_lab.optimization.performance_cache import PerformanceCache
            cache = PerformanceCache(max_size=100)
            
            # Test basic cache operations
            cache.set("test_key", "test_value")
            value = cache.get("test_key")
            checks.append({
                'check': 'Performance caching system',
                'passed': value == "test_value"
            })
            
        except Exception as e:
            checks.append({
                'check': 'Performance caching system',
                'passed': False,
                'error': str(e)
            })
        
        passed_checks = sum(1 for check in checks if check['passed'])
        return {
            'passed': passed_checks >= len(checks) * 0.7,  # 70% threshold 
            'checks': checks,
            'summary': f'{passed_checks}/{len(checks)} performance checks passed'
        }
    
    def validate_documentation_check(self) -> Dict[str, Any]:
        """Validate documentation completeness."""
        checks = []
        
        # Check README exists and has content
        readme_path = Path('README.md')
        if readme_path.exists():
            content = readme_path.read_text()
            checks.append({
                'check': 'README.md exists and has content',
                'passed': len(content) > 1000,  # Substantial content
                'details': f'{len(content)} characters'
            })
            
            # Check for key sections
            key_sections = ['Installation', 'Usage', 'API', 'Examples']
            for section in key_sections:
                has_section = section.lower() in content.lower()
                checks.append({
                    'check': f'README has {section} section',
                    'passed': has_section
                })
        else:
            checks.append({
                'check': 'README.md exists',
                'passed': False
            })
        
        # Check docs directory
        docs_path = Path('docs')
        if docs_path.exists():
            doc_files = list(docs_path.rglob('*.md'))
            checks.append({
                'check': 'Documentation directory has content',
                'passed': len(doc_files) > 3,
                'details': f'{len(doc_files)} documentation files'
            })
        else:
            checks.append({
                'check': 'Documentation directory exists',
                'passed': False
            })
        
        # Check pyproject.toml
        pyproject_path = Path('pyproject.toml')
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            has_metadata = all(key in content for key in ['name', 'version', 'description'])
            checks.append({
                'check': 'pyproject.toml has complete metadata',
                'passed': has_metadata
            })
        else:
            checks.append({
                'check': 'pyproject.toml exists',
                'passed': False
            })
        
        passed_checks = sum(1 for check in checks if check['passed'])
        return {
            'passed': passed_checks >= len(checks) * 0.8,
            'checks': checks,
            'summary': f'{passed_checks}/{len(checks)} documentation checks passed'
        }
    
    def validate_deployment_readiness(self) -> Dict[str, Any]:
        """Validate deployment readiness."""
        checks = []
        
        # Check Docker files
        dockerfile_exists = Path('Dockerfile').exists()
        checks.append({
            'check': 'Dockerfile exists',
            'passed': dockerfile_exists
        })
        
        docker_compose_exists = Path('docker-compose.yml').exists()
        checks.append({
            'check': 'docker-compose.yml exists',
            'passed': docker_compose_exists
        })
        
        # Check Kubernetes configs
        k8s_path = Path('kubernetes')
        if k8s_path.exists():
            k8s_files = list(k8s_path.glob('*.yaml')) + list(k8s_path.glob('*.yml'))
            checks.append({
                'check': 'Kubernetes configurations exist',
                'passed': len(k8s_files) > 0,
                'details': f'{len(k8s_files)} K8s files'
            })
        else:
            checks.append({
                'check': 'Kubernetes directory exists',
                'passed': True  # Optional for basic deployment
            })
        
        # Check monitoring configs
        monitoring_path = Path('monitoring')
        if monitoring_path.exists():
            monitoring_files = list(monitoring_path.rglob('*.yml')) + list(monitoring_path.rglob('*.yaml'))
            checks.append({
                'check': 'Monitoring configurations exist',
                'passed': len(monitoring_files) > 0,
                'details': f'{len(monitoring_files)} monitoring files'
            })
        else:
            checks.append({
                'check': 'Monitoring configurations exist',
                'passed': False
            })
        
        # Check for security configs
        security_files = [
            '.github/workflows/security.yml',
            'sonar-project.properties',
            'renovate.json'
        ]
        
        security_count = sum(1 for f in security_files if Path(f).exists())
        checks.append({
            'check': 'Security configurations exist',
            'passed': security_count >= 2,
            'details': f'{security_count}/{len(security_files)} security files'
        })
        
        passed_checks = sum(1 for check in checks if check['passed'])
        return {
            'passed': passed_checks >= len(checks) * 0.7,
            'checks': checks,
            'summary': f'{passed_checks}/{len(checks)} deployment checks passed'
        }
    
    def _generate_summary(self):
        """Generate validation summary."""
        passed_gates = sum(1 for gate in self.results['gates'].values() if gate['passed'])
        total_gates = len(self.results['gates'])
        
        self.results['summary'] = {
            'passed_gates': passed_gates,
            'total_gates': total_gates,
            'success_rate': (passed_gates / total_gates) * 100 if total_gates > 0 else 0,
            'status': 'production_ready' if passed_gates == total_gates else 'needs_attention'
        }
    
    def save_results(self, filename: str = 'quality_validation_results.json'):
        """Save validation results to file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n📊 Results saved to {filename}")


def main():
    """Main validation execution."""
    validator = QualityGateValidator()
    results = validator.run_all_gates()
    validator.save_results()
    
    # Exit with appropriate code
    exit_code = 0 if results['overall_status'] == 'passed' else 1
    sys.exit(exit_code)


if __name__ == '__main__':
    main()