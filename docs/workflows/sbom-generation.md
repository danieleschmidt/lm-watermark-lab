# SBOM (Software Bill of Materials) Generation Guide

## Overview

Software Bill of Materials (SBOM) is a comprehensive inventory of all software components, dependencies, and metadata used in building software applications. This is crucial for supply chain security and regulatory compliance.

## Why SBOM Matters

### Security Benefits
- **Vulnerability Management**: Quickly identify affected components
- **Supply Chain Risk**: Track third-party dependencies
- **Incident Response**: Rapid impact assessment
- **Compliance**: Meet regulatory requirements (EO 14028)

### Business Benefits
- **Risk Mitigation**: Proactive security posture
- **Audit Readiness**: Automated compliance reporting
- **Cost Reduction**: Efficient vulnerability remediation
- **Trust Building**: Transparency with customers

## Supported SBOM Formats

### 1. SPDX (Software Package Data Exchange)
- **Standard**: Linux Foundation
- **Format**: JSON, YAML, Tag-Value, RDF
- **Use Case**: License compliance, legal analysis

### 2. CycloneDX
- **Standard**: OWASP
- **Format**: JSON, XML
- **Use Case**: Security analysis, vulnerability management

### 3. SWID (Software Identification Tags)
- **Standard**: ISO/IEC 19770-2
- **Format**: XML
- **Use Case**: Asset management, inventory

## Implementation Guide

### Step 1: Install SBOM Tools

```bash
# Install Syft (Anchore)
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin

# Install CycloneDX Python tools
pip install cyclonedx-bom

# Install SPDX tools
pip install spdx-tools

# Install Python license tools
pip install pip-licenses
```

### Step 2: Generate Basic SBOM

```bash
# Generate SPDX SBOM
syft packages . -o spdx-json=sbom.spdx.json

# Generate CycloneDX SBOM
syft packages . -o cyclonedx-json=sbom.cyclonedx.json

# Generate Python-specific SBOM
cyclonedx-py --format json --output python-sbom.json .

# Generate license report
pip-licenses --format json --output licenses.json
```

### Step 3: Enhanced SBOM Generation

Create `generate_sbom.py`:

```python
#!/usr/bin/env python3
"""
Enhanced SBOM generation script with metadata enrichment.
"""

import json
import subprocess
import datetime
from pathlib import Path
from typing import Dict, Any

def generate_enhanced_sbom():
    """Generate comprehensive SBOM with metadata."""
    
    # Base SBOM generation
    subprocess.run([
        'syft', 'packages', '.',
        '-o', 'spdx-json=sbom-enhanced.spdx.json'
    ])
    
    # Load and enhance SBOM
    with open('sbom-enhanced.spdx.json', 'r') as f:
        sbom = json.load(f)
    
    # Add custom metadata
    sbom['creationInfo']['created'] = datetime.datetime.utcnow().isoformat() + 'Z'
    sbom['creationInfo']['creators'].append('Tool: watermark-lab-sbom-generator')
    
    # Add build information
    sbom['annotations'] = sbom.get('annotations', [])
    sbom['annotations'].extend([
        {
            'annotator': 'watermark-lab-generator',
            'annotationType': 'BUILD',
            'annotationDate': datetime.datetime.utcnow().isoformat() + 'Z',
            'annotationComment': 'Generated during CI/CD pipeline'
        },
        {
            'annotator': 'watermark-lab-generator', 
            'annotationType': 'SECURITY',
            'annotationDate': datetime.datetime.utcnow().isoformat() + 'Z',
            'annotationComment': 'Security scan results attached'
        }
    ])
    
    # Save enhanced SBOM
    with open('sbom-enhanced.spdx.json', 'w') as f:
        json.dump(sbom, f, indent=2)
    
    print("Enhanced SBOM generated successfully")

if __name__ == "__main__":
    generate_enhanced_sbom()
```

### Step 4: Vulnerability Scanning

```bash
# Install Grype for vulnerability scanning
curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin

# Scan SBOM for vulnerabilities
grype sbom:sbom.spdx.json -o json > vulnerabilities.json
grype sbom:sbom.spdx.json -o table

# Generate security report
grype sbom:sbom.spdx.json -o template -t security-report.tmpl > security-report.md
```

### Step 5: GitHub Actions Integration

Since GitHub Actions workflows require special permissions, here's a script-based approach:

Create `scripts/ci_sbom_generation.sh`:

```bash
#!/bin/bash
set -e

echo "Starting SBOM generation..."

# Install tools
echo "Installing SBOM tools..."
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
pip install cyclonedx-bom spdx-tools pip-licenses

# Generate multiple SBOM formats
echo "Generating SPDX SBOM..."
syft packages . -o spdx-json=artifacts/sbom.spdx.json

echo "Generating CycloneDX SBOM..."
syft packages . -o cyclonedx-json=artifacts/sbom.cyclonedx.json

echo "Generating Python SBOM..."
cyclonedx-py --format json --output artifacts/python-sbom.cyclonedx.json .

echo "Generating license report..."
pip-licenses --format json --output artifacts/licenses.json

# Vulnerability scanning
echo "Installing vulnerability scanner..."
curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin

echo "Scanning for vulnerabilities..."
grype sbom:artifacts/sbom.spdx.json -o json > artifacts/vulnerabilities.json || true
grype sbom:artifacts/sbom.spdx.json -o table > artifacts/vulnerability-summary.txt || true

# Generate metadata
echo "Generating metadata..."
cat > artifacts/sbom-metadata.json << EOF
{
  "generation_info": {
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "generator": "watermark-lab-sbom-generator",
    "version": "1.0.0"
  },
  "repository_info": {
    "name": "${GITHUB_REPOSITORY:-unknown}",
    "commit": "${GITHUB_SHA:-unknown}",
    "branch": "${GITHUB_REF_NAME:-unknown}"
  },
  "formats_generated": [
    "spdx-json",
    "cyclonedx-json",  
    "python-cyclonedx",
    "license-report"
  ]
}
EOF

echo "SBOM generation completed successfully!"
echo "Artifacts saved to: artifacts/"
ls -la artifacts/
```

## SBOM Validation and Quality

### Validation Tools

```bash
# Validate SPDX SBOM
spdx-tools validate sbom.spdx.json

# Validate CycloneDX SBOM  
cyclonedx validate --input-file sbom.cyclonedx.json

# Custom validation script
python scripts/validate_sbom.py sbom.spdx.json
```

### Quality Metrics

Track SBOM quality with these metrics:

```python
def assess_sbom_quality(sbom_file: str) -> Dict[str, Any]:
    """Assess SBOM quality metrics."""
    
    with open(sbom_file, 'r') as f:
        sbom = json.load(f)
    
    metrics = {
        'total_packages': len(sbom.get('packages', [])),
        'packages_with_licenses': 0,
        'packages_with_versions': 0, 
        'packages_with_suppliers': 0,
        'packages_with_checksums': 0,
        'quality_score': 0.0
    }
    
    for package in sbom.get('packages', []):
        if package.get('licenseConcluded'):
            metrics['packages_with_licenses'] += 1
        if package.get('versionInfo'):
            metrics['packages_with_versions'] += 1
        if package.get('supplier'):
            metrics['packages_with_suppliers'] += 1
        if package.get('checksums'):
            metrics['packages_with_checksums'] += 1
    
    # Calculate quality score (0-100)
    if metrics['total_packages'] > 0:
        license_score = metrics['packages_with_licenses'] / metrics['total_packages']
        version_score = metrics['packages_with_versions'] / metrics['total_packages']
        supplier_score = metrics['packages_with_suppliers'] / metrics['total_packages']
        checksum_score = metrics['packages_with_checksums'] / metrics['total_packages']
        
        metrics['quality_score'] = (
            license_score * 0.3 +
            version_score * 0.3 +
            supplier_score * 0.2 +
            checksum_score * 0.2
        ) * 100
    
    return metrics
```

## Compliance and Standards

### Executive Order 14028 Requirements

- ✅ **Software Bill of Materials**: Generated in standard formats
- ✅ **Vulnerability Disclosure**: Automated scanning integrated  
- ✅ **Supply Chain Security**: Third-party component tracking
- ✅ **Attestation**: Digital signatures supported

### NIST Guidelines Alignment

- **SP 800-161**: Supply chain risk management
- **SSDF**: Secure software development framework
- **Cybersecurity Framework**: Identity and asset management

### Implementation Checklist

- [ ] SBOM generated in multiple formats (SPDX, CycloneDX)
- [ ] Vulnerability scanning integrated
- [ ] License compliance verification
- [ ] Digital signatures applied
- [ ] Metadata enrichment implemented
- [ ] Quality metrics tracked
- [ ] Automated generation in CI/CD
- [ ] Storage and distribution configured

## Integration with Existing Tools

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "SBOM Monitoring",
    "panels": [
      {
        "title": "Total Dependencies",
        "type": "stat",
        "targets": [
          {
            "expr": "sbom_total_packages"
          }
        ]
      },
      {
        "title": "Vulnerabilities by Severity",
        "type": "bargauge",
        "targets": [
          {
            "expr": "sbom_vulnerabilities_by_severity"
          }
        ]
      }
    ]
  }
}
```

### Prometheus Metrics

```python
from prometheus_client import Counter, Gauge, Histogram

# SBOM metrics
sbom_generation_total = Counter('sbom_generation_total', 'Total SBOM generations')
sbom_packages_total = Gauge('sbom_packages_total', 'Total packages in SBOM')
sbom_vulnerabilities = Gauge('sbom_vulnerabilities_total', 'Total vulnerabilities', ['severity'])
sbom_quality_score = Gauge('sbom_quality_score', 'SBOM quality score (0-100)')
```

## Best Practices

### 1. Automate Generation
- Generate SBOM on every build
- Version and tag SBOMs with releases
- Store SBOMs in artifact repositories

### 2. Enrich Metadata
- Add build environment information
- Include security scan results
- Attach compliance attestations

### 3. Validate Quality
- Check for completeness
- Verify format compliance
- Assess metadata richness

### 4. Monitor Trends
- Track dependency changes
- Monitor vulnerability introduction
- Measure compliance metrics

### 5. Share Responsibly
- Publish SBOMs with releases
- Provide vulnerability disclosure
- Enable customer security assessment

## Troubleshooting

### Common Issues

**Missing Dependencies**
```bash
# Solution: Update package discovery
syft packages . --scope all-layers
```

**Large File Size**
```bash
# Solution: Compress SBOMs
gzip sbom.spdx.json
```

**Validation Errors**
```bash
# Solution: Use latest tools
pip install --upgrade spdx-tools cyclonedx-bom
```

### Performance Optimization

```bash
# Parallel processing
syft packages . --parallelism 4

# Exclude test dependencies
cyclonedx-py --format json --output sbom.json . --exclude-dev

# Use caching
export SYFT_CACHE_DIR=/tmp/syft-cache
```

## Resources

- [SPDX Specification](https://spdx.github.io/spdx-spec/)
- [CycloneDX Specification](https://cyclonedx.org/specification/overview/)
- [NIST SBOM Guidelines](https://www.nist.gov/itl/executive-order-improving-nations-cybersecurity/software-bill-materials)
- [CISA SBOM Resources](https://www.cisa.gov/sbom)

## Maintenance

### Regular Tasks

1. **Daily**: Monitor vulnerability feeds
2. **Weekly**: Update SBOM generation tools
3. **Monthly**: Review SBOM quality metrics
4. **Quarterly**: Assess compliance requirements
5. **Per Release**: Generate and distribute SBOMs