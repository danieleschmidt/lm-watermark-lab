#!/usr/bin/env python3
"""
License compliance checker for LM Watermark Lab
"""

import json
import sys
from typing import Dict, List, Set
import argparse

# Define license categories and policies
LICENSE_POLICIES = {
    "approved": {
        "MIT",
        "Apache Software License",
        "Apache 2.0",
        "Apache License 2.0",
        "BSD License",
        "BSD 3-Clause",
        "BSD 2-Clause",
        "ISC License",
        "ISC",
        "Python Software Foundation License",
        "Mozilla Public License 2.0",
        "MPL-2.0"
    },
    "copyleft_weak": {
        "GNU Lesser General Public License v2 or later (LGPLv2+)",
        "GNU Lesser General Public License v3 or later (LGPLv3+)",
        "LGPL-2.1",
        "LGPL-3.0",
        "Mozilla Public License 2.0 (MPL 2.0)"
    },
    "copyleft_strong": {
        "GNU General Public License v2 (GPLv2)",
        "GNU General Public License v3 (GPLv3)",
        "GPL-2.0",
        "GPL-3.0",
        "GNU Affero General Public License v3",
        "AGPL-3.0"
    },
    "unknown": {
        "UNKNOWN"
    }
}

# Reverse mapping for quick lookup
LICENSE_CATEGORY_MAP = {}
for category, licenses in LICENSE_POLICIES.items():
    for license_name in licenses:
        LICENSE_CATEGORY_MAP[license_name.lower()] = category


def normalize_license_name(license_name: str) -> str:
    """Normalize license name for comparison"""
    if not license_name:
        return "UNKNOWN"
    
    # Clean up common variations
    normalized = license_name.strip()
    
    # Handle common variations
    mappings = {
        "apache": "Apache 2.0",
        "apache license": "Apache 2.0",
        "apache software license": "Apache 2.0",
        "bsd": "BSD License",
        "bsd license (bsd)": "BSD License",
        "new bsd license": "BSD 3-Clause",
        "mit license": "MIT",
        "isc license (iscl)": "ISC",
        "python software foundation license": "Python Software Foundation License"
    }
    
    normalized_lower = normalized.lower()
    for pattern, replacement in mappings.items():
        if pattern in normalized_lower:
            return replacement
    
    return normalized


def get_license_category(license_name: str) -> str:
    """Get the category of a license"""
    normalized = normalize_license_name(license_name)
    return LICENSE_CATEGORY_MAP.get(normalized.lower(), "unknown")


def analyze_licenses(licenses_data: List[Dict]) -> Dict:
    """Analyze license data and categorize packages"""
    results = {
        "total_packages": len(licenses_data),
        "by_category": {
            "approved": [],
            "copyleft_weak": [],
            "copyleft_strong": [],
            "unknown": []
        },
        "summary": {
            "approved": 0,
            "copyleft_weak": 0,
            "copyleft_strong": 0,
            "unknown": 0
        },
        "violations": [],
        "warnings": []
    }
    
    for package_info in licenses_data:
        name = package_info.get("Name", "Unknown")
        version = package_info.get("Version", "Unknown")
        license_name = package_info.get("License", "UNKNOWN")
        
        category = get_license_category(license_name)
        
        package_entry = {
            "name": name,
            "version": version,
            "license": license_name,
            "normalized_license": normalize_license_name(license_name)
        }
        
        results["by_category"][category].append(package_entry)
        results["summary"][category] += 1
        
        # Check for violations and warnings
        if category == "copyleft_strong":
            results["violations"].append({
                "package": f"{name}=={version}",
                "license": license_name,
                "issue": "Strong copyleft license may require source code disclosure",
                "severity": "high"
            })
        elif category == "copyleft_weak":
            results["warnings"].append({
                "package": f"{name}=={version}",
                "license": license_name,
                "issue": "Weak copyleft license may have linking restrictions",
                "severity": "medium"
            })
        elif category == "unknown":
            results["warnings"].append({
                "package": f"{name}=={version}",
                "license": license_name,
                "issue": "Unknown or unrecognized license",
                "severity": "medium"
            })
    
    return results


def generate_report(results: Dict, output_format: str = "text") -> str:
    """Generate a license compliance report"""
    if output_format == "json":
        return json.dumps(results, indent=2)
    
    # Text format
    report_lines = []
    report_lines.append("# License Compliance Report")
    report_lines.append("")
    
    # Summary
    report_lines.append("## Summary")
    report_lines.append(f"Total packages: {results['total_packages']}")
    report_lines.append(f"✅ Approved licenses: {results['summary']['approved']}")
    report_lines.append(f"⚠️  Weak copyleft: {results['summary']['copyleft_weak']}")
    report_lines.append(f"❌ Strong copyleft: {results['summary']['copyleft_strong']}")
    report_lines.append(f"❓ Unknown licenses: {results['summary']['unknown']}")
    report_lines.append("")
    
    # Violations
    if results["violations"]:
        report_lines.append("## ❌ License Violations")
        for violation in results["violations"]:
            report_lines.append(f"- **{violation['package']}**: {violation['license']}")
            report_lines.append(f"  Issue: {violation['issue']}")
        report_lines.append("")
    
    # Warnings
    if results["warnings"]:
        report_lines.append("## ⚠️ License Warnings")
        for warning in results["warnings"]:
            report_lines.append(f"- **{warning['package']}**: {warning['license']}")
            report_lines.append(f"  Issue: {warning['issue']}")
        report_lines.append("")
    
    # Detailed breakdown
    for category, packages in results["by_category"].items():
        if not packages:
            continue
        
        emoji_map = {
            "approved": "✅",
            "copyleft_weak": "⚠️",
            "copyleft_strong": "❌",
            "unknown": "❓"
        }
        
        category_name = category.replace("_", " ").title()
        report_lines.append(f"## {emoji_map[category]} {category_name} Licenses")
        
        for package in packages:
            report_lines.append(f"- {package['name']} ({package['version']}): {package['license']}")
        
        report_lines.append("")
    
    return "\n".join(report_lines)


def check_compliance(results: Dict, strict_mode: bool = False) -> bool:
    """Check if the license compliance is acceptable"""
    # Always fail on strong copyleft
    if results["summary"]["copyleft_strong"] > 0:
        return False
    
    # In strict mode, also fail on weak copyleft and unknown licenses
    if strict_mode:
        if results["summary"]["copyleft_weak"] > 0 or results["summary"]["unknown"] > 0:
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Check license compliance")
    parser.add_argument("licenses_file", help="Path to licenses JSON file from pip-licenses")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--output", help="Output file (default: stdout)")
    parser.add_argument("--strict", action="store_true", help="Strict mode: fail on any non-approved licenses")
    parser.add_argument("--fail-on-violations", action="store_true", help="Exit with error code if violations found")
    
    args = parser.parse_args()
    
    # Load license data
    try:
        with open(args.licenses_file, 'r') as f:
            licenses_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: License file '{args.licenses_file}' not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in license file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Analyze licenses
    results = analyze_licenses(licenses_data)
    
    # Generate report
    report = generate_report(results, args.format)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"License report written to {args.output}")
    else:
        print(report)
    
    # Check compliance and exit accordingly
    if args.fail_on_violations:
        compliant = check_compliance(results, args.strict)
        if not compliant:
            print("❌ License compliance check failed!", file=sys.stderr)
            if results["violations"]:
                print(f"Found {len(results['violations'])} license violations", file=sys.stderr)
            if args.strict and (results["warnings"]):
                print(f"Found {len(results['warnings'])} license warnings (strict mode)", file=sys.stderr)
            sys.exit(1)
        else:
            print("✅ License compliance check passed!")
    

if __name__ == "__main__":
    main()