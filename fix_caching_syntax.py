#!/usr/bin/env python3
"""Quick fix for caching.py syntax issues."""

import re

# Read the caching.py file
with open('src/watermark_lab/optimization/caching.py', 'r') as f:
    content = f.read()

# Fix escaped quotes
content = content.replace('\\"', '"')

# Fix method duplication by finding and removing duplicate methods
lines = content.split('\n')
cleaned_lines = []
skip_until_next_def = False

for i, line in enumerate(lines):
    # Skip duplicate method definitions
    if skip_until_next_def:
        if line.strip().startswith('def ') and 'exists' in line:
            skip_until_next_def = False
            cleaned_lines.append(line)
        elif line.strip().startswith('def ') and line.strip() != 'def exists(self, key: str) -> bool:':
            skip_until_next_def = False
            cleaned_lines.append(line)
        continue
        
    # Check for specific patterns that need fixing
    if '"""Check if key exists in memory cache."""' in line:
        skip_until_next_def = True
        continue
        
    cleaned_lines.append(line)

# Write back the cleaned content
with open('src/watermark_lab/optimization/caching.py', 'w') as f:
    f.write('\n'.join(cleaned_lines))

print("Fixed caching.py syntax issues")