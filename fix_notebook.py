#!/usr/bin/env python3
"""Auto-fix common notebook corruption issues."""

import json
import re
import sys

def fix_cell_source(source_lines):
    """Fix common issues in cell source."""
    # Join into single string first
    code = ''.join(source_lines)

    # Fix 1: Remove garbage text appended to valid lines
    # Pattern: valid_code)garbage or valid_code]garbage
    code = re.sub(r'(\)\s*)(ETA\s+Bu.*?)(\n|$)', r')\3', code)
    code = re.sub(r'(\]\s*)(ETA\s+Bu.*?)(\n|$)', r']\3', code)
    code = re.sub(r'(#\s+CB)(ETA\s+Bu.*?)(\n|$)', r'\3', code)

    # Fix 2: Remove duplicate lines
    lines = code.split('\n')
    seen = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        # Allow empty lines and common patterns
        if not stripped or stripped in ('', 'pass', 'continue', 'break'):
            new_lines.append(line)
        elif stripped not in seen:
            seen.add(stripped)
            new_lines.append(line)
        # If duplicate, skip but warn
        else:
            print(f"  Removed duplicate: {stripped[:50]}...")

    # Fix 3: Fix merged statements (two statements on one line without proper separator)
    fixed_lines = []
    for line in new_lines:
        # Check for patterns like: statement1)statement2
        if re.search(r'\)[a-z_]+\(', line) or re.search(r'\)print\(', line):
            # Try to split
            parts = re.split(r'(\)[a-z])', line)
            if len(parts) > 1:
                print(f"  Splitting merged line: {line[:50]}...")
                # Reconstruct
                reconstructed = []
                for i, part in enumerate(parts):
                    if re.match(r'\)[a-z]', part):
                        reconstructed.append(part[0])  # The )
                        if i + 1 < len(parts):
                            fixed_lines.append(''.join(reconstructed))
                            reconstructed = [part[1]]  # Start of next statement
                    else:
                        reconstructed.append(part)
                if reconstructed:
                    fixed_lines.append(''.join(reconstructed))
                continue
        fixed_lines.append(line)

    # Convert back to source format
    result = [line + '\n' for line in fixed_lines[:-1]]
    if fixed_lines and fixed_lines[-1]:
        result.append(fixed_lines[-1])

    return result

def fix_notebook(path):
    """Fix notebook and return True if changes made."""
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue

        cell_id = cell.get('id', 'unknown')
        source = cell.get('source', [])

        # Check for syntax errors
        code = ''.join(source)
        try:
            compile(code, f'<{cell_id}>', 'exec')
            continue  # No errors, skip
        except SyntaxError:
            pass

        print(f"Fixing {cell_id}...")
        fixed = fix_cell_source(source)

        # Verify fix worked
        fixed_code = ''.join(fixed)
        try:
            compile(fixed_code, f'<{cell_id}>', 'exec')
            cell['source'] = fixed
            changed = True
            print(f"  Fixed successfully")
        except SyntaxError as e:
            print(f"  Auto-fix failed: {e.msg} at line {e.lineno}")

    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Saved changes to {path}")

    return changed

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "BIP_v10.12.ipynb"
    fix_notebook(path)
