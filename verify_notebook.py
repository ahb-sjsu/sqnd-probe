#!/usr/bin/env python3
"""Verify BIP notebook integrity - run before and after any changes."""

import json
import sys

def verify_notebook(path):
    errors = []
    warnings = []

    # 1. Check JSON validity
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        print(f"FATAL: Invalid JSON - {e}")
        return False

    print(f"Notebook: {path}")
    print(f"Cells: {len(nb.get('cells', []))}")

    # 2. Check each code cell for Python syntax
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue

        cell_id = cell.get('id', 'unknown')
        source = cell.get('source', [])

        # Check source format (should be list of strings)
        if not isinstance(source, list):
            errors.append(f"{cell_id}: source is not a list")
            continue

        # Check each line ends with \n (except possibly last)
        for i, line in enumerate(source[:-1] if source else []):
            if not line.endswith('\n'):
                warnings.append(f"{cell_id} line {i}: missing trailing newline")

        # Join and check Python syntax
        code = ''.join(source)
        try:
            compile(code, f'<{cell_id}>', 'exec')
            print(f"  {cell_id}: OK ({len(source)} lines)")
        except SyntaxError as e:
            errors.append(f"{cell_id} line {e.lineno}: {e.msg}")
            # Show context
            lines = code.split('\n')
            if e.lineno and e.lineno <= len(lines):
                print(f"  {cell_id}: SYNTAX ERROR at line {e.lineno}")
                for offset in range(-2, 3):
                    ln = e.lineno + offset - 1
                    if 0 <= ln < len(lines):
                        marker = ">>> " if offset == 0 else "    "
                        print(f"    {marker}{ln+1}: {lines[ln][:70]}")

    # 3. Check cell IDs
    ids = [c.get('id') for c in nb.get('cells', []) if c.get('id')]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate cell IDs found")
    if len(ids) != len(nb.get('cells', [])):
        warnings.append("Some cells missing IDs")

    # Summary
    print()
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings[:5]:
            print(f"  - {w}")
        if len(warnings) > 5:
            print(f"  - ... and {len(warnings) - 5} more")

    if not errors:
        print("VERIFICATION PASSED")
        return True
    else:
        print("VERIFICATION FAILED")
        return False

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "BIP_v10.12.ipynb"
    success = verify_notebook(path)
    sys.exit(0 if success else 1)
