import json

nb = json.load(open('BIP_v10.5_expanded.ipynb', encoding='utf-8'))
cell7 = ''.join(nb['cells'][7]['source'])

# More robust warning suppression that handles "Exception ignored" messages
warning_suppression = '''#@title 7. Train BIP Model { display-mode: "form" }
#@markdown Training with tuned adversarial weights and hardware-optimized parameters

# ===== SUPPRESS DATALOADER MULTIPROCESSING WARNINGS =====
# These occur during garbage collection and bypass normal exception handling
import warnings
import sys
import os
import io
import logging

# Method 1: Filter warnings
warnings.filterwarnings('ignore', message='.*can only test a child process.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.utils.data')

# Method 2: Suppress logging
logging.getLogger('torch.utils.data.dataloader').setLevel(logging.CRITICAL)

# Method 3: Redirect stderr during DataLoader cleanup (most effective)
class StderrFilter(io.TextIOWrapper):
    """Filters out DataLoader multiprocessing cleanup messages from stderr"""
    def __init__(self, original):
        self.original = original
        self.buffer_lines = []

    def write(self, text):
        # Filter out the specific error patterns
        skip_patterns = [
            'can only test a child process',
            '_MultiProcessingDataLoaderIter.__del__',
            '_shutdown_workers',
            'Exception ignored in:',
            'w.is_alive()',
        ]
        # Buffer multi-line error messages
        if any(p in text for p in skip_patterns):
            return len(text)  # Pretend we wrote it
        # Also skip if it looks like part of a traceback for these errors
        if text.strip().startswith('^') and len(text.strip()) < 80:
            return len(text)
        if text.strip().startswith('File "/usr') and 'dataloader.py' in text:
            return len(text)
        if text.strip() == 'Traceback (most recent call last):':
            self.buffer_lines = [text]
            return len(text)
        if self.buffer_lines:
            self.buffer_lines.append(text)
            # Check if this is the DataLoader error traceback
            full_msg = ''.join(self.buffer_lines)
            if any(p in full_msg for p in skip_patterns):
                self.buffer_lines = []
                return len(text)
            # After 10 lines, flush if not the target error
            if len(self.buffer_lines) > 10:
                for line in self.buffer_lines:
                    self.original.write(line)
                self.buffer_lines = []
        return self.original.write(text)

    def flush(self):
        if self.buffer_lines:
            # Flush any remaining buffered content
            for line in self.buffer_lines:
                self.original.write(line)
            self.buffer_lines = []
        self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)

# Install the stderr filter
_original_stderr = sys.stderr
sys.stderr = StderrFilter(_original_stderr)

# Method 4: Patch the DataLoader cleanup function directly
try:
    import torch.utils.data.dataloader as dl_module
    _original_del = dl_module._MultiProcessingDataLoaderIter.__del__

    def _patched_del(self):
        try:
            _original_del(self)
        except (AssertionError, AttributeError, RuntimeError):
            pass  # Silently ignore cleanup errors

    dl_module._MultiProcessingDataLoaderIter.__del__ = _patched_del
except Exception:
    pass  # If patching fails, the stderr filter will still work

from sklearn.metrics import f1_score
import gc'''

# Find and replace the header section
# Look for the current header pattern
current_patterns = [
    '''#@title 7. Train BIP Model { display-mode: "form" }
#@markdown Training with tuned adversarial weights and hardware-optimized parameters

# Suppress PyTorch DataLoader multiprocessing warnings''',
    '''#@title 7. Train BIP Model { display-mode: "form" }
#@markdown Training with tuned adversarial weights and hardware-optimized parameters

from sklearn.metrics import f1_score'''
]

replaced = False
for pattern in current_patterns:
    if pattern in cell7:
        # Find where the pattern ends and where we should continue
        idx = cell7.find(pattern)
        # Find the line with "from sklearn" or after the existing suppression block
        sklearn_line = 'from sklearn.metrics import f1_score'
        sklearn_idx = cell7.find(sklearn_line, idx)
        if sklearn_idx > 0:
            # Find end of gc import
            gc_line = 'import gc'
            gc_idx = cell7.find(gc_line, sklearn_idx)
            if gc_idx > 0:
                gc_end = gc_idx + len(gc_line)
                cell7 = warning_suppression + cell7[gc_end:]
                replaced = True
                print(f"Replaced header (pattern found)")
                break

if not replaced:
    print("WARNING: Could not find header pattern to replace")
    print("First 500 chars of cell7:")
    print(cell7[:500])

# Save back
lines = cell7.split('\n')
nb['cells'][7]['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

with open('BIP_v10.5_expanded.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n=== Robust Warning Suppression Added ===")
print("Four-layer suppression:")
print("  1. warnings.filterwarnings() - catches warning module calls")
print("  2. logging.setLevel(CRITICAL) - suppresses torch logging")
print("  3. StderrFilter - filters 'Exception ignored' messages from stderr")
print("  4. Patched __del__ - catches errors at source")
