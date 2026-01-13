import json

nb = json.load(open('BIP_v10.5_expanded.ipynb', encoding='utf-8'))
cell7 = ''.join(nb['cells'][7]['source'])

# Add warning suppression at the very top of Cell 7
warning_suppression = '''#@title 7. Train BIP Model { display-mode: "form" }
#@markdown Training with tuned adversarial weights and hardware-optimized parameters

# Suppress PyTorch DataLoader multiprocessing warnings
import warnings
import sys
import logging

# Suppress the specific multiprocessing cleanup warnings
warnings.filterwarnings('ignore', message='.*can only test a child process.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.utils.data.dataloader')

# Suppress stderr for DataLoader cleanup exceptions
class SuppressDataLoaderWarnings:
    """Context manager to suppress DataLoader cleanup exceptions"""
    def __init__(self):
        self._original_excepthook = None

    def __enter__(self):
        self._original_excepthook = sys.excepthook
        def custom_excepthook(exc_type, exc_value, exc_tb):
            # Suppress multiprocessing cleanup errors
            if 'can only test a child process' in str(exc_value):
                return
            if exc_type.__name__ == 'AssertionError' and 'child process' in str(exc_value):
                return
            self._original_excepthook(exc_type, exc_value, exc_tb)
        sys.excepthook = custom_excepthook
        return self

    def __exit__(self, *args):
        sys.excepthook = self._original_excepthook

# Also suppress via logging
logging.getLogger('torch.utils.data.dataloader').setLevel(logging.ERROR)

from sklearn.metrics import f1_score
import gc'''

# Find the old header and replace it
old_header = '''#@title 7. Train BIP Model { display-mode: "form" }
#@markdown Training with tuned adversarial weights and hardware-optimized parameters

from sklearn.metrics import f1_score
import gc'''

if old_header in cell7:
    cell7 = cell7.replace(old_header, warning_suppression)
    print("Added warning suppression header")
else:
    print("Could not find old header - checking alternative...")
    # Try alternative pattern
    alt_header = '#@title 7. Train BIP Model { display-mode: "form" }'
    if alt_header in cell7:
        idx = cell7.find(alt_header)
        # Find the "from sklearn" line
        sklearn_idx = cell7.find('from sklearn.metrics import f1_score')
        if sklearn_idx > idx:
            # Replace from title to after gc import
            gc_idx = cell7.find('import gc', sklearn_idx)
            if gc_idx > 0:
                gc_end = gc_idx + len('import gc')
                cell7 = warning_suppression + cell7[gc_end:]
                print("Added warning suppression (alternative method)")

# Save back
lines = cell7.split('\n')
nb['cells'][7]['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

with open('BIP_v10.5_expanded.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n=== Warning Suppression Added ===")
print("Suppresses:")
print("  - 'can only test a child process' AssertionErrors")
print("  - DataLoader multiprocessing cleanup warnings")
print("  - torch.utils.data.dataloader UserWarnings")
