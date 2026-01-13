import json

nb = json.load(open('BIP_v10.5_expanded.ipynb', encoding='utf-8'))

new_cell1 = '''#@title 1. Configuration & Setup { display-mode: "form" }
#@markdown ## Data Source Configuration
#@markdown Choose where to load data from:

USE_DRIVE_DATA = True  #@param {type:"boolean"}
#@markdown If True, load pre-processed data from persistent storage (faster)

REFRESH_DATA_FROM_SOURCE = False  #@param {type:"boolean"}
#@markdown If True, re-download from online sources even if cached data exists

DRIVE_FOLDER = "BIP_v10"  #@param {type:"string"}
#@markdown Folder name for persistent storage

#@markdown ---
#@markdown ## Run Setup

import time
import os
import sys
EXPERIMENT_START = time.time()

print("="*60)
print("BIP v10.5 - ENVIRONMENT DETECTION")
print("="*60)

# ===== ENVIRONMENT DETECTION =====
# Detect which cloud platform we're running on

ENV_NAME = "UNKNOWN"
ENV_GPU_QUOTA = "Unknown"
PERSISTENT_STORAGE = None
DATA_DIR = "/content"  # Default

def detect_environment():
    """Detect cloud environment and return (name, gpu_quota, storage_path, data_dir)"""

    # 1. Google Colab
    try:
        import google.colab
        return ("COLAB", "Free: T4 ~12h/day, Pro: L4/A100", "/content/drive/MyDrive", "/content")
    except ImportError:
        pass

    # 2. Kaggle Kernels
    if os.path.exists('/kaggle'):
        # Kaggle has /kaggle/input for datasets, /kaggle/working for output
        return ("KAGGLE", "Free: 2xT4 30h/week, TPU 30h/week", "/kaggle/working", "/kaggle/working")

    # 3. Lightning.ai Studios
    if os.environ.get('LIGHTNING_CLOUDSPACE_HOST') or os.path.exists('/teamspace'):
        # Lightning.ai has /teamspace/studios for persistent storage
        return ("LIGHTNING_AI", "Free: 22h/month GPU, Pro: A10G/H100", "/teamspace/studios", "/teamspace/studios")

    # 4. Paperspace Gradient
    if os.environ.get('PAPERSPACE_NOTEBOOK_REPO_ID') or os.path.exists('/notebooks'):
        return ("PAPERSPACE", "Free: M4000 6h, Pro: A100/H100", "/storage", "/notebooks")

    # 5. Saturn Cloud
    if os.environ.get('SATURN_RESOURCE_ID') or 'saturn' in os.environ.get('HOSTNAME', '').lower():
        return ("SATURN_CLOUD", "Free: T4 10h/month, Pro: A10G/A100", "/home/jovyan/workspace", "/home/jovyan")

    # 6. HuggingFace Spaces
    if os.environ.get('SPACE_ID') or os.environ.get('HF_SPACE_ID'):
        return ("HUGGINGFACE_SPACES", "Free: CPU only, ZeroGPU: A10G/A100 quota", "/data", "/home/user/app")

    # 7. AWS SageMaker Studio Lab
    if os.path.exists('/home/studio-lab-user'):
        return ("SAGEMAKER_STUDIO_LAB", "Free: T4 4h/session, 24h max/day", "/home/studio-lab-user", "/home/studio-lab-user")

    # 8. Deepnote
    if os.environ.get('DEEPNOTE_PROJECT_ID'):
        return ("DEEPNOTE", "Free: CPU, Pro: T4/A10G", "/work", "/work")

    # 9. Local/Unknown
    return ("LOCAL", "Depends on local hardware", os.getcwd(), os.getcwd())

ENV_NAME, ENV_GPU_QUOTA, PERSISTENT_STORAGE, DATA_DIR = detect_environment()

print(f"\\nEnvironment: {ENV_NAME}")
print(f"GPU Quota:   {ENV_GPU_QUOTA}")
print(f"Storage:     {PERSISTENT_STORAGE}")
print(f"Data Dir:    {DATA_DIR}")

# Environment-specific setup
ENV_TIPS = {
    "COLAB": [
        "Tip: Use GPU runtime (Runtime -> Change runtime type -> T4 GPU)",
        "Tip: Colab Pro gives L4 GPU access (~2x faster than T4)"
    ],
    "KAGGLE": [
        "Tip: Enable GPU (Settings -> Accelerator -> GPU T4 x2)",
        "Tip: 30h/week GPU quota resets every Saturday",
        "Tip: Upload data as a Kaggle Dataset for persistence"
    ],
    "LIGHTNING_AI": [
        "Tip: Select GPU studio (A10G recommended for this workload)",
        "Tip: /teamspace/studios persists across sessions"
    ],
    "PAPERSPACE": [
        "Tip: Use /storage for persistent data across runs",
        "Tip: Free tier has 6h/month GPU limit"
    ],
    "SATURN_CLOUD": [
        "Tip: Start a T4 instance from the Resources tab",
        "Tip: 10h/month free GPU quota"
    ],
    "HUGGINGFACE_SPACES": [
        "Tip: ZeroGPU provides A10G/A100 access with quota system",
        "Tip: Use Gradio/Streamlit for interactive demos"
    ],
    "SAGEMAKER_STUDIO_LAB": [
        "Tip: Request GPU runtime from the launcher",
        "Tip: Sessions timeout after 4h, max 24h/day"
    ],
    "LOCAL": [
        "Tip: Running locally - ensure CUDA is installed for GPU support"
    ]
}

print(f"\\n" + "-"*60)
print("ENVIRONMENT TIPS:")
for tip in ENV_TIPS.get(ENV_NAME, ["No specific tips for this environment"]):
    print(f"  {tip}")
print("-"*60)

# ===== INSTALL DEPENDENCIES =====
import subprocess

print("\\nInstalling dependencies...")
for pkg in ["transformers", "sentence-transformers", "pandas", "tqdm", "scikit-learn", "pyyaml", "psutil"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import torch
import psutil

print("\\n" + "="*60)
print("GPU DETECTION & RESOURCE ALLOCATION")
print("="*60)

# Detect hardware
if torch.cuda.is_available():
    GPU_NAME = torch.cuda.get_device_name(0)
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    GPU_COUNT = torch.cuda.device_count()
else:
    GPU_NAME = "CPU"
    VRAM_GB = 0
    GPU_COUNT = 0

RAM_GB = psutil.virtual_memory().total / 1e9

print(f"\\nDetected Hardware:")
print(f"  GPU:  {GPU_NAME}" + (f" (x{GPU_COUNT})" if GPU_COUNT > 1 else ""))
print(f"  VRAM: {VRAM_GB:.1f} GB" + (f" (total: {VRAM_GB*GPU_COUNT:.1f} GB)" if GPU_COUNT > 1 else ""))
print(f"  RAM:  {RAM_GB:.1f} GB")

# Set optimal parameters based on hardware
if VRAM_GB >= 22:      # L4 (24GB) or A100
    BATCH_SIZE = 512
    GPU_TIER = "L4/A100"
elif VRAM_GB >= 14:    # T4 (16GB)
    BATCH_SIZE = 256
    GPU_TIER = "T4"
elif VRAM_GB >= 10:
    BATCH_SIZE = 128
    GPU_TIER = "SMALL"
else:
    BATCH_SIZE = 64
    GPU_TIER = "MINIMAL/CPU"

# Kaggle with 2xT4 can use larger batch
if ENV_NAME == "KAGGLE" and GPU_COUNT >= 2:
    BATCH_SIZE = min(512, BATCH_SIZE * 2)
    GPU_TIER = "2xT4"
    print(f"  ** Kaggle 2xT4 detected - increased batch size **")

MAX_PER_LANG = 50000  # Language sample limit
CPU_CORES = os.cpu_count() or 2
NUM_WORKERS = min(4, CPU_CORES - 1) if RAM_GB >= 24 and VRAM_GB >= 14 else 0
MAX_TEST_SAMPLES = 20000
LR = 2e-5 * (BATCH_SIZE / 256)

print(f"\\n" + "-"*60)
print(f"OPTIMAL SETTINGS:")
print(f"-"*60)
print(f"  Environment:     {ENV_NAME}")
print(f"  GPU Tier:        {GPU_TIER}")
print(f"  Batch size:      {BATCH_SIZE}")
print(f"  Max per lang:    {MAX_PER_LANG:,}")
print(f"  DataLoader workers: {NUM_WORKERS}")
print(f"  Learning rate:   {LR:.2e}")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

# ===== PERSISTENT STORAGE SETUP =====
print("\\n" + "="*60)
print("PERSISTENT STORAGE SETUP")
print("="*60)

SAVE_DIR = None
DRIVE_HAS_DATA = False
DRIVE_FILES = []

if ENV_NAME == "COLAB":
    # Google Colab - mount Drive
    try:
        from google.colab import drive
        DRIVE_MOUNT_PATH = '/content/drive'

        if os.path.exists(f'{DRIVE_MOUNT_PATH}/MyDrive'):
            print("Google Drive already mounted")
        else:
            try:
                drive.mount(DRIVE_MOUNT_PATH, force_remount=False)
                print("Google Drive mounted successfully")
            except Exception as e:
                print(f"Drive mount issue: {e}")
                try:
                    drive.mount(DRIVE_MOUNT_PATH, force_remount=True)
                    print("Google Drive mounted (force remount)")
                except Exception as e2:
                    print(f"WARNING: Could not mount Drive: {e2}")
                    print("Falling back to local storage")
                    PERSISTENT_STORAGE = DATA_DIR

        SAVE_DIR = f'{DRIVE_MOUNT_PATH}/MyDrive/{DRIVE_FOLDER}'
    except Exception as e:
        print(f"Colab Drive setup failed: {e}")
        SAVE_DIR = f'{DATA_DIR}/{DRIVE_FOLDER}'

elif ENV_NAME == "KAGGLE":
    # Kaggle - use working directory
    SAVE_DIR = f'{PERSISTENT_STORAGE}/{DRIVE_FOLDER}'
    print(f"Using Kaggle working directory: {SAVE_DIR}")
    print("Note: Data persists until kernel is reset")
    # Check for uploaded datasets
    if os.path.exists('/kaggle/input'):
        datasets = os.listdir('/kaggle/input')
        if datasets:
            print(f"Available datasets: {datasets[:5]}")

elif ENV_NAME == "LIGHTNING_AI":
    SAVE_DIR = f'{PERSISTENT_STORAGE}/{DRIVE_FOLDER}'
    print(f"Using Lightning.ai studio storage: {SAVE_DIR}")

elif ENV_NAME == "PAPERSPACE":
    SAVE_DIR = f'{PERSISTENT_STORAGE}/{DRIVE_FOLDER}'
    print(f"Using Paperspace /storage: {SAVE_DIR}")

elif ENV_NAME == "HUGGINGFACE_SPACES":
    # HF Spaces has limited persistent storage
    SAVE_DIR = f'{PERSISTENT_STORAGE}/{DRIVE_FOLDER}'
    print(f"Using HuggingFace Spaces storage: {SAVE_DIR}")
    print("Warning: HF Spaces storage is limited")

else:
    SAVE_DIR = f'{PERSISTENT_STORAGE}/{DRIVE_FOLDER}'
    print(f"Using local storage: {SAVE_DIR}")

os.makedirs(SAVE_DIR, exist_ok=True)

# Check what's available in storage
if os.path.exists(SAVE_DIR):
    DRIVE_FILES = os.listdir(SAVE_DIR)
    DRIVE_HAS_DATA = 'passages.jsonl' in DRIVE_FILES and 'bonds.jsonl' in DRIVE_FILES

print(f"\\n" + "-"*60)
print(f"STORAGE STATUS:")
print(f"-"*60)
print(f"  Folder: {SAVE_DIR}")
print(f"  Files found: {len(DRIVE_FILES)}")
if DRIVE_FILES:
    for f in DRIVE_FILES[:10]:
        print(f"    - {f}")
    if len(DRIVE_FILES) > 10:
        print(f"    ... and {len(DRIVE_FILES)-10} more")
print(f"  Pre-processed data available: {DRIVE_HAS_DATA}")

# Decide data loading strategy
LOAD_FROM_DRIVE = USE_DRIVE_DATA and DRIVE_HAS_DATA and not REFRESH_DATA_FROM_SOURCE

print(f"\\n" + "="*60)
print(f"DATA LOADING STRATEGY:")
if LOAD_FROM_DRIVE:
    print(f"  -> Will load pre-processed data from {ENV_NAME} storage")
    print(f"     (Set REFRESH_DATA_FROM_SOURCE=True to re-download)")
else:
    print(f"  -> Will download and process data from online sources")
    if USE_DRIVE_DATA and not DRIVE_HAS_DATA:
        print(f"     (Cached data not found, downloading fresh)")
print("="*60)

# Create local directories
for d in ["data/processed", "data/splits", "data/raw", "models/checkpoints", "results"]:
    os.makedirs(d, exist_ok=True)

print(f"\\n" + "="*60)
print(f"SETUP COMPLETE")
print(f"="*60)
print(f"  Environment: {ENV_NAME}")
print(f"  GPU:         {GPU_NAME} ({GPU_TIER})")
print(f"  Storage:     {SAVE_DIR}")
print(f"  Ready to run: Cell 2 (Imports)")
'''

# Update the cell
lines = new_cell1.split('\n')
nb['cells'][1]['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

# Save notebook
with open('BIP_v10.5_expanded.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("="*60)
print("ENVIRONMENT DETECTION ADDED")
print("="*60)
print("""
Supported environments:
  1. Google Colab      - T4/L4/A100, Google Drive storage
  2. Kaggle            - 2xT4, /kaggle/working storage
  3. Lightning.ai      - A10G/H100, /teamspace/studios storage
  4. Paperspace        - M4000/A100, /storage persistent
  5. Saturn Cloud      - T4/A10G, workspace storage
  6. HuggingFace Spaces- ZeroGPU, limited /data storage
  7. AWS SageMaker Lab - T4, home directory storage
  8. Deepnote          - T4/A10G, /work storage
  9. Local             - Uses current directory

Features:
  - Auto-detects environment
  - Configures optimal batch size for multi-GPU (Kaggle 2xT4)
  - Sets up appropriate persistent storage paths
  - Shows environment-specific tips
  - Graceful fallback if detection fails
""")
