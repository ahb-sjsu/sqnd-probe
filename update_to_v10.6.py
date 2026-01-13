"""
Update BIP notebook to v10.6 with backbone selection (MiniLM, LaBSE, XLM-R)
"""
import json

# Load notebook
nb = json.load(open('BIP_v10.5_expanded.ipynb', encoding='utf-8'))

# ========== UPDATE CELL 0: Version ==========
cell0 = ''.join(nb['cells'][0]['source'])
cell0 = cell0.replace('BIP v10.5', 'BIP v10.6')
cell0 = cell0.replace('v10.5', 'v10.6')
lines = cell0.split('\n')
nb['cells'][0]['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
print("Cell 0: Updated version to v10.6")

# ========== UPDATE CELL 1: Add backbone selection ==========
cell1 = ''.join(nb['cells'][1]['source'])

# Find insertion point (after DRIVE_FOLDER line)
backbone_config = '''
#@markdown ---
#@markdown ## Model Backbone
BACKBONE = "MiniLM"  #@param ["MiniLM", "LaBSE", "XLM-R-base", "XLM-R-large"]
#@markdown - **MiniLM**: Fast, 118M params, good baseline
#@markdown - **LaBSE**: Best cross-lingual alignment, 471M params (recommended)
#@markdown - **XLM-R-base**: Strong multilingual, 270M params
#@markdown - **XLM-R-large**: Strongest representations, 550M params

# Backbone configurations
BACKBONE_CONFIGS = {
    "MiniLM": {
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "hidden_size": 384,
        "recommended_batch": {"L4/A100": 512, "T4": 256, "2xT4": 512, "SMALL": 128, "MINIMAL/CPU": 64},
    },
    "LaBSE": {
        "model_name": "sentence-transformers/LaBSE",
        "hidden_size": 768,
        "recommended_batch": {"L4/A100": 256, "T4": 128, "2xT4": 256, "SMALL": 64, "MINIMAL/CPU": 32},
    },
    "XLM-R-base": {
        "model_name": "xlm-roberta-base",
        "hidden_size": 768,
        "recommended_batch": {"L4/A100": 256, "T4": 128, "2xT4": 256, "SMALL": 64, "MINIMAL/CPU": 32},
    },
    "XLM-R-large": {
        "model_name": "xlm-roberta-large",
        "hidden_size": 1024,
        "recommended_batch": {"L4/A100": 128, "T4": 64, "2xT4": 128, "SMALL": 32, "MINIMAL/CPU": 16},
    },
}

BACKBONE_CONFIG = BACKBONE_CONFIGS[BACKBONE]
MODEL_NAME = BACKBONE_CONFIG["model_name"]
BACKBONE_HIDDEN = BACKBONE_CONFIG["hidden_size"]
'''

# Insert after DRIVE_FOLDER line
drive_folder_line = '#@markdown Folder name for persistent storage'
if drive_folder_line in cell1:
    idx = cell1.find(drive_folder_line) + len(drive_folder_line)
    cell1 = cell1[:idx] + backbone_config + cell1[idx:]
    print("Cell 1: Added backbone selection parameter")
else:
    print("Cell 1: WARNING - Could not find insertion point for backbone config")

# Update version in print statement
cell1 = cell1.replace('BIP v10.5', 'BIP v10.6')

# Update batch size selection to use backbone-specific recommendations
old_batch_logic = '''# Set optimal parameters based on hardware
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
    print(f"  ** Kaggle 2xT4 detected - increased batch size **")'''

new_batch_logic = '''# Set optimal parameters based on hardware
if VRAM_GB >= 22:      # L4 (24GB) or A100
    GPU_TIER = "L4/A100"
elif VRAM_GB >= 14:    # T4 (16GB)
    GPU_TIER = "T4"
elif VRAM_GB >= 10:
    GPU_TIER = "SMALL"
else:
    GPU_TIER = "MINIMAL/CPU"

# Kaggle with 2xT4 can use larger batch
if ENV_NAME == "KAGGLE" and GPU_COUNT >= 2:
    GPU_TIER = "2xT4"
    print(f"  ** Kaggle 2xT4 detected **")

# Get backbone-specific batch size
BATCH_SIZE = BACKBONE_CONFIG["recommended_batch"].get(GPU_TIER, 64)
print(f"  Backbone: {BACKBONE} -> batch size {BATCH_SIZE}")'''

if old_batch_logic in cell1:
    cell1 = cell1.replace(old_batch_logic, new_batch_logic)
    print("Cell 1: Updated batch size logic for backbone-specific recommendations")
else:
    print("Cell 1: WARNING - Could not find batch size logic to update")

# Update the settings output to include backbone
old_settings_print = '''print(f"  Environment:     {ENV_NAME}")
print(f"  GPU Tier:        {GPU_TIER}")
print(f"  Batch size:      {BATCH_SIZE}")'''

new_settings_print = '''print(f"  Environment:     {ENV_NAME}")
print(f"  GPU Tier:        {GPU_TIER}")
print(f"  Backbone:        {BACKBONE}")
print(f"  Batch size:      {BATCH_SIZE}")'''

if old_settings_print in cell1:
    cell1 = cell1.replace(old_settings_print, new_settings_print)
    print("Cell 1: Updated settings output to include backbone")

lines = cell1.split('\n')
nb['cells'][1]['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

# ========== UPDATE CELL 6: Model Architecture ==========
cell6_new = '''#@title 6. Model Architecture { display-mode: "form" }
#@markdown BIP model with configurable backbone and adversarial heads

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
import json

print("="*60)
print("MODEL ARCHITECTURE")
print("="*60)
print(f"Backbone: {BACKBONE} ({MODEL_NAME})")
print(f"Hidden size: {BACKBONE_HIDDEN}")

# Index mappings
BOND_TO_IDX = {bt.name: i for i, bt in enumerate(BondType)}
IDX_TO_BOND = {i: bt.name for i, bt in enumerate(BondType)}
LANG_TO_IDX = {'hebrew': 0, 'aramaic': 1, 'classical_chinese': 2, 'arabic': 3, 'english': 4}
IDX_TO_LANG = {i: l for l, i in LANG_TO_IDX.items()}
PERIOD_TO_IDX = {'BIBLICAL': 0, 'TANNAITIC': 1, 'AMORAIC': 2, 'RISHONIM': 3, 'ACHRONIM': 4,
                 'CONFUCIAN': 5, 'DAOIST': 6, 'QURANIC': 7, 'HADITH': 8, 'DEAR_ABBY': 9}
IDX_TO_PERIOD = {i: p for p, i in PERIOD_TO_IDX.items()}
HOHFELD_TO_IDX = {hs.name: i for i, hs in enumerate(HohfeldState)}
IDX_TO_HOHFELD = {i: hs.name for i, hs in enumerate(HohfeldState)}
CONTEXT_TO_IDX = {'prescriptive': 0, 'descriptive': 1, 'unknown': 2}
IDX_TO_CONTEXT = {i: c for c, i in CONTEXT_TO_IDX.items()}
CONFIDENCE_TO_WEIGHT = {'high': 2.0, 'medium': 1.0, 'low': 0.5}

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class BIPModel(nn.Module):
    def __init__(self, model_name=None, hidden_size=None, z_dim=64):
        super().__init__()
        # Use global config if not specified
        model_name = model_name or MODEL_NAME
        hidden_size = hidden_size or BACKBONE_HIDDEN

        print(f"  Loading encoder: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)

        # Get actual hidden size from model config
        actual_hidden = self.encoder.config.hidden_size
        if actual_hidden != hidden_size:
            print(f"  Note: Using actual hidden size {actual_hidden}")
            hidden_size = actual_hidden

        self.hidden_size = hidden_size
        self.model_name = model_name

        # Projection to z_bond space (scales with backbone size)
        proj_hidden = min(512, hidden_size)
        self.z_proj = nn.Sequential(
            nn.Linear(hidden_size, proj_hidden),
            nn.LayerNorm(proj_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_hidden, z_dim),
        )

        # Task heads
        self.bond_head = nn.Linear(z_dim, len(BondType))
        self.hohfeld_head = nn.Linear(z_dim, len(HohfeldState))

        # Adversarial heads
        self.language_head = nn.Linear(z_dim, len(LANG_TO_IDX))
        self.period_head = nn.Linear(z_dim, len(PERIOD_TO_IDX))

        # Context prediction head (auxiliary task)
        self.context_head = nn.Linear(z_dim, len(CONTEXT_TO_IDX))

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")

    def forward(self, input_ids, attention_mask, adv_lambda=1.0):
        enc = self.encoder(input_ids, attention_mask)

        # Handle different pooling strategies
        if hasattr(enc, 'pooler_output') and enc.pooler_output is not None:
            pooled = enc.pooler_output
        else:
            pooled = enc.last_hidden_state[:, 0]

        z = self.z_proj(pooled)

        # Bond prediction (main task)
        bond_pred = self.bond_head(z)
        hohfeld_pred = self.hohfeld_head(z)

        # Adversarial predictions (gradient reversal)
        z_rev = GradientReversalLayer.apply(z, adv_lambda)
        language_pred = self.language_head(z_rev)
        period_pred = self.period_head(z_rev)

        return {
            'bond_pred': bond_pred,
            'hohfeld_pred': hohfeld_pred,
            'language_pred': language_pred,
            'period_pred': period_pred,
            'context_pred': self.context_head(z),
            'z': z,
        }

# Initialize tokenizer for selected backbone
print(f"\\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"  Vocab size: {tokenizer.vocab_size:,}")

# Dataset with Hohfeld support
class NativeDataset(Dataset):
    def __init__(self, ids_set, passages_file, bonds_file, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        bonds_by_id = {}
        with open(bonds_file) as fb:
            for line in fb:
                b = json.loads(line)
                bonds_by_id[b['passage_id']] = b

        with open(passages_file) as fp:
            for line in tqdm(fp, desc="Loading", unit="line"):
                p = json.loads(line)
                if p['id'] in ids_set and p['id'] in bonds_by_id:
                    b = bonds_by_id[p['id']]
                    self.data.append({
                        'text': p['text'][:1000],
                        'language': p['language'],
                        'period': p['time_period'],
                        'bond': b.get('bond_type') or b.get('bonds', {}).get('primary_bond'),
                        'hohfeld': None,
                        'context': b.get('context') or b.get('bonds', {}).get('context', 'unknown'),
                        'confidence': b.get('confidence') or b.get('bonds', {}).get('confidence', 'medium'),
                    })
        print(f"  Loaded {len(self.data):,} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(item['text'], truncation=True, max_length=self.max_len,
                            padding='max_length', return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'bond_label': BOND_TO_IDX.get(item['bond'], 9),
            'language_label': LANG_TO_IDX.get(item['language'], 4),
            'period_label': PERIOD_TO_IDX.get(item['period'], 9),
            'hohfeld_label': HOHFELD_TO_IDX.get(item['hohfeld'], 0) if item['hohfeld'] else 0,
            'context_label': CONTEXT_TO_IDX.get(item['context'], 2),
            'sample_weight': CONFIDENCE_TO_WEIGHT.get(item['confidence'], 1.0),
            'language': item['language'],
            'context': item['context'],
            'confidence': item['confidence'],
        }

def collate_fn(batch):
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'bond_labels': torch.tensor([x['bond_label'] for x in batch]),
        'language_labels': torch.tensor([x['language_label'] for x in batch]),
        'period_labels': torch.tensor([x['period_label'] for x in batch]),
        'hohfeld_labels': torch.tensor([x['hohfeld_label'] for x in batch]),
        'context_labels': torch.tensor([x['context_label'] for x in batch]),
        'sample_weights': torch.tensor([x['sample_weight'] for x in batch], dtype=torch.float),
        'languages': [x['language'] for x in batch],
        'contexts': [x['context'] for x in batch],
        'confidences': [x['confidence'] for x in batch],
    }

print(f"\\nArchitecture ready for {BACKBONE}")
print(f"  Bond classes: {len(BondType)}")
print(f"  Languages: {len(LANG_TO_IDX)}")
print("\\n" + "="*60)'''

lines = cell6_new.split('\n')
nb['cells'][6]['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
print("Cell 6: Updated model architecture with backbone support")

# ========== UPDATE CELL 7: Remove hardcoded tokenizer ==========
cell7 = ''.join(nb['cells'][7]['source'])

# Remove the hardcoded tokenizer line
old_tokenizer = 'tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")'
new_tokenizer = '# tokenizer loaded in Cell 6 based on BACKBONE selection'

if old_tokenizer in cell7:
    cell7 = cell7.replace(old_tokenizer, new_tokenizer)
    print("Cell 7: Removed hardcoded tokenizer (now uses Cell 6 tokenizer)")
else:
    print("Cell 7: WARNING - Could not find hardcoded tokenizer to remove")

# Add backbone info to training output
old_train_header = '''print("="*60)
print("TRAINING BIP MODEL")
print("="*60)
print(f"\\nHardware-optimized settings:")
print(f"  GPU Tier:     {GPU_TIER}")'''

new_train_header = '''print("="*60)
print("TRAINING BIP MODEL")
print("="*60)
print(f"\\nSettings:")
print(f"  Backbone:     {BACKBONE}")
print(f"  GPU Tier:     {GPU_TIER}")'''

if old_train_header in cell7:
    cell7 = cell7.replace(old_train_header, new_train_header)
    print("Cell 7: Added backbone to training header")

lines = cell7.split('\n')
nb['cells'][7]['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

# ========== SAVE ==========
with open('BIP_v10.6_expanded.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n" + "="*60)
print("BIP v10.6 CREATED")
print("="*60)
print("""
Changes:
  - Cell 0: Version updated to v10.6
  - Cell 1: Added BACKBONE dropdown (MiniLM, LaBSE, XLM-R-base, XLM-R-large)
  - Cell 1: Batch sizes now backbone-specific
  - Cell 6: Model uses configurable backbone
  - Cell 6: Tokenizer initialized based on BACKBONE
  - Cell 7: Uses tokenizer from Cell 6

To use LaBSE:
  1. Set BACKBONE = "LaBSE" in Cell 1
  2. Run all cells

Saved to: BIP_v10.6_expanded.ipynb
""")
