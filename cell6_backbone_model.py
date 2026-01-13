# Updated Cell 6 with backbone support

CELL6_NEW = '''#@title 6. Model Architecture { display-mode: "form" }
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

        # Get actual hidden size from model config (in case of mismatch)
        actual_hidden = self.encoder.config.hidden_size
        if actual_hidden != hidden_size:
            print(f"  Note: Using actual hidden size {actual_hidden} (config said {hidden_size})")
            hidden_size = actual_hidden

        self.hidden_size = hidden_size
        self.model_name = model_name

        # Projection to z_bond space (scales with backbone size)
        proj_hidden = min(512, hidden_size)  # Cap intermediate size
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
        print(f"  Trainable params: {trainable_params:,}")

    def forward(self, input_ids, attention_mask, adv_lambda=1.0):
        enc = self.encoder(input_ids, attention_mask)

        # Handle different pooling strategies
        if hasattr(enc, 'pooler_output') and enc.pooler_output is not None:
            # BERT-style models (XLM-R, LaBSE) have pooler_output
            pooled = enc.pooler_output
        else:
            # Sentence-transformers style: use CLS token
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
print(f"\\nLoading tokenizer for {BACKBONE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"  Vocab size: {tokenizer.vocab_size:,}")

# Dataset with Hohfeld support
class NativeDataset(Dataset):
    def __init__(self, ids_set, passages_file, bonds_file, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        # Build bonds lookup by passage_id
        bonds_by_id = {}
        with open(bonds_file) as fb:
            for line in fb:
                b = json.loads(line)
                bonds_by_id[b['passage_id']] = b

        # Load passages that have bonds
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

print(f"\\nModel architecture defined for {BACKBONE}")
print(f"  Backbone: {MODEL_NAME}")
print(f"  Hidden size: {BACKBONE_HIDDEN}")
print(f"  Bond classes: {len(BondType)}")
print(f"  Hohfeld states: {len(HohfeldState)}")
print(f"  Languages: {len(LANG_TO_IDX)}")
print(f"  Periods: {len(PERIOD_TO_IDX)}")
print(f"  Context classes: {len(CONTEXT_TO_IDX)}")
print("\\n" + "="*60)'''

print("Cell 6 backbone model ready")
