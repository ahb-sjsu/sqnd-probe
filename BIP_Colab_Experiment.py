"""
BIP TEMPORAL INVARIANCE EXPERIMENT - Google Colab Version
==========================================================

Copy this entire file to a Google Colab notebook and run.

SETUP INSTRUCTIONS:
1. Go to https://colab.research.google.com
2. Create a new notebook
3. Change runtime to GPU: Runtime -> Change runtime type -> GPU
4. Copy and paste this entire script into a cell and run

Expected runtime: ~2-4 hours with free GPU
"""

# ============================================================================
# CELL 1: SETUP AND INSTALL
# ============================================================================
# Run this cell first

print("Setting up BIP Temporal Invariance Experiment...")
print("=" * 60)

# Install required packages
!pip install -q torch transformers sentence-transformers scipy scikit-learn pandas numpy tqdm pyyaml

import os
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/splits', exist_ok=True)
os.makedirs('models/checkpoints', exist_ok=True)

# Check GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# CELL 2: DOWNLOAD DATA
# ============================================================================

print("\n" + "=" * 60)
print("DOWNLOADING DATA")
print("=" * 60)

# Clone Sefaria (works on Linux without path issues)
print("\n[1/2] Cloning Sefaria corpus (~8GB, takes 5-10 minutes)...")
print("=" * 60)
!cd data/raw && git clone --depth 1 --progress https://github.com/Sefaria/Sefaria-Export.git 2>&1 || echo "Sefaria already exists"
print("=" * 60)

# Count files
!echo "Sefaria JSON files:" && find data/raw/Sefaria-Export/json -name "*.json" 2>/dev/null | wc -l

# Dear Abby data - from the sqnd-probe repo
print("\n[2/2] Dear Abby data...")

import pandas as pd
from pathlib import Path

# Clone the sqnd-probe repo to get the Dear Abby data
print("Cloning sqnd-probe repo to get Dear Abby dataset...")
print("=" * 60)
!git clone --depth 1 --progress https://github.com/ahb-sjsu/sqnd-probe.git sqnd-probe-data 2>&1 || echo "Repo already cloned"
print("=" * 60)

# Copy Dear Abby data from repo
dear_abby_source = Path('sqnd-probe-data/dear_abby_data/raw_da_qs.csv')
dear_abby_path = Path('data/raw/dear_abby.csv')

if dear_abby_source.exists():
    !cp "{dear_abby_source}" "{dear_abby_path}"
    print(f"Copied Dear Abby data from repo")
elif not dear_abby_path.exists():
    raise FileNotFoundError(
        "EXPERIMENT HALTED: Dear Abby dataset not found.\n"
        "Expected at: sqnd-probe-data/dear_abby_data/raw_da_qs.csv\n"
        "Make sure the repo is public or upload the file manually."
    )

# Verify the data
df_check = pd.read_csv(dear_abby_path)
print(f"Dear Abby dataset loaded: {len(df_check):,} entries")
print(f"Columns: {list(df_check.columns)}")
print(f"Year range: {df_check['year'].min():.0f} - {df_check['year'].max():.0f}")

# ============================================================================
# CELL 3: DATA CLASSES AND PREPROCESSING
# ============================================================================

import json
import hashlib
import re
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
from collections import defaultdict
from tqdm.auto import tqdm

class TimePeriod(Enum):
    BIBLICAL = 0
    SECOND_TEMPLE = 1
    TANNAITIC = 2
    AMORAIC = 3
    GEONIC = 4
    RISHONIM = 5
    ACHRONIM = 6
    MODERN_HEBREW = 7
    DEAR_ABBY = 8

class BondType(Enum):
    HARM_PREVENTION = 0
    RECIPROCITY = 1
    AUTONOMY = 2
    PROPERTY = 3
    FAMILY = 4
    AUTHORITY = 5
    EMERGENCY = 6
    CONTRACT = 7
    CARE = 8
    FAIRNESS = 9

class HohfeldianState(Enum):
    RIGHT = 0
    OBLIGATION = 1
    LIBERTY = 2
    NO_RIGHT = 3

class ConsentStatus(Enum):
    EXPLICIT_YES = 0
    IMPLICIT_YES = 1
    CONTESTED = 2
    IMPLICIT_NO = 3
    EXPLICIT_NO = 4
    IMPOSSIBLE = 5

@dataclass
class Passage:
    id: str
    text_original: str
    text_english: str
    time_period: str
    century: int
    source: str
    source_type: str
    category: str
    language: str = "hebrew"
    word_count: int = 0
    has_dispute: bool = False
    consensus_tier: str = "unknown"
    bond_types: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

# ============================================================================
# CELL 4: LOADERS
# ============================================================================

CATEGORY_TO_PERIOD = {
    'Tanakh': TimePeriod.BIBLICAL,
    'Torah': TimePeriod.BIBLICAL,
    'Mishnah': TimePeriod.TANNAITIC,
    'Tosefta': TimePeriod.TANNAITIC,
    'Talmud': TimePeriod.AMORAIC,
    'Bavli': TimePeriod.AMORAIC,
    'Midrash': TimePeriod.AMORAIC,
    'Halakhah': TimePeriod.RISHONIM,
    'Chasidut': TimePeriod.ACHRONIM,
}

PERIOD_TO_CENTURY = {
    TimePeriod.BIBLICAL: -6,
    TimePeriod.SECOND_TEMPLE: -2,
    TimePeriod.TANNAITIC: 2,
    TimePeriod.AMORAIC: 4,
    TimePeriod.GEONIC: 8,
    TimePeriod.RISHONIM: 12,
    TimePeriod.ACHRONIM: 17,
    TimePeriod.MODERN_HEBREW: 20,
}

def load_sefaria(base_path: str, max_passages: int = None) -> List[Passage]:
    """Load Sefaria corpus."""
    passages = []
    json_path = Path(base_path) / "json"

    if not json_path.exists():
        print(f"Warning: {json_path} not found")
        return []

    print("Loading Sefaria corpus...")

    # Find all JSON files
    json_files = list(json_path.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    for json_file in tqdm(json_files[:max_passages] if max_passages else json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue

        # Infer period from path
        rel_path = json_file.relative_to(json_path)
        category = str(rel_path.parts[0]) if rel_path.parts else "unknown"
        time_period = CATEGORY_TO_PERIOD.get(category, TimePeriod.AMORAIC)
        century = PERIOD_TO_CENTURY.get(time_period, 0)

        # Extract text
        if isinstance(data, dict):
            hebrew = data.get('he', data.get('text', []))
            english = data.get('text', data.get('en', []))

            def flatten(h, e, ref=""):
                if isinstance(h, str) and isinstance(e, str):
                    h_clean = re.sub(r'<[^>]+>', '', h).strip()
                    e_clean = re.sub(r'<[^>]+>', '', e).strip()

                    if 50 <= len(e_clean) <= 2000:
                        pid = hashlib.md5(f"{json_file.stem}:{ref}:{h_clean[:50]}".encode()).hexdigest()[:12]
                        return [Passage(
                            id=f"sefaria_{pid}",
                            text_original=h_clean,
                            text_english=e_clean,
                            time_period=time_period.name,
                            century=century,
                            source=f"{json_file.stem} {ref}".strip(),
                            source_type="sefaria",
                            category=category,
                            language="hebrew",
                            word_count=len(e_clean.split())
                        )]
                    return []
                elif isinstance(h, list) and isinstance(e, list):
                    result = []
                    for i, (hh, ee) in enumerate(zip(h, e)):
                        result.extend(flatten(hh, ee, f"{ref}.{i+1}" if ref else str(i+1)))
                    return result
                return []

            passages.extend(flatten(hebrew, english))

    print(f"Loaded {len(passages)} Sefaria passages")
    return passages

def load_dear_abby(path: str, max_passages: int = None) -> List[Passage]:
    """Load Dear Abby corpus."""
    passages = []

    print("Loading Dear Abby corpus...")
    df = pd.read_csv(path)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = str(row.get('question_only', ''))
        if not question or question == 'nan' or len(question) < 50:
            continue

        if len(question) > 2000:
            continue

        year = int(row.get('year', 1990))
        pid = hashlib.md5(f"abby:{idx}:{question[:50]}".encode()).hexdigest()[:12]

        passages.append(Passage(
            id=f"abby_{pid}",
            text_original=question,
            text_english=question,
            time_period=TimePeriod.DEAR_ABBY.name,
            century=20 if year < 2000 else 21,
            source=f"Dear Abby {year}",
            source_type="dear_abby",
            category="general",
            language="english",
            word_count=len(question.split())
        ))

        if max_passages and len(passages) >= max_passages:
            break

    print(f"Loaded {len(passages)} Dear Abby passages")
    return passages

# ============================================================================
# CELL 5: BOND EXTRACTION
# ============================================================================

RELATION_PATTERNS = {
    BondType.HARM_PREVENTION: [r'\b(kill|murder|harm|hurt|save|rescue|protect|danger)\b'],
    BondType.RECIPROCITY: [r'\b(return|repay|owe|debt|mutual|exchange)\b'],
    BondType.AUTONOMY: [r'\b(choose|decision|consent|agree|force|coerce|right)\b'],
    BondType.PROPERTY: [r'\b(property|own|steal|theft|buy|sell|land)\b'],
    BondType.FAMILY: [r'\b(honor|parent|marry|divorce|inherit|family)\b'],
    BondType.AUTHORITY: [r'\b(obey|command|law|judge|rule|teach)\b'],
    BondType.CARE: [r'\b(care|help|assist|feed|clothe|visit)\b'],
    BondType.FAIRNESS: [r'\b(fair|just|equal|deserve|bias)\b'],
}

HOHFELD_PATTERNS = {
    HohfeldianState.OBLIGATION: [r'\b(must|shall|duty|require|should)\b'],
    HohfeldianState.RIGHT: [r'\b(right to|entitled|deserve)\b'],
    HohfeldianState.LIBERTY: [r'\b(may|can|permitted|allowed)\b'],
}

def extract_bond_structure(passage: Passage) -> Dict:
    """Extract bond structure from passage."""
    text = passage.text_english.lower()

    # Find relations
    relations = []
    for rel_type, patterns in RELATION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                relations.append(rel_type.name)
                break

    if not relations:
        relations = ['CARE']

    # Find Hohfeldian state
    hohfeld = None
    for state, patterns in HOHFELD_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                hohfeld = state.name
                break
        if hohfeld:
            break

    # Create signature
    signature = "|".join(sorted(set(relations)))

    return {
        'bonds': [{'relation': r} for r in relations],
        'primary_relation': relations[0],
        'hohfeld_state': hohfeld,
        'signature': signature
    }

# ============================================================================
# CELL 6: RUN PREPROCESSING
# ============================================================================

print("\n" + "=" * 60)
print("PREPROCESSING DATA")
print("=" * 60)

# Load corpora
sefaria_passages = load_sefaria("data/raw/Sefaria-Export")
abby_passages = load_dear_abby("data/raw/dear_abby.csv")

all_passages = sefaria_passages + abby_passages
print(f"\nTotal passages: {len(all_passages)}")

# Extract bonds
print("\nExtracting bond structures...")
bond_structures = []
for passage in tqdm(all_passages):
    bond_struct = extract_bond_structure(passage)
    passage.bond_types = [b['relation'] for b in bond_struct['bonds']]
    bond_structures.append({
        'passage_id': passage.id,
        'bond_structure': bond_struct
    })

# Save
print("\nSaving processed data...")
with open("data/processed/passages.jsonl", 'w') as f:
    for p in all_passages:
        f.write(json.dumps(p.to_dict()) + '\n')

with open("data/processed/bond_structures.jsonl", 'w') as f:
    for bs in bond_structures:
        f.write(json.dumps(bs) + '\n')

# Statistics
by_period = defaultdict(int)
by_source = defaultdict(int)
for p in all_passages:
    by_period[p.time_period] += 1
    by_source[p.source_type] += 1

print("\nBy source:")
for source, count in sorted(by_source.items()):
    print(f"  {source}: {count:,}")

print("\nBy time period:")
for period, count in sorted(by_period.items()):
    print(f"  {period}: {count:,}")

# ============================================================================
# CELL 7: GENERATE SPLITS
# ============================================================================

print("\n" + "=" * 60)
print("GENERATING SPLITS")
print("=" * 60)

import random
random.seed(42)

# Index passages
passages_by_period = defaultdict(list)
passages_by_source = defaultdict(list)

for p in all_passages:
    passages_by_period[p.time_period].append(p)
    passages_by_source[p.source_type].append(p)

# TEMPORAL HOLDOUT SPLIT (Primary BIP Test)
train_periods = {'BIBLICAL', 'SECOND_TEMPLE', 'TANNAITIC', 'AMORAIC', 'GEONIC', 'RISHONIM'}
valid_periods = {'ACHRONIM'}
test_periods = {'MODERN_HEBREW', 'DEAR_ABBY'}

train = [p for p in all_passages if p.time_period in train_periods]
valid = [p for p in all_passages if p.time_period in valid_periods]
test = [p for p in all_passages if p.time_period in test_periods]

random.shuffle(train)
random.shuffle(valid)
random.shuffle(test)

temporal_holdout = {
    'name': 'temporal_holdout',
    'train_ids': [p.id for p in train],
    'valid_ids': [p.id for p in valid],
    'test_ids': [p.id for p in test],
    'train_size': len(train),
    'valid_size': len(valid),
    'test_size': len(test)
}

print(f"Temporal holdout split:")
print(f"  Train (ancient/medieval): {len(train):,}")
print(f"  Valid (early modern): {len(valid):,}")
print(f"  Test (modern): {len(test):,}")

# STRATIFIED RANDOM (Control)
random.shuffle(all_passages)
n = len(all_passages)
n_train = int(0.7 * n)
n_valid = int(0.15 * n)

stratified = {
    'name': 'stratified_random',
    'train_ids': [p.id for p in all_passages[:n_train]],
    'valid_ids': [p.id for p in all_passages[n_train:n_train+n_valid]],
    'test_ids': [p.id for p in all_passages[n_train+n_valid:]],
    'train_size': n_train,
    'valid_size': n_valid,
    'test_size': n - n_train - n_valid
}

print(f"\nStratified random split:")
print(f"  Train: {stratified['train_size']:,}")
print(f"  Valid: {stratified['valid_size']:,}")
print(f"  Test: {stratified['test_size']:,}")

# Save splits
splits = {
    'temporal_holdout': temporal_holdout,
    'stratified_random': stratified
}

with open("data/splits/all_splits.json", 'w') as f:
    json.dump(splits, f, indent=2)

print("\nSplits saved!")

# ============================================================================
# CELL 8: MODEL DEFINITION
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def gradient_reversal(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)

class BIPEncoder(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", d_model=384):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.d_model = d_model

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled

class BIPModel(nn.Module):
    def __init__(self, d_model=384, d_bond=64, d_label=32, n_periods=9, n_hohfeld=4):
        super().__init__()

        self.encoder = BIPEncoder()

        # Disentangle
        self.bond_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_bond)
        )

        self.label_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_label)
        )

        # Classifiers
        self.time_from_bond = nn.Linear(d_bond, n_periods)
        self.time_from_label = nn.Linear(d_label, n_periods)
        self.hohfeld_classifier = nn.Linear(d_bond, n_hohfeld)

    def forward(self, input_ids, attention_mask, adversarial_lambda=1.0):
        h = self.encoder(input_ids, attention_mask)

        z_bond = self.bond_proj(h)
        z_label = self.label_proj(h)

        # Time prediction (adversarial from z_bond)
        if adversarial_lambda > 0:
            z_bond_adv = gradient_reversal(z_bond, adversarial_lambda)
            time_pred_bond = self.time_from_bond(z_bond_adv)
        else:
            time_pred_bond = self.time_from_bond(z_bond)

        time_pred_label = self.time_from_label(z_label)
        hohfeld_pred = self.hohfeld_classifier(z_bond)

        return {
            'z_bond': z_bond,
            'z_label': z_label,
            'time_pred_bond': time_pred_bond,
            'time_pred_label': time_pred_label,
            'hohfeld_pred': hohfeld_pred
        }

# ============================================================================
# CELL 9: DATASET AND TRAINING
# ============================================================================

from torch.utils.data import Dataset, DataLoader

PERIOD_TO_IDX = {p.name: i for i, p in enumerate(TimePeriod)}
HOHFELD_TO_IDX = {h.name: i for i, h in enumerate(HohfeldianState)}

class MoralDataset(Dataset):
    def __init__(self, passage_ids, passages_file, bonds_file, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.passages = {}
        with open(passages_file) as f:
            for line in f:
                p = json.loads(line)
                if p['id'] in passage_ids:
                    self.passages[p['id']] = p

        self.bonds = {}
        with open(bonds_file) as f:
            for line in f:
                b = json.loads(line)
                self.bonds[b['passage_id']] = b['bond_structure']

        self.ids = [pid for pid in passage_ids if pid in self.passages]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        passage = self.passages[pid]

        encoded = self.tokenizer(
            passage['text_english'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        time_label = PERIOD_TO_IDX.get(passage['time_period'], 0)
        hohfeld = self.bonds.get(pid, {}).get('hohfeld_state')
        hohfeld_label = HOHFELD_TO_IDX.get(hohfeld, 0) if hohfeld else 0

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'time_label': time_label,
            'hohfeld_label': hohfeld_label
        }

def collate_fn(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'time_labels': torch.tensor([b['time_label'] for b in batch]),
        'hohfeld_labels': torch.tensor([b['hohfeld_label'] for b in batch])
    }

# ============================================================================
# CELL 10: TRAINING LOOP
# ============================================================================

print("\n" + "=" * 60)
print("TRAINING BIP MODEL")
print("=" * 60)

# Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Choose split based on available data
if temporal_holdout['train_size'] > 0:
    split_name = 'temporal_holdout'
    split = temporal_holdout
    print("Using TEMPORAL HOLDOUT split (primary BIP test)")
else:
    split_name = 'stratified_random'
    split = stratified
    print("Using STRATIFIED RANDOM split (Dear Abby only)")

# Initialize
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = BIPModel().to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Datasets
train_dataset = MoralDataset(
    split['train_ids'],
    "data/processed/passages.jsonl",
    "data/processed/bond_structures.jsonl",
    tokenizer
)
valid_dataset = MoralDataset(
    split['valid_ids'],
    "data/processed/passages.jsonl",
    "data/processed/bond_structures.jsonl",
    tokenizer
)
test_dataset = MoralDataset(
    split['test_ids'],
    "data/processed/passages.jsonl",
    "data/processed/bond_structures.jsonl",
    tokenizer
)

print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

# Skip if no training data
if len(train_dataset) == 0:
    print("\nWARNING: No training data in this split!")
    print("The BIP test requires data from multiple time periods.")
    print("Make sure Sefaria data was loaded successfully.")
else:
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Training
    n_epochs = 5
    best_valid_loss = float('inf')

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            time_labels = batch['time_labels'].to(device)
            hohfeld_labels = batch['hohfeld_labels'].to(device)

            outputs = model(input_ids, attention_mask, adversarial_lambda=1.0)

            # Loss: adversarial (maximize entropy), time from label, hohfeld
            time_probs = F.softmax(outputs['time_pred_bond'], dim=-1)
            entropy = -torch.sum(time_probs * torch.log(time_probs + 1e-8), dim=-1)
            loss_adv = -entropy.mean()

            loss_time = F.cross_entropy(outputs['time_pred_label'], time_labels)
            loss_hohfeld = F.cross_entropy(outputs['hohfeld_pred'], hohfeld_labels)

            loss = loss_adv + loss_time + loss_hohfeld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Validation
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                time_labels = batch['time_labels'].to(device)
                hohfeld_labels = batch['hohfeld_labels'].to(device)

                outputs = model(input_ids, attention_mask, adversarial_lambda=0)
                loss = F.cross_entropy(outputs['hohfeld_pred'], hohfeld_labels)
                valid_loss += loss.item()

            valid_loss /= len(valid_loader)

        print(f"Epoch {epoch}: Train loss = {total_loss/len(train_loader):.4f}, Valid loss = {valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "models/checkpoints/best_model.pt")
            print("  Saved best model!")

# ============================================================================
# CELL 11: EVALUATION
# ============================================================================

print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

if len(train_dataset) > 0:
    # Load best model
    model.load_state_dict(torch.load("models/checkpoints/best_model.pt"))
    model.eval()

    # Test evaluation
    all_time_preds = []
    all_time_labels = []
    all_hohfeld_preds = []
    all_hohfeld_labels = []
    all_z_bonds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask, adversarial_lambda=0)

            all_time_preds.extend(outputs['time_pred_bond'].argmax(dim=-1).cpu().tolist())
            all_time_labels.extend(batch['time_labels'].tolist())
            all_hohfeld_preds.extend(outputs['hohfeld_pred'].argmax(dim=-1).cpu().tolist())
            all_hohfeld_labels.extend(batch['hohfeld_labels'].tolist())
            all_z_bonds.append(outputs['z_bond'].cpu())

    # Metrics
    time_acc = sum(p == l for p, l in zip(all_time_preds, all_time_labels)) / len(all_time_preds)
    hohfeld_acc = sum(p == l for p, l in zip(all_hohfeld_preds, all_hohfeld_labels)) / len(all_hohfeld_preds)

    print("\n" + "=" * 60)
    print("BIP TEST RESULTS")
    print("=" * 60)

    print(f"\n1. TIME PREDICTION FROM z_bond:")
    print(f"   Accuracy: {time_acc:.1%}")
    print(f"   Chance level: {1/9:.1%}")
    if abs(time_acc - 1/9) < 0.05:
        print("   RESULT: z_bond IS time-invariant (BIP SUPPORTED)")
    else:
        print(f"   RESULT: z_bond retains temporal info (diff from chance: {abs(time_acc - 1/9):.1%})")

    print(f"\n2. HOHFELDIAN CLASSIFICATION FROM z_bond:")
    print(f"   Accuracy: {hohfeld_acc:.1%}")
    if hohfeld_acc > 0.5:
        print("   RESULT: z_bond captures moral structure")
    else:
        print("   RESULT: Weak moral structure encoding")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if abs(time_acc - 1/9) < 0.05 and hohfeld_acc > 0.4:
        print("""

    BIP TEMPORAL INVARIANCE: SUPPORTED

    The bond embedding (z_bond) successfully captured moral structure
    while remaining invariant to temporal/cultural context.

    This suggests that moral cognition has a geometry that is stable
    across 2000+ years of human ethical reasoning.

    """)
    else:
        print("""

    BIP TEST: INCONCLUSIVE

    Either:
    - Not enough data from different time periods
    - Model needs more training
    - Bond extraction needs improvement

    For definitive results, ensure:
    1. Sefaria corpus loaded successfully
    2. At least 10,000 passages per major time period
    3. Model trained for 10+ epochs

    """)

print("EXPERIMENT COMPLETE")
