#!/usr/bin/env python3
"""
eval_checkpoints.py - Evaluate BIP v10.14.4 model checkpoints

Self-contained script to load and evaluate trained models without
requiring the notebook environment.

Usage:
    python eval_checkpoints.py --checkpoint-dir /path/to/checkpoints
    python eval_checkpoints.py --checkpoint /path/to/best_hebrew_to_others.pt
"""

import argparse
import json
import os
from enum import Enum, auto
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm


# =============================================================================
# ENUMS (from Cell 4)
# =============================================================================

class BondType(Enum):
    HARM_PREVENTION = auto()
    RECIPROCITY = auto()
    AUTONOMY = auto()
    PROPERTY = auto()
    FAMILY = auto()
    AUTHORITY = auto()
    CARE = auto()
    FAIRNESS = auto()
    CONTRACT = auto()
    NONE = auto()


class HohfeldState(Enum):
    OBLIGATION = auto()
    RIGHT = auto()
    LIBERTY = auto()
    NO_RIGHT = auto()


# =============================================================================
# INDEX MAPPINGS (from Cell 6)
# =============================================================================

BOND_TO_IDX = {bt.name: i for i, bt in enumerate(BondType)}
IDX_TO_BOND = {i: bt.name for i, bt in enumerate(BondType)}

LANG_TO_IDX = {
    "hebrew": 0,
    "aramaic": 1,
    "classical_chinese": 2,
    "arabic": 3,
    "english": 4,
    "sanskrit": 5,
    "pali": 6,
    "greek": 7,
}
IDX_TO_LANG = {i: l for l, i in LANG_TO_IDX.items()}

PERIOD_TO_IDX = {
    "BIBLICAL": 0, "TANNAITIC": 1, "AMORAIC": 2, "RISHONIM": 3, "ACHRONIM": 4,
    "CONFUCIAN": 5, "DAOIST": 6, "MOHIST": 7, "LEGALIST": 8, "BUDDHIST": 9,
    "NEO_CONFUCIAN": 10, "QURANIC": 11, "HADITH": 12, "FIQH": 13, "SUFI": 14,
    "FALSAFA": 15, "DHARMA": 16, "UPANISHAD": 17, "GITA": 18, "ARTHA": 19,
    "PALI": 20, "WESTERN_CLASSICAL": 21, "MEDIEVAL": 22, "DEAR_ABBY": 23,
    "MODERN": 24, "CLASSICAL": 25,
}
IDX_TO_PERIOD = {i: p for p, i in PERIOD_TO_IDX.items()}

HOHFELD_TO_IDX = {hs.name: i for i, hs in enumerate(HohfeldState)}
IDX_TO_HOHFELD = {i: hs.name for i, hs in enumerate(HohfeldState)}

CONTEXT_TO_IDX = {"prescriptive": 0, "descriptive": 1, "unknown": 2}
IDX_TO_CONTEXT = {i: c for c, i in CONTEXT_TO_IDX.items()}


# =============================================================================
# BACKBONE CONFIGS (from Cell 1)
# =============================================================================

BACKBONE_CONFIGS = {
    "MiniLM": {
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "hidden_size": 384,
    },
    "LaBSE": {
        "model_name": "sentence-transformers/LaBSE",
        "hidden_size": 768,
    },
    "XLM-R-base": {
        "model_name": "xlm-roberta-base",
        "hidden_size": 768,
    },
    "XLM-R-large": {
        "model_name": "xlm-roberta-large",
        "hidden_size": 1024,
    },
}

DEFAULT_BACKBONE = "LaBSE"


# =============================================================================
# MODEL (from Cell 6)
# =============================================================================

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class BIPModel(nn.Module):
    def __init__(self, model_name=None, hidden_size=None, z_dim=64,
                 adv_hidden_dim=512, adv_dropout=0.3):
        super().__init__()

        if model_name is None:
            config = BACKBONE_CONFIGS[DEFAULT_BACKBONE]
            model_name = config["model_name"]
            hidden_size = config["hidden_size"]

        print(f"  Loading encoder: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)

        actual_hidden = self.encoder.config.hidden_size
        if hidden_size and actual_hidden != hidden_size:
            print(f"  Note: Using actual hidden size {actual_hidden}")
        hidden_size = actual_hidden

        self.hidden_size = hidden_size
        self.model_name = model_name

        proj_hidden = min(512, hidden_size)
        self.z_proj = nn.Sequential(
            nn.Linear(hidden_size, proj_hidden),
            nn.LayerNorm(proj_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_hidden, z_dim),
        )

        self.bond_head = nn.Linear(z_dim, len(BondType))
        self.hohfeld_head = nn.Linear(z_dim, len(HohfeldState))

        self.language_head = nn.Sequential(
            nn.Linear(z_dim, adv_hidden_dim),
            nn.ReLU(),
            nn.Dropout(adv_dropout),
            nn.Linear(adv_hidden_dim, adv_hidden_dim),
            nn.ReLU(),
            nn.Dropout(adv_dropout),
            nn.Linear(adv_hidden_dim, len(LANG_TO_IDX)),
        )
        self.period_head = nn.Sequential(
            nn.Linear(z_dim, adv_hidden_dim),
            nn.ReLU(),
            nn.Dropout(adv_dropout),
            nn.Linear(adv_hidden_dim, adv_hidden_dim),
            nn.ReLU(),
            nn.Dropout(adv_dropout),
            nn.Linear(adv_hidden_dim, len(PERIOD_TO_IDX)),
        )

        self.context_head = nn.Linear(z_dim, len(CONTEXT_TO_IDX))

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")

    def forward(self, input_ids, attention_mask, adv_lambda=1.0):
        enc = self.encoder(input_ids, attention_mask)

        if hasattr(enc, "pooler_output") and enc.pooler_output is not None:
            pooled = enc.pooler_output
        else:
            pooled = enc.last_hidden_state[:, 0]

        z = self.z_proj(pooled)

        bond_pred = self.bond_head(z)
        hohfeld_pred = self.hohfeld_head(z)

        z_rev = GradientReversalLayer.apply(z, adv_lambda)
        language_pred = self.language_head(z_rev)
        period_pred = self.period_head(z_rev)

        return {
            "bond_pred": bond_pred,
            "hohfeld_pred": hohfeld_pred,
            "language_pred": language_pred,
            "period_pred": period_pred,
            "context_pred": self.context_head(z),
            "z": z,
        }

    def get_bond_embedding(self, input_ids, attention_mask):
        enc = self.encoder(input_ids, attention_mask)
        if hasattr(enc, "pooler_output") and enc.pooler_output is not None:
            pooled = enc.pooler_output
        else:
            pooled = enc.last_hidden_state[:, 0]
        return self.z_proj(pooled)


# =============================================================================
# DATASET (from Cell 6)
# =============================================================================

def get_confidence_weight(conf):
    if isinstance(conf, str):
        return {"high": 2.0, "medium": 1.0, "low": 0.5}.get(conf, 1.0)
    elif isinstance(conf, (int, float)):
        return 2.0 if conf >= 0.8 else 1.0
    return 1.0


class NativeDataset(Dataset):
    def __init__(self, ids_set, passages_file, bonds_file, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        bonds_by_id = {}
        with open(bonds_file, encoding="utf-8") as fb:
            for line in fb:
                b = json.loads(line)
                bonds_by_id[b["passage_id"]] = b

        with open(passages_file, encoding="utf-8") as fp:
            for line in tqdm(fp, desc="Loading data", unit="line"):
                p = json.loads(line)
                if p["id"] in ids_set and p["id"] in bonds_by_id:
                    b = bonds_by_id[p["id"]]
                    self.data.append({
                        "text": p["text"][:1000],
                        "language": p["language"],
                        "period": p.get("time_periods", ["UNKNOWN"])[0],
                        "bond": b.get("bond_type") or b.get("bonds", {}).get("primary_bond"),
                        "hohfeld": None,
                        "context": b.get("context") or b.get("bonds", {}).get("context", "unknown"),
                        "confidence": b.get("confidence") or b.get("bonds", {}).get("confidence", "medium"),
                    })
        print(f"  Loaded {len(self.data):,} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "bond_label": BOND_TO_IDX.get(item["bond"], 9),
            "language_label": LANG_TO_IDX.get(item["language"], 4),
            "period_label": PERIOD_TO_IDX.get(item["period"], 9),
            "hohfeld_label": HOHFELD_TO_IDX.get(item["hohfeld"], 0) if item["hohfeld"] else 0,
            "context_label": CONTEXT_TO_IDX.get(item["context"], 2),
            "sample_weight": get_confidence_weight(item["confidence"]),
            "language": item["language"],
            "text": item["text"],
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "bond_labels": torch.tensor([x["bond_label"] for x in batch]),
        "language_labels": torch.tensor([x["language_label"] for x in batch]),
        "period_labels": torch.tensor([x["period_label"] for x in batch]),
        "hohfeld_labels": torch.tensor([x["hohfeld_label"] for x in batch]),
        "context_labels": torch.tensor([x["context_label"] for x in batch]),
        "sample_weights": torch.tensor([x["sample_weight"] for x in batch], dtype=torch.float),
        "languages": [x["language"] for x in batch],
        "texts": [x["text"] for x in batch],
    }


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()

    total_bond_correct = 0
    total_lang_correct = 0
    total_period_correct = 0
    total_samples = 0

    all_bond_preds = []
    all_bond_labels = []
    all_lang_preds = []
    all_lang_labels = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        bond_labels = batch["bond_labels"].to(device)
        lang_labels = batch["language_labels"].to(device)
        period_labels = batch["period_labels"].to(device)

        outputs = model(input_ids, attention_mask, adv_lambda=0.0)

        bond_preds = outputs["bond_pred"].argmax(dim=1)
        lang_preds = outputs["language_pred"].argmax(dim=1)
        period_preds = outputs["period_pred"].argmax(dim=1)

        total_bond_correct += (bond_preds == bond_labels).sum().item()
        total_lang_correct += (lang_preds == lang_labels).sum().item()
        total_period_correct += (period_preds == period_labels).sum().item()
        total_samples += len(bond_labels)

        all_bond_preds.extend(bond_preds.cpu().tolist())
        all_bond_labels.extend(bond_labels.cpu().tolist())
        all_lang_preds.extend(lang_preds.cpu().tolist())
        all_lang_labels.extend(lang_labels.cpu().tolist())

    metrics = {
        "bond_accuracy": total_bond_correct / total_samples if total_samples > 0 else 0,
        "language_accuracy": total_lang_correct / total_samples if total_samples > 0 else 0,
        "period_accuracy": total_period_correct / total_samples if total_samples > 0 else 0,
        "total_samples": total_samples,
    }

    for bond_idx, bond_name in IDX_TO_BOND.items():
        mask = [i for i, l in enumerate(all_bond_labels) if l == bond_idx]
        if mask:
            correct = sum(1 for i in mask if all_bond_preds[i] == bond_idx)
            metrics[f"bond_{bond_name}_acc"] = correct / len(mask)

    for lang_idx, lang_name in IDX_TO_LANG.items():
        mask = [i for i, l in enumerate(all_lang_labels) if l == lang_idx]
        if mask:
            correct = sum(1 for i in mask if all_lang_preds[i] == lang_idx)
            metrics[f"lang_{lang_name}_acc"] = correct / len(mask)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate BIP v10.14.4 checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory containing checkpoints")
    parser.add_argument("--checkpoint", type=str, help="Single checkpoint file to evaluate")
    parser.add_argument("--splits-file", type=str, default="data/splits/all_splits.json",
                        help="Path to splits JSON file")
    parser.add_argument("--passages-file", type=str, default="data/processed/passages.jsonl",
                        help="Path to passages JSONL file")
    parser.add_argument("--bonds-file", type=str, default="data/processed/bonds.jsonl",
                        help="Path to bonds JSONL file")
    parser.add_argument("--backbone", type=str, default="LaBSE",
                        choices=list(BACKBONE_CONFIGS.keys()),
                        help="Model backbone")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max samples per split")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoints = []
    if args.checkpoint:
        checkpoints = [args.checkpoint]
    elif args.checkpoint_dir:
        checkpoints = [
            os.path.join(args.checkpoint_dir, f)
            for f in os.listdir(args.checkpoint_dir)
            if f.endswith(".pt") and f.startswith("best_")
        ]
    else:
        parser.error("Must specify --checkpoint or --checkpoint-dir")

    if not checkpoints:
        print("No checkpoints found!")
        return

    print(f"Found {len(checkpoints)} checkpoint(s)")

    config = BACKBONE_CONFIGS[args.backbone]
    model_name = config["model_name"]
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if os.path.exists(args.splits_file):
        with open(args.splits_file, encoding="utf-8") as f:
            all_splits = json.load(f)
        print(f"Loaded splits from {args.splits_file}")
    else:
        print(f"Warning: Splits file not found: {args.splits_file}")
        all_splits = {}

    results = {}
    for ckpt_path in checkpoints:
        ckpt_name = os.path.basename(ckpt_path)
        split_name = ckpt_name.replace("best_", "").replace(".pt", "")

        print(f"\n{'='*60}")
        print(f"Evaluating: {split_name}")
        print(f"Checkpoint: {ckpt_path}")
        print("="*60)

        print("\nInitializing model...")
        model = BIPModel(model_name=model_name).to(device)

        print(f"Loading weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        if split_name in all_splits:
            test_ids = set(all_splits[split_name].get("test_ids", [])[:args.max_samples])
        else:
            print(f"  Warning: Split '{split_name}' not found in splits file")
            print(f"  Available splits: {list(all_splits.keys())}")
            continue

        if not test_ids:
            print(f"  No test IDs for split {split_name}")
            continue

        if not os.path.exists(args.passages_file) or not os.path.exists(args.bonds_file):
            print(f"  Data files not found")
            continue

        print(f"\nLoading test data ({len(test_ids)} IDs)...")
        test_dataset = NativeDataset(
            test_ids, args.passages_file, args.bonds_file, tokenizer
        )

        if len(test_dataset) < 10:
            print(f"  Skip - only {len(test_dataset)} samples")
            continue

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=0
        )

        print(f"\nRunning evaluation on {len(test_dataset)} samples...")
        metrics = evaluate_model(model, test_loader, device)
        results[split_name] = metrics

        print(f"\n--- Results for {split_name} ---")
        print(f"  Bond Accuracy:     {metrics['bond_accuracy']:.4f}")
        print(f"  Language Accuracy: {metrics['language_accuracy']:.4f}")
        print(f"  Period Accuracy:   {metrics['period_accuracy']:.4f}")
        print(f"  Total Samples:     {metrics['total_samples']}")

        bip_score = metrics['bond_accuracy'] - metrics['language_accuracy']
        print(f"\n  BIP Score (bond - lang): {bip_score:.4f}")
        if bip_score > 0.3:
            print("  [GOOD] Model learns bonds without encoding language")
        elif bip_score > 0.1:
            print("  [OK] Some language leakage")
        else:
            print("  [POOR] Language strongly encoded in bond space")

        del model
        torch.cuda.empty_cache()

    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print("="*60)
        for split_name, metrics in results.items():
            bip_score = metrics['bond_accuracy'] - metrics['language_accuracy']
            print(f"{split_name:30} Bond: {metrics['bond_accuracy']:.3f}  "
                  f"Lang: {metrics['language_accuracy']:.3f}  BIP: {bip_score:+.3f}")


if __name__ == "__main__":
    main()
