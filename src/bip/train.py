#!/usr/bin/env python3
"""
Training script for BIP temporal invariance model.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Conditional imports
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bip.models.bip_model import BIPTemporalInvarianceModel, BIPLoss, get_tokenizer

# =============================================================================
# DATASET
# =============================================================================


class MoralPassageDataset(Dataset):
    """Dataset for moral passages."""

    # Map time periods to indices
    PERIOD_TO_IDX = {
        "BIBLICAL": 0,
        "SECOND_TEMPLE": 1,
        "TANNAITIC": 2,
        "AMORAIC": 3,
        "GEONIC": 4,
        "RISHONIM": 5,
        "ACHRONIM": 6,
        "MODERN_HEBREW": 7,
        "DEAR_ABBY": 8,
    }

    HOHFELD_TO_IDX = {"RIGHT": 0, "OBLIGATION": 1, "LIBERTY": 2, "NO_RIGHT": 3}

    BOND_TYPES = [
        "HARM_PREVENTION",
        "RECIPROCITY",
        "AUTONOMY",
        "PROPERTY",
        "FAMILY",
        "AUTHORITY",
        "EMERGENCY",
        "CONTRACT",
        "CARE",
        "FAIRNESS",
    ]

    def __init__(
        self,
        passage_ids: List[str],
        passages_file: str,
        bonds_file: str,
        tokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load passages
        self.passages = {}
        with open(passages_file, encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                if p["id"] in passage_ids or not passage_ids:
                    self.passages[p["id"]] = p

        # Load bond structures
        self.bonds = {}
        if bonds_file and Path(bonds_file).exists():
            with open(bonds_file, encoding="utf-8") as f:
                for line in f:
                    b = json.loads(line)
                    self.bonds[b["passage_id"]] = b["bond_structure"]

        # Filter to requested IDs
        if passage_ids:
            self.ids = [pid for pid in passage_ids if pid in self.passages]
        else:
            self.ids = list(self.passages.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        passage = self.passages[pid]

        # Tokenize
        text = passage["text_english"]
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Time period label
        time_label = self.PERIOD_TO_IDX.get(passage["time_period"], 0)

        # Hohfeldian label (if available)
        bond_struct = self.bonds.get(pid, {})
        hohfeld_str = bond_struct.get("hohfeld_state")
        hohfeld_label = self.HOHFELD_TO_IDX.get(hohfeld_str, 0) if hohfeld_str else 0

        # Bond type labels (multi-hot)
        bond_labels = torch.zeros(len(self.BOND_TYPES))
        for bt in passage.get("bond_types", []):
            if bt in self.BOND_TYPES:
                bond_labels[self.BOND_TYPES.index(bt)] = 1

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "time_label": time_label,
            "hohfeld_label": hohfeld_label,
            "bond_labels": bond_labels,
            "passage_id": pid,
        }


def collate_fn(batch):
    """Collate batch of samples."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "time_labels": torch.tensor([b["time_label"] for b in batch]),
        "hohfeld_labels": torch.tensor([b["hohfeld_label"] for b in batch]),
        "bond_labels": torch.stack([b["bond_labels"] for b in batch]),
        "passage_ids": [b["passage_id"] for b in batch],
    }


# =============================================================================
# TRAINER
# =============================================================================


class BIPTrainer:
    """Trainer for BIP model."""

    def __init__(self, config: dict, split_name: str = "temporal_holdout"):
        self.config = config
        self.split_name = split_name

        # Device selection
        device_str = config["experiment"].get("device", "cuda")
        if device_str == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device_str = "cpu"
        self.device = torch.device(device_str)
        print(f"Using device: {self.device}")

        # Create output directories
        self.output_dir = Path(f"models/checkpoints/{split_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = BIPTemporalInvarianceModel(config["model"]).to(self.device)
        self.tokenizer = get_tokenizer(config["model"]["encoder"])

        # Loss
        self.criterion = BIPLoss(config["training"])

        # Load splits
        splits_file = Path("data/splits/all_splits.json")
        if not splits_file.exists():
            raise FileNotFoundError(
                f"Splits file not found: {splits_file}. Run generate_splits first."
            )

        with open(splits_file) as f:
            all_splits = json.load(f)

        if split_name in all_splits:
            split = all_splits[split_name]
        else:
            raise ValueError(f"Unknown split: {split_name}")

        # Create datasets
        passages_file = f"{config['data']['processed_path']}/passages.jsonl"
        bonds_file = f"{config['data']['processed_path']}/bond_structures.jsonl"

        self.train_dataset = MoralPassageDataset(
            split["train_ids"],
            passages_file,
            bonds_file,
            self.tokenizer,
            config["model"].get("max_length", 512),
        )
        self.valid_dataset = MoralPassageDataset(
            split["valid_ids"],
            passages_file,
            bonds_file,
            self.tokenizer,
            config["model"].get("max_length", 512),
        )
        self.test_dataset = MoralPassageDataset(
            split["test_ids"],
            passages_file,
            bonds_file,
            self.tokenizer,
            config["model"].get("max_length", 512),
        )

        # Dataloaders
        batch_size = config["training"]["batch_size"]
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # Use 0 for Windows compatibility
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # Scheduler
        total_steps = len(self.train_loader) * config["training"]["max_epochs"]
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config["training"]["learning_rate"],
            total_steps=total_steps,
            pct_start=0.1,
        )

        # Tracking
        self.best_valid_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0

        # Wandb
        if HAS_WANDB and config["logging"].get("wandb_project"):
            wandb.init(
                project=config["logging"]["wandb_project"],
                name=f"{split_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config,
            )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        loss_components = {k: 0 for k in ["adv", "time", "kl", "hohfeld", "bond", "bip"]}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            time_labels = batch["time_labels"].to(self.device)
            hohfeld_labels = batch["hohfeld_labels"].to(self.device)
            bond_labels = batch["bond_labels"].to(self.device)

            # Forward
            outputs = self.model(
                input_ids,
                attention_mask,
                time_labels=time_labels,
                adversarial_lambda=self.config["training"]["lambda_adversarial"],
            )

            # Loss
            loss, loss_dict = self.criterion(outputs, time_labels, hohfeld_labels, bond_labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["training"]["gradient_clip"]
            )

            self.optimizer.step()
            self.scheduler.step()

            # Track
            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k in loss_components:
                    loss_components[k] += v

            self.global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log
            if HAS_WANDB and self.global_step % self.config["logging"]["log_interval"] == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": self.scheduler.get_last_lr()[0],
                        **{f"train/{k}": v for k, v in loss_dict.items()},
                    },
                    step=self.global_step,
                )

        n_batches = len(self.train_loader)
        return {
            "loss": total_loss / n_batches,
            **{k: v / n_batches for k, v in loss_components.items()},
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, prefix: str = "valid") -> Dict[str, float]:
        """Evaluate on a dataset."""
        self.model.eval()

        total_loss = 0
        all_time_preds = []
        all_time_labels = []
        all_hohfeld_preds = []
        all_hohfeld_labels = []
        all_z_bonds = []

        for batch in tqdm(loader, desc=f"Evaluating {prefix}"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            time_labels = batch["time_labels"].to(self.device)
            hohfeld_labels = batch["hohfeld_labels"].to(self.device)
            bond_labels = batch["bond_labels"].to(self.device)

            outputs = self.model(input_ids, attention_mask, adversarial_lambda=0.0)

            loss, _ = self.criterion(outputs, time_labels, hohfeld_labels, bond_labels)
            total_loss += loss.item()

            # Collect predictions
            all_time_preds.append(outputs["time_pred_bond"].argmax(dim=-1).cpu())
            all_time_labels.append(time_labels.cpu())
            all_hohfeld_preds.append(outputs["hohfeld_pred"].argmax(dim=-1).cpu())
            all_hohfeld_labels.append(hohfeld_labels.cpu())
            all_z_bonds.append(outputs["z_bond"].cpu())

        # Concatenate
        time_preds = torch.cat(all_time_preds)
        time_labels_cat = torch.cat(all_time_labels)
        hohfeld_preds = torch.cat(all_hohfeld_preds)
        hohfeld_labels_cat = torch.cat(all_hohfeld_labels)
        z_bonds = torch.cat(all_z_bonds)

        # Metrics
        n_batches = len(loader)

        # Time prediction accuracy (from z_bond - should be low!)
        time_acc = (time_preds == time_labels_cat).float().mean().item()

        # Hohfeldian accuracy
        hohfeld_acc = (hohfeld_preds == hohfeld_labels_cat).float().mean().item()

        metrics = {
            f"{prefix}/loss": total_loss / n_batches,
            f"{prefix}/time_acc_from_bond": time_acc,
            f"{prefix}/hohfeld_acc": hohfeld_acc,
        }

        return metrics, z_bonds, time_labels_cat

    def train(self):
        """Full training loop."""
        print(f"\nTraining on split: {self.split_name}")
        print(
            f"Train: {len(self.train_dataset)}, Valid: {len(self.valid_dataset)}, Test: {len(self.test_dataset)}"
        )

        max_epochs = self.config["training"]["max_epochs"]
        patience = self.config["training"]["early_stopping_patience"]

        for epoch in range(1, max_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{max_epochs}")
            print("=" * 60)

            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Train loss: {train_metrics['loss']:.4f}")

            # Validate
            valid_metrics, _, _ = self.evaluate(self.valid_loader, "valid")
            print(f"Valid loss: {valid_metrics['valid/loss']:.4f}")
            print(f"Valid time_acc_from_bond: {valid_metrics['valid/time_acc_from_bond']:.4f}")
            print(f"Valid hohfeld_acc: {valid_metrics['valid/hohfeld_acc']:.4f}")

            # Log
            if HAS_WANDB:
                wandb.log(
                    {"epoch": epoch, "train/epoch_loss": train_metrics["loss"], **valid_metrics},
                    step=self.global_step,
                )

            # Early stopping
            if valid_metrics["valid/loss"] < self.best_valid_loss:
                self.best_valid_loss = valid_metrics["valid/loss"]
                self.patience_counter = 0

                # Save best model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_valid_loss": self.best_valid_loss,
                        "config": self.config,
                    },
                    self.output_dir / "best_model.pt",
                )
                print("Saved best model")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Final test evaluation
        print("\n" + "=" * 60)
        print("FINAL TEST EVALUATION")
        print("=" * 60)

        # Load best model
        checkpoint = torch.load(self.output_dir / "best_model.pt")
        self.model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics, test_z_bonds, test_time_labels = self.evaluate(self.test_loader, "test")

        print(f"\nTest Results:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")

        # Save test embeddings for analysis
        torch.save(
            {"z_bonds": test_z_bonds, "time_labels": test_time_labels},
            self.output_dir / "test_embeddings.pt",
        )

        # Save final metrics
        with open(self.output_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

        if HAS_WANDB:
            wandb.log(test_metrics, step=self.global_step)
            wandb.finish()

        return test_metrics


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_bip.yaml")
    parser.add_argument(
        "--split",
        type=str,
        default="temporal_holdout",
        choices=["temporal_holdout", "stratified_random"],
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set seed
    seed = config["experiment"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Train
    trainer = BIPTrainer(config, args.split)
    metrics = trainer.train()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
