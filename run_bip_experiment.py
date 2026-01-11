#!/usr/bin/env python3
"""
BIP Temporal Invariance Experiment - Master Script

This script runs the complete BIP experiment pipeline:
1. Preprocess corpora (Sefaria + Dear Abby)
2. Extract bond structures
3. Generate train/valid/test splits
4. Train the BIP model
5. Run evaluation tests
"""

import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_step(step_name: str, module_path: str):
    """Run a step and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print('='*60)

    try:
        exec(open(module_path).read())
        print(f"[OK] {step_name} completed successfully")
        return True
    except Exception as e:
        print(f"[ERROR] {step_name} failed: {e}")
        return False


def main():
    print("="*60)
    print("BIP TEMPORAL INVARIANCE EXPERIMENT")
    print("Testing the Bond Invariance Principle")
    print("="*60)

    # Check for data
    sefaria_path = Path("data/raw/Sefaria-Export")
    dear_abby_path = Path("data/raw/dear_abby.csv")

    if not dear_abby_path.exists():
        print(f"\n[ERROR] Dear Abby data not found at {dear_abby_path}")
        print("Please ensure dear_abby.csv is in data/raw/")
        return

    if not sefaria_path.exists():
        print(f"\n[WARNING] Sefaria corpus not found at {sefaria_path}")
        print("The experiment will proceed with Dear Abby data only.")
        print("For full experiment, clone Sefaria-Export:")
        print("  cd data/raw && git clone --depth 1 https://github.com/Sefaria/Sefaria-Export.git")
        print("\nNote: On Windows, you may need to use WSL due to Hebrew filename issues.")

    # Step 1: Preprocess
    print("\n[1/5] Preprocessing corpora...")
    from bip.data.preprocess import preprocess_all
    preprocess_all("config_bip.yaml")

    # Step 2: Extract bonds
    print("\n[2/5] Extracting bond structures...")
    from bip.data.extract_bonds import extract_bonds_all
    extract_bonds_all("config_bip.yaml")

    # Step 3: Generate splits
    print("\n[3/5] Generating train/valid/test splits...")
    from bip.data.generate_splits import generate_splits
    generate_splits("config_bip.yaml")

    # Step 4: Train model
    print("\n[4/5] Training BIP model...")
    print("This may take a while depending on your hardware...")

    import yaml
    with open("config_bip.yaml") as f:
        config = yaml.safe_load(f)

    from bip.train import BIPTrainer
    import torch
    import numpy as np

    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    trainer = BIPTrainer(config, "temporal_holdout")
    test_metrics = trainer.train()

    # Step 5: Print results
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)

    print("\n[Test Metrics]")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Interpret results for BIP
    time_acc = test_metrics.get('test/time_acc_from_bond', 0)
    hohfeld_acc = test_metrics.get('test/hohfeld_acc', 0)

    print("\n[BIP Analysis]")

    # Time prediction from bond embedding should be near chance (1/9 = 11.1%)
    chance_level = 1/9
    if abs(time_acc - chance_level) < 0.05:
        print(f"  Time prediction: {time_acc:.1%} (near chance {chance_level:.1%})")
        print("  -> z_bond IS time-invariant (BIP SUPPORTED)")
    else:
        print(f"  Time prediction: {time_acc:.1%} (chance = {chance_level:.1%})")
        print("  -> z_bond contains temporal information")

    # Hohfeldian classification should work well
    if hohfeld_acc > 0.5:
        print(f"  Hohfeld classification: {hohfeld_acc:.1%}")
        print("  -> z_bond captures moral structure")
    else:
        print(f"  Hohfeld classification: {hohfeld_acc:.1%}")
        print("  -> Need more data or better bond extraction")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print("\nResults saved to:")
    print("  - models/checkpoints/temporal_holdout/best_model.pt")
    print("  - models/checkpoints/temporal_holdout/test_metrics.json")
    print("  - models/checkpoints/temporal_holdout/test_embeddings.pt")


if __name__ == "__main__":
    main()
