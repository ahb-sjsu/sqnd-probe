# @title 7. Train BIP Model { display-mode: "form" }
# @markdown Training with tuned adversarial weights and hardware-optimized parameters
# @markdown v10.9: Added new splits (confucian_to_buddhist, all_to_sanskrit, etc.)

# ===== SUPPRESS DATALOADER MULTIPROCESSING WARNINGS =====
# These occur during garbage collection and bypass normal exception handling
import warnings
import sys
import os
import io
import logging

# Method 1: Filter warnings
warnings.filterwarnings("ignore", message=".*can only test a child process.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data")

# Method 2: Suppress logging
logging.getLogger("torch.utils.data.dataloader").setLevel(logging.CRITICAL)


# Method 3: Redirect stderr during DataLoader cleanup (most effective)
class StderrFilter(io.TextIOWrapper):
    """Filters out DataLoader multiprocessing cleanup messages from stderr"""

    def __init__(self, original):
        self.original = original
        self.buffer_lines = []

    def write(self, text):
        # Filter out the specific error patterns
        skip_patterns = [
            "can only test a child process",
            "_MultiProcessingDataLoaderIter.__del__",
            "_shutdown_workers",
            "Exception ignored in:",
            "w.is_alive()",
        ]
        # Buffer multi-line error messages
        if any(p in text for p in skip_patterns):
            return len(text)  # Pretend we wrote it
        # Also skip if it looks like part of a traceback for these errors
        if text.strip().startswith("^") and len(text.strip()) < 80:
            return len(text)
        if text.strip().startswith('File "/usr') and "dataloader.py" in text:
            return len(text)
        if text.strip() == "Traceback (most recent call last):":
            self.buffer_lines = [text]
            return len(text)
        if self.buffer_lines:
            self.buffer_lines.append(text)
            # Check if this is the DataLoader error traceback
            full_msg = "".join(self.buffer_lines)
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
import gc


# @markdown **Splits to train:**
TRAIN_HEBREW_TO_OTHERS = True  # @param {type:"boolean"}
TRAIN_SEMITIC_TO_NON_SEMITIC = True  # @param {type:"boolean"}
TRAIN_ANCIENT_TO_MODERN = True  # @param {type:"boolean"}
TRAIN_MIXED_BASELINE = True  # @param {type:"boolean"}
TRAIN_ABBY_TO_CHINESE = True  # @param {type:"boolean"}

# @markdown **v10.9 New Splits:**
TRAIN_CONFUCIAN_TO_BUDDHIST = True  # @param {type:"boolean"}
TRAIN_CONFUCIAN_TO_LEGALIST = True  # @param {type:"boolean"}
TRAIN_ALL_TO_SANSKRIT = True  # @param {type:"boolean"}
TRAIN_SEMITIC_TO_INDIC = True  # @param {type:"boolean"}
TRAIN_QURAN_TO_FIQH = True  # @param {type:"boolean"}

# @markdown **Hyperparameters:**
LANG_WEIGHT = 0.1  # @param {type:"number"}
PERIOD_WEIGHT = 0.066  # @param {type:"number"}
N_EPOCHS = 10  # @param {type:"integer"}

# @markdown **Context-Aware Training:**
USE_CONFIDENCE_WEIGHTING = True  # @param {type:"boolean"}
# @markdown Weight prescriptive (high confidence) examples 2x in loss

USE_CONTEXT_AUXILIARY = True  # @param {type:"boolean"}
# @markdown Add context prediction as auxiliary training target

CONTEXT_LOSS_WEIGHT = 0.33  # @param {type:"number"}
# @markdown Weight for context prediction loss

STRICT_PRESCRIPTIVE_TEST = False  # @param {type:"boolean"}
# @markdown Only evaluate on prescriptive examples (reduces test set ~97%!)

print("=" * 60)
print("TRAINING BIP MODEL")
print("=" * 60)
print(f"\nSettings:")
print(f"  Backbone:     {BACKBONE}")
print(f"  GPU Tier:     {GPU_TIER}")
print(f"  Batch size:   {BATCH_SIZE}")
print(f"  Workers:      {NUM_WORKERS}")
print(f"  Learning rate: {LR:.2e}")
print(f"  Adv weights:  lang={LANG_WEIGHT}, period={PERIOD_WEIGHT}")
print("(0.01 prevents loss explosion while maintaining invariance)")
print(f"  Confidence weighting: {USE_CONFIDENCE_WEIGHTING}")
print(f"  Context auxiliary: {USE_CONTEXT_AUXILIARY} (weight={CONTEXT_LOSS_WEIGHT})")
print(f"  Strict prescriptive test: {STRICT_PRESCRIPTIVE_TEST}")

# tokenizer loaded in Cell 6 based on BACKBONE selection

with open("data/splits/all_splits.json") as f:
    all_splits = json.load(f)

splits_to_train = []
if TRAIN_HEBREW_TO_OTHERS:
    splits_to_train.append("hebrew_to_others")
if TRAIN_SEMITIC_TO_NON_SEMITIC:
    splits_to_train.append("semitic_to_non_semitic")
if TRAIN_ANCIENT_TO_MODERN:
    splits_to_train.append("ancient_to_modern")
if TRAIN_MIXED_BASELINE:
    splits_to_train.append("mixed_baseline")
if TRAIN_ABBY_TO_CHINESE:
    splits_to_train.append("abby_to_chinese")
# v10.9 new splits
if TRAIN_CONFUCIAN_TO_BUDDHIST:
    splits_to_train.append("confucian_to_buddhist")
if TRAIN_CONFUCIAN_TO_LEGALIST:
    splits_to_train.append("confucian_to_legalist")
if TRAIN_ALL_TO_SANSKRIT:
    splits_to_train.append("all_to_sanskrit")
if TRAIN_SEMITIC_TO_INDIC:
    splits_to_train.append("semitic_to_indic")
if TRAIN_QURAN_TO_FIQH:
    splits_to_train.append("quran_to_fiqh")

print(f"\nTraining {len(splits_to_train)} splits: {splits_to_train}")

all_results = {}
MIN_TEST_SIZE = 100  # Lowered to allow smaller test sets like Chinese

for split_idx, split_name in enumerate(splits_to_train):
    split_start = time.time()
    print("\n" + "=" * 60)
    print(f"[{split_idx+1}/{len(splits_to_train)}] {split_name}")
    print("=" * 60)

    split = all_splits[split_name]
    print(f"Train: {split['train_size']:,} | Test: {split['test_size']:,}")

    if split["test_size"] < MIN_TEST_SIZE:
        print(f"WARNING: Test set only {split['test_size']} samples (need {MIN_TEST_SIZE})")
        print("Skipping this split - results would be unreliable")
        print("To fix: Add more data to the test languages/periods")
        continue

    # Create model with OOM recovery
    def create_model_with_retry():
        """Create model, cleaning up GPU memory if OOM occurs."""
        try:
            return BIPModel().to(device)
        except torch.cuda.OutOfMemoryError:
            print("  OOM on model creation - cleaning up and retrying...")
            # Clean up any existing model in globals
            _g = globals()
            for _var in ["model", "analyzer", "encoder"]:
                if _var in _g and _g[_var] is not None:
                    try:
                        if hasattr(_g[_var], "cpu"):
                            _g[_var].cpu()
                        _g[_var] = None
                    except:
                        pass
            # Force cleanup
            gc.collect()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Retry
            return BIPModel().to(device)

    model = create_model_with_retry()

    train_dataset = NativeDataset(
        set(split["train_ids"]),
        "data/processed/passages.jsonl",
        "data/processed/bonds.jsonl",
        tokenizer,
    )

    test_ids_to_use = split["test_ids"][:MAX_TEST_SAMPLES]

    # Optional: strict prescriptive-only test
    if STRICT_PRESCRIPTIVE_TEST:
        print("Filtering to prescriptive examples only...")
        # Load bonds to filter
        prescriptive_ids = set()
        with open("data/processed/bonds.jsonl") as f:
            for line in f:
                b = json.loads(line)
                if b.get("context") == "prescriptive":
                    prescriptive_ids.add(b["passage_id"])
        test_ids_to_use = [tid for tid in test_ids_to_use if tid in prescriptive_ids]
        print(f"  Filtered to {len(test_ids_to_use):,} prescriptive samples")

    test_dataset = NativeDataset(
        set(test_ids_to_use),
        "data/processed/passages.jsonl",
        "data/processed/bonds.jsonl",
        tokenizer,
    )

    if len(train_dataset) == 0:
        print("ERROR: No training data!")
        continue

    # Use hardware-optimized batch size
    actual_batch = min(BATCH_SIZE, max(32, len(train_dataset) // 20))
    print(f"Actual batch size: {actual_batch}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_batch,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=actual_batch * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    def get_adv_lambda(epoch, warmup=3):
        """Ramp adversarial strength: 0.1 -> 1.0 over warmup, then hold at 1.0"""
        if epoch <= warmup:
            return 0.1 + 0.9 * (epoch / warmup)
        return 1.0

    best_loss = float("inf")
    start_epoch = 1

    # Check for existing checkpoint to resume from
    checkpoint_path = f"models/checkpoints/latest_{split_name}.pt"
    if os.path.exists(checkpoint_path):
        print(f"  Found checkpoint, resuming...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        print(f"  Resuming from epoch {start_epoch}, best_loss={best_loss:.4f}")

    for epoch in range(start_epoch, N_EPOCHS + 1):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bond_labels = batch["bond_labels"].to(device)
            language_labels = batch["language_labels"].to(device)
            period_labels = batch["period_labels"].to(device)

            adv_lambda = get_adv_lambda(epoch)

            # Use new autocast API
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                out = model(input_ids, attention_mask, adv_lambda=adv_lambda)

                # Weighted bond loss
                if USE_CONFIDENCE_WEIGHTING:
                    sample_weights = batch["sample_weights"].to(device)
                    loss_bond = F.cross_entropy(out["bond_pred"], bond_labels, reduction="none")
                    loss_bond = (loss_bond * sample_weights).mean()
                else:
                    loss_bond = F.cross_entropy(out["bond_pred"], bond_labels)

                # Context auxiliary loss
                if USE_CONTEXT_AUXILIARY:
                    context_labels = batch["context_labels"].to(device)
                    loss_context = F.cross_entropy(out["context_pred"], context_labels)
                else:
                    loss_context = 0

                loss_lang = F.cross_entropy(out["language_pred"], language_labels)
                loss_period = F.cross_entropy(out["period_pred"], period_labels)

            loss = (
                loss_bond
                + LANG_WEIGHT * loss_lang
                + PERIOD_WEIGHT * loss_period
                + CONTEXT_LOSS_WEIGHT * loss_context
            )

            if USE_AMP and scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Clear CUDA cache after each epoch to prevent memory accumulation
        torch.cuda.empty_cache()

        mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"Epoch {epoch}: Loss={avg_loss:.4f} (adv_lambda={adv_lambda:.2f}) [GPU: {mem_gb:.1f}GB]")

        # Save checkpoint every epoch (for crash recovery)
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": avg_loss,
            "best_loss": best_loss,
        }
        torch.save(checkpoint, f"models/checkpoints/latest_{split_name}.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"models/checkpoints/best_{split_name}.pt")
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_{split_name}.pt")

    # Evaluate
    print("\nEvaluating...")
    model.load_state_dict(torch.load(f"models/checkpoints/best_{split_name}.pt"))
    model.eval()

    all_preds = {"bond": [], "lang": []}
    all_labels = {"bond": [], "lang": []}
    all_languages = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), 0)
            all_preds["bond"].extend(out["bond_pred"].argmax(-1).cpu().tolist())
            all_preds["lang"].extend(out["language_pred"].argmax(-1).cpu().tolist())
            all_labels["bond"].extend(batch["bond_labels"].tolist())
            all_labels["lang"].extend(batch["language_labels"].tolist())
            all_languages.extend(batch["languages"])

    bond_f1 = f1_score(all_labels["bond"], all_preds["bond"], average="macro", zero_division=0)
    bond_acc = sum(p == l for p, l in zip(all_preds["bond"], all_labels["bond"])) / len(
        all_preds["bond"]
    )
    lang_acc = sum(p == l for p, l in zip(all_preds["lang"], all_labels["lang"])) / len(
        all_preds["lang"]
    )

    # Per-language F1
    lang_f1 = {}
    for lang in set(all_languages):
        mask = [l == lang for l in all_languages]
        if sum(mask) > 10:
            preds = [p for p, m in zip(all_preds["bond"], mask) if m]
            labels = [l for l, m in zip(all_labels["bond"], mask) if m]
            lang_f1[lang] = {
                "f1": f1_score(labels, preds, average="macro", zero_division=0),
                "n": sum(mask),
            }

    all_results[split_name] = {
        "bond_f1_macro": bond_f1,
        "bond_acc": bond_acc,
        "language_acc": lang_acc,
        "per_language_f1": lang_f1,
        "training_time": time.time() - split_start,
    }

    print(f"\n{split_name} RESULTS:")
    print(f"  Bond F1 (macro): {bond_f1:.3f} ({bond_f1/0.1:.1f}x chance)")
    print(f"  Bond accuracy:   {bond_acc:.1%}")
    print(f"  Language acc:    {lang_acc:.1%} (want ~20% = invariant)")
    print("  Per-language:")
    for lang, m in sorted(lang_f1.items(), key=lambda x: -x[1]["n"]):
        print(f"    {lang:20s}: F1={m['f1']:.3f} (n={m['n']:,})")

    # Context analysis
    high_conf = sum(1 for c in test_dataset.data if c["confidence"] == "high")
    prescriptive = sum(1 for c in test_dataset.data if c["context"] == "prescriptive")
    print(
        f"  Context: {prescriptive:,}/{len(test_dataset):,} prescriptive ({prescriptive/len(test_dataset)*100:.1f}%)"
    )
    print(
        f"  High confidence: {high_conf:,}/{len(test_dataset):,} ({high_conf/len(test_dataset)*100:.1f}%)"
    )

    # GPU memory usage before cleanup
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"\n  GPU memory (before cleanup): {mem:.1f} GB / {VRAM_GB:.1f} GB ({mem/VRAM_GB*100:.0f}%)")

    # Aggressive memory cleanup between splits
    # Step 1: Move model to CPU to release GPU memory
    model.cpu()

    # Step 2: Delete all references
    del model, train_dataset, test_dataset, train_loader, test_loader, optimizer
    if USE_AMP and scaler:
        del scaler

    # Step 3: Force garbage collection (multiple passes)
    for _ in range(3):
        gc.collect()

    # Step 4: Clear CUDA cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Step 5: Re-create scaler for next split
    if USE_AMP:
        scaler = torch.amp.GradScaler("cuda")

    # GPU memory after cleanup
    if torch.cuda.is_available():
        mem_after = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory (after cleanup): {mem_after:.1f} GB (freed {mem - mem_after:.1f} GB)")
        if mem_after > 1.0:
            print(f"  WARNING: {mem_after:.1f} GB still allocated - may cause OOM on next split")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
