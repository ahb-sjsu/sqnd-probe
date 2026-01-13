#@title 7. Train BIP Model { display-mode: "form" }
#@markdown Training with tuned adversarial weights and hardware-optimized parameters

from sklearn.metrics import f1_score
import gc

#@markdown **Splits to train:**
TRAIN_HEBREW_TO_OTHERS = True  #@param {type:"boolean"}
TRAIN_SEMITIC_TO_NON_SEMITIC = True  #@param {type:"boolean"}
TRAIN_ANCIENT_TO_MODERN = True  #@param {type:"boolean"}
TRAIN_MIXED_BASELINE = True  #@param {type:"boolean"}
TRAIN_ABBY_TO_CHINESE = True  #@param {type:"boolean"}

#@markdown **Hyperparameters:**
LANG_WEIGHT = 1.0  #@param {type:"number"}
PERIOD_WEIGHT = 0.5  #@param {type:"number"}
N_EPOCHS = 10  #@param {type:"integer"}

#@markdown **Context-Aware Training:**
USE_CONFIDENCE_WEIGHTING = True  #@param {type:"boolean"}
#@markdown Weight prescriptive (high confidence) examples 2x in loss

USE_CONTEXT_AUXILIARY = True  #@param {type:"boolean"}
#@markdown Add context prediction as auxiliary training target

CONTEXT_LOSS_WEIGHT = 0.1  #@param {type:"number"}
#@markdown Weight for context prediction loss

STRICT_PRESCRIPTIVE_TEST = False  #@param {type:"boolean"}
#@markdown Only evaluate on prescriptive examples (strict test)

print("="*60)
print("TRAINING BIP MODEL")
print("="*60)
print(f"\nHardware-optimized settings:")
print(f"  GPU Tier:     {GPU_TIER}")
print(f"  Batch size:   {BATCH_SIZE}")
print(f"  Workers:      {NUM_WORKERS}")
print(f"  Learning rate: {LR:.2e}")
print(f"  Adv weights:  lang={LANG_WEIGHT}, period={PERIOD_WEIGHT}")
print("(0.01 prevents loss explosion while maintaining invariance)")
print(f"  Confidence weighting: {USE_CONFIDENCE_WEIGHTING}")
print(f"  Context auxiliary: {USE_CONTEXT_AUXILIARY} (weight={CONTEXT_LOSS_WEIGHT})")
print(f"  Strict prescriptive test: {STRICT_PRESCRIPTIVE_TEST}")

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

with open('data/splits/all_splits.json') as f:
    all_splits = json.load(f)

splits_to_train = []
if TRAIN_HEBREW_TO_OTHERS: splits_to_train.append('hebrew_to_others')
if TRAIN_SEMITIC_TO_NON_SEMITIC: splits_to_train.append('semitic_to_non_semitic')
if TRAIN_ANCIENT_TO_MODERN: splits_to_train.append('ancient_to_modern')
if TRAIN_MIXED_BASELINE: splits_to_train.append('mixed_baseline')
if TRAIN_ABBY_TO_CHINESE: splits_to_train.append('abby_to_chinese')

print(f"\nTraining {len(splits_to_train)} splits: {splits_to_train}")

all_results = {}
MIN_TEST_SIZE = 100  # Lowered to allow smaller test sets like Chinese

for split_idx, split_name in enumerate(splits_to_train):
    split_start = time.time()
    print("\n" + "="*60)
    print(f"[{split_idx+1}/{len(splits_to_train)}] {split_name}")
    print("="*60)

    split = all_splits[split_name]
    print(f"Train: {split['train_size']:,} | Test: {split['test_size']:,}")

    if split['test_size'] < MIN_TEST_SIZE:
        print(f"WARNING: Test set only {split['test_size']} samples (need {MIN_TEST_SIZE})")
        print("Skipping this split - results would be unreliable")
        print("To fix: Add more data to the test languages/periods")
        continue

    model = BIPModel().to(device)

    train_dataset = NativeDataset(set(split['train_ids']), 'data/processed/passages.jsonl',
                                   'data/processed/bonds.jsonl', tokenizer)

    test_ids_to_use = split['test_ids'][:MAX_TEST_SAMPLES]

    # Optional: strict prescriptive-only test
    if STRICT_PRESCRIPTIVE_TEST:
        print("Filtering to prescriptive examples only...")
        # Load bonds to filter
        prescriptive_ids = set()
        with open('data/processed/bonds.jsonl') as f:
            for line in f:
                b = json.loads(line)
                if b.get('context') == 'prescriptive':
                    prescriptive_ids.add(b['passage_id'])
        test_ids_to_use = [tid for tid in test_ids_to_use if tid in prescriptive_ids]
        print(f"  Filtered to {len(test_ids_to_use):,} prescriptive samples")

    test_dataset = NativeDataset(set(test_ids_to_use), 'data/processed/passages.jsonl',
                                  'data/processed/bonds.jsonl', tokenizer)

    if len(train_dataset) == 0:
        print("ERROR: No training data!")
        continue

    # Use hardware-optimized batch size
    actual_batch = min(BATCH_SIZE, max(32, len(train_dataset) // 20))
    print(f"Actual batch size: {actual_batch}")

    train_loader = DataLoader(train_dataset, batch_size=actual_batch, shuffle=True,
                              collate_fn=collate_fn, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=actual_batch*2, shuffle=False,
                             collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    def get_adv_lambda(epoch, warmup=3):
        """Ramp adversarial strength: 0.1 -> 2.0 over warmup, then hold at 2.0"""
        if epoch <= warmup:
            return 0.1 + 1.9 * (epoch / warmup)
        return 2.0

    best_loss = float('inf')

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bond_labels = batch['bond_labels'].to(device)
            language_labels = batch['language_labels'].to(device)
            period_labels = batch['period_labels'].to(device)

            adv_lambda = get_adv_lambda(epoch)

            # Use new autocast API
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                out = model(input_ids, attention_mask, adv_lambda=adv_lambda)

                # Weighted bond loss
                if USE_CONFIDENCE_WEIGHTING:
                    sample_weights = batch['sample_weights'].to(device)
                    loss_bond = F.cross_entropy(out['bond_pred'], bond_labels, reduction='none')
                    loss_bond = (loss_bond * sample_weights).mean()
                else:
                    loss_bond = F.cross_entropy(out['bond_pred'], bond_labels)

                # Context auxiliary loss
                if USE_CONTEXT_AUXILIARY:
                    context_labels = batch['context_labels'].to(device)
                    loss_context = F.cross_entropy(out['context_pred'], context_labels)
                else:
                    loss_context = 0

                loss_lang = F.cross_entropy(out['language_pred'], language_labels)
                loss_period = F.cross_entropy(out['period_pred'], period_labels)

            loss = loss_bond + LANG_WEIGHT * loss_lang + PERIOD_WEIGHT * loss_period + CONTEXT_LOSS_WEIGHT * loss_context

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
        print(f"Epoch {epoch}: Loss={avg_loss:.4f} (adv_lambda={adv_lambda:.2f})")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'models/checkpoints/best_{split_name}.pt')
            torch.save(model.state_dict(), f'{SAVE_DIR}/best_{split_name}.pt')

    # Evaluate
    print("\nEvaluating...")
    model.load_state_dict(torch.load(f'models/checkpoints/best_{split_name}.pt'))
    model.eval()

    all_preds = {'bond': [], 'lang': []}
    all_labels = {'bond': [], 'lang': []}
    all_languages = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            out = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), 0)
            all_preds['bond'].extend(out['bond_pred'].argmax(-1).cpu().tolist())
            all_preds['lang'].extend(out['language_pred'].argmax(-1).cpu().tolist())
            all_labels['bond'].extend(batch['bond_labels'].tolist())
            all_labels['lang'].extend(batch['language_labels'].tolist())
            all_languages.extend(batch['languages'])

    bond_f1 = f1_score(all_labels['bond'], all_preds['bond'], average='macro', zero_division=0)
    bond_acc = sum(p == l for p, l in zip(all_preds['bond'], all_labels['bond'])) / len(all_preds['bond'])
    lang_acc = sum(p == l for p, l in zip(all_preds['lang'], all_labels['lang'])) / len(all_preds['lang'])

    # Per-language F1
    lang_f1 = {}
    for lang in set(all_languages):
        mask = [l == lang for l in all_languages]
        if sum(mask) > 10:
            preds = [p for p, m in zip(all_preds['bond'], mask) if m]
            labels = [l for l, m in zip(all_labels['bond'], mask) if m]
            lang_f1[lang] = {'f1': f1_score(labels, preds, average='macro', zero_division=0), 'n': sum(mask)}

    all_results[split_name] = {
        'bond_f1_macro': bond_f1,
        'bond_acc': bond_acc,
        'language_acc': lang_acc,
        'per_language_f1': lang_f1,
        'training_time': time.time() - split_start
    }

    print(f"\n{split_name} RESULTS:")
    print(f"  Bond F1 (macro): {bond_f1:.3f} ({bond_f1/0.1:.1f}x chance)")
    print(f"  Bond accuracy:   {bond_acc:.1%}")
    print(f"  Language acc:    {lang_acc:.1%} (want ~20% = invariant)")
    print("  Per-language:")
    for lang, m in sorted(lang_f1.items(), key=lambda x: -x[1]['n']):
        print(f"    {lang:20s}: F1={m['f1']:.3f} (n={m['n']:,})")

    # Context analysis
    high_conf = sum(1 for c in test_dataset.data if c['confidence'] == 'high')
    prescriptive = sum(1 for c in test_dataset.data if c['context'] == 'prescriptive')
    print(f"  Context: {prescriptive:,}/{len(test_dataset):,} prescriptive ({prescriptive/len(test_dataset)*100:.1f}%)")
    print(f"  High confidence: {high_conf:,}/{len(test_dataset):,} ({high_conf/len(test_dataset)*100:.1f}%)")

    # GPU memory usage
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"\n  GPU memory: {mem:.1f} GB / {VRAM_GB:.1f} GB ({mem/VRAM_GB*100:.0f}%)")

    del model, train_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)