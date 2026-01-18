import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

all_results = {}

saved_splits = [
"hebrew_to_others", "semitic_to_indic", "confucian_to_buddhist",
"ancient_to_modern", "semitic_to_chinese", "jewish_to_islamic",
"daoist_to_buddhist", "stoic_to_confucian", "hindu_to_buddhist",
"mixed_baseline"
]

for split_name in saved_splits:
    split = all_splits[split_name]
    print(f"\n{'='*60}")
    print(f"Evaluating: {split_name}")
    model = BIPModel(len(BOND_TYPES), len(LANGUAGE_MAP)).to(device)
    ckpt_path = f"{SAVE_DIR}/best_{split_name}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"  Loaded checkpoint")
    test_dataset = NativeDataset(split["test"], bonds, tokenizer, BOND_TYPES, LANGUAGE_MAP)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    all_preds = {"bond": [], "lang": []}
    all_labels = {"bond": [], "lang": []}
    all_languages = []
    torch.cuda.empty_cache()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), 0)
            all_preds["bond"].extend(out["bond_pred"].argmax(-1).cpu().tolist())
            all_preds["lang"].extend(out["language_pred"].argmax(-1).cpu().tolist())
            all_labels["bond"].extend(batch["bond_labels"].tolist())
            all_labels["lang"].extend(batch["language_labels"].tolist())
            all_languages.extend(batch["languages"])
    bond_f1 = f1_score(all_labels["bond"], all_preds["bond"], average="macro", zero_division=0)
    bond_acc = sum(p==l for p,l in zip(all_preds["bond"], all_labels["bond"])) / len(all_preds["bond"])
    lang_acc = sum(p==l for p,l in zip(all_preds["lang"], all_labels["lang"])) / len(all_preds["lang"])
    all_results[split_name] = {"bond_f1": bond_f1, "bond_acc": bond_acc, "lang_acc": lang_acc}
    print(f"  Bond F1: {bond_f1:.4f} | Bond Acc: {bond_acc:.4f}")
    print(f"  Lang Acc: {lang_acc:.4f} (lower = better disentanglement)")
    del model
    torch.cuda.empty_cache()

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
for name, res in all_results.items():
    print(f"{name:25s} | Bond F1: {res['bond_f1']:.4f} | Lang Acc: {res['lang_acc']:.4f}")
