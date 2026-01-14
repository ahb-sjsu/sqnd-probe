# @title 5. Generate Splits { display-mode: "form" }
# @markdown Creates train/test splits for cross-lingual experiments
# @markdown v10.9: Added confucian_to_buddhist, confucian_to_legalist,
# @markdown        all_to_sanskrit, semitic_to_indic, quran_to_fiqh

import json
import random
import shutil
from collections import defaultdict

print("=" * 60)
print("GENERATING SPLITS")
print("=" * 60)

# Check if splits already exist from Drive
# Check if splits are valid (IDs match current passages)
splits_valid = False
if os.path.exists("data/splits/all_splits.json"):
    try:
        with open("data/splits/all_splits.json") as f:
            cached_splits = json.load(f)
        # Get sample of IDs from splits
        sample_ids = set()
        for split in cached_splits.values():
            sample_ids.update(split["train_ids"][:100])
            sample_ids.update(split["test_ids"][:100])
        # Check if they exist in current passages
        passage_ids = set()
        with open("data/processed/passages.jsonl") as f:
            for line in f:
                p = json.loads(line)
                passage_ids.add(p["id"])
                if len(passage_ids) > 10000:
                    break
        matches = len(sample_ids & passage_ids)
        splits_valid = matches > len(sample_ids) * 0.9  # 90% match
        if not splits_valid:
            print(f"Splits invalid: only {matches}/{len(sample_ids)} IDs match current passages")
    except Exception as e:
        print(f"Error validating splits: {e}")

if splits_valid and not REFRESH_DATA_FROM_SOURCE:
    print("\nSplits already loaded from Drive")
    with open("data/splits/all_splits.json") as f:
        all_splits = json.load(f)
    for name, split in all_splits.items():
        print(f"  {name}: train={split['train_size']:,}, test={split['test_size']:,}")
else:
    random.seed(42)

    # Read passage metadata
    passage_meta = []
    with open("data/processed/passages.jsonl", "r") as f:
        for line in f:
            p = json.loads(line)
            passage_meta.append(p)

    print(f"Total passages: {len(passage_meta):,}")

    by_lang = defaultdict(list)
    by_period = defaultdict(list)
    for p in passage_meta:
        by_lang[p["language"]].append(p["id"])
        by_period[p["time_period"]].append(p["id"])

    print("\nBy language:")
    for lang, ids in sorted(by_lang.items(), key=lambda x: -len(x[1])):
        print(f"  {lang}: {len(ids):,}")

    print("\nBy period:")
    for period, ids in sorted(by_period.items(), key=lambda x: -len(x[1])):
        print(f"  {period}: {len(ids):,}")

    all_splits = {}
    # ===== SPLIT 1: Hebrew -> Others =====
    print("\n" + "-" * 60)
    print("SPLIT 1: HEBREW -> OTHERS")
    hebrew_ids = by_lang.get("hebrew", [])
    other_ids = [p["id"] for p in passage_meta if p["language"] != "hebrew"]
    random.shuffle(hebrew_ids)
    random.shuffle(other_ids)

    all_splits["hebrew_to_others"] = {
        "train_ids": hebrew_ids,
        "test_ids": other_ids,
        "train_size": len(hebrew_ids),
        "test_size": len(other_ids),
    }
    print(f"  Train (Hebrew): {len(hebrew_ids):,}")
    print(f"  Test (Others): {len(other_ids):,}")

    # ===== SPLIT 2: Semitic -> Non-Semitic =====
    print("\n" + "-" * 60)
    print("SPLIT 2: SEMITIC -> NON-SEMITIC")
    semitic_ids = by_lang.get("hebrew", []) + by_lang.get("aramaic", []) + by_lang.get("arabic", [])
    non_semitic_ids = by_lang.get("classical_chinese", []) + by_lang.get("english", [])
    random.shuffle(semitic_ids)
    random.shuffle(non_semitic_ids)

    all_splits["semitic_to_non_semitic"] = {
        "train_ids": semitic_ids,
        "test_ids": non_semitic_ids,
        "train_size": len(semitic_ids),
        "test_size": len(non_semitic_ids),
    }
    print(f"  Train (Semitic): {len(semitic_ids):,}")
    print(f"  Test (Non-Semitic): {len(non_semitic_ids):,}")

    # ===== SPLIT 3: Ancient -> Modern =====
    print("\n" + "-" * 60)
    print("SPLIT 3: ANCIENT -> MODERN")
    # Define modern periods explicitly, derive ancient dynamically
    modern_periods = {"MODERN", "DEAR_ABBY"}
    all_periods = set(by_period.keys())
    ancient_periods = all_periods - modern_periods

    print(f"  Ancient periods: {sorted(ancient_periods)}")
    print(f"  Modern periods: {sorted(modern_periods)}")

    ancient_ids = [p["id"] for p in passage_meta if p["time_period"] in ancient_periods]
    modern_ids = [p["id"] for p in passage_meta if p["time_period"] in modern_periods]
    random.shuffle(ancient_ids)
    random.shuffle(modern_ids)

    all_splits["ancient_to_modern"] = {
        "train_ids": ancient_ids,
        "test_ids": modern_ids,
        "train_size": len(ancient_ids),
        "test_size": len(modern_ids),
    }
    print(f"  Train (Ancient): {len(ancient_ids):,}")
    print(f"  Test (Modern): {len(modern_ids):,}")

    # ===== SPLIT 4: Mixed Baseline =====
    print("\n" + "-" * 60)
    print("SPLIT 4: MIXED BASELINE")
    all_ids = [p["id"] for p in passage_meta]
    random.shuffle(all_ids)
    split_idx = int(0.7 * len(all_ids))

    all_splits["mixed_baseline"] = {
        "train_ids": all_ids[:split_idx],
        "test_ids": all_ids[split_idx:],
        "train_size": split_idx,
        "test_size": len(all_ids) - split_idx,
    }
    print(f"  Train: {split_idx:,}")
    print(f"  Test: {len(all_ids) - split_idx:,}")

    # ===== SPLIT 5: Dear Abby -> Classical Chinese =====
    print("\n" + "-" * 60)
    print("SPLIT 5: DEAR ABBY -> CHINESE")
    abby_ids = [p["id"] for p in passage_meta if p["time_period"] == "DEAR_ABBY"]
    chinese_ids = [p["id"] for p in passage_meta if p["language"] == "classical_chinese"]
    random.shuffle(abby_ids)
    random.shuffle(chinese_ids)

    all_splits["abby_to_chinese"] = {
        "train_ids": abby_ids,
        "test_ids": chinese_ids,
        "train_size": len(abby_ids),
        "test_size": len(chinese_ids),
    }
    print(f"  Train (Dear Abby): {len(abby_ids):,}")
    print(f"  Test (Chinese): {len(chinese_ids):,}")

    # ===== SPLIT 6: Western Classical -> Eastern =====
    print("\n" + "-" * 60)
    print("SPLIT 6: WESTERN CLASSICAL -> EASTERN")
    western_ids = [p["id"] for p in passage_meta if p["time_period"] == "WESTERN_CLASSICAL"]
    eastern_ids = [
        p["id"] for p in passage_meta if p["language"] in ("classical_chinese", "hebrew")
    ]
    random.shuffle(western_ids)
    random.shuffle(eastern_ids)

    all_splits["western_to_eastern"] = {
        "train_ids": western_ids,
        "test_ids": eastern_ids,
        "train_size": len(western_ids),
        "test_size": len(eastern_ids),
    }
    print(f"  Train (Western - Plato, Aristotle, Stoics): {len(western_ids):,}")
    print(f"  Test (Eastern - Chinese, Hebrew): {len(eastern_ids):,}")

    # ===== SPLIT 7: Confucian -> Buddhist (v10.9) =====
    print("\n" + "-" * 60)
    print("SPLIT 7: CONFUCIAN -> BUDDHIST (Chinese intra-tradition)")
    confucian_daoist_ids = [
        p["id"]
        for p in passage_meta
        if p["time_period"] in ("CONFUCIAN", "DAOIST") and p["language"] == "classical_chinese"
    ]
    buddhist_ids = [
        p["id"]
        for p in passage_meta
        if p["time_period"] == "BUDDHIST" and p["language"] == "classical_chinese"
    ]
    random.shuffle(confucian_daoist_ids)
    random.shuffle(buddhist_ids)

    all_splits["confucian_to_buddhist"] = {
        "train_ids": confucian_daoist_ids,
        "test_ids": buddhist_ids,
        "train_size": len(confucian_daoist_ids),
        "test_size": len(buddhist_ids),
        "description": "Test if Chinese performance is tradition-specific",
    }
    print(f"  Train (Confucian+Daoist): {len(confucian_daoist_ids):,}")
    print(f"  Test (Buddhist): {len(buddhist_ids):,}")

    # ===== SPLIT 8: Confucian -> Legalist/Mohist (v10.9) =====
    print("\n" + "-" * 60)
    print("SPLIT 8: CONFUCIAN -> LEGALIST/MOHIST (virtue vs consequentialist)")
    confucian_only_ids = [
        p["id"]
        for p in passage_meta
        if p["time_period"] == "CONFUCIAN" and p["language"] == "classical_chinese"
    ]
    legalist_mohist_ids = [
        p["id"]
        for p in passage_meta
        if p["time_period"] in ("LEGALIST", "MOHIST") and p["language"] == "classical_chinese"
    ]
    random.shuffle(confucian_only_ids)
    random.shuffle(legalist_mohist_ids)

    all_splits["confucian_to_legalist"] = {
        "train_ids": confucian_only_ids,
        "test_ids": legalist_mohist_ids,
        "train_size": len(confucian_only_ids),
        "test_size": len(legalist_mohist_ids),
        "description": "Virtue ethics → consequentialist/legalist",
    }
    print(f"  Train (Confucian): {len(confucian_only_ids):,}")
    print(f"  Test (Legalist+Mohist): {len(legalist_mohist_ids):,}")

    # ===== SPLIT 9: All -> Sanskrit/Pali (v10.9) =====
    print("\n" + "-" * 60)
    print("SPLIT 9: ALL -> SANSKRIT/PALI (ultimate transfer test)")
    non_indic_ids = [
        p["id"]
        for p in passage_meta
        if p["language"] in ("hebrew", "aramaic", "classical_chinese", "arabic", "english")
    ]
    indic_ids = [p["id"] for p in passage_meta if p["language"] in ("sanskrit", "pali")]
    random.shuffle(non_indic_ids)
    random.shuffle(indic_ids)

    all_splits["all_to_sanskrit"] = {
        "train_ids": non_indic_ids,
        "test_ids": indic_ids,
        "train_size": len(non_indic_ids),
        "test_size": len(indic_ids),
        "description": "Ultimate transfer test: completely held-out language family",
    }
    print(f"  Train (non-Indic): {len(non_indic_ids):,}")
    print(f"  Test (Sanskrit+Pali): {len(indic_ids):,}")

    # ===== SPLIT 10: Semitic -> Indic (v10.9) =====
    print("\n" + "-" * 60)
    print("SPLIT 10: SEMITIC -> INDIC")
    semitic_only_ids = [
        p["id"] for p in passage_meta if p["language"] in ("hebrew", "aramaic", "arabic")
    ]
    indic_only_ids = [p["id"] for p in passage_meta if p["language"] in ("sanskrit", "pali")]
    random.shuffle(semitic_only_ids)
    random.shuffle(indic_only_ids)

    all_splits["semitic_to_indic"] = {
        "train_ids": semitic_only_ids,
        "test_ids": indic_only_ids,
        "train_size": len(semitic_only_ids),
        "test_size": len(indic_only_ids),
        "description": "Semitic → Indo-Aryan transfer",
    }
    print(f"  Train (Semitic): {len(semitic_only_ids):,}")
    print(f"  Test (Indic): {len(indic_only_ids):,}")

    # ===== SPLIT 11: Quran -> Fiqh (v10.9) =====
    print("\n" + "-" * 60)
    print("SPLIT 11: QURAN -> FIQH (religious to legal/philosophical)")
    quranic_ids = [
        p["id"]
        for p in passage_meta
        if p["time_period"] in ("QURANIC", "HADITH") and p["language"] == "arabic"
    ]
    fiqh_ids = [
        p["id"]
        for p in passage_meta
        if p["time_period"] in ("FIQH", "SUFI", "FALSAFA") and p["language"] == "arabic"
    ]
    random.shuffle(quranic_ids)
    random.shuffle(fiqh_ids)

    all_splits["quran_to_fiqh"] = {
        "train_ids": quranic_ids,
        "test_ids": fiqh_ids,
        "train_size": len(quranic_ids),
        "test_size": len(fiqh_ids),
        "description": "Religious → legal/philosophical Arabic",
    }
    print(f"  Train (Quranic+Hadith): {len(quranic_ids):,}")
    print(f"  Test (Fiqh+Sufi+Falsafa): {len(fiqh_ids):,}")

    # Save splits
    with open("data/splits/all_splits.json", "w") as f:
        json.dump(all_splits, f, indent=2)

    # Save to Drive
    shutil.copy("data/splits/all_splits.json", f"{SAVE_DIR}/all_splits.json")

print("\n" + "=" * 60)
print("Splits saved to local and Drive")
print("=" * 60)
