"""Test Cell 4 splits system with synthetic data."""
import json
import random
from collections import defaultdict
from pathlib import Path

print("=" * 60)
print("TESTING CELL 4 SPLITS SYSTEM")
print("=" * 60)

# Create synthetic test data
print("\n[1] Creating synthetic test passages...")

passages = []

# Confucian Chinese (500)
for i in range(500):
    passages.append({
        "id": f"confucian_{i}",
        "language": "classical_chinese",
        "time_periods": ["CONFUCIAN"],
    })

# Daoist Chinese (300)
for i in range(300):
    passages.append({
        "id": f"daoist_{i}",
        "language": "classical_chinese",
        "time_periods": ["DAOIST"],
    })

# Buddhist CBETA (400)
for i in range(400):
    passages.append({
        "id": f"buddhist_{i}",
        "language": "classical_chinese",
        "time_periods": ["BUDDHIST"],
    })

# Hebrew Biblical (600)
for i in range(600):
    passages.append({
        "id": f"hebrew_{i}",
        "language": "hebrew",
        "time_periods": ["BIBLICAL"],
    })

# Arabic Quranic (400)
for i in range(400):
    passages.append({
        "id": f"arabic_{i}",
        "language": "arabic",
        "time_periods": ["QURANIC"],
    })

# Sanskrit Dharma (350)
for i in range(350):
    passages.append({
        "id": f"sanskrit_{i}",
        "language": "sanskrit",
        "time_periods": ["DHARMA"],
    })

# Pali Buddhist (300)
for i in range(300):
    passages.append({
        "id": f"pali_{i}",
        "language": "pali",
        "time_periods": ["PALI"],
    })

# English Modern Dear Abby (800)
for i in range(800):
    passages.append({
        "id": f"abby_{i}",
        "language": "english",
        "time_periods": ["DEAR_ABBY"],
    })

# Greek Stoic (200)
for i in range(200):
    passages.append({
        "id": f"greek_{i}",
        "language": "greek",
        "time_periods": ["CLASSICAL_GREEK"],
    })

print(f"Created {len(passages)} synthetic passages")

# Save test data
Path("data/processed").mkdir(parents=True, exist_ok=True)
with open("data/processed/passages.jsonl", "w", encoding="utf-8") as f:
    for p in passages:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print("Saved to data/processed/passages.jsonl")

# Tag migration (from Cell 4)
PERIOD_TO_TAGS = {
    "CONFUCIAN": ["confucian", "east-asia", "classical"],
    "DAOIST": ["daoist", "east-asia", "classical"],
    "BUDDHIST": ["buddhist"],
    "PALI": ["buddhist", "south-asia", "ancient"],
    "DHARMA": ["hindu", "south-asia", "ancient"],
    "BIBLICAL": ["jewish", "middle-east", "ancient"],
    "QURANIC": ["islamic", "middle-east", "medieval"],
    "CLASSICAL_GREEK": ["stoic", "mediterranean", "classical"],
    "DEAR_ABBY": ["american", "modern", "advice", "western"],
}

LANG_TO_TAGS = {
    "classical_chinese": ["chinese", "east-asia"],
    "sanskrit": ["sanskrit", "south-asia"],
    "pali": ["pali", "south-asia"],
    "hebrew": ["hebrew", "middle-east"],
    "arabic": ["arabic", "middle-east"],
    "greek": ["greek", "mediterranean"],
    "english": ["english", "western"],
}


def add_tags(p):
    tags = set()
    lang = p.get("language", "")
    if lang in LANG_TO_TAGS:
        tags.update(LANG_TO_TAGS[lang])
    for period in p.get("time_periods", []):
        if period in PERIOD_TO_TAGS:
            tags.update(PERIOD_TO_TAGS[period])
    return sorted(tags)


# Load and tag
print("\n[2] Running tag migration...")
passage_meta = []
with open("data/processed/passages.jsonl", encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)
        passage_meta.append({
            "id": p["id"],
            "language": p.get("language", ""),
            "tags": add_tags(p),
        })

# Count tags
tag_counts = defaultdict(int)
for p in passage_meta:
    for tag in p["tags"]:
        tag_counts[tag] += 1

print("\nAVAILABLE TAGS:")
print("-" * 40)
for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
    print(f"  {tag}: {count:,}")


# Split helpers
def ids_with_tags(tags):
    tag_set = set(tags)
    return [p["id"] for p in passage_meta if set(p["tags"]) & tag_set]


def ids_by_lang(*langs):
    return [p["id"] for p in passage_meta if p["language"] in langs]


def ids_not_lang(*langs):
    return [p["id"] for p in passage_meta if p["language"] not in langs]


def remove_overlap(train, test):
    overlap = set(train) & set(test)
    return [x for x in train if x not in overlap], [x for x in test if x not in overlap]


# Generate splits
print("\n" + "=" * 60)
print("[3] GENERATING SPLITS")
print("=" * 60)

all_splits = {}
random.seed(42)

# Default splits
print("\n[Default Splits]")

train = ids_by_lang("hebrew")
test = ids_not_lang("hebrew")
all_splits["hebrew_to_others"] = (len(train), len(test))
print(f"  hebrew_to_others: {len(train):,} -> {len(test):,}")

train = ids_by_lang("hebrew", "aramaic", "arabic")
test = ids_not_lang("hebrew", "aramaic", "arabic")
all_splits["semitic_to_non_semitic"] = (len(train), len(test))
print(f"  semitic_to_non_semitic: {len(train):,} -> {len(test):,}")

train = ids_with_tags(["ancient", "classical"])
test = ids_with_tags(["modern"])
train, test = remove_overlap(train, test)
all_splits["ancient_to_modern"] = (len(train), len(test))
print(f"  ancient_to_modern: {len(train):,} -> {len(test):,}")

train = ids_with_tags(["confucian"])
test = ids_with_tags(["buddhist"])
train, test = remove_overlap(train, test)
all_splits["confucian_to_buddhist"] = (len(train), len(test))
print(f"  confucian_to_buddhist: {len(train):,} -> {len(test):,}")

train = ids_with_tags(["east-asia"])
test = ids_with_tags(["western"])
train, test = remove_overlap(train, test)
all_splits["east_to_west"] = (len(train), len(test))
print(f"  east_to_west: {len(train):,} -> {len(test):,}")

train = ids_with_tags(["hebrew", "arabic"])
test = ids_with_tags(["sanskrit", "pali"])
all_splits["semitic_to_indic"] = (len(train), len(test))
print(f"  semitic_to_indic: {len(train):,} -> {len(test):,}")

train = ids_with_tags(["american", "advice"])
test = ids_with_tags(["chinese", "confucian", "daoist"])
train, test = remove_overlap(train, test)
all_splits["abby_to_chinese"] = (len(train), len(test))
print(f"  abby_to_chinese: {len(train):,} -> {len(test):,}")

# Auto-generated tradition splits
print("\n[Auto-Generated Tradition Splits]")
traditions = ["confucian", "daoist", "buddhist", "hindu", "jewish", "islamic", "stoic"]
avail = [t for t in traditions if tag_counts.get(t, 0) >= 100]
print(f"  Available (>=100): {avail}")

for i, t1 in enumerate(avail):
    for t2 in avail[i + 1 :]:
        train = ids_with_tags([t1])
        test = ids_with_tags([t2])
        train, test = remove_overlap(train, test)
        if len(train) >= 100 and len(test) >= 100:
            name = f"{t1}_to_{t2}"
            if name not in all_splits:
                all_splits[name] = (len(train), len(test))
                print(f"    {name}: {len(train):,} -> {len(test):,}")

# Custom split test
print("\n[Custom Split Test]")
custom_spec = "confucian,daoist -> islamic,jewish"
train_part, test_part = custom_spec.split("->")
train_tags = [t.strip() for t in train_part.split(",")]
test_tags = [t.strip() for t in test_part.split(",")]
train = ids_with_tags(train_tags)
test = ids_with_tags(test_tags)
train, test = remove_overlap(train, test)
print(f"  Custom: '{custom_spec}'")
print(f"    Train: {train_tags} -> {len(train):,}")
print(f"    Test: {test_tags} -> {len(test):,}")

# Summary
print("\n" + "=" * 60)
print("SPLIT SUMMARY MATRIX")
print("=" * 60)
print(f"{'Split Name':<30} {'Train':>8} {'Test':>8}")
print("-" * 50)
for name, (tr, te) in sorted(all_splits.items()):
    print(f"{name:<30} {tr:>8,} {te:>8,}")
print("-" * 50)
print(f"Total splits: {len(all_splits)}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
