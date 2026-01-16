# @title 4. Generate Splits { display-mode: "form" }
# @markdown v10.13: Tag-based splits with matrix selection

# @markdown ---
# @markdown ## Split Matrix
# @markdown Select train/test tags using dropdowns. Use "none" to disable.

# @markdown ### Experiment 1
EXP1_ENABLE = True  # @param {type:"boolean"}
EXP1_NAME = "hebrew_to_others"  # @param {type:"string"}
EXP1_TRAIN = "hebrew"  # @param ["none", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]
EXP1_TEST = "all-other"  # @param ["none", "all-other", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]

# @markdown ### Experiment 2
EXP2_ENABLE = True  # @param {type:"boolean"}
EXP2_NAME = "semitic_to_indic"  # @param {type:"string"}
EXP2_TRAIN = "semitic"  # @param ["none", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]
EXP2_TEST = "indic"  # @param ["none", "all-other", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]

# @markdown ### Experiment 3
EXP3_ENABLE = True  # @param {type:"boolean"}
EXP3_NAME = "confucian_to_buddhist"  # @param {type:"string"}
EXP3_TRAIN = "confucian"  # @param ["none", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]
EXP3_TEST = "buddhist"  # @param ["none", "all-other", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]

# @markdown ### Experiment 4
EXP4_ENABLE = True  # @param {type:"boolean"}
EXP4_NAME = "ancient_to_modern"  # @param {type:"string"}
EXP4_TRAIN = "ancient"  # @param ["none", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]
EXP4_TEST = "modern"  # @param ["none", "all-other", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]

# @markdown ### Experiment 5
EXP5_ENABLE = True  # @param {type:"boolean"}
EXP5_NAME = "east_to_west"  # @param {type:"string"}
EXP5_TRAIN = "east-asia"  # @param ["none", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]
EXP5_TEST = "western"  # @param ["none", "all-other", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]

# @markdown ### Experiment 6
EXP6_ENABLE = True  # @param {type:"boolean"}
EXP6_NAME = "semitic_to_chinese"  # @param {type:"string"}
EXP6_TRAIN = "semitic"  # @param ["none", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]
EXP6_TEST = "chinese"  # @param ["none", "all-other", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]

# @markdown ### Experiment 7
EXP7_ENABLE = True  # @param {type:"boolean"}
EXP7_NAME = "jewish_to_islamic"  # @param {type:"string"}
EXP7_TRAIN = "hebrew"  # @param ["none", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]
EXP7_TEST = "arabic"  # @param ["none", "all-other", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]

# @markdown ### Experiment 8
EXP8_ENABLE = True  # @param {type:"boolean"}
EXP8_NAME = "stoic_to_confucian"  # @param {type:"string"}
EXP8_TRAIN = "stoic"  # @param ["none", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]
EXP8_TEST = "confucian"  # @param ["none", "all-other", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]

# @markdown ### Experiment 9
EXP9_ENABLE = True  # @param {type:"boolean"}
EXP9_NAME = "daoist_to_buddhist"  # @param {type:"string"}
EXP9_TRAIN = "daoist"  # @param ["none", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]
EXP9_TEST = "buddhist"  # @param ["none", "all-other", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]

# @markdown ### Experiment 10
EXP10_ENABLE = True  # @param {type:"boolean"}
EXP10_NAME = "hindu_to_buddhist"  # @param {type:"string"}
EXP10_TRAIN = "hindu"  # @param ["none", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]
EXP10_TEST = "buddhist"  # @param ["none", "all-other", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]

# @markdown ### Experiment 11
EXP11_ENABLE = False  # @param {type:"boolean"}
EXP11_NAME = "custom_11"  # @param {type:"string"}
EXP11_TRAIN = "none"  # @param ["none", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]
EXP11_TEST = "none"  # @param ["none", "all-other", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]

# @markdown ### Experiment 12
EXP12_ENABLE = False  # @param {type:"boolean"}
EXP12_NAME = "custom_12"  # @param {type:"string"}
EXP12_TRAIN = "none"  # @param ["none", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]
EXP12_TEST = "none"  # @param ["none", "all-other", "hebrew", "aramaic", "arabic", "semitic", "chinese", "confucian", "daoist", "buddhist", "hindu", "sanskrit", "pali", "indic", "greek", "latin", "stoic", "english", "western", "modern", "ancient", "classical", "east-asia", "south-asia", "middle-east"]

# @markdown ---
# @markdown ## Options
INCLUDE_MIXED_BASELINE = True  # @param {type:"boolean"}
MIN_SPLIT_SIZE = 50  # @param {type:"integer"}

import json
import random
from collections import defaultdict
from pathlib import Path

print("=" * 60)
print("GENERATING SPLITS (v10.13)")
print("=" * 60)

# =============================================================================
# TAG DEFINITIONS
# =============================================================================

# Compound tag groups
TAG_GROUPS = {
    "semitic": ["hebrew", "aramaic", "arabic"],
    "indic": ["sanskrit", "pali", "hindi"],
    "east-asia": ["chinese", "confucian", "daoist"],
    "south-asia": ["sanskrit", "pali", "hindu", "buddhist"],
    "middle-east": ["hebrew", "aramaic", "arabic", "jewish", "islamic"],
    "western": ["english", "greek", "latin", "stoic"],
    "ancient": ["ancient", "classical"],
    "modern": ["modern", "advice", "american"],
}

# Period to tags mapping
PERIOD_TO_TAGS = {
    "CONFUCIAN": ["confucian", "east-asia", "classical", "chinese"],
    "DAOIST": ["daoist", "east-asia", "classical", "chinese"],
    "BUDDHIST": ["buddhist"],
    "PALI": ["buddhist", "south-asia", "ancient", "pali"],
    "DHARMA": ["hindu", "south-asia", "ancient", "sanskrit"],
    "BIBLICAL": ["jewish", "middle-east", "ancient", "hebrew"],
    "TANNAITIC": ["jewish", "middle-east", "classical", "hebrew"],
    "AMORAIC": ["jewish", "middle-east", "classical", "aramaic"],
    "QURANIC": ["islamic", "middle-east", "medieval", "arabic"],
    "HADITH": ["islamic", "middle-east", "medieval", "arabic"],
    "CLASSICAL_GREEK": ["stoic", "mediterranean", "classical", "greek"],
    "DEAR_ABBY": ["american", "modern", "advice", "english", "western"],
    "MODERN_ETHICS": ["western", "modern", "ethics", "english"],
}

LANG_TO_TAGS = {
    "classical_chinese": ["chinese", "east-asia"],
    "hebrew": ["hebrew", "middle-east"],
    "aramaic": ["aramaic", "middle-east"],
    "arabic": ["arabic", "middle-east"],
    "sanskrit": ["sanskrit", "south-asia"],
    "pali": ["pali", "south-asia"],
    "greek": ["greek", "mediterranean"],
    "latin": ["latin", "mediterranean"],
    "english": ["english", "western"],
}


def add_tags(p: dict) -> list:
    """Generate tags for a passage."""
    tags = set()

    lang = p.get("language", "")
    if lang in LANG_TO_TAGS:
        tags.update(LANG_TO_TAGS[lang])

    for period in p.get("time_periods", []):
        if period in PERIOD_TO_TAGS:
            tags.update(PERIOD_TO_TAGS[period])

    return sorted(tags)


# =============================================================================
# LOAD PASSAGES
# =============================================================================

passages_file = Path("data/processed/passages.jsonl")
if not passages_file.exists():
    raise FileNotFoundError("Run Cell 2 first to generate passages.jsonl")

passage_meta = []
with open(passages_file, encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)
        passage_meta.append({
            "id": p["id"],
            "language": p.get("language", ""),
            "tags": add_tags(p),
        })

print(f"Loaded {len(passage_meta):,} passages")

# Count tags
tag_counts = defaultdict(int)
for p in passage_meta:
    for tag in p["tags"]:
        tag_counts[tag] += 1

print("\nTag counts:")
for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:15]:
    print(f"  {tag}: {count:,}")


# =============================================================================
# SPLIT HELPERS
# =============================================================================

def expand_tag(tag: str) -> list:
    """Expand compound tags like 'semitic' to individual tags."""
    if tag in TAG_GROUPS:
        return TAG_GROUPS[tag]
    return [tag]


def ids_with_tags(tags: list) -> list:
    """Get passage IDs with ANY of the tags."""
    tag_set = set()
    for t in tags:
        tag_set.update(expand_tag(t))
    return [p["id"] for p in passage_meta if set(p["tags"]) & tag_set]


def ids_without_tags(tags: list) -> list:
    """Get passage IDs with NONE of the tags."""
    tag_set = set()
    for t in tags:
        tag_set.update(expand_tag(t))
    return [p["id"] for p in passage_meta if not (set(p["tags"]) & tag_set)]


# =============================================================================
# GENERATE SPLITS FROM MATRIX
# =============================================================================

print("\n" + "=" * 60)
print("GENERATING SPLITS")
print("=" * 60)

all_splits = {}
random.seed(42)

experiments = [
    (EXP1_ENABLE, EXP1_NAME, EXP1_TRAIN, EXP1_TEST),
    (EXP2_ENABLE, EXP2_NAME, EXP2_TRAIN, EXP2_TEST),
    (EXP3_ENABLE, EXP3_NAME, EXP3_TRAIN, EXP3_TEST),
    (EXP4_ENABLE, EXP4_NAME, EXP4_TRAIN, EXP4_TEST),
    (EXP5_ENABLE, EXP5_NAME, EXP5_TRAIN, EXP5_TEST),
    (EXP6_ENABLE, EXP6_NAME, EXP6_TRAIN, EXP6_TEST),
    (EXP7_ENABLE, EXP7_NAME, EXP7_TRAIN, EXP7_TEST),
    (EXP8_ENABLE, EXP8_NAME, EXP8_TRAIN, EXP8_TEST),
    (EXP9_ENABLE, EXP9_NAME, EXP9_TRAIN, EXP9_TEST),
    (EXP10_ENABLE, EXP10_NAME, EXP10_TRAIN, EXP10_TEST),
    (EXP11_ENABLE, EXP11_NAME, EXP11_TRAIN, EXP11_TEST),
    (EXP12_ENABLE, EXP12_NAME, EXP12_TRAIN, EXP12_TEST),
]

for enabled, name, train_tag, test_tag in experiments:
    if not enabled or train_tag == "none" or not name.strip():
        continue

    name = name.strip().replace(" ", "_")

    # Get train IDs
    train_ids = ids_with_tags([train_tag])

    # Get test IDs
    if test_tag == "all-other":
        test_ids = ids_without_tags([train_tag])
    elif test_tag == "none":
        continue
    else:
        test_ids = ids_with_tags([test_tag])
        # Remove overlap
        overlap = set(train_ids) & set(test_ids)
        train_ids = [x for x in train_ids if x not in overlap]
        test_ids = [x for x in test_ids if x not in overlap]

    if len(train_ids) < MIN_SPLIT_SIZE or len(test_ids) < MIN_SPLIT_SIZE:
        print(f"  SKIP {name}: insufficient data (train={len(train_ids)}, test={len(test_ids)})")
        continue

    random.shuffle(train_ids)
    random.shuffle(test_ids)

    all_splits[name] = {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "train_size": len(train_ids),
        "test_size": len(test_ids),
        "train_tags": expand_tag(train_tag),
        "test_tags": expand_tag(test_tag) if test_tag != "all-other" else ["*"],
    }
    print(f"  {name}: {len(train_ids):,} -> {len(test_ids):,}")

# Add mixed baseline
if INCLUDE_MIXED_BASELINE:
    all_ids = [p["id"] for p in passage_meta]
    random.shuffle(all_ids)
    split_pt = int(len(all_ids) * 0.7)
    all_splits["mixed_baseline"] = {
        "train_ids": all_ids[:split_pt],
        "test_ids": all_ids[split_pt:],
        "train_size": split_pt,
        "test_size": len(all_ids) - split_pt,
        "train_tags": ["*"],
        "test_tags": ["*"],
    }
    print(f"  mixed_baseline: {split_pt:,} -> {len(all_ids)-split_pt:,}")


# =============================================================================
# SAVE
# =============================================================================

splits_file = Path("data/splits/all_splits.json")
splits_file.parent.mkdir(parents=True, exist_ok=True)

with open(splits_file, "w", encoding="utf-8") as f:
    json.dump(all_splits, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print(f"SAVED {len(all_splits)} SPLITS")
print("=" * 60)

print("\n" + "-" * 50)
print(f"{'Experiment':<25} {'Train':>10} {'Test':>10}")
print("-" * 50)
for name, split in sorted(all_splits.items()):
    print(f"{name:<25} {split['train_size']:>10,} {split['test_size']:>10,}")
print("-" * 50)
