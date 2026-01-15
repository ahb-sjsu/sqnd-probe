# BIP v10.10 Integration Notes
# This file summarizes all changes for v10.10 upgrade

"""
=============================================================================
BIP v10.10 CHANGES SUMMARY
=============================================================================

1. ROLE AUGMENTATION (cell6_v10.9.py, cell7_v10.9.py)
   - COMPLETED: Added "text" field to dataset __getitem__ return
   - COMPLETED: Added "texts" field to collate_fn return
   - COMPLETED: Added swap_roles_simple() function in cell7
   - COMPLETED: Added role contrastive loss computation in training loop
   - Settings: USE_ROLE_AUGMENTATION, ROLE_AUGMENT_PROB,
               ROLE_CONTRASTIVE_WEIGHT, ROLE_CONTRASTIVE_MARGIN
   - Purpose: Fix weak role_swap sensitivity (0.003) from fuzz testing

2. EXPANDED CORPORA (temp_expanded_corpus_v10.10.py)
   - Sanskrit: ~258 passages (from ~80 original)
   - Pali: ~193 passages (from ~75 original)
   - Fiqh: ~76 passages (from ~70 original)
   - Sufi: ~54 passages (from ~45 original)
   - Falsafa: ~42 passages (from ~35 original)
   - Buddhist Chinese: ~98 passages (from ~86 original)

   Target sizes (10x expansion):
   - Sanskrit: 700 passages
   - Pali: 700 passages
   - Arabic total: 600 passages
   - Buddhist Chinese: 300 passages

   Status: Partial expansion complete. More passages needed.

3. VERSION UPDATES
   - cell10_v10.9.py: Fixed zip filename to v10.9

=============================================================================
INTEGRATION INSTRUCTIONS
=============================================================================

To integrate v10.10 changes:

1. Role augmentation is already integrated in cell6 and cell7.
   Run as-is to use new role contrastive loss.

2. To use expanded corpora, in cell4_v10.9.py:

   Replace:
   - SANSKRIT_DHARMA with SANSKRIT_DHARMA_EXPANDED from temp_expanded_corpus_v10.10.py
   - PALI_ETHICS with PALI_ETHICS_EXPANDED from temp_expanded_corpus_v10.10.py

   Add new lists for:
   - FIQH_EXPANDED
   - SUFI_EXPANDED
   - FALSAFA_EXPANDED
   - BUDDHIST_CHINESE_EXPANDED (merge with existing)

3. Create cell4_v10.10.py by copying cell4_v10.9.py and:
   - Replace corpus lists with expanded versions
   - Update version comments

=============================================================================
EXPECTED IMPROVEMENTS
=============================================================================

1. Role augmentation should improve role_swap sensitivity from 0.003 to >0.1
2. Expanded Sanskrit/Pali should allow all_to_sanskrit split to run
3. Expanded Arabic should allow quran_to_fiqh split to run
4. Expanded Buddhist Chinese should improve confucian_to_buddhist F1

Current issues to fix:
- Sanskrit/Pali corpus too small (76+75=151 samples) -> ~700 each
- Arabic quran_to_fiqh didn't run (test set too small) -> 600+
- Chinese diversity tests failed (96 Buddhist) -> 300+
- Role swap sensitivity very weak (0.003) -> >0.1 via augmentation
"""

# Quick test of role swap function (from cell7)
def swap_roles_simple(text, language):
    """Simple role swap using word order reversal for common patterns."""
    import re

    patterns = {
        "english": [
            (r"(\w+) must (\w+) (\w+)", r"\3 must \2 \1"),
            (r"(\w+) should (\w+) (\w+)", r"\3 should \2 \1"),
        ],
    }

    lang_patterns = patterns.get(language, patterns["english"])
    for pattern, replacement in lang_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            swapped = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            if swapped != text:
                return swapped
    return None

# Test examples
if __name__ == "__main__":
    test_cases = [
        ("The parent must protect the child", "english"),
        ("One should help another", "english"),
        ("The teacher shall guide the student", "english"),
    ]

    print("Role swap examples:")
    for text, lang in test_cases:
        swapped = swap_roles_simple(text, lang)
        print(f"  Original: {text}")
        print(f"  Swapped:  {swapped}")
        print()
