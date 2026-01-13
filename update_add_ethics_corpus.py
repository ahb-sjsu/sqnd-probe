"""
Add ETHICS and Social Chemistry 101 datasets to Cell 4
Generic augmentation: downloads missing datasets for ANY under-represented language
"""
import json

nb = json.load(open('BIP_v10.6_expanded.ipynb', encoding='utf-8'))
cell4 = ''.join(nb['cells'][4]['source'])

# First, update the SKIP_PROCESSING logic
old_skip_check = '''# Check if we should skip processing (data loaded from Drive)
# Check if we should use cached data or download fresh
SKIP_PROCESSING = LOAD_FROM_DRIVE  # Re-evaluate based on current settings
if SKIP_PROCESSING:'''

new_skip_check = '''# Check if we should skip processing (data loaded from Drive)
# Check if we should use cached data or download fresh
SKIP_PROCESSING = LOAD_FROM_DRIVE  # Re-evaluate based on current settings

# Minimum thresholds for balanced experiments
MIN_CORPUS_SIZE = {
    'english': 100000,       # Need 100K+ for Modern→Ancient direction
    'classical_chinese': 50000,  # Need for chinese splits
    'hebrew': 10000,
    'aramaic': 5000,
    'arabic': 5000,
}

# Available augmentation datasets by language
AUGMENTATION_DATASETS = {
    'english': [
        ('hendrycks/ethics', 'ETHICS'),           # ~130K moral scenarios
        ('allenai/social_chem_101', 'SocialChem'), # ~292K social norms
    ],
    'classical_chinese': [
        ('wikisource_zh_classical', 'WikisourceZH'),  # If available
    ],
}

if SKIP_PROCESSING:'''

if old_skip_check in cell4:
    cell4 = cell4.replace(old_skip_check, new_skip_check)
    print("Updated SKIP_PROCESSING section with generic augmentation config")

# Update the cached data validation with generic augmentation
old_validation = '''    # Validate corpus sizes
    MIN_RECOMMENDED = {'hebrew': 10000, 'aramaic': 5000, 'classical_chinese': 500, 'arabic': 500, 'english': 5000}
    print("\\nCorpus adequacy check:")
    for lang, min_size in MIN_RECOMMENDED.items():
        actual = by_lang.get(lang, 0)
        status = "OK" if actual >= min_size else "LOW"
        print(f"  {lang}: {actual:,} (need {min_size:,}) - {status}")

    if by_lang.get('english', 0) < 1000:
        print("\\nWARNING: English corpus too small for reliable semitic_to_non_semitic test!")
    if by_lang.get('classical_chinese', 0) < 100:
        print("WARNING: Chinese corpus too small!")'''

new_validation = '''    # Validate corpus sizes and identify what needs augmentation
    print("\\nCorpus adequacy check:")
    languages_to_augment = []
    for lang, min_size in MIN_CORPUS_SIZE.items():
        actual = by_lang.get(lang, 0)
        status = "OK" if actual >= min_size else "NEED MORE"
        print(f"  {lang}: {actual:,} / {min_size:,} - {status}")
        if actual < min_size and lang in AUGMENTATION_DATASETS:
            languages_to_augment.append((lang, min_size - actual))

    # Augment any under-represented languages that have available datasets
    if languages_to_augment:
        print(f"\\n" + "="*60)
        print(f"AUGMENTING UNDER-REPRESENTED CORPORA")
        print(f"="*60)
        print(f"Languages to augment: {[l for l, _ in languages_to_augment]}")

        # Load existing passages
        all_passages = []
        with open('data/processed/passages.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                all_passages.append(json.loads(line))

        # Normalize field names
        for p in all_passages:
            if 'lang' not in p and 'language' in p:
                p['lang'] = p['language']
            if 'period' not in p and 'time_period' in p:
                p['period'] = p['time_period']

        print(f"Loaded {len(all_passages):,} existing passages")

        from datasets import load_dataset

        for lang, needed in languages_to_augment:
            lang_count = by_lang.get(lang, 0)
            print(f"\\n--- Augmenting {lang} (need {needed:,} more) ---")

            for dataset_name, short_name in AUGMENTATION_DATASETS.get(lang, []):
                if lang_count >= MIN_CORPUS_SIZE[lang]:
                    break

                print(f"  Loading {short_name}...")
                try:
                    if dataset_name == 'hendrycks/ethics':
                        # ETHICS has multiple categories
                        categories = ['commonsense', 'deontology', 'justice', 'utilitarianism', 'virtue']
                        for cat in categories:
                            if lang_count >= MIN_CORPUS_SIZE[lang]:
                                break
                            try:
                                ds = load_dataset(dataset_name, cat, split='train', trust_remote_code=True)
                                cat_count = 0
                                for item in ds:
                                    if lang_count >= MIN_CORPUS_SIZE[lang]:
                                        break
                                    if cat == 'commonsense':
                                        text = item.get('input', '')
                                    elif cat == 'justice':
                                        text = item.get('scenario', '')
                                    elif cat == 'deontology':
                                        text = item.get('scenario', '') + ' ' + item.get('excuse', '')
                                    elif cat == 'virtue':
                                        text = item.get('scenario', '')
                                    else:
                                        text = str(item.get('baseline', '')) + ' vs ' + str(item.get('less_pleasant', ''))

                                    if text and len(text) > 30:
                                        all_passages.append({
                                            'id': f"ethics_{cat}_{len(all_passages)}",
                                            'text': text[:1000],
                                            'lang': lang,
                                            'language': lang,
                                            'source': f'ETHICS_{cat}',
                                            'period': 'MODERN',
                                            'time_period': 'MODERN'
                                        })
                                        lang_count += 1
                                        cat_count += 1
                                print(f"    {cat}: +{cat_count:,}")
                            except Exception as e:
                                print(f"    {cat} error: {e}")

                    elif dataset_name == 'allenai/social_chem_101':
                        ds = load_dataset(dataset_name, split='train', trust_remote_code=True)
                        sc_count = 0
                        for item in ds:
                            if lang_count >= MIN_CORPUS_SIZE[lang]:
                                break
                            action = item.get('action', '')
                            situation = item.get('situation', '')
                            rot = item.get('rot', '')

                            if rot and len(rot) > 20:
                                text = f"{situation} {action}".strip() if situation else action
                                text = f"{text}. {rot}" if text else rot

                                all_passages.append({
                                    'id': f"socialchem_{len(all_passages)}",
                                    'text': text[:1000],
                                    'lang': lang,
                                    'language': lang,
                                    'source': 'Social_Chemistry_101',
                                    'period': 'MODERN',
                                    'time_period': 'MODERN'
                                })
                                lang_count += 1
                                sc_count += 1
                        print(f"    Social Chemistry: +{sc_count:,}")

                    else:
                        # Generic HuggingFace dataset
                        try:
                            ds = load_dataset(dataset_name, split='train', trust_remote_code=True)
                            gen_count = 0
                            for item in ds:
                                if lang_count >= MIN_CORPUS_SIZE[lang]:
                                    break
                                text = item.get('text', '') or item.get('content', '') or str(item)
                                if text and len(text) > 50:
                                    all_passages.append({
                                        'id': f"{short_name.lower()}_{len(all_passages)}",
                                        'text': text[:1000],
                                        'lang': lang,
                                        'language': lang,
                                        'source': short_name,
                                        'period': 'MODERN',
                                        'time_period': 'MODERN'
                                    })
                                    lang_count += 1
                                    gen_count += 1
                            print(f"    {short_name}: +{gen_count:,}")
                        except Exception as e:
                            print(f"    {short_name} failed: {e}")

                except Exception as e:
                    print(f"    {short_name} failed: {e}")

            by_lang[lang] = lang_count
            print(f"  {lang} now: {lang_count:,}")

        # Extract bonds for new passages
        print("\\nExtracting bonds for new passages...")
        new_bonds = []
        new_sources = {'ETHICS_commonsense', 'ETHICS_deontology', 'ETHICS_justice',
                       'ETHICS_utilitarianism', 'ETHICS_virtue', 'Social_Chemistry_101'}

        for p in tqdm(all_passages, desc="Processing"):
            src = p.get('source', '')
            if any(src.startswith(s.split('_')[0]) for s in new_sources) or src in new_sources:
                text_lower = p['text'].lower()
                if any(w in text_lower for w in ['wrong', 'bad', "shouldn't", 'immoral', 'rude', 'unethical']):
                    bond_type = 'PROHIBITION'
                elif any(w in text_lower for w in ['should', 'must', 'duty', 'obligat', 'need to']):
                    bond_type = 'OBLIGATION'
                elif any(w in text_lower for w in ['okay', 'fine', 'acceptable', 'can', 'may', 'allowed']):
                    bond_type = 'PERMISSION'
                else:
                    bond_type = 'NEUTRAL'

                new_bonds.append({
                    'passage_id': p['id'],
                    'bond_type': bond_type,
                    'language': p.get('language', p.get('lang')),
                    'time_period': p.get('time_period', p.get('period', 'MODERN')),
                    'source': src,
                    'text': p['text'][:500],
                    'context': 'prescriptive',
                    'confidence': 'high'
                })

        # Load existing bonds and merge
        existing_bonds = []
        with open('data/processed/bonds.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                existing_bonds.append(json.loads(line))

        all_bonds = existing_bonds + new_bonds
        print(f"Total bonds: {len(all_bonds):,} ({len(new_bonds):,} new)")

        # Save updated passages
        with open('data/processed/passages.jsonl', 'w', encoding='utf-8') as f:
            for p in all_passages:
                p_out = {
                    'id': p['id'],
                    'text': p['text'],
                    'language': p.get('language', p.get('lang', 'english')),
                    'source': p.get('source', ''),
                    'time_period': p.get('time_period', p.get('period', 'MODERN'))
                }
                f.write(json.dumps(p_out, ensure_ascii=False) + '\\n')

        # Save updated bonds
        with open('data/processed/bonds.jsonl', 'w', encoding='utf-8') as f:
            for b in all_bonds:
                f.write(json.dumps(b, ensure_ascii=False) + '\\n')

        print("Saved augmented data")

        # Copy to Drive
        if USE_DRIVE_DATA:
            try:
                shutil.copy('data/processed/passages.jsonl', f'{SAVE_DIR}/passages.jsonl')
                shutil.copy('data/processed/bonds.jsonl', f'{SAVE_DIR}/bonds.jsonl')
                print(f"Updated Drive cache: {SAVE_DIR}")
            except Exception as e:
                print(f"Drive update failed: {e}")

        # Final summary
        print(f"\\nFinal corpus sizes:")
        for lang, cnt in sorted(by_lang.items(), key=lambda x: -x[1]):
            target = MIN_CORPUS_SIZE.get(lang, 0)
            status = "OK" if cnt >= target else "LOW"
            print(f"  {lang}: {cnt:,} ({status})")
        n_passages = len(all_passages)'''

if old_validation in cell4:
    cell4 = cell4.replace(old_validation, new_validation)
    print("Updated validation with generic augmentation logic")
else:
    print("WARNING: Could not find validation section to update")

# Add datasets to dependencies
cell1 = ''.join(nb['cells'][1]['source'])
old_deps = 'for pkg in ["transformers", "sentence-transformers", "pandas", "tqdm", "scikit-learn", "pyyaml", "psutil"]:'
new_deps = 'for pkg in ["transformers", "sentence-transformers", "pandas", "tqdm", "scikit-learn", "pyyaml", "psutil", "datasets"]:'
if old_deps in cell1:
    cell1 = cell1.replace(old_deps, new_deps)
    print("Added 'datasets' to Cell 1 dependencies")

# Update version
cell0 = ''.join(nb['cells'][0]['source'])
cell0 = cell0.replace('v10.6', 'v10.7')
lines = cell0.split('\n')
nb['cells'][0]['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

cell1 = cell1.replace('v10.6', 'v10.7')
lines = cell1.split('\n')
nb['cells'][1]['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

lines = cell4.split('\n')
nb['cells'][4]['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

with open('BIP_v10.7_expanded.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n" + "="*60)
print("BIP v10.7 CREATED")
print("="*60)
print("""
Generic augmentation logic:
  1. Loads cached data from Drive
  2. Checks EACH language against MIN_CORPUS_SIZE thresholds
  3. For any under-represented language with available datasets:
     - Downloads from HuggingFace
     - Extracts bonds
     - Saves augmented data
  4. Updates Drive cache with augmented corpus

Configurable thresholds (MIN_CORPUS_SIZE):
  - english: 100,000 (ETHICS + Social Chemistry)
  - classical_chinese: 50,000
  - hebrew: 10,000
  - aramaic: 5,000
  - arabic: 5,000

Available augmentation datasets (AUGMENTATION_DATASETS):
  - english: ETHICS (~130K), Social Chemistry 101 (~292K)
  - (easily extensible for other languages)

Saved to: BIP_v10.7_expanded.ipynb
""")
