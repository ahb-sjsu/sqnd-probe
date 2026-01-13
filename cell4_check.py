
LOAD_FROM_DRIVE = False
SKIP_PROCESSING = False  
GPU_TIER = 'L4'
MAX_PER_LANG = 50000
REFRESH_DATA_FROM_SOURCE = True
USE_DRIVE_DATA = True
DRIVE_AVAILABLE = True
SAVE_DIR = '/tmp'
def normalize_text(t, l): return t
def detect_context(t, l, p): return ('prescriptive', None)
ALL_BOND_PATTERNS = {}
import os
#@title 4. Parallel Download + Stream Processing { display-mode: "form" }
#@markdown Loads ALL corpora including expanded Arabic, Chinese, and Western classics

import json
import re
import random
import gc
import shutil
import requests
import time
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm

# Thread-safe queue for passages
passage_queue = Queue(maxsize=100000)
download_complete = threading.Event()
corpus_stats = defaultdict(int)
stats_lock = threading.Lock()

def update_stats(lang, count):
    with stats_lock:
        corpus_stats[lang] += count
        total = sum(corpus_stats.values())
        if total % 1000 == 0:
            print(".", end="", flush=True)

# Check if we should skip processing (data loaded from Drive)
# Check if we should use cached data or download fresh
SKIP_PROCESSING = LOAD_FROM_DRIVE  # Re-evaluate based on current settings
if SKIP_PROCESSING:
    print("="*60)
    print("USING CACHED DATA - Run with REFRESH_DATA_FROM_SOURCE=True to use v10.4 loaders")
    print("="*60)

    # Count passages by language
    by_lang = defaultdict(int)
    with open('data/processed/passages.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            p = json.loads(line)
            by_lang[p['language']] += 1

    print("\nPassages by language:")
    for lang, cnt in sorted(by_lang.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {cnt:,}")

    n_passages = sum(by_lang.values())
    print(f"\nTotal: {n_passages:,} passages")

    # Validate corpus sizes
    MIN_RECOMMENDED = {'hebrew': 10000, 'aramaic': 5000, 'classical_chinese': 500, 'arabic': 500, 'english': 5000}
    print("\nCorpus adequacy check:")
    for lang, min_size in MIN_RECOMMENDED.items():
        actual = by_lang.get(lang, 0)
        status = "OK" if actual >= min_size else "LOW"
        print(f"  {lang}: {actual:,} (need {min_size:,}) - {status}")

    if by_lang.get('english', 0) < 1000:
        print("\nWARNING: English corpus too small for reliable semitic_to_non_semitic test!")
    if by_lang.get('classical_chinese', 0) < 100:
        print("WARNING: Chinese corpus too small!")

else:
    print("="*60)
    print("LOADING CORPORA")
    print(f"GPU Tier: {GPU_TIER}")
    print(f"Max per language: {MAX_PER_LANG:,}")
    print("="*60)

    random.seed(42)
    all_passages = []

    # ===== SEFARIA (Hebrew/Aramaic) =====
    print("\nLoading Sefaria...")
    sefaria_path = Path('data/raw/Sefaria-Export/json')

    CATEGORY_TO_PERIOD = {
        'Tanakh': 'BIBLICAL', 'Torah': 'BIBLICAL', 'Prophets': 'BIBLICAL', 'Writings': 'BIBLICAL',
        'Mishnah': 'TANNAITIC', 'Tosefta': 'TANNAITIC', 'Sifra': 'TANNAITIC', 'Sifrei': 'TANNAITIC',
        'Talmud': 'TALMUDIC', 'Bavli': 'TALMUDIC', 'Yerushalmi': 'TALMUDIC',
        'Midrash': 'MIDRASHIC', 'Midrash Rabbah': 'MIDRASHIC', 'Midrash Aggadah': 'MIDRASHIC',
        'Halakhah': 'MEDIEVAL', 'Shulchan Arukh': 'MEDIEVAL', 'Mishneh Torah': 'MEDIEVAL',
        'Musar': 'MODERN', 'Chasidut': 'MODERN', 'Modern': 'MODERN'
    }

    lang_counts = {'hebrew': 0, 'aramaic': 0}

    if sefaria_path.exists():
        for json_file in tqdm(list(sefaria_path.rglob('*.json'))[:5000], desc="Sefaria"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, dict) and 'text' in data:
                    # Determine period from path
                    path_parts = str(json_file.relative_to(sefaria_path)).split('/')
                    period = 'CLASSICAL'
                    for part in path_parts:
                        if part in CATEGORY_TO_PERIOD:
                            period = CATEGORY_TO_PERIOD[part]
                            break

                    # Determine language (heuristic: Talmud is primarily Aramaic)
                    is_talmud = any(t in str(json_file) for t in ['Talmud', 'Bavli', 'Yerushalmi'])
                    lang = 'aramaic' if is_talmud else 'hebrew'

                    def extract_texts(obj, texts):
                        if isinstance(obj, str) and len(obj) > 20:
                            texts.append(obj)
                        elif isinstance(obj, list):
                            for item in obj:
                                extract_texts(item, texts)

                    texts = []
                    extract_texts(data['text'], texts)

                    for txt in texts[:50]:  # Limit per file
                        if lang_counts[lang] < MAX_PER_LANG:
                            all_passages.append({
                                'id': f"sefaria_{len(all_passages)}",
                                'text': txt,
                                'lang': lang,
                                'source': json_file.stem,
                                'period': period
                            })
                            lang_counts[lang] += 1

            except Exception as e:
                continue
    else:
        print("  Sefaria not found - will download")

    print(f"  Hebrew: {lang_counts['hebrew']:,}, Aramaic: {lang_counts['aramaic']:,}")

    # ===== CLASSICAL CHINESE (ctext.org API) =====
    print("\nLoading Classical Chinese from ctext.org API...")

    CHINESE_TEXTS = [
        ('ctp:analects', '論語', 'CLASSICAL'),       # Analects
        ('ctp:mengzi', '孟子', 'CLASSICAL'),         # Mencius
        ('ctp:dao-de-jing', '道德經', 'CLASSICAL'),  # Tao Te Ching
        ('ctp:xunzi', '荀子', 'CLASSICAL'),          # Xunzi
        ('ctp:xiaojing', '孝經', 'CLASSICAL'),       # Classic of Filial Piety
        ('ctp:liji', '禮記', 'CLASSICAL'),           # Book of Rites
        ('ctp:mozi', '墨子', 'CLASSICAL'),           # Mozi
        ('ctp:zhuangzi', '莊子', 'CLASSICAL'),       # Zhuangzi
        ('ctp:lunheng', '論衡', 'CLASSICAL'),        # Lunheng
        ('ctp:hanfeizi', '韓非子', 'CLASSICAL'),     # Han Feizi
    ]

    chinese_count = 0
    ctext_base = "https://api.ctext.org/gettext"

    for urn, title, period in CHINESE_TEXTS:
        if chinese_count >= MAX_PER_LANG:
            break
        try:
            resp = requests.get(f"{ctext_base}?urn={urn}", timeout=15)
            if resp.status_code != 200:
                continue

            data = resp.json()
            subsections = data.get('subsections', [])[:30]  # Limit chapters

            for sub_urn in subsections:
                if chinese_count >= MAX_PER_LANG:
                    break
                try:
                    resp2 = requests.get(f"{ctext_base}?urn={sub_urn}", timeout=15)
                    if resp2.status_code == 200:
                        data2 = resp2.json()
                        for para in data2.get('fulltext', []):
                            if para and len(para) > 10 and chinese_count < MAX_PER_LANG:
                                all_passages.append({
                                    'id': f"ctext_{len(all_passages)}",
                                    'text': para,
                                    'lang': 'classical_chinese',
                                    'source': title,
                                    'period': period
                                })
                                chinese_count += 1
                    time.sleep(0.2)  # Rate limit
                except:
                    continue
        except Exception as e:
            print(f"  Error with {title}: {e}")
            continue

    print(f"  Classical Chinese: {chinese_count:,}")

    # ===== ARABIC/ISLAMIC (Kaggle quran-nlp) =====
    print("\nLoading Arabic from Kaggle quran-nlp...")

    arabic_count = 0
    kaggle_path = Path('data/raw/quran-nlp')

    # Try to download from Kaggle
    if not kaggle_path.exists() and REFRESH_DATA_FROM_SOURCE:
        try:
            import subprocess
            import zipfile
            subprocess.run(['pip', 'install', '-q', 'kaggle'], check=True)
            subprocess.run([
                'kaggle', 'datasets', 'download',
                '-d', 'alizahidraja/quran-nlp',
                '-p', 'data/raw'
            ], check=True, timeout=300)

            with zipfile.ZipFile('data/raw/quran-nlp.zip', 'r') as z:
                z.extractall(kaggle_path)
            print("  Downloaded from Kaggle!")
        except Exception as e:
            print(f"  Kaggle download failed: {e}")

    # Load if available
    if kaggle_path.exists():
        import pandas as pd

        # Load Quran
        quran_files = list(kaggle_path.rglob('*quran*.csv'))
        for qf in quran_files:
            if arabic_count >= MAX_PER_LANG:
                break
            try:
                df = pd.read_csv(qf, nrows=MAX_PER_LANG - arabic_count)
                for _, row in df.iterrows():
                    text = str(row.get('arabic', row.get('text', row.get('Arabic', ''))))
                    if text and len(text) > 10 and text != 'nan':
                        all_passages.append({
                            'id': f"quran_{len(all_passages)}",
                            'text': text,
                            'lang': 'arabic',
                            'source': 'Quran',
                            'period': 'CLASSICAL'
                        })
                        arabic_count += 1
            except:
                continue

        # Load Hadith
        hadith_files = list(kaggle_path.rglob('*hadith*.csv'))
        for hf in hadith_files:
            if arabic_count >= MAX_PER_LANG:
                break
            try:
                df = pd.read_csv(hf, nrows=MAX_PER_LANG - arabic_count)
                for _, row in df.iterrows():
                    text = str(row.get('hadith', row.get('text', row.get('Arabic', ''))))
                    if text and len(text) > 10 and text != 'nan':
                        all_passages.append({
                            'id': f"hadith_{len(all_passages)}",
                            'text': text,
                            'lang': 'arabic',
                            'source': 'Hadith',
                            'period': 'CLASSICAL'
                        })
                        arabic_count += 1
            except:
                continue
    else:
        # Try Tanzil.net (simple direct download)
        print("  Trying Tanzil.net for Quran text...")
        try:
            tanzil_url = "https://tanzil.net/pub/download/index.php?quranType=uthmani&outType=txt-2&agree=true"
            resp = requests.get(tanzil_url, timeout=60)
            if resp.status_code == 200:
                lines = resp.text.strip().split('\n')
                for line in lines:
                    if '|' in line and arabic_count < MAX_PER_LANG:
                        parts = line.split('|')
                        if len(parts) >= 3:
                            text = parts[2].strip()
                            if len(text) > 10:
                                all_passages.append({
                                    'id': f"tanzil_{len(all_passages)}",
                                    'text': text,
                                    'lang': 'arabic',
                                    'source': 'Quran (Tanzil)',
                                    'period': 'CLASSICAL'
                                })
                                arabic_count += 1
                print(f"    Loaded {arabic_count} verses from Tanzil")
        except Exception as e:
            print(f"    Tanzil failed: {e}")

        # Final fallback: expanded hardcoded corpus
        if arabic_count < 100:
            print("  Using expanded hardcoded Arabic corpus...")
        ARABIC_CORPUS = [
            # Quran excerpts (moral/ethical content)
            "وَلَا تَقْتُلُوا النَّفْسَ الَّتِي حَرَّمَ اللَّهُ إِلَّا بِالْحَقِّ",
            "وَبِالْوَالِدَيْنِ إِحْسَانًا",
            "وَأَوْفُوا بِالْعَهْدِ إِنَّ الْعَهْدَ كَانَ مَسْئُولًا",
            "إِنَّ اللَّهَ يَأْمُرُ بِالْعَدْلِ وَالْإِحْسَانِ",
            "وَلَا تَبْخَسُوا النَّاسَ أَشْيَاءَهُمْ",
            "وَأَقِيمُوا الْوَزْنَ بِالْقِسْطِ وَلَا تُخْسِرُوا الْمِيزَانَ",
            "يَا أَيُّهَا الَّذِينَ آمَنُوا أَوْفُوا بِالْعُقُودِ",
            "وَتَعَاوَنُوا عَلَى الْبِرِّ وَالتَّقْوَى",
            # ... more can be added
        ]
        for i, txt in enumerate(ARABIC_CORPUS):
            all_passages.append({
                'id': f"arabic_{len(all_passages)}",
                'text': txt,
                'lang': 'arabic',
                'source': 'Quran/Hadith',
                'period': 'CLASSICAL'
            })
            arabic_count += 1

    print(f"  Arabic: {arabic_count:,}")

    # ===== DEAR ABBY (English) =====
    print("
Loading Dear Abby...")

    english_count = 0
    abby_path = Path('data/raw/dear_abby.csv')

    # Check Drive first
    drive_abby = f'{SAVE_DIR}/dear_abby.csv'
    if not abby_path.exists() and os.path.exists(drive_abby):
        os.makedirs('data/raw', exist_ok=True)
        shutil.copy(drive_abby, abby_path)
        print("  Copied from Drive")


    if not abby_path.exists() and REFRESH_DATA_FROM_SOURCE:
        try:
            import subprocess
            subprocess.run(['pip', 'install', '-q', 'kaggle'], check=True)
            subprocess.run([
                'kaggle', 'datasets', 'download',
                '-d', 'thedevastator/20000-dear-abby-questions',
                '-p', 'data/raw',
                '-f', 'dear_abby.csv'
            ], check=True, timeout=120)
            print("  Downloaded from Kaggle!")
        except Exception as e:
            print(f"  Kaggle download failed: {e}")

    if abby_path.exists():
        import pandas as pd
        df = pd.read_csv(abby_path, nrows=MAX_PER_LANG)
        for _, row in df.iterrows():
            question = str(row.get('question', ''))
            answer = str(row.get('answer', ''))
            if len(answer) > 50:
                all_passages.append({
                    'id': f"abby_{len(all_passages)}",
                    'text': answer,
                    'lang': 'english',
                    'source': 'Dear Abby',
                    'period': 'MODERN'
                })
                english_count += 1
    else:
        print("  Dear Abby not found")

    print(f"  Dear Abby: {english_count:,}")

    # ===== WESTERN CLASSICS (English translations) =====
    print("\nLoading Western Classics from MIT Classics Archive...")

    WESTERN_TEXTS = [
        ('https://classics.mit.edu/Aristotle/nicomachaen.mb.txt', 'Nicomachean Ethics', 'CLASSICAL'),
        ('https://classics.mit.edu/Aristotle/politics.mb.txt', 'Politics', 'CLASSICAL'),
        ('https://classics.mit.edu/Plato/republic.mb.txt', 'Republic', 'CLASSICAL'),
        ('https://classics.mit.edu/Plato/laws.mb.txt', 'Laws', 'CLASSICAL'),
        ('https://classics.mit.edu/Antoninus/meditations.mb.txt', 'Meditations', 'CLASSICAL'),
        ('https://classics.mit.edu/Epictetus/epicench.mb.txt', 'Enchiridion', 'CLASSICAL'),
        ('https://classics.mit.edu/Cicero/duties.mb.txt', 'De Officiis', 'CLASSICAL'),
    ]

    western_count = 0
    for url, title, period in WESTERN_TEXTS:
        if english_count + western_count >= MAX_PER_LANG:
            break
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                text = resp.text
                # Split into paragraphs
                paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
                for para in paragraphs[:500]:  # Limit per text
                    if english_count + western_count >= MAX_PER_LANG:
                        break
                    # Clean up
                    para = re.sub(r'\s+', ' ', para).strip()
                    if len(para) > 50:
                        all_passages.append({
                            'id': f"western_{len(all_passages)}",
                            'text': para,
                            'lang': 'english',
                            'source': title,
                            'period': period
                        })
                        western_count += 1
                print(f"  {title}: {min(500, len(paragraphs))} paragraphs")
        except Exception as e:
            print(f"  Error loading {title}: {e}")

    print(f"  Western Classics: {western_count:,}")

    # ===== UNIMORAL (HuggingFace - multilingual moral dilemmas) =====
    print("\nLoading UniMoral from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("shivaniku/UniMoral", split="train", streaming=True)

        unimoral_count = 0
        lang_map = {'ar': 'arabic', 'zh': 'classical_chinese', 'en': 'english'}

        for item in ds:
            if unimoral_count >= min(MAX_PER_LANG, 5000):
                break

            lang_code = item.get('language', 'en')
            if lang_code not in lang_map:
                continue

            text = item.get('scenario', '') or item.get('text', '')
            if len(str(text)) > 20:
                all_passages.append({
                    'id': f"unimoral_{len(all_passages)}",
                    'text': str(text),
                    'lang': lang_map[lang_code],
                    'source': 'UniMoral',
                    'period': 'MODERN'
                })
                unimoral_count += 1

        print(f"  UniMoral: {unimoral_count:,}")
    except Exception as e:
        print(f"  UniMoral error: {e}")

    # ===== UN PARALLEL CORPUS (HuggingFace streaming) =====
    print("\nLoading UN Corpus from HuggingFace (streaming)...")
    try:
        from datasets import load_dataset

        pairs = [('ar', 'en'), ('zh', 'en')]
        un_count = 0
        lang_map = {'ar': 'arabic', 'zh': 'classical_chinese', 'en': 'english'}

        for src, tgt in pairs:
            if un_count >= MAX_PER_LANG:
                break
            try:
                config = f"{src}-{tgt}"
                ds = load_dataset("Helsinki-NLP/un_pc", config, split="train", streaming=True)

                pair_count = 0
                for item in ds:
                    if pair_count >= min(MAX_PER_LANG // 4, 5000):
                        break

                    translation = item.get('translation', {})
                    for lang_code in [src, tgt]:
                        text = translation.get(lang_code, '')
                        if len(text) > 30 and lang_code in lang_map:
                            all_passages.append({
                                'id': f"un_{len(all_passages)}",
                                'text': text,
                                'lang': lang_map[lang_code],
                                'source': 'UN Corpus',
                                'period': 'MODERN'
                            })
                            pair_count += 1
                            un_count += 1

                print(f"  UN {config}: {pair_count:,}")
            except Exception as e:
                print(f"  UN {config} error: {e}")

        print(f"  UN Corpus total: {un_count:,}")
    except Exception as e:
        print(f"  UN Corpus error: {e}")

    # ===== BIBLE PARALLEL CORPUS (GitHub) =====
    print("\nLoading Bible Parallel Corpus...")
    try:
        base_url = "https://raw.githubusercontent.com/christos-c/bible-corpus/master/bibles"
        bible_files = [
            ('Hebrew.xml', 'hebrew'),
            ('Arabic.xml', 'arabic'),
            ('Chinese.xml', 'classical_chinese'),
        ]

        bible_count = 0
        for filename, lang in bible_files:
            if bible_count >= MAX_PER_LANG * 3:
                break
            try:
                url = f"{base_url}/{filename}"
                resp = requests.get(url, timeout=60)
                if resp.status_code == 200:
                    verses = re.findall(r'<seg[^>]*>([^<]+)</seg>', resp.text)
                    file_count = 0
                    for verse in verses:
                        if file_count >= MAX_PER_LANG:
                            break
                        verse = verse.strip()
                        if len(verse) > 10:
                            all_passages.append({
                                'id': f"bible_{len(all_passages)}",
                                'text': verse,
                                'lang': lang,
                                'source': 'Bible',
                                'period': 'CLASSICAL'
                            })
                            file_count += 1
                            bible_count += 1
                    print(f"  Bible {lang}: {file_count:,}")
            except Exception as e:
                print(f"  Bible {filename} error: {e}")

        print(f"  Bible total: {bible_count:,}")
    except Exception as e:
        print(f"  Bible error: {e}")

    # ===== SUMMARY =====
    print(f"\nTOTAL: {len(all_passages):,}")

    # Count by language
    by_lang = defaultdict(int)
    for p in all_passages:
        by_lang[p['lang']] += 1
    print("\nBy language:")
    for lang, cnt in sorted(by_lang.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {cnt:,}")

    # ===== EXTRACT BONDS =====
    print("\n" + "="*60)
    print("EXTRACTING BONDS")
    print("="*60)

    def extract_bond(text, language):
        """Extract bond type with context awareness."""
        tn = normalize_text(text, language)

        for bt, pats in ALL_BOND_PATTERNS.get(language, {}).items():
            for p in pats:
                match = re.search(p, tn)
                if match:
                    # Check context around the match
                    context, marker_type = detect_context(text, language, match.start())
                    confidence = 0.9 if context == 'prescriptive' else 0.5
                    return bt, context, confidence
        return None, 'unknown', 0.5

    bonds = []
    for p in tqdm(all_passages, desc="Extracting bonds"):
        bt, ctx, conf = extract_bond(p['text'], p['lang'])
        if bt:
            bonds.append({
                'passage_id': p['id'],
                'bond_type': bt,
                'language': p['lang'],
                'time_period': p['period'],
                'source': p['source'],
                'text': p['text'][:500],
                'context': ctx,
                'confidence': conf
            })

    print(f"\nExtracted {len(bonds):,} bonds from {len(all_passages):,} passages")

    # Count by bond type
    by_bond = defaultdict(int)
    for b in bonds:
        by_bond[b['bond_type']] += 1
    print("\nBy bond type:")
    for bt, cnt in sorted(by_bond.items(), key=lambda x: -x[1]):
        print(f"  {bt}: {cnt:,}")

    # Count by context
    by_ctx = defaultdict(int)
    for b in bonds:
        by_ctx[b['context']] += 1
    print("\nBy context:")
    for ctx, cnt in sorted(by_ctx.items(), key=lambda x: -x[1]):
        print(f"  {ctx}: {cnt:,}")

    # ===== SAVE =====
    print("\n" + "="*60)
    print("SAVING DATA")
    print("="*60)

    # Save passages
    with open('data/processed/passages.jsonl', 'w', encoding='utf-8') as f:
        for p in all_passages:
            # Normalize field names
            p_out = {
                'id': p['id'],
                'text': p['text'],
                'language': p['lang'],
                'source': p['source'],
                'time_period': p['period']
            }
            f.write(json.dumps(p_out, ensure_ascii=False) + '\n')
    print(f"  Saved {len(all_passages):,} passages to data/processed/passages.jsonl")

    # Save bonds
    with open('data/processed/bonds.jsonl', 'w', encoding='utf-8') as f:
        for b in bonds:
            f.write(json.dumps(b, ensure_ascii=False) + '\n')
    print(f"  Saved {len(bonds):,} bonds to data/processed/bonds.jsonl")

    # Copy to Drive if enabled
    if USE_DRIVE_DATA and DRIVE_AVAILABLE:
        try:
            shutil.copy('data/processed/passages.jsonl', f'{SAVE_DIR}/passages.jsonl')
            shutil.copy('data/processed/bonds.jsonl', f'{SAVE_DIR}/bonds.jsonl')
            print(f"  Copied to Drive: {SAVE_DIR}")
        except Exception as e:
            print(f"  Drive copy failed: {e}")

    gc.collect()
    print("\nDone!")
