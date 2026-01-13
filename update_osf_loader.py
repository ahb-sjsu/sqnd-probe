import json

nb = json.load(open('BIP_v10.5_expanded.ipynb', encoding='utf-8'))
cell4 = nb['cells'][4]['source']
cell4_str = ''.join(cell4)

old_section = '''    # ===== KAGGLE: Ancient Chinese from Wikisource =====
    if chinese_count < MAX_PER_LANG:
        print("  Checking for Kaggle zh-wenyanwen-wikisource...")
        wikisource_path = Path('data/raw/zh-wenyanwen-wikisource')
        if not wikisource_path.exists():
            try:
                subprocess.run(['kaggle', 'datasets', 'download', '-d',
                               'raynardj/zh-wenyanwen-wikisource',
                               '-p', 'data/raw/', '--unzip'], check=True, timeout=120)
                print("    Downloaded from Kaggle!")
            except Exception as e:
                print(f"    Kaggle download failed: {e}")

        # Load if available
        wikisource_files = list(Path('data/raw').glob('*wenyanwen*.txt')) + list(Path('data/raw').glob('*wenyanwen*.csv'))
        ws_count = 0
        for ws_file in wikisource_files[:5]:
            try:
                with open(ws_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if chinese_count >= MAX_PER_LANG:
                            break
                        line = line.strip()
                        if len(line) > 20 and len(line) < 2000:
                            all_passages.append({
                                'id': f"wikisource_{len(all_passages)}",
                                'text': line,
                                'lang': 'classical_chinese',
                                'source': 'Wikisource',
                                'period': 'CONFUCIAN'
                            })
                            chinese_count += 1
                            ws_count += 1
            except Exception as e:
                continue
        if ws_count > 0:
            print(f"    Added {ws_count:,} from Wikisource")

    print(f"  Total Classical Chinese: {chinese_count:,}")'''

new_section = '''    # ===== OSF: Chinese Classics & Dynastic Histories =====
    if chinese_count < MAX_PER_LANG:
        print("  Loading from OSF Chinese Classics dataset...")
        osf_zip_name = 'osfstorage-archive.zip'
        osf_local_zip = Path(f'data/raw/{osf_zip_name}')
        osf_drive_zip = Path(f'{SAVE_DIR}/{osf_zip_name}') if DRIVE_AVAILABLE else None
        osf_extract_dir = Path('data/raw/osf_chinese')

        # Check if already extracted
        if not osf_extract_dir.exists():
            # Try Drive first, then local, then download
            zip_path = None
            if osf_drive_zip and osf_drive_zip.exists():
                zip_path = osf_drive_zip
                print("    Found zip in Google Drive")
            elif osf_local_zip.exists():
                zip_path = osf_local_zip
                print("    Found zip locally")
            else:
                print("    Downloading from OSF (may take a few minutes)...")
                try:
                    subprocess.run(['curl', '-L', '-o', str(osf_local_zip),
                        'https://files.osf.io/v1/resources/tp729/providers/osfstorage/?zip='],
                        check=True, timeout=600)
                    zip_path = osf_local_zip
                    print("    Downloaded!")
                except Exception as e:
                    print(f"    OSF download failed: {e}")

            # Extract if we have the zip
            if zip_path:
                try:
                    import zipfile
                    print("    Extracting...")
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        z.extractall(osf_extract_dir)
                    print("    Extracted!")
                except Exception as e:
                    print(f"    Extraction failed: {e}")

        # Load texts from extracted data
        osf_count = 0
        if osf_extract_dir.exists():
            # Load 13 Classics first (philosophy)
            classics_dir = osf_extract_dir / '_13_classics'
            if classics_dir.exists():
                for txt_file in classics_dir.glob('13_*.txt'):
                    if chinese_count >= MAX_PER_LANG:
                        break
                    try:
                        with open(txt_file, 'r', encoding='utf-8-sig') as f:
                            content = f.read()
                        # Split by line, skip headers (lines starting with G* or *)
                        paragraphs = content.split('\\n')
                        for para in paragraphs:
                            if chinese_count >= MAX_PER_LANG:
                                break
                            para = para.strip()
                            # Skip header lines and short lines
                            if para.startswith('G*') or para.startswith('*') or len(para) < 20:
                                continue
                            if len(para) < 2000:
                                all_passages.append({
                                    'id': f"osf_classics_{len(all_passages)}",
                                    'text': para,
                                    'lang': 'classical_chinese',
                                    'source': txt_file.stem.replace('13_', '').replace('_mod', ''),
                                    'period': 'CONFUCIAN'
                                })
                                chinese_count += 1
                                osf_count += 1
                    except Exception as e:
                        continue
                print(f"    13 Classics: {osf_count:,} passages")

            # Load 24 Histories if we need more
            histories_dir = osf_extract_dir / '_24_histories_source_texts'
            hist_count = 0
            if histories_dir.exists() and chinese_count < MAX_PER_LANG:
                for txt_file in sorted(histories_dir.glob('*_full.txt'))[:5]:  # First 5 histories
                    if chinese_count >= MAX_PER_LANG:
                        break
                    try:
                        with open(txt_file, 'r', encoding='utf-8-sig') as f:
                            content = f.read()
                        paragraphs = content.split('\\n')
                        for para in paragraphs:
                            if chinese_count >= MAX_PER_LANG:
                                break
                            para = para.strip()
                            if para.startswith('*') or len(para) < 30:
                                continue
                            if len(para) < 2000:
                                all_passages.append({
                                    'id': f"osf_hist_{len(all_passages)}",
                                    'text': para,
                                    'lang': 'classical_chinese',
                                    'source': txt_file.stem,
                                    'period': 'CONFUCIAN'
                                })
                                chinese_count += 1
                                hist_count += 1
                    except Exception as e:
                        continue
                print(f"    24 Histories: {hist_count:,} passages")

    print(f"  Total Classical Chinese: {chinese_count:,}")'''

if old_section in cell4_str:
    cell4_str = cell4_str.replace(old_section, new_section)
    lines = cell4_str.split('\n')
    nb['cells'][4]['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    print('SUCCESS: Replaced Wikisource with OSF loader')
else:
    print('ERROR: Could not find Wikisource section to replace')
    print('Searching for partial matches...')
    if 'Wikisource' in cell4_str:
        print('  Found "Wikisource"')
    if 'KAGGLE' in cell4_str:
        print('  Found "KAGGLE"')

with open('BIP_v10.5_expanded.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
