import json

nb = json.load(open('BIP_v10.5_expanded.ipynb', encoding='utf-8'))
cell4 = ''.join(nb['cells'][4]['source'])

old_section = '''# ===== KAGGLE: Ancient Chinese from Wikisource =====
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

new_section = '''# ===== KAGGLE: Ancient Chinese Wenyanwen (132K texts, 552M chars) =====
    if chinese_count < MAX_PER_LANG:
        print("  Loading from Kaggle Wenyanwen dataset...")
        wenyan_zip_name = 'Ancient_Chinese_Text_(wenyanwen)_archive.zip'
        wenyan_csv_name = 'cn_wenyan.csv'
        wenyan_local_zip = Path(f'data/raw/{wenyan_zip_name}')
        wenyan_drive_zip = Path(f'{SAVE_DIR}/{wenyan_zip_name}') if DRIVE_AVAILABLE else None
        wenyan_local_csv = Path(f'data/raw/{wenyan_csv_name}')
        wenyan_drive_csv = Path(f'{SAVE_DIR}/{wenyan_csv_name}') if DRIVE_AVAILABLE else None

        # Find the CSV (extracted or in zip)
        csv_path = None
        if wenyan_local_csv.exists():
            csv_path = wenyan_local_csv
            print("    Found CSV locally")
        elif wenyan_drive_csv and wenyan_drive_csv.exists():
            csv_path = wenyan_drive_csv
            print("    Found CSV in Drive")
        else:
            # Need to extract from zip
            zip_path = None
            if wenyan_local_zip.exists():
                zip_path = wenyan_local_zip
                print("    Found zip locally")
            elif wenyan_drive_zip and wenyan_drive_zip.exists():
                zip_path = wenyan_drive_zip
                print("    Found zip in Drive")

            if zip_path:
                try:
                    import zipfile
                    print("    Extracting CSV from zip...")
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        z.extract(wenyan_csv_name, 'data/raw/')
                    csv_path = wenyan_local_csv
                    print("    Extracted!")
                except Exception as e:
                    print(f"    Extraction failed: {e}")

        # Load texts from CSV
        wenyan_count = 0
        if csv_path and csv_path.exists():
            import csv
            csv.field_size_limit(10000000)  # Some texts are very long
            try:
                with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if chinese_count >= MAX_PER_LANG:
                            break
                        text = row.get('text', '')
                        title = row.get('title', '')
                        # Split long texts into passages (max 2000 chars each)
                        # Use paragraph breaks or every 1500 chars
                        paragraphs = text.split('\\n')
                        current_para = ''
                        for para in paragraphs:
                            para = para.strip()
                            if not para:
                                continue
                            if len(current_para) + len(para) < 1500:
                                current_para += para
                            else:
                                if len(current_para) > 50:
                                    all_passages.append({
                                        'id': f"wenyan_{len(all_passages)}",
                                        'text': current_para,
                                        'lang': 'classical_chinese',
                                        'source': title.split('/')[0] if '/' in title else title,
                                        'period': 'CONFUCIAN'
                                    })
                                    chinese_count += 1
                                    wenyan_count += 1
                                    if chinese_count >= MAX_PER_LANG:
                                        break
                                current_para = para
                        # Don't forget last paragraph
                        if current_para and len(current_para) > 50 and chinese_count < MAX_PER_LANG:
                            all_passages.append({
                                'id': f"wenyan_{len(all_passages)}",
                                'text': current_para,
                                'lang': 'classical_chinese',
                                'source': title.split('/')[0] if '/' in title else title,
                                'period': 'CONFUCIAN'
                            })
                            chinese_count += 1
                            wenyan_count += 1
                print(f"    Added {wenyan_count:,} passages from Wenyanwen")
            except Exception as e:
                print(f"    Error loading Wenyanwen: {e}")

    print(f"  Total Classical Chinese: {chinese_count:,}")'''

if old_section in cell4:
    cell4 = cell4.replace(old_section, new_section)
    lines = cell4.split('\n')
    nb['cells'][4]['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    print('SUCCESS: Replaced Wikisource with Wenyanwen loader')
else:
    print('ERROR: Could not find Wikisource section to replace')
    # Debug
    if 'KAGGLE: Ancient Chinese from Wikisource' in cell4:
        print('  Found header but full match failed')
        # Show what differs
        idx = cell4.find('# ===== KAGGLE: Ancient Chinese from Wikisource =====')
        print(f'  Found at index {idx}')
        print('  Actual content:')
        print(repr(cell4[idx:idx+200]))

with open('BIP_v10.5_expanded.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
