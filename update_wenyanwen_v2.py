import json

nb = json.load(open('BIP_v10.5_expanded.ipynb', encoding='utf-8'))
cell4 = ''.join(nb['cells'][4]['source'])

# Find section boundaries by markers
start = cell4.find('# ===== KAGGLE: Ancient Chinese from Wikisource =====')
end_marker = 'print(f"  Total Classical Chinese: {chinese_count:,}")'
end = cell4.find(end_marker, start) + len(end_marker)

if start < 0 or end <= start:
    print("ERROR: Could not find section markers")
    exit(1)

old_section = cell4[start:end]
print(f"Found old section: {len(old_section)} chars")

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

# Replace
cell4_new = cell4[:start] + new_section + cell4[end:]
lines = cell4_new.split('\n')
nb['cells'][4]['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

with open('BIP_v10.5_expanded.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('SUCCESS: Updated notebook with Wenyanwen loader')
