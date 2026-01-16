###############################################################################
# CBETA Buddhist loader
# Copy this into Cell 2, AFTER the ctext loader function
###############################################################################


def load_chinese_buddhist() -> list[dict]:
    """Load Chinese Buddhist texts from CBETA via CLTK GitHub mirrors."""
    passages = []
    cache_file = CACHE_DIR / "cbeta_buddhist.json"

    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            passages = json.load(f)
            print(f"  CBETA Buddhist: {len(passages):,} passages (cached)")
            return passages

    print("  Fetching Chinese Buddhist texts from CBETA/CLTK...")

    texts = [
        (
            "T08/T0235",
            "Diamond Sutra",
            "https://raw.githubusercontent.com/cltk/chinese_text_cbeta_02/master/cltk_json/T08/T0235.json",
        ),
        (
            "T08/T0251",
            "Heart Sutra",
            "https://raw.githubusercontent.com/cltk/chinese_text_cbeta_02/master/cltk_json/T08/T0251.json",
        ),
        (
            "T17/T0784",
            "42 Sections",
            "https://raw.githubusercontent.com/cltk/chinese_text_cbeta_02/master/cltk_json/T17/T0784.json",
        ),
        (
            "T48/T2008",
            "Platform Sutra",
            "https://raw.githubusercontent.com/cltk/chinese_text_cbeta_02/master/cltk_json/T48/T2008.json",
        ),
        (
            "T14/T0475",
            "Vimalakirti",
            "https://raw.githubusercontent.com/cltk/chinese_text_cbeta_02/master/cltk_json/T14/T0475.json",
        ),
        (
            "T09/T0262",
            "Lotus Sutra",
            "https://raw.githubusercontent.com/cltk/chinese_text_cbeta_02/master/cltk_json/T09/T0262.json",
        ),
    ]

    for text_id, name, url in texts:
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code != 200:
                print(f"    {name}: HTTP {resp.status_code}")
                continue

            data = resp.json()
            count = 0

            if isinstance(data, list):
                for item in data:
                    text = item.get("text", "") if isinstance(item, dict) else str(item)
                    text = text.strip()
                    if text and len(text) > 15 and not text.startswith("【"):
                        passages.append(
                            {
                                "id": f"cbeta_{text_id.replace('/', '_')}_{count}",
                                "text": text,
                                "language": "classical_chinese",
                                "source": f"CBETA/{name}",
                                "time_periods": ["BUDDHIST"],
                            }
                        )
                        count += 1

            if count > 0:
                print(f"    {name}: {count} passages")
            else:
                print(f"    {name}: 0 (format issue)")

        except requests.exceptions.Timeout:
            print(f"    {name}: timeout")
        except Exception as e:
            print(f"    {name}: {type(e).__name__}")

    if passages:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(passages, f, ensure_ascii=False)

    print(f"  CBETA Buddhist: {len(passages):,} passages total")
    return passages


###############################################################################
# ALSO ADD this line in the MAIN LOADER section (after ctext loader call):
###############################################################################
#
# print()
# print("[CHINESE BUDDHIST]")
# by_language["classical_chinese"].extend(load_chinese_buddhist())
#
###############################################################################
