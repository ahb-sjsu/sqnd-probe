###############################################################################
# CLEAN ctext loader with error logging
# Copy this into Cell 2, replacing the existing load_chinese_ctext function
###############################################################################


def load_chinese_ctext() -> list[dict]:
    """Load Chinese classics from ctext.org API."""
    passages = []
    cache_file = CACHE_DIR / "ctext.json"

    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            passages = json.load(f)
            # Show breakdown by period
            by_period = {}
            for p in passages:
                period = p.get("time_periods", ["UNKNOWN"])[0]
                by_period[period] = by_period.get(period, 0) + 1
            print(f"  ctext.org: {len(passages):,} passages (cached)")
            for period, count in sorted(by_period.items()):
                print(f"    {period}: {count}")
            return passages

    print("  Fetching from ctext.org API...")

    texts = [
        (
            "ctp:analects",
            "Analects",
            "CONFUCIAN",
            [
                "xue-er",
                "wei-zheng",
                "ba-yi",
                "li-ren",
                "gong-ye-chang",
                "yong-ye",
                "shu-er",
                "tai-bo",
                "zi-han",
                "xiang-dang",
                "xian-jin",
                "yan-yuan",
                "zi-lu",
                "xian-wen",
                "wei-ling-gong",
                "ji-shi",
                "yang-huo",
                "wei-zi",
                "zi-zhang",
                "yao-yue",
            ],
        ),
        (
            "ctp:mengzi",
            "Mencius",
            "CONFUCIAN",
            [
                "liang-hui-wang-i",
                "liang-hui-wang-ii",
                "gong-sun-chou-i",
                "gong-sun-chou-ii",
                "teng-wen-gong-i",
                "teng-wen-gong-ii",
                "li-lou-i",
                "li-lou-ii",
                "wan-zhang-i",
                "wan-zhang-ii",
                "gao-zi-i",
                "gao-zi-ii",
                "jin-xin-i",
                "jin-xin-ii",
            ],
        ),
        ("ctp:dao-de-jing", "Daodejing", "DAOIST", [str(i) for i in range(1, 82)]),
        (
            "ctp:zhuangzi",
            "Zhuangzi",
            "DAOIST",
            [
                "xiao-yao-you",
                "qi-wu-lun",
                "yang-sheng-zhu",
                "ren-jian-shi",
                "de-chong-fu",
                "da-zong-shi",
                "ying-di-wang",
            ],
        ),
        (
            "ctp:xunzi",
            "Xunzi",
            "CONFUCIAN",
            [
                "quan-xue",
                "xiu-shen",
                "bu-gou",
                "rong-ru",
                "fei-xiang",
                "wang-zhi",
            ],
        ),
        (
            "ctp:mozi",
            "Mozi",
            "MOHIST",
            [
                "qin-shi",
                "xiu-shen",
                "suo-ran",
                "fa-yi",
                "qi-huan",
                "ci-guo",
            ],
        ),
        (
            "ctp:hanfeizi",
            "Han Feizi",
            "LEGALIST",
            [
                "chu-jian",
                "cun-han",
                "nan-yan",
                "ai-chen",
                "zhu-dao",
                "yang-quan",
                "er-bing",
                "jian-jie-shi-chen",
            ],
        ),
    ]

    errors_by_text = {}

    for text_id, name, period, chapters in texts:
        count = 0
        errors = []
        for chapter in chapters:
            try:
                CTEXT_LIMITER.wait()
                urn = f"{text_id}/{chapter}"
                url = f"https://api.ctext.org/gettext?urn={urn}"
                resp = requests.get(url, timeout=30)

                if resp.status_code != 200:
                    errors.append(f"{chapter}: HTTP {resp.status_code}")
                    continue

                data = resp.json()

                if isinstance(data, dict) and "error" in data:
                    errors.append(f"{chapter}: {data['error']}")
                    continue

                if isinstance(data, dict) and "fulltext" in data:
                    for text in data["fulltext"]:
                        if text and len(text) > 10:
                            passages.append(
                                {
                                    "id": f"ctext_{len(passages)}",
                                    "text": text,
                                    "language": "classical_chinese",
                                    "source": f"{name}/{chapter}",
                                    "time_periods": [period],
                                }
                            )
                            count += 1
                elif isinstance(data, list):
                    for item in data:
                        text = item.get("text", "") if isinstance(item, dict) else str(item)
                        if text and len(text) > 10:
                            passages.append(
                                {
                                    "id": f"ctext_{len(passages)}",
                                    "text": text,
                                    "language": "classical_chinese",
                                    "source": f"{name}/{chapter}",
                                    "time_periods": [period],
                                }
                            )
                            count += 1
                else:
                    errors.append(f"{chapter}: unexpected format")

            except requests.exceptions.Timeout:
                errors.append(f"{chapter}: timeout")
            except requests.exceptions.RequestException as e:
                errors.append(f"{chapter}: {type(e).__name__}")
            except json.JSONDecodeError:
                errors.append(f"{chapter}: invalid JSON")
            except Exception as e:
                errors.append(f"{chapter}: {type(e).__name__}")

        # Always print status
        if count > 0:
            print(f"    {name} ({period}): {count} passages")
        else:
            print(f"    {name} ({period}): 0 passages [FAILED]")

        if errors:
            errors_by_text[name] = errors

    # Error summary
    if errors_by_text:
        print("\n  ctext.org API errors:")
        for name, errs in errors_by_text.items():
            print(f"    {name}: {len(errs)} failed")
            for err in errs[:3]:
                print(f"      - {err}")

    if passages:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(passages, f, ensure_ascii=False)

    print(f"  ctext.org: {len(passages):,} passages total")
    return passages
