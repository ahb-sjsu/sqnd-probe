# @title 2. Download/Load Corpora { display-mode: "form" }
# @markdown Downloads from online sources OR loads from Google Drive

import subprocess
import json
import pandas as pd
import shutil
from pathlib import Path

print("=" * 60)
print("LOADING CORPORA")
print("=" * 60)

# Force Google Drive sync refresh (workaround for stale FUSE mount)
if ENV_NAME == "COLAB" and SAVE_DIR and os.path.exists(os.path.dirname(SAVE_DIR)):
    try:
        # Accessing the directory forces FUSE to refresh
        _ = os.listdir(SAVE_DIR)
        # Also touch parent to wake up sync
        _ = os.listdir(os.path.dirname(SAVE_DIR))
        print("  [Drive sync refreshed]")
    except Exception as e:
        print(f"  [Drive sync warning: {e}]")

if LOAD_FROM_DRIVE:
    # ===== LOAD FROM DRIVE =====
    print("\nLoading pre-processed data from Google Drive...")

    # Copy files from Drive to local
    for fname in ["passages.jsonl", "bonds.jsonl"]:
        src = f"{SAVE_DIR}/{fname}"
        dst = f"data/processed/{fname}"
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  Copied {fname}")

    if os.path.exists(f"{SAVE_DIR}/all_splits.json"):
        shutil.copy(f"{SAVE_DIR}/all_splits.json", "data/splits/all_splits.json")
        print(f"  Copied all_splits.json")

    # Load Dear Abby from Drive if available (check filesystem, not cached set)
    abby_drive_path = f"{SAVE_DIR}/dear_abby.csv"
    if os.path.exists(abby_drive_path):
        shutil.copy(abby_drive_path, "data/raw/dear_abby.csv")
        print(f"  Copied dear_abby.csv from {abby_drive_path}")

    # Count loaded data
    if os.path.exists("data/processed/passages.jsonl"):
        with open("data/processed/passages.jsonl") as f:
            n_passages = sum(1 for _ in f)
        print(f"\nLoaded {n_passages:,} passages from Drive")

    SKIP_PROCESSING = True
    print("\n" + "=" * 60)
    print("Drive data loaded - skipping download/processing")
    print("=" * 60)

else:
    # ===== DOWNLOAD/UPDATE FROM ONLINE =====
    SKIP_PROCESSING = False

    # Check if CACHE_ONLY mode but cache is missing
    if CACHE_ONLY:
        print("\n" + "=" * 60)
        print("ERROR: CACHE_ONLY mode but cached data not found!")
        print("=" * 60)
        print("Options:")
        print("  1. Change DATA_MODE to 'Update missing' or 'Refresh all'")
        print("  2. Ensure Drive has: passages.jsonl, bonds.jsonl")
        raise RuntimeError("Cache-only mode requires cached data. Change DATA_MODE.")

    # SEFARIA - with Drive caching
    sefaria_local = "data/raw/Sefaria-Export/json"
    sefaria_drive = f"{SAVE_DIR}/Sefaria-Export-json.tar.gz" if USE_DRIVE_DATA else None

    if os.path.exists(sefaria_local):
        print("\n[1/4] Sefaria already exists locally")
    elif sefaria_drive and os.path.exists(sefaria_drive):
        print("\n[1/4] Restoring Sefaria from Drive cache...")
        import tarfile

        os.makedirs("data/raw/Sefaria-Export", exist_ok=True)
        with tarfile.open(sefaria_drive, "r:gz") as tar:
            tar.extractall("data/raw/Sefaria-Export")
        print("  Restored from Drive!")
    else:
        print("\n[1/4] Downloading Sefaria (~2GB)...")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/Sefaria/Sefaria-Export.git",
                "data/raw/Sefaria-Export",
            ],
            check=True,
        )
        print("  Done!")
        # Cache to Drive for next time
        if USE_DRIVE_DATA and SAVE_DIR:
            print("  Caching Sefaria to Drive (this may take a minute)...")
            import tarfile

            with tarfile.open(sefaria_drive, "w:gz") as tar:
                tar.add("data/raw/Sefaria-Export/json", arcname="json")
            print(f"  Cached to {sefaria_drive}")

    # CHINESE - 200+ REAL CLASSICAL TEXTS
    print("\n[2/4] Chinese classics (200+ real passages)...")
    os.makedirs("data/raw/chinese", exist_ok=True)

    chinese = []

    # === ANALECTS (論語) - 50+ passages ===
    analects = [
        ("子曰：己所不欲，勿施於人。", "Analects 15.24"),
        ("孝悌也者，其為仁之本與。", "Analects 1.2"),
        ("父母在，不遠游，遊必有方。", "Analects 4.19"),
        ("君子喻於義，小人喻於利。", "Analects 4.16"),
        ("不義而富且貴，於我如浮雲。", "Analects 7.16"),
        ("學而時習之，不亦說乎。", "Analects 1.1"),
        ("有朋自遠方來，不亦樂乎。", "Analects 1.1"),
        ("人不知而不慍，不亦君子乎。", "Analects 1.1"),
        ("巧言令色，鮮矣仁。", "Analects 1.3"),
        ("吾日三省吾身。", "Analects 1.4"),
        ("為人謀而不忠乎，與朋友交而不信乎。", "Analects 1.4"),
        ("弟子入則孝，出則悌。", "Analects 1.6"),
        ("謹而信，汎愛眾，而親仁。", "Analects 1.6"),
        ("君子不重則不威，學則不固。", "Analects 1.8"),
        ("主忠信，無友不如己者。", "Analects 1.8"),
        ("過則勿憚改。", "Analects 1.8"),
        ("慎終追遠，民德歸厚矣。", "Analects 1.9"),
        ("禮之用，和為貴。", "Analects 1.12"),
        ("信近於義，言可復也。", "Analects 1.13"),
        ("君子食無求飽，居無求安。", "Analects 1.14"),
        ("敏於事而慎於言，就有道而正焉。", "Analects 1.14"),
        ("不患人之不己知，患不知人也。", "Analects 1.16"),
        ("為政以德，譬如北辰。", "Analects 2.1"),
        ("道之以政，齊之以刑，民免而無恥。", "Analects 2.3"),
        ("道之以德，齊之以禮，有恥且格。", "Analects 2.3"),
        ("吾十有五而志于學。", "Analects 2.4"),
        ("三十而立，四十而不惑。", "Analects 2.4"),
        ("五十而知天命，六十而耳順。", "Analects 2.4"),
        ("七十而從心所欲，不逾矩。", "Analects 2.4"),
        ("生，事之以禮；死，葬之以禮，祭之以禮。", "Analects 2.5"),
        ("父母唯其疾之憂。", "Analects 2.6"),
        ("今之孝者，是謂能養。", "Analects 2.7"),
        ("至於犬馬，皆能有養；不敬，何以別乎。", "Analects 2.7"),
        ("色難。有事，弟子服其勞。", "Analects 2.8"),
        ("視其所以，觀其所由，察其所安。", "Analects 2.10"),
        ("溫故而知新，可以為師矣。", "Analects 2.11"),
        ("君子不器。", "Analects 2.12"),
        ("先行其言而後從之。", "Analects 2.13"),
        ("君子周而不比，小人比而不周。", "Analects 2.14"),
        ("學而不思則罔，思而不學則殆。", "Analects 2.15"),
        ("知之為知之，不知為不知，是知也。", "Analects 2.17"),
        ("多聞闕疑，慎言其餘，則寡尤。", "Analects 2.18"),
        ("舉直錯諸枉，則民服。", "Analects 2.19"),
        ("人而無信，不知其可也。", "Analects 2.22"),
        ("見義不為，無勇也。", "Analects 2.24"),
        ("非其鬼而祭之，諂也。", "Analects 2.24"),
        ("是可忍也，孰不可忍也。", "Analects 3.1"),
        ("人而不仁，如禮何。", "Analects 3.3"),
        ("人而不仁，如樂何。", "Analects 3.3"),
        ("里仁為美。擇不處仁，焉得知。", "Analects 4.1"),
        ("不仁者不可以久處約，不可以長處樂。", "Analects 4.2"),
        ("仁者安仁，知者利仁。", "Analects 4.2"),
        ("唯仁者能好人，能惡人。", "Analects 4.3"),
        ("苟志於仁矣，無惡也。", "Analects 4.4"),
    ]
    for i, (text, source) in enumerate(analects):
        chinese.append(
            {
                "id": f"cn_analects_{i}",
                "text": text,
                "source": source,
                "period": "CONFUCIAN",
                "century": -5,
            }
        )
    print(f"    - Analects: {len([x for x in chinese if 'analects' in x['id']]):,} passages")

    # === MENCIUS (孟子) - 40+ passages ===
    mencius = [
        ("惻隱之心，仁之端也。", "Mencius 2A.6"),
        ("羞惡之心，義之端也。", "Mencius 2A.6"),
        ("辭讓之心，禮之端也。", "Mencius 2A.6"),
        ("是非之心，智之端也。", "Mencius 2A.6"),
        ("人皆有不忍人之心。", "Mencius 2A.6"),
        ("無惻隱之心，非人也。", "Mencius 2A.6"),
        ("無羞惡之心，非人也。", "Mencius 2A.6"),
        ("無辭讓之心，非人也。", "Mencius 2A.6"),
        ("無是非之心，非人也。", "Mencius 2A.6"),
        ("仁義禮智，非由外鑠我也，我固有之也。", "Mencius 6A.6"),
        ("人性之善也，猶水之就下也。", "Mencius 6A.2"),
        ("人無有不善，水無有不下。", "Mencius 6A.2"),
        ("惟仁者宜在高位。", "Mencius 4A.1"),
        ("不仁而在高位，是播其惡於眾也。", "Mencius 4A.1"),
        ("民為貴，社稷次之，君為輕。", "Mencius 7B.14"),
        ("得道者多助，失道者寡助。", "Mencius 2B.1"),
        ("寡助之至，親戚畔之。", "Mencius 2B.1"),
        ("多助之至，天下順之。", "Mencius 2B.1"),
        ("天時不如地利，地利不如人和。", "Mencius 2B.1"),
        ("生於憂患，死於安樂。", "Mencius 6B.15"),
        ("天將降大任於是人也，必先苦其心志。", "Mencius 6B.15"),
        ("勞其筋骨，餓其體膚。", "Mencius 6B.15"),
        ("空乏其身，行拂亂其所為。", "Mencius 6B.15"),
        ("所以動心忍性，曾益其所不能。", "Mencius 6B.15"),
        ("老吾老，以及人之老。", "Mencius 1A.7"),
        ("幼吾幼，以及人之幼。", "Mencius 1A.7"),
        ("窮則獨善其身，達則兼善天下。", "Mencius 7A.9"),
        ("魚，我所欲也；熊掌，亦我所欲也。", "Mencius 6A.10"),
        ("二者不可得兼，舍魚而取熊掌者也。", "Mencius 6A.10"),
        ("生，亦我所欲也；義，亦我所欲也。", "Mencius 6A.10"),
        ("二者不可得兼，舍生而取義者也。", "Mencius 6A.10"),
        ("養心莫善於寡欲。", "Mencius 7B.35"),
        ("仁者無敵於天下。", "Mencius 1A.5"),
        ("以力服人者，非心服也。", "Mencius 2A.3"),
        ("以德服人者，中心悅而誠服也。", "Mencius 2A.3"),
        ("人之患在好為人師。", "Mencius 4A.23"),
        ("盡信書，則不如無書。", "Mencius 7B.3"),
        ("不以規矩，不能成方圓。", "Mencius 4A.1"),
        ("孝子之至，莫大乎尊親。", "Mencius 5A.4"),
        ("父子有親，君臣有義，夫婦有別，長幼有序，朋友有信。", "Mencius 3A.4"),
        ("人有不為也，而後可以有為。", "Mencius 4B.8"),
    ]
    for i, (text, source) in enumerate(mencius):
        chinese.append(
            {
                "id": f"cn_mencius_{i}",
                "text": text,
                "source": source,
                "period": "CONFUCIAN",
                "century": -4,
            }
        )
    print(f"    - Mencius: {len([x for x in chinese if 'mencius' in x['id']]):,} passages")

    # === DAODEJING (道德經) - 40+ passages ===
    daodejing = [
        ("道可道，非常道。名可名，非常名。", "Daodejing 1"),
        ("天下皆知美之為美，斯惡已。", "Daodejing 2"),
        ("皆知善之為善，斯不善已。", "Daodejing 2"),
        ("有無相生，難易相成。", "Daodejing 2"),
        ("長短相較，高下相傾。", "Daodejing 2"),
        ("是以聖人處無為之事，行不言之教。", "Daodejing 2"),
        ("不尚賢，使民不爭。", "Daodejing 3"),
        ("不貴難得之貨，使民不為盜。", "Daodejing 3"),
        ("上善若水。水善利萬物而不爭。", "Daodejing 8"),
        ("處眾人之所惡，故幾於道。", "Daodejing 8"),
        ("居善地，心善淵，與善仁。", "Daodejing 8"),
        ("言善信，政善治，事善能，動善時。", "Daodejing 8"),
        ("夫唯不爭，故無尤。", "Daodejing 8"),
        ("金玉滿堂，莫之能守。", "Daodejing 9"),
        ("富貴而驕，自遺其咎。", "Daodejing 9"),
        ("功成身退，天之道也。", "Daodejing 9"),
        ("知人者智，自知者明。", "Daodejing 33"),
        ("勝人者有力，自勝者強。", "Daodejing 33"),
        ("知足者富，強行者有志。", "Daodejing 33"),
        ("不失其所者久，死而不亡者壽。", "Daodejing 33"),
        ("大道廢，有仁義。", "Daodejing 18"),
        ("智慧出，有大偽。", "Daodejing 18"),
        ("六親不和，有孝慈。", "Daodejing 18"),
        ("國家昏亂，有忠臣。", "Daodejing 18"),
        ("禍兮福之所倚，福兮禍之所伏。", "Daodejing 58"),
        ("天長地久。", "Daodejing 7"),
        ("天地所以能長且久者，以其不自生。", "Daodejing 7"),
        ("是以聖人後其身而身先。", "Daodejing 7"),
        ("外其身而身存。", "Daodejing 7"),
        ("非以其無私耶，故能成其私。", "Daodejing 7"),
        ("柔弱勝剛強。", "Daodejing 36"),
        ("大方無隅，大器晚成。", "Daodejing 41"),
        ("大音希聲，大象無形。", "Daodejing 41"),
        ("道生一，一生二，二生三，三生萬物。", "Daodejing 42"),
        ("天下萬物生於有，有生於無。", "Daodejing 40"),
        ("千里之行，始於足下。", "Daodejing 64"),
        ("合抱之木，生於毫末。", "Daodejing 64"),
        ("九層之臺，起於累土。", "Daodejing 64"),
        ("民不畏死，奈何以死懼之。", "Daodejing 74"),
        ("信言不美，美言不信。", "Daodejing 81"),
        ("善者不辯，辯者不善。", "Daodejing 81"),
        ("知者不博，博者不知。", "Daodejing 81"),
    ]
    for i, (text, source) in enumerate(daodejing):
        chinese.append(
            {
                "id": f"cn_daodejing_{i}",
                "text": text,
                "source": source,
                "period": "DAOIST",
                "century": -4,
            }
        )
    print(f"    - Daodejing: {len([x for x in chinese if 'daodejing' in x['id']]):,} passages")

    # === GREAT LEARNING (大學) - 20+ passages ===
    daxue = [
        ("大學之道，在明明德，在親民，在止於至善。", "Great Learning 1"),
        ("知止而後有定，定而後能靜。", "Great Learning 1"),
        ("靜而後能安，安而後能慮，慮而後能得。", "Great Learning 1"),
        ("物有本末，事有終始。", "Great Learning 1"),
        ("知所先後，則近道矣。", "Great Learning 1"),
        ("古之欲明明德於天下者，先治其國。", "Great Learning 1"),
        ("欲治其國者，先齊其家。", "Great Learning 1"),
        ("欲齊其家者，先修其身。", "Great Learning 1"),
        ("欲修其身者，先正其心。", "Great Learning 1"),
        ("欲正其心者，先誠其意。", "Great Learning 1"),
        ("欲誠其意者，先致其知。", "Great Learning 1"),
        ("致知在格物。", "Great Learning 1"),
        ("物格而後知至，知至而後意誠。", "Great Learning 1"),
        ("意誠而後心正，心正而後身修。", "Great Learning 1"),
        ("身修而後家齊，家齊而後國治。", "Great Learning 1"),
        ("國治而後天下平。", "Great Learning 1"),
        ("自天子以至於庶人，壹是皆以修身為本。", "Great Learning 1"),
        ("其本亂而末治者否矣。", "Great Learning 1"),
        ("所謂誠其意者，毋自欺也。", "Great Learning 6"),
        ("如惡惡臭，如好好色，此之謂自謙。", "Great Learning 6"),
        ("故君子必慎其獨也。", "Great Learning 6"),
        ("富潤屋，德潤身，心廣體胖。", "Great Learning 6"),
    ]
    for i, (text, source) in enumerate(daxue):
        chinese.append(
            {
                "id": f"cn_daxue_{i}",
                "text": text,
                "source": source,
                "period": "CONFUCIAN",
                "century": -5,
            }
        )
    print(f"    - Great Learning: {len([x for x in chinese if 'daxue' in x['id']]):,} passages")

    # === DOCTRINE OF THE MEAN (中庸) - 20+ passages ===
    zhongyong = [
        ("天命之謂性，率性之謂道，修道之謂教。", "Doctrine of the Mean 1"),
        ("道也者，不可須臾離也；可離，非道也。", "Doctrine of the Mean 1"),
        ("是故君子戒慎乎其所不睹，恐懼乎其所不聞。", "Doctrine of the Mean 1"),
        ("莫見乎隱，莫顯乎微，故君子慎其獨也。", "Doctrine of the Mean 1"),
        ("喜怒哀樂之未發，謂之中。", "Doctrine of the Mean 1"),
        ("發而皆中節，謂之和。", "Doctrine of the Mean 1"),
        ("中也者，天下之大本也。", "Doctrine of the Mean 1"),
        ("和也者，天下之達道也。", "Doctrine of the Mean 1"),
        ("致中和，天地位焉，萬物育焉。", "Doctrine of the Mean 1"),
        ("君子中庸，小人反中庸。", "Doctrine of the Mean 2"),
        ("君子之中庸也，君子而時中。", "Doctrine of the Mean 2"),
        ("小人之反中庸也，小人而無忌憚也。", "Doctrine of the Mean 2"),
        ("中庸其至矣乎！民鮮能久矣。", "Doctrine of the Mean 3"),
        ("道之不行也，我知之矣：知者過之，愚者不及也。", "Doctrine of the Mean 4"),
        ("道之不明也，我知之矣：賢者過之，不肖者不及也。", "Doctrine of the Mean 4"),
        ("人莫不飲食也，鮮能知味也。", "Doctrine of the Mean 4"),
        ("誠者，天之道也。誠之者，人之道也。", "Doctrine of the Mean 20"),
        ("誠者，不勉而中，不思而得，從容中道，聖人也。", "Doctrine of the Mean 20"),
        ("誠之者，擇善而固執之者也。", "Doctrine of the Mean 20"),
        ("博學之，審問之，慎思之，明辨之，篤行之。", "Doctrine of the Mean 20"),
        ("人一能之，己百之；人十能之，己千之。", "Doctrine of the Mean 20"),
        ("果能此道矣，雖愚必明，雖柔必強。", "Doctrine of the Mean 20"),
    ]
    for i, (text, source) in enumerate(zhongyong):
        chinese.append(
            {
                "id": f"cn_zhongyong_{i}",
                "text": text,
                "source": source,
                "period": "CONFUCIAN",
                "century": -5,
            }
        )
    print(
        f"    - Doctrine of Mean: {len([x for x in chinese if 'zhongyong' in x['id']]):,} passages"
    )

    # === BOOK OF RITES (禮記) - 30+ passages ===
    liji = [
        ("禮尚往來。往而不來，非禮也；來而不往，亦非禮也。", "Book of Rites - Quli"),
        ("敖不可長，欲不可從，志不可滿，樂不可極。", "Book of Rites - Quli"),
        ("臨財毋茍得，臨難毋茍免。", "Book of Rites - Quli"),
        ("夫禮者，自卑而尊人。", "Book of Rites - Quli"),
        ("雖負販者，必有尊也，而況富貴乎。", "Book of Rites - Quli"),
        ("富貴而知好禮，則不驕不淫。", "Book of Rites - Quli"),
        ("貧賤而知好禮，則志不懾。", "Book of Rites - Quli"),
        ("大道之行也，天下為公。", "Book of Rites - Liyun"),
        ("選賢與能，講信修睦。", "Book of Rites - Liyun"),
        ("故人不獨親其親，不獨子其子。", "Book of Rites - Liyun"),
        ("使老有所終，壯有所用，幼有所長。", "Book of Rites - Liyun"),
        ("矜寡孤獨廢疾者皆有所養。", "Book of Rites - Liyun"),
        ("男有分，女有歸。", "Book of Rites - Liyun"),
        ("貨惡其棄於地也，不必藏於己。", "Book of Rites - Liyun"),
        ("力惡其不出於身也，不必為己。", "Book of Rites - Liyun"),
        ("是故謀閉而不興，盜竊亂賊而不作。", "Book of Rites - Liyun"),
        ("故外戶而不閉，是謂大同。", "Book of Rites - Liyun"),
        ("玉不琢，不成器；人不學，不知道。", "Book of Rites - Xueji"),
        ("是故學然後知不足，教然後知困。", "Book of Rites - Xueji"),
        ("知不足，然後能自反也。", "Book of Rites - Xueji"),
        ("知困，然後能自強也。", "Book of Rites - Xueji"),
        ("故曰：教學相長也。", "Book of Rites - Xueji"),
        ("凡學之道，嚴師為難。", "Book of Rites - Xueji"),
        ("師嚴然後道尊，道尊然後民知敬學。", "Book of Rites - Xueji"),
        ("善歌者使人繼其聲，善教者使人繼其志。", "Book of Rites - Xueji"),
        ("記問之學，不足以為人師。", "Book of Rites - Xueji"),
        ("必也其聽語乎，力不能問，然後語之。", "Book of Rites - Xueji"),
        ("語之而不知，雖舍之可也。", "Book of Rites - Xueji"),
        ("博學而不窮，篤行而不倦。", "Book of Rites - Ruxing"),
        ("君子之於學也，藏焉，修焉，息焉，游焉。", "Book of Rites - Xueji"),
    ]
    for i, (text, source) in enumerate(liji):
        chinese.append(
            {
                "id": f"cn_liji_{i}",
                "text": text,
                "source": source,
                "period": "CONFUCIAN",
                "century": -3,
            }
        )
    print(f"    - Book of Rites: {len([x for x in chinese if 'liji' in x['id']]):,} passages")

    with open("data/raw/chinese/chinese_native.json", "w", encoding="utf-8") as f:
        json.dump(chinese, f, ensure_ascii=False, indent=2)
    print(f"  Created {len(chinese)} Chinese passages")

    # ISLAMIC - 150+ REAL PASSAGES
    print("\n[3/4] Islamic texts (150+ real passages)...")
    os.makedirs("data/raw/islamic", exist_ok=True)

    islamic = []

    # === QURANIC VERSES (40+) ===
    quran = [
        ("وَلَا تَقْتُلُوا النَّفْسَ الَّتِي حَرَّمَ اللَّهُ إِلَّا بِالْحَقِّ", "Quran 6:151"),
        ("وَبِالْوَالِدَيْنِ إِحْسَانًا", "Quran 17:23"),
        ("إِمَّا يَبْلُغَنَّ عِندَكَ الْكِبَرَ أَحَدُهُمَا أَوْ كِلَاهُمَا فَلَا تَقُل لَّهُمَا أُفٍّ", "Quran 17:23"),
        ("وَلَا تَنْهَرْهُمَا وَقُل لَّهُمَا قَوْلًا كَرِيمًا", "Quran 17:23"),
        ("وَاخْفِضْ لَهُمَا جَنَاحَ الذُّلِّ مِنَ الرَّحْمَةِ", "Quran 17:24"),
        ("وَقُل رَّبِّ ارْحَمْهُمَا كَمَا رَبَّيَانِي صَغِيرًا", "Quran 17:24"),
        ("وَآتِ ذَا الْقُرْبَىٰ حَقَّهُ وَالْمِسْكِينَ وَابْنَ السَّبِيلِ", "Quran 17:26"),
        ("وَلَا تُبَذِّرْ تَبْذِيرًا", "Quran 17:26"),
        ("إِنَّ الْمُبَذِّرِينَ كَانُوا إِخْوَانَ الشَّيَاطِينِ", "Quran 17:27"),
        ("وَلَا تَجْعَلْ يَدَكَ مَغْلُولَةً إِلَىٰ عُنُقِكَ وَلَا تَبْسُطْهَا كُلَّ الْبَسْطِ", "Quran 17:29"),
        ("وَلَا تَقْرَبُوا الزِّنَا ۖ إِنَّهُ كَانَ فَاحِشَةً وَسَاءَ سَبِيلًا", "Quran 17:32"),
        ("وَلَا تَقْتُلُوا أَوْلَادَكُمْ خَشْيَةَ إِمْلَاقٍ", "Quran 17:31"),
        ("وَلَا تَقْرَبُوا مَالَ الْيَتِيمِ إِلَّا بِالَّتِي هِيَ أَحْسَنُ", "Quran 17:34"),
        ("وَأَوْفُوا بِالْعَهْدِ ۖ إِنَّ الْعَهْدَ كَانَ مَسْئُولًا", "Quran 17:34"),
        ("وَأَوْفُوا الْكَيْلَ إِذَا كِلْتُمْ وَزِنُوا بِالْقِسْطَاسِ الْمُسْتَقِيمِ", "Quran 17:35"),
        ("وَلَا تَقْفُ مَا لَيْسَ لَكَ بِهِ عِلْمٌ", "Quran 17:36"),
        ("إِنَّ السَّمْعَ وَالْبَصَرَ وَالْفُؤَادَ كُلُّ أُولَٰئِكَ كَانَ عَنْهُ مَسْئُولًا", "Quran 17:36"),
        ("وَلَا تَمْشِ فِي الْأَرْضِ مَرَحًا", "Quran 17:37"),
        ("إِنَّ اللَّهَ يَأْمُرُ بِالْعَدْلِ وَالْإِحْسَانِ وَإِيتَاءِ ذِي الْقُرْبَىٰ", "Quran 16:90"),
        ("وَيَنْهَىٰ عَنِ الْفَحْشَاءِ وَالْمُنكَرِ وَالْبَغْيِ", "Quran 16:90"),
        ("يَا أَيُّهَا الَّذِينَ آمَنُوا كُونُوا قَوَّامِينَ بِالْقِسْطِ", "Quran 4:135"),
        ("شُهَدَاءَ لِلَّهِ وَلَوْ عَلَىٰ أَنفُسِكُمْ أَوِ الْوَالِدَيْنِ وَالْأَقْرَبِينَ", "Quran 4:135"),
        ("وَإِذَا حَكَمْتُم بَيْنَ النَّاسِ أَن تَحْكُمُوا بِالْعَدْلِ", "Quran 4:58"),
        ("يَا أَيُّهَا الَّذِينَ آمَنُوا أَوْفُوا بِالْعُقُودِ", "Quran 5:1"),
        ("وَتَعَاوَنُوا عَلَى الْبِرِّ وَالتَّقْوَىٰ ۖ وَلَا تَعَاوَنُوا عَلَى الْإِثْمِ وَالْعُدْوَانِ", "Quran 5:2"),
        ("مَن قَتَلَ نَفْسًا بِغَيْرِ نَفْسٍ أَوْ فَسَادٍ فِي الْأَرْضِ فَكَأَنَّمَا قَتَلَ النَّاسَ جَمِيعًا", "Quran 5:32"),
        ("وَمَنْ أَحْيَاهَا فَكَأَنَّمَا أَحْيَا النَّاسَ جَمِيعًا", "Quran 5:32"),
        ("وَلَا يَجْرِمَنَّكُمْ شَنَآنُ قَوْمٍ عَلَىٰ أَلَّا تَعْدِلُوا", "Quran 5:8"),
        ("اعْدِلُوا هُوَ أَقْرَبُ لِلتَّقْوَىٰ", "Quran 5:8"),
        ("لَّيْسَ الْبِرَّ أَن تُوَلُّوا وُجُوهَكُمْ قِبَلَ الْمَشْرِقِ وَالْمَغْرِبِ", "Quran 2:177"),
        ("وَلَٰكِنَّ الْبِرَّ مَنْ آمَنَ بِاللَّهِ وَالْيَوْمِ الْآخِرِ", "Quran 2:177"),
        ("وَآتَى الْمَالَ عَلَىٰ حُبِّهِ ذَوِي الْقُرْبَىٰ وَالْيَتَامَىٰ وَالْمَسَاكِينَ", "Quran 2:177"),
        ("وَابْنَ السَّبِيلِ وَالسَّائِلِينَ وَفِي الرِّقَابِ", "Quran 2:177"),
        ("وَأَقَامَ الصَّلَاةَ وَآتَى الزَّكَاةَ", "Quran 2:177"),
        ("وَالْمُوفُونَ بِعَهْدِهِمْ إِذَا عَاهَدُوا", "Quran 2:177"),
        ("وَالصَّابِرِينَ فِي الْبَأْسَاءِ وَالضَّرَّاءِ وَحِينَ الْبَأْسِ", "Quran 2:177"),
        ("خُذِ الْعَفْوَ وَأْمُرْ بِالْعُرْفِ وَأَعْرِضْ عَنِ الْجَاهِلِينَ", "Quran 7:199"),
        ("وَالْكَاظِمِينَ الْغَيْظَ وَالْعَافِينَ عَنِ النَّاسِ", "Quran 3:134"),
        ("وَاللَّهُ يُحِبُّ الْمُحْسِنِينَ", "Quran 3:134"),
        ("ادْفَعْ بِالَّتِي هِيَ أَحْسَنُ فَإِذَا الَّذِي بَيْنَكَ وَبَيْنَهُ عَدَاوَةٌ كَأَنَّهُ وَلِيٌّ حَمِيمٌ", "Quran 41:34"),
        ("وَمَا يُلَقَّاهَا إِلَّا الَّذِينَ صَبَرُوا وَمَا يُلَقَّاهَا إِلَّا ذُو حَظٍّ عَظِيمٍ", "Quran 41:35"),
        ("إِنَّ اللَّهَ يَأْمُرُكُمْ أَن تُؤَدُّوا الْأَمَانَاتِ إِلَىٰ أَهْلِهَا", "Quran 4:58"),
    ]
    for i, (text, source) in enumerate(quran):
        islamic.append(
            {"id": f"quran_{i}", "text": text, "source": source, "period": "QURANIC", "century": 7}
        )
    print(f"    - Quranic verses: {len([x for x in islamic if 'quran' in x['id']]):,} passages")

    # === HADITH (110+) ===
    hadith = [
        ("لا ضرر ولا ضرار", "Hadith - Ibn Majah"),
        ("إنما الأعمال بالنيات وإنما لكل امرئ ما نوى", "Hadith - Bukhari 1"),
        ("المسلم من سلم المسلمون من لسانه ويده", "Hadith - Bukhari 10"),
        ("لا يؤمن أحدكم حتى يحب لأخيه ما يحب لنفسه", "Hadith - Bukhari 13"),
        ("من كان يؤمن بالله واليوم الآخر فليقل خيرا أو ليصمت", "Hadith - Bukhari 6018"),
        ("من كان يؤمن بالله واليوم الآخر فليكرم ضيفه", "Hadith - Bukhari 6019"),
        ("من كان يؤمن بالله واليوم الآخر فليصل رحمه", "Hadith - Bukhari 6138"),
        ("ارحموا من في الأرض يرحمكم من في السماء", "Hadith - Tirmidhi 1924"),
        ("الراحمون يرحمهم الرحمن", "Hadith - Abu Dawud 4941"),
        ("ليس منا من لم يرحم صغيرنا ويوقر كبيرنا", "Hadith - Tirmidhi 1919"),
        ("خيركم خيركم لأهله وأنا خيركم لأهلي", "Hadith - Tirmidhi 3895"),
        ("اتق الله حيثما كنت وأتبع السيئة الحسنة تمحها", "Hadith - Tirmidhi 1987"),
        ("وخالق الناس بخلق حسن", "Hadith - Tirmidhi 1987"),
        ("أكمل المؤمنين إيمانا أحسنهم خلقا", "Hadith - Abu Dawud 4682"),
        ("إن من أحبكم إلي وأقربكم مني مجلسا يوم القيامة أحاسنكم أخلاقا", "Hadith - Tirmidhi 2018"),
        ("ما من شيء أثقل في ميزان المؤمن يوم القيامة من حسن الخلق", "Hadith - Tirmidhi 2002"),
        ("البر حسن الخلق والإثم ما حاك في صدرك وكرهت أن يطلع عليه الناس", "Hadith - Muslim 2553"),
        ("الحياء من الإيمان", "Hadith - Bukhari 24"),
        ("الحياء لا يأتي إلا بخير", "Hadith - Bukhari 6117"),
        ("إن الله رفيق يحب الرفق في الأمر كله", "Hadith - Bukhari 6927"),
        ("ما كان الرفق في شيء إلا زانه وما نزع من شيء إلا شانه", "Hadith - Muslim 2594"),
        ("من يحرم الرفق يحرم الخير كله", "Hadith - Muslim 2592"),
        ("أد الأمانة إلى من ائتمنك ولا تخن من خانك", "Hadith - Abu Dawud 3535"),
        ("آية المنافق ثلاث إذا حدث كذب وإذا وعد أخلف وإذا اؤتمن خان", "Hadith - Bukhari 33"),
        ("الصدق يهدي إلى البر والبر يهدي إلى الجنة", "Hadith - Bukhari 6094"),
        ("وإن الكذب يهدي إلى الفجور والفجور يهدي إلى النار", "Hadith - Bukhari 6094"),
        ("عليكم بالصدق فإن الصدق يهدي إلى البر", "Hadith - Muslim 2607"),
        ("إياكم والكذب فإن الكذب يهدي إلى الفجور", "Hadith - Muslim 2607"),
        ("من غشنا فليس منا", "Hadith - Muslim 101"),
        ("كلكم راع وكلكم مسؤول عن رعيته", "Hadith - Bukhari 893"),
        ("الإمام راع ومسؤول عن رعيته", "Hadith - Bukhari 893"),
        ("والرجل راع في أهله ومسؤول عن رعيته", "Hadith - Bukhari 893"),
        ("والمرأة راعية في بيت زوجها ومسؤولة عن رعيتها", "Hadith - Bukhari 893"),
        ("انصر أخاك ظالما أو مظلوما", "Hadith - Bukhari 2444"),
        (
            "تنصره إذا كان مظلوما أفرأيت إذا كان ظالما كيف تنصره قال تحجزه أو تمنعه من الظلم فإن ذلك نصره",
            "Hadith - Bukhari 2444",
        ),
        ("المؤمن للمؤمن كالبنيان يشد بعضه بعضا", "Hadith - Bukhari 481"),
        ("مثل المؤمنين في توادهم وتراحمهم وتعاطفهم مثل الجسد الواحد", "Hadith - Muslim 2586"),
        ("إذا اشتكى منه عضو تداعى له سائر الجسد بالسهر والحمى", "Hadith - Muslim 2586"),
        ("المسلم أخو المسلم لا يظلمه ولا يسلمه", "Hadith - Bukhari 2442"),
        ("من كان في حاجة أخيه كان الله في حاجته", "Hadith - Bukhari 2442"),
        ("ومن فرج عن مسلم كربة فرج الله عنه كربة من كربات يوم القيامة", "Hadith - Bukhari 2442"),
        ("ومن ستر مسلما ستره الله يوم القيامة", "Hadith - Bukhari 2442"),
        ("لا تحاسدوا ولا تناجشوا ولا تباغضوا ولا تدابروا", "Hadith - Muslim 2564"),
        ("ولا يبع بعضكم على بيع بعض وكونوا عباد الله إخوانا", "Hadith - Muslim 2564"),
        ("بحسب امرئ من الشر أن يحقر أخاه المسلم", "Hadith - Muslim 2564"),
        ("كل المسلم على المسلم حرام دمه وماله وعرضه", "Hadith - Muslim 2564"),
        ("إياكم والظن فإن الظن أكذب الحديث", "Hadith - Bukhari 6064"),
        ("ولا تجسسوا ولا تحسسوا ولا تنافسوا", "Hadith - Bukhari 6064"),
        ("الظلم ظلمات يوم القيامة", "Hadith - Bukhari 2447"),
        ("اتقوا الظلم فإن الظلم ظلمات يوم القيامة", "Hadith - Muslim 2578"),
        ("واتقوا الشح فإن الشح أهلك من كان قبلكم", "Hadith - Muslim 2578"),
        ("أفضل الجهاد كلمة عدل عند سلطان جائر", "Hadith - Abu Dawud 4344"),
        (
            "سيد الشهداء حمزة بن عبد المطلب ورجل قام إلى إمام جائر فأمره ونهاه فقتله",
            "Hadith - Hakim 4884",
        ),
        ("إذا رأيت أمتي تهاب أن تقول للظالم يا ظالم فقد تودع منهم", "Hadith - Ahmad 6521"),
        ("من رأى منكم منكرا فليغيره بيده", "Hadith - Muslim 49"),
        ("فإن لم يستطع فبلسانه فإن لم يستطع فبقلبه وذلك أضعف الإيمان", "Hadith - Muslim 49"),
        ("أحب الناس إلى الله أنفعهم للناس", "Hadith - Tabarani 6026"),
        ("وأحب الأعمال إلى الله سرور تدخله على مسلم", "Hadith - Tabarani 6026"),
        ("أو تكشف عنه كربة أو تقضي عنه دينا أو تطرد عنه جوعا", "Hadith - Tabarani 6026"),
        (
            "ولأن أمشي مع أخي في حاجة أحب إلي من أن أعتكف في هذا المسجد شهرا",
            "Hadith - Tabarani 6026",
        ),
        (
            "الدين النصيحة قلنا لمن قال لله ولكتابه ولرسوله ولأئمة المسلمين وعامتهم",
            "Hadith - Muslim 55",
        ),
        ("ما نقصت صدقة من مال", "Hadith - Muslim 2588"),
        ("وما زاد الله عبدا بعفو إلا عزا", "Hadith - Muslim 2588"),
        ("وما تواضع أحد لله إلا رفعه الله", "Hadith - Muslim 2588"),
        ("اليد العليا خير من اليد السفلى", "Hadith - Bukhari 1427"),
        ("وابدأ بمن تعول", "Hadith - Bukhari 1427"),
        ("وخير الصدقة ما كان عن ظهر غنى", "Hadith - Bukhari 1427"),
        ("من استطاع منكم الباءة فليتزوج", "Hadith - Bukhari 5066"),
        ("فإنه أغض للبصر وأحصن للفرج", "Hadith - Bukhari 5066"),
        ("ومن لم يستطع فعليه بالصوم فإنه له وجاء", "Hadith - Bukhari 5066"),
        ("استوصوا بالنساء خيرا", "Hadith - Bukhari 3331"),
        (
            "خذوا عني خذوا عني قد جعل الله لهن سبيلا البكر بالبكر جلد مائة ونفي سنة",
            "Hadith - Muslim 1690",
        ),
        ("لا يفرك مؤمن مؤمنة إن كره منها خلقا رضي منها آخر", "Hadith - Muslim 1469"),
        ("أكمل المؤمنين إيمانا أحسنهم خلقا وخياركم خياركم لنسائهم", "Hadith - Tirmidhi 1162"),
        ("ما أكرمهن إلا كريم وما أهانهن إلا لئيم", "Hadith - Ibn Asakir"),
        ("اللهم إني أحرج حق الضعيفين اليتيم والمرأة", "Hadith - Ahmad 9664"),
        ("ألا أخبركم بخياركم قالوا بلى قال خياركم أحاسنكم أخلاقا", "Hadith - Bukhari 6035"),
        ("إنكم لن تسعوا الناس بأموالكم فليسعهم منكم بسط الوجه وحسن الخلق", "Hadith - Hakim 422"),
        ("تبسمك في وجه أخيك صدقة", "Hadith - Tirmidhi 1956"),
        ("وأمرك بالمعروف ونهيك عن المنكر صدقة", "Hadith - Tirmidhi 1956"),
        ("وإرشادك الرجل في أرض الضلال لك صدقة", "Hadith - Tirmidhi 1956"),
        ("وإماطتك الأذى والشوك والعظم عن الطريق لك صدقة", "Hadith - Tirmidhi 1956"),
        ("وإفراغك من دلوك في دلو أخيك لك صدقة", "Hadith - Tirmidhi 1956"),
        ("الكلمة الطيبة صدقة", "Hadith - Bukhari 2989"),
        ("وكل خطوة تمشيها إلى الصلاة صدقة", "Hadith - Bukhari 2989"),
        ("من دل على خير فله مثل أجر فاعله", "Hadith - Muslim 1893"),
        ("ليس الشديد بالصرعة إنما الشديد الذي يملك نفسه عند الغضب", "Hadith - Bukhari 6114"),
        ("لا تغضب فردد مرارا قال لا تغضب", "Hadith - Bukhari 6116"),
        ("إن الغضب من الشيطان وإن الشيطان خلق من النار", "Hadith - Abu Dawud 4784"),
        ("وإنما تطفأ النار بالماء فإذا غضب أحدكم فليتوضأ", "Hadith - Abu Dawud 4784"),
        ("لا يحل لمسلم أن يهجر أخاه فوق ثلاث ليال", "Hadith - Bukhari 6077"),
        ("يلتقيان فيعرض هذا ويعرض هذا وخيرهما الذي يبدأ بالسلام", "Hadith - Bukhari 6077"),
        ("أفشوا السلام بينكم", "Hadith - Muslim 54"),
        ("والذي نفسي بيده لا تدخلوا الجنة حتى تؤمنوا", "Hadith - Muslim 54"),
        (
            "ولا تؤمنوا حتى تحابوا أولا أدلكم على شيء إذا فعلتموه تحاببتم أفشوا السلام بينكم",
            "Hadith - Muslim 54",
        ),
        ("طعام الاثنين كافي الثلاثة وطعام الثلاثة كافي الأربعة", "Hadith - Bukhari 5392"),
        ("ما ملأ آدمي وعاء شرا من بطن", "Hadith - Tirmidhi 2380"),
        ("بحسب ابن آدم أكلات يقمن صلبه", "Hadith - Tirmidhi 2380"),
        ("فإن كان لا محالة فثلث لطعامه وثلث لشرابه وثلث لنفسه", "Hadith - Tirmidhi 2380"),
        ("إن الله كتب الإحسان على كل شيء", "Hadith - Muslim 1955"),
        ("فإذا قتلتم فأحسنوا القتلة وإذا ذبحتم فأحسنوا الذبح", "Hadith - Muslim 1955"),
        ("وليحد أحدكم شفرته وليرح ذبيحته", "Hadith - Muslim 1955"),
        ("عذبت امرأة في هرة سجنتها حتى ماتت", "Hadith - Bukhari 3318"),
        (
            "فلا هي أطعمتها ولا سقتها إذ حبستها ولا هي تركتها تأكل من خشاش الأرض",
            "Hadith - Bukhari 3318",
        ),
        ("بينما رجل يمشي بطريق اشتد عليه العطش فوجد بئرا فنزل فيها فشرب", "Hadith - Bukhari 2466"),
        ("ثم خرج فإذا كلب يلهث يأكل الثرى من العطش", "Hadith - Bukhari 2466"),
        ("فقال لقد بلغ هذا الكلب من العطش مثل الذي كان بلغ مني", "Hadith - Bukhari 2466"),
        ("فنزل البئر فملأ خفه ماء ثم أمسكه بفيه حتى رقي فسقى الكلب", "Hadith - Bukhari 2466"),
        ("فشكر الله له فغفر له", "Hadith - Bukhari 2466"),
        ("في كل كبد رطبة أجر", "Hadith - Bukhari 2466"),
    ]
    for i, (text, source) in enumerate(hadith):
        islamic.append(
            {"id": f"hadith_{i}", "text": text, "source": source, "period": "HADITH", "century": 9}
        )
    print(f"    - Hadith: {len([x for x in islamic if 'hadith' in x['id']]):,} passages")

    with open("data/raw/islamic/islamic_native.json", "w", encoding="utf-8") as f:
        json.dump(islamic, f, ensure_ascii=False, indent=2)
    print(f"  Created {len(islamic)} Islamic passages")

    # DEAR ABBY
    print("\n[4/4] Dear Abby...")
    abby_count = 0
    if (
        not os.path.exists("data/raw/dear_abby.csv")
        or os.path.getsize("data/raw/dear_abby.csv") < 10000
    ):
        # Check if in Drive (with retry for stale FUSE mount)
        drive_abby_path = f"{SAVE_DIR}/dear_abby.csv"
        found_in_drive = False

        # First attempt
        if os.path.exists(drive_abby_path):
            found_in_drive = True
        else:
            # Retry after refreshing Drive mount (FUSE can be stale)
            print(f"  First check failed, refreshing Drive...")
            try:
                _ = os.listdir(SAVE_DIR)  # Force FUSE refresh
                import time

                time.sleep(0.5)  # Brief pause for sync
                if os.path.exists(drive_abby_path):
                    found_in_drive = True
                    print(f"  Found after refresh!")
            except Exception as e:
                print(f"  Drive refresh error: {e}")

        if found_in_drive:
            shutil.copy(drive_abby_path, "data/raw/dear_abby.csv")
            print(f"  Loaded from Drive: {drive_abby_path}")
        else:
            print(f"  Not found in Drive at: {drive_abby_path}")
            # Show what IS in the Drive folder
            try:
                contents = os.listdir(SAVE_DIR) if os.path.exists(SAVE_DIR) else []
                print(f"  Drive folder contents: {contents[:10]}")
            except:
                pass
            try:
                subprocess.run(
                    [
                        "kaggle",
                        "datasets",
                        "download",
                        "-d",
                        "thedevastator/20000-dear-abby-questions",
                        "-p",
                        "data/raw/",
                        "--unzip",
                    ],
                    check=True,
                    timeout=120,
                )
                print("  Downloaded from Kaggle")
            except:
                print("  Kaggle failed - creating minimal fallback")
                fallback = [
                    {"question_only": f"Dear Abby, I have a problem {i}", "year": 1990 + i % 30}
                    for i in range(100)
                ]
                pd.DataFrame(fallback).to_csv("data/raw/dear_abby.csv", index=False)
    else:
        print("  Already exists")

    # Count Dear Abby samples
    try:
        df = pd.read_csv("data/raw/dear_abby.csv")
        abby_count = len(
            [
                1
                for _, row in df.iterrows()
                if str(row.get("question_only", "")) != "nan"
                and 50 <= len(str(row.get("question_only", ""))) <= 2000
            ]
        )
    except:
        abby_count = 0

    # Warning for insufficient Dear Abby data
    if abby_count < 1000:
        print("\n" + "!" * 60)
        print("CRITICAL: Dear Abby corpus is too small!")
        print("The semitic_to_non_semitic split WILL FAIL without this data.")
        print("\nTo fix:")
        print("1. Download from: kaggle.com/datasets/thedevastator/20000-dear-abby-questions")
        print("2. Upload dear_abby.csv to your Google Drive BIP_v10 folder")
        print("3. Set REFRESH_DATA_FROM_SOURCE = True and rerun")
        print("!" * 60 + "\n")

    print("\n" + "=" * 60)
    print("Downloads complete")
    print("=" * 60)
