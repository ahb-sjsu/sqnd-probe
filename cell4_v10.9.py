# @title 4. Parallel Download + Stream Processing { display-mode: "form" }
# @markdown BIP v10.9: Loads ALL corpora including expanded Chinese (Buddhist, Legalist, Mohist, Neo-Confucian),
# @markdown Arabic (Fiqh, Sufi, Falsafa), and NEW Sanskrit/Pali texts

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

# Minimum thresholds for balanced experiments
MIN_CORPUS_SIZE = {
    "english": 20000,  # Lowered - HF augmentation datasets deprecated
    "classical_chinese": 20000,  # Lowered
    "hebrew": 5000,
    "aramaic": 2000,
    "arabic": 2000,
    "sanskrit": 100,  # NEW in v10.9 - smaller initial corpus
    "pali": 50,  # NEW in v10.9 - smaller initial corpus
}

# Available augmentation datasets by language
AUGMENTATION_DATASETS = {
    "english": [
        ("hendrycks/ethics", "ETHICS"),  # ~130K moral scenarios
        ("allenai/social_chem_101", "SocialChem"),  # ~292K social norms
    ],
    "classical_chinese": [
        ("wikisource_zh_classical", "WikisourceZH"),  # If available
    ],
}

# ===== v10.9 NEW CORPORA =====
# Buddhist Chinese (佛教漢文)
BUDDHIST_CHINESE = [
    ("諸惡莫作，眾善奉行，自淨其意，是諸佛教。", "Dhammapada 183", "BUDDHIST"),
    ("以恨止恨，恨終不滅；唯以忍止恨，此古聖常法。", "Dhammapada 5", "BUDDHIST"),
    ("勝者生怨，負者自鄙；去勝負心，無諍自安。", "Dhammapada 201", "BUDDHIST"),
    ("不以財物施，唯以法布施，法施勝財施。", "Dhammapada 354", "BUDDHIST"),
    ("若以色見我，以音聲求我，是人行邪道，不能見如來。", "Diamond Sutra 26", "BUDDHIST"),
    ("應無所住而生其心。", "Diamond Sutra 10", "BUDDHIST"),
    ("一切有為法，如夢幻泡影，如露亦如電，應作如是觀。", "Diamond Sutra 32", "BUDDHIST"),
    ("凡所有相，皆是虛妄。若見諸相非相，即見如來。", "Diamond Sutra 5", "BUDDHIST"),
    ("諸佛世尊唯以一大事因緣故，出現於世。", "Lotus Sutra 2", "BUDDHIST"),
    ("十方佛土中，唯有一乘法，無二亦無三。", "Lotus Sutra 2", "BUDDHIST"),
    ("色不異空，空不異色，色即是空，空即是色。", "Heart Sutra", "BUDDHIST"),
    ("無苦集滅道，無智亦無得，以無所得故。", "Heart Sutra", "BUDDHIST"),
    ("殺生之罪，能令眾生墮三惡道。", "Sutra of Golden Light 4", "BUDDHIST"),
    ("一切眾生皆有佛性，悉能成佛。", "Nirvana Sutra", "BUDDHIST"),
    ("慈悲喜捨，名為四無量心。", "Brahma Net Sutra", "BUDDHIST"),
    ("不殺生，是菩薩波羅夷罪。", "Brahma Net Sutra 1", "BUDDHIST"),
    ("不偷盜，是菩薩波羅夷罪。", "Brahma Net Sutra 2", "BUDDHIST"),
    ("不邪淫，是菩薩波羅夷罪。", "Brahma Net Sutra 3", "BUDDHIST"),
    ("不妄語，是菩薩波羅夷罪。", "Brahma Net Sutra 4", "BUDDHIST"),
    ("不飲酒，是菩薩波羅夷罪。", "Brahma Net Sutra 5", "BUDDHIST"),
    ("若佛子，以慈心故，行放生業。", "Brahma Net Sutra 20", "BUDDHIST"),
    ("一切男子是我父，一切女人是我母。", "Brahma Net Sutra 9", "BUDDHIST"),
    ("菩薩病者，以大悲起。", "Vimalakirti Sutra 5", "BUDDHIST"),
    ("眾生病，是故我病。", "Vimalakirti Sutra 5", "BUDDHIST"),
    ("菩提本無樹，明鏡亦非臺，本來無一物，何處惹塵埃。", "Platform Sutra", "BUDDHIST"),
    ("何期自性，本自清淨；何期自性，本不生滅。", "Platform Sutra", "BUDDHIST"),
]

# Legalist Chinese (法家)
LEGALIST_CHINESE = [
    ("法不阿貴，繩不撓曲。", "Han Feizi 6", "LEGALIST"),
    ("刑過不避大臣，賞善不遺匹夫。", "Han Feizi 50", "LEGALIST"),
    ("以法為教，以吏為師。", "Han Feizi 49", "LEGALIST"),
    ("明主之國，無書簡之文，以法為教。", "Han Feizi 49", "LEGALIST"),
    ("法者，編著之圖籍，設之於官府，而布之於百姓者也。", "Han Feizi 38", "LEGALIST"),
    ("術者，藏之於胸中，以偶眾端，而潛御群臣者也。", "Han Feizi 38", "LEGALIST"),
    ("法莫如顯，而術不欲見。", "Han Feizi 38", "LEGALIST"),
    ("賞罰不信，則禁令不行。", "Han Feizi 46", "LEGALIST"),
    ("刑重則不敢以惡犯，罰輕則民不畏。", "Han Feizi 46", "LEGALIST"),
    ("夫嚴刑重罰者，民之所惡也；而國之所以治也。", "Han Feizi 49", "LEGALIST"),
    ("法之所加，智者弗能辭，勇者弗敢爭。", "Han Feizi 6", "LEGALIST"),
    ("一民之軌，莫如法。", "Han Feizi 6", "LEGALIST"),
    ("故明主使法擇人，不自舉也。", "Han Feizi 6", "LEGALIST"),
    ("使法量功，不自度也。", "Han Feizi 6", "LEGALIST"),
    ("國之所以興者，農戰也。", "Shang Jun Shu 3", "LEGALIST"),
    ("民弱國強，民強國弱。故有道之國，務在弱民。", "Shang Jun Shu 20", "LEGALIST"),
    ("聖人之為國也，壹賞，壹刑，壹教。", "Shang Jun Shu 17", "LEGALIST"),
    ("治國者，貴分明而不可相舉。", "Shang Jun Shu 14", "LEGALIST"),
    ("行罰重其輕者，輕者不至，重者不來。", "Shang Jun Shu 17", "LEGALIST"),
    ("倉廩實則知禮節，衣食足則知榮辱。", "Guanzi 1", "LEGALIST"),
    ("禮義廉恥，國之四維；四維不張，國乃滅亡。", "Guanzi 1", "LEGALIST"),
    ("政之所興，在順民心；政之所廢，在逆民心。", "Guanzi 1", "LEGALIST"),
]

# Mohist Chinese (墨家)
MOHIST_CHINESE = [
    ("兼相愛，交相利。", "Mozi 15", "MOHIST"),
    ("天下之人皆相愛，強不執弱，眾不劫寡，富不侮貧，貴不傲賤。", "Mozi 15", "MOHIST"),
    ("殺一人謂之不義，必有一死罪矣。", "Mozi 17", "MOHIST"),
    ("今至大為攻國，則弗知非，從而譽之，謂之義。", "Mozi 17", "MOHIST"),
    ("天下之利，是為天下之義。", "Mozi 26", "MOHIST"),
    ("非攻，墨子之道也。", "Mozi 17", "MOHIST"),
    ("節用，墨子之教也。", "Mozi 20", "MOHIST"),
    ("聖人以治天下為事者也，必知亂之所自起，焉能治之。", "Mozi 14", "MOHIST"),
    ("天下之所以亂者，生於不相愛。", "Mozi 14", "MOHIST"),
    ("臣子之不孝君父，所謂亂也。", "Mozi 14", "MOHIST"),
    ("子自愛不愛父，故虧父而自利。", "Mozi 14", "MOHIST"),
    ("弟自愛不愛兄，故虧兄而自利。", "Mozi 14", "MOHIST"),
    ("若使天下兼相愛，愛人若愛其身，猶有不孝者乎？", "Mozi 15", "MOHIST"),
    ("視人之國若視其國，視人之家若視其家，視人之身若視其身。", "Mozi 15", "MOHIST"),
    ("是故諸侯相愛則不野戰，家主相愛則不相篡。", "Mozi 15", "MOHIST"),
    ("人與人相愛則不相賊。", "Mozi 15", "MOHIST"),
    ("君臣相愛則惠忠，父子相愛則慈孝。", "Mozi 15", "MOHIST"),
    ("兄弟相愛則和調。", "Mozi 15", "MOHIST"),
]

# Neo-Confucian Chinese (宋明理學)
NEO_CONFUCIAN_CHINESE = [
    ("存天理，滅人欲。", "Zhu Xi - Analects Commentary", "NEO_CONFUCIAN"),
    ("格物致知，誠意正心。", "Zhu Xi - Great Learning Commentary", "NEO_CONFUCIAN"),
    ("天理人欲，同行異情。", "Zhu Xi - Classified Conversations", "NEO_CONFUCIAN"),
    (
        "聖人千言萬語，只是教人明天理，滅人欲。",
        "Zhu Xi - Classified Conversations",
        "NEO_CONFUCIAN",
    ),
    ("敬者，聖學之所以成始而成終者也。", "Zhu Xi - Collected Writings", "NEO_CONFUCIAN"),
    ("窮理以致其知，反躬以踐其實。", "Zhu Xi - Collected Writings", "NEO_CONFUCIAN"),
    ("涵養須用敬，進學則在致知。", "Zhu Xi - Classified Conversations", "NEO_CONFUCIAN"),
    ("知行合一。", "Wang Yangming - Instructions for Practical Living", "NEO_CONFUCIAN"),
    ("致良知。", "Wang Yangming - Instructions for Practical Living", "NEO_CONFUCIAN"),
    ("無善無惡心之體，有善有惡意之動。", "Wang Yangming - Four Maxims", "NEO_CONFUCIAN"),
    ("知善知惡是良知，為善去惡是格物。", "Wang Yangming - Four Maxims", "NEO_CONFUCIAN"),
    ("心即理也。", "Wang Yangming - Instructions for Practical Living", "NEO_CONFUCIAN"),
    (
        "吾心之良知，即所謂天理也。",
        "Wang Yangming - Instructions for Practical Living",
        "NEO_CONFUCIAN",
    ),
    (
        "知是行之始，行是知之成。",
        "Wang Yangming - Instructions for Practical Living",
        "NEO_CONFUCIAN",
    ),
    ("知而不行，只是未知。", "Wang Yangming - Instructions for Practical Living", "NEO_CONFUCIAN"),
    ("破山中賊易，破心中賊難。", "Wang Yangming - Letters", "NEO_CONFUCIAN"),
    ("誠者，聖人之本。", "Zhou Dunyi - Tongshu", "NEO_CONFUCIAN"),
    ("誠，五常之本，百行之源也。", "Zhou Dunyi - Tongshu", "NEO_CONFUCIAN"),
    ("民吾同胞，物吾與也。", "Zhang Zai - Western Inscription", "NEO_CONFUCIAN"),
    (
        "為天地立心，為生民立命，為往聖繼絕學，為萬世開太平。",
        "Zhang Zai - Attributed",
        "NEO_CONFUCIAN",
    ),
]

# Islamic Legal Maxims (قواعد فقهية)
ISLAMIC_LEGAL_MAXIMS = [
    ("الأمور بمقاصدها", "Al-Qawa'id - Major 1", "FIQH"),
    ("اليقين لا يزول بالشك", "Al-Qawa'id - Major 2", "FIQH"),
    ("المشقة تجلب التيسير", "Al-Qawa'id - Major 3", "FIQH"),
    ("الضرر يزال", "Al-Qawa'id - Major 4", "FIQH"),
    ("العادة محكمة", "Al-Qawa'id - Major 5", "FIQH"),
    ("لا ضرر ولا ضرار", "Al-Qawa'id", "FIQH"),
    ("الضرر لا يزال بالضرر", "Al-Qawa'id", "FIQH"),
    ("الضرر الأشد يزال بالضرر الأخف", "Al-Qawa'id", "FIQH"),
    ("درء المفاسد أولى من جلب المصالح", "Al-Qawa'id", "FIQH"),
    ("يتحمل الضرر الخاص لدفع الضرر العام", "Al-Qawa'id", "FIQH"),
    ("إذا تعارضت مفسدتان روعي أعظمهما ضررا بارتكاب أخفهما", "Al-Qawa'id", "FIQH"),
    ("الأصل في الأشياء الإباحة", "Al-Qawa'id", "FIQH"),
    ("الأصل في العقود الصحة", "Al-Qawa'id", "FIQH"),
    ("الأصل بقاء ما كان على ما كان", "Al-Qawa'id", "FIQH"),
    ("ما حرم أخذه حرم إعطاؤه", "Al-Qawa'id", "FIQH"),
    ("ما حرم فعله حرم طلبه", "Al-Qawa'id", "FIQH"),
    ("الضرورات تبيح المحظورات", "Al-Qawa'id", "FIQH"),
    ("الضرورة تقدر بقدرها", "Al-Qawa'id", "FIQH"),
    ("ما أبيح للضرورة يقدر بقدرها", "Al-Qawa'id", "FIQH"),
    ("الحاجة تنزل منزلة الضرورة عامة كانت أو خاصة", "Al-Qawa'id", "FIQH"),
    ("إذا ضاق الأمر اتسع", "Al-Qawa'id", "FIQH"),
    ("الجواز الشرعي ينافي الضمان", "Al-Qawa'id", "FIQH"),
    ("المباشر ضامن وإن لم يتعمد", "Al-Qawa'id", "FIQH"),
    ("المتسبب لا يضمن إلا بالتعمد", "Al-Qawa'id", "FIQH"),
    ("إذا اجتمع المباشر والمتسبب يضاف الحكم إلى المباشر", "Al-Qawa'id", "FIQH"),
    ("الإذن العام كالإذن الخاص", "Al-Qawa'id", "FIQH"),
    ("لا عبرة بالدلالة في مقابلة التصريح", "Al-Qawa'id", "FIQH"),
    ("إعمال الكلام أولى من إهماله", "Al-Qawa'id", "FIQH"),
    ("الأصل في الكلام الحقيقة", "Al-Qawa'id", "FIQH"),
]

# Sufi Ethics (الأخلاق الصوفية)
SUFI_ETHICS = [
    ("التصوف كله أخلاق", "Al-Ghazali - Ihya", "SUFI"),
    ("من لم يؤثر فيه علم أخلاقه فقد غفل عن الفقه", "Al-Ghazali - Ihya", "SUFI"),
    ("الخلق الحسن جماع الدين كله", "Al-Ghazali - Ihya", "SUFI"),
    ("أصل الأخلاق المحمودة كلها أربعة: الحكمة والشجاعة والعفة والعدل", "Al-Ghazali - Ihya", "SUFI"),
    ("العلم بلا عمل جنون، والعمل بغير علم لا يكون", "Al-Ghazali - Ihya", "SUFI"),
    ("من عرف نفسه عرف ربه", "Al-Ghazali - Attributed", "SUFI"),
    ("قلب المؤمن بين أصبعين من أصابع الرحمن", "Al-Ghazali - Ihya", "SUFI"),
    ("التصوف هو الخلق، فمن زاد عليك في الخلق زاد عليك في التصوف", "Al-Junayd", "SUFI"),
    ("الصوفي من صفا قلبه لله", "Al-Junayd", "SUFI"),
    ("أفضل الأعمال مخالفة النفس والهوى", "Al-Junayd", "SUFI"),
    (
        "من لم يزن أفعاله وأحواله في كل وقت بالكتاب والسنة فلا تعده في ديوان الرجال",
        "Al-Qushayri - Risala",
        "SUFI",
    ),
    ("الصدق سيف الله في أرضه، ما وضع على شيء إلا قطعه", "Al-Qushayri - Risala", "SUFI"),
    ("ما خلقت الخلق إلا ليعرفوني", "Rumi - Attributed", "SUFI"),
    ("كن كالشمس للرحمة والشفقة، وكالليل في ستر عيوب الغير", "Rumi - Masnavi", "SUFI"),
    ("من عرف نفسه فقد عرف ربه", "Ibn Arabi - Fusus", "SUFI"),
    ("الإنسان الكامل مرآة الحق", "Ibn Arabi - Fusus", "SUFI"),
]

# Arabic Philosophy (الفلسفة العربية)
ARABIC_PHILOSOPHY = [
    ("الإنسان مدني بالطبع", "Al-Farabi - Ara Ahl al-Madina", "FALSAFA"),
    ("السعادة هي الخير المطلوب لذاته", "Al-Farabi - Tahsil al-Sa'ada", "FALSAFA"),
    ("الفضيلة هي الحال التي بها يفعل الإنسان الأفعال الجميلة", "Al-Farabi - Fusul", "FALSAFA"),
    ("العقل العملي هو الذي يدبر البدن", "Ibn Sina - Shifa", "FALSAFA"),
    ("النفس جوهر روحاني", "Ibn Sina - Shifa", "FALSAFA"),
    ("العدل هو فضيلة من الفضائل العامة", "Ibn Rushd - Commentary on Republic", "FALSAFA"),
    ("الحكمة والشريعة أختان رضيعتان", "Ibn Rushd - Fasl al-Maqal", "FALSAFA"),
    ("الحق لا يضاد الحق بل يوافقه ويشهد له", "Ibn Rushd - Fasl al-Maqal", "FALSAFA"),
    ("الإنسان مدني بالطبع، أي لا بد له من الاجتماع", "Ibn Khaldun - Muqaddima", "FALSAFA"),
    ("العصبية هي الرابطة الاجتماعية", "Ibn Khaldun - Muqaddima", "FALSAFA"),
    ("الظلم مؤذن بخراب العمران", "Ibn Khaldun - Muqaddima", "FALSAFA"),
]

# Sanskrit Dharmashastra (धर्मशास्त्र)
SANSKRIT_DHARMA = [
    ("अहिंसा परमो धर्मः", "Mahabharata 13.117.37", "DHARMA"),
    ("धर्म एव हतो हन्ति धर्मो रक्षति रक्षितः", "Mahabharata 8.69.57", "DHARMA"),
    ("न हि प्रियं मे स्यात् आत्मनः प्रतिकूलं परेषाम्", "Mahabharata 5.15.17", "DHARMA"),
    ("सत्यं ब्रूयात् प्रियं ब्रूयात्", "Mahabharata", "DHARMA"),
    ("आत्मनः प्रतिकूलानि परेषां न समाचरेत्", "Mahabharata 5.15.17", "DHARMA"),
    ("अहिंसा सत्यमस्तेयं शौचमिन्द्रियनिग्रहः", "Manusmriti 10.63", "DHARMA"),
    ("सर्वभूतेषु चात्मानं सर्वभूतानि चात्मनि", "Manusmriti", "DHARMA"),
    ("सत्यं वद धर्मं चर", "Taittiriya Upanishad 1.11", "UPANISHAD"),
    ("मातृदेवो भव। पितृदेवो भव। आचार्यदेवो भव। अतिथिदेवो भव।", "Taittiriya Upanishad 1.11", "UPANISHAD"),
    ("ईशावास्यमिदं सर्वं यत्किञ्च जगत्यां जगत्", "Isha Upanishad 1", "UPANISHAD"),
    ("तेन त्यक्तेन भुञ्जीथा मा गृधः कस्यस्विद्धनम्", "Isha Upanishad 1", "UPANISHAD"),
    ("कर्मण्येवाधिकारस्ते मा फलेषु कदाचन", "Bhagavad Gita 2.47", "GITA"),
    ("योगः कर्मसु कौशलम्", "Bhagavad Gita 2.50", "GITA"),
    ("समत्वं योग उच्यते", "Bhagavad Gita 2.48", "GITA"),
    ("सर्वधर्मान्परित्यज्य मामेकं शरणं व्रज", "Bhagavad Gita 18.66", "GITA"),
    ("अद्वेष्टा सर्वभूतानां मैत्रः करुण एव च", "Bhagavad Gita 12.13", "GITA"),
    ("प्रजासुखे सुखं राज्ञः प्रजानां च हिते हितम्", "Arthashastra 1.19", "ARTHA"),
    ("राज्ञो हि व्रतं कार्याणां चेष्टा राष्ट्रसंग्रहः", "Arthashastra", "ARTHA"),
]

# Pali Canon Ethics
PALI_ETHICS = [
    ("Sabbe sattā bhavantu sukhitattā", "Metta Sutta", "PALI"),
    ("Dhammo have rakkhati dhammacāriṃ", "Theragatha 303", "PALI"),
    ("Sabba pāpassa akaraṇaṃ, kusalassa upasampadā", "Dhammapada 183", "PALI"),
    ("Manopubbaṅgamā dhammā manoseṭṭhā manomayā", "Dhammapada 1", "PALI"),
    ("Na hi verena verāni sammantīdha kudācanaṃ", "Dhammapada 5", "PALI"),
    ("Averena ca sammanti esa dhammo sanantano", "Dhammapada 5", "PALI"),
    ("Attā hi attano nātho ko hi nātho paro siyā", "Dhammapada 160", "PALI"),
    ("Pāṇātipātā veramaṇī sikkhāpadaṃ samādiyāmi", "Vinaya", "PALI"),
    ("Adinnādānā veramaṇī sikkhāpadaṃ samādiyāmi", "Vinaya", "PALI"),
    ("Kāmesumicchācārā veramaṇī sikkhāpadaṃ samādiyāmi", "Vinaya", "PALI"),
    ("Musāvādā veramaṇī sikkhāpadaṃ samādiyāmi", "Vinaya", "PALI"),
    ("Surāmerayamajjapamādaṭṭhānā veramaṇī sikkhāpadaṃ samādiyāmi", "Vinaya", "PALI"),
    ("Mettañca sabbalokasmiṃ mānasaṃ bhāvaye aparimāṇaṃ", "Metta Sutta", "PALI"),
]

if SKIP_PROCESSING:
    print("=" * 60)
    print("USING CACHED DATA - Run with REFRESH_DATA_FROM_SOURCE=True to use v10.4 loaders")
    print("=" * 60)

    # Count passages by language
    by_lang = defaultdict(int)
    with open("data/processed/passages.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            by_lang[p["language"]] += 1

    print("\nPassages by language:")
    for lang, cnt in sorted(by_lang.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {cnt:,}")

    n_passages = sum(by_lang.values())
    print(f"\nTotal: {n_passages:,} passages")

    # ===== CHECK FOR v10.9 CORPORA =====
    # If cached data is missing v10.9 hardcoded corpora, add them
    has_v109 = by_lang.get("sanskrit", 0) > 0 or by_lang.get("pali", 0) > 0
    if not has_v109:
        print("\n" + "=" * 60)
        print("ADDING v10.9 CORPORA TO CACHED DATA")
        print("=" * 60)
        print("(Sanskrit, Pali, Buddhist Chinese, Legalist, Fiqh, Sufi, etc.)")

        # Load existing passages
        all_passages = []
        with open("data/processed/passages.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                all_passages.append(json.loads(line))
        print(f"Loaded {len(all_passages):,} existing passages")

        # Load existing bonds
        existing_bonds = []
        with open("data/processed/bonds.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                existing_bonds.append(json.loads(line))
        print(f"Loaded {len(existing_bonds):,} existing bonds")

        v109_start = len(all_passages)

        # Add Chinese philosophical traditions
        for corpus, period, label in [
            (BUDDHIST_CHINESE, "BUDDHIST", "Buddhist Chinese"),
            (LEGALIST_CHINESE, "LEGALIST", "Legalist Chinese"),
            (MOHIST_CHINESE, "MOHIST", "Mohist Chinese"),
            (NEO_CONFUCIAN_CHINESE, "NEO_CONFUCIAN", "Neo-Confucian"),
        ]:
            for text_content, source_ref, _ in corpus:
                all_passages.append(
                    {
                        "id": f"v109_{label.lower().replace(' ', '_')}_{len(all_passages)}",
                        "text": text_content,
                        "language": "classical_chinese",
                        "source": source_ref,
                        "time_period": period,
                    }
                )

        # Add Arabic/Islamic traditions
        for corpus, period, label in [
            (ISLAMIC_LEGAL_MAXIMS, "FIQH", "Islamic Legal Maxims"),
            (SUFI_ETHICS, "SUFI", "Sufi Ethics"),
            (ARABIC_PHILOSOPHY, "FALSAFA", "Arabic Philosophy"),
        ]:
            for text_content, source_ref, _ in corpus:
                all_passages.append(
                    {
                        "id": f"v109_{label.lower().replace(' ', '_')}_{len(all_passages)}",
                        "text": text_content,
                        "language": "arabic",
                        "source": source_ref,
                        "time_period": period,
                    }
                )

        # Add Sanskrit
        for text_content, source_ref, period_tag in SANSKRIT_DHARMA:
            all_passages.append(
                {
                    "id": f"v109_sanskrit_{len(all_passages)}",
                    "text": text_content,
                    "language": "sanskrit",
                    "source": source_ref,
                    "time_period": period_tag,
                }
            )

        # Add Pali
        for text_content, source_ref, period_tag in PALI_ETHICS:
            all_passages.append(
                {
                    "id": f"v109_pali_{len(all_passages)}",
                    "text": text_content,
                    "language": "pali",
                    "source": source_ref,
                    "time_period": period_tag,
                }
            )

        v109_count = len(all_passages) - v109_start
        print(f"Added {v109_count} v10.9 passages")

        # Extract bonds for new passages
        print("Extracting bonds for v10.9 passages...")
        new_bonds = []
        for p in all_passages[v109_start:]:
            # Simple bond extraction for hardcoded corpora (all are prescriptive)
            new_bonds.append(
                {
                    "passage_id": p["id"],
                    "bond_type": "AUTHORITY",  # Default, will be refined by patterns
                    "language": p["language"],
                    "time_period": p["time_period"],
                    "source": p["source"],
                    "text": p["text"][:500],
                    "context": "prescriptive",
                    "confidence": "high",
                }
            )

        all_bonds = existing_bonds + new_bonds
        print(f"Total bonds: {len(all_bonds):,}")

        # Save updated passages
        with open("data/processed/passages.jsonl", "w", encoding="utf-8") as f:
            for p in all_passages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        # Save updated bonds
        with open("data/processed/bonds.jsonl", "w", encoding="utf-8") as f:
            for b in all_bonds:
                f.write(json.dumps(b, ensure_ascii=False) + "\n")

        # Update Drive cache
        if USE_DRIVE_DATA:
            try:
                shutil.copy("data/processed/passages.jsonl", f"{SAVE_DIR}/passages.jsonl")
                shutil.copy("data/processed/bonds.jsonl", f"{SAVE_DIR}/bonds.jsonl")
                print(f"Updated Drive cache with v10.9 corpora")
            except Exception as e:
                print(f"Drive update failed: {e}")

        # Update counts
        by_lang["sanskrit"] = len(SANSKRIT_DHARMA)
        by_lang["pali"] = len(PALI_ETHICS)
        by_lang["classical_chinese"] += sum(
            len(c)
            for c, _, _ in [
                (BUDDHIST_CHINESE, "", ""),
                (LEGALIST_CHINESE, "", ""),
                (MOHIST_CHINESE, "", ""),
                (NEO_CONFUCIAN_CHINESE, "", ""),
            ]
        )
        by_lang["arabic"] += sum(
            len(c)
            for c, _, _ in [
                (ISLAMIC_LEGAL_MAXIMS, "", ""),
                (SUFI_ETHICS, "", ""),
                (ARABIC_PHILOSOPHY, "", ""),
            ]
        )
        n_passages = len(all_passages)

        print(f"\nUpdated corpus sizes:")
        for lang, cnt in sorted(by_lang.items(), key=lambda x: -x[1]):
            print(f"  {lang}: {cnt:,}")
    else:
        print("\nv10.9 corpora already present (Sanskrit/Pali detected)")

    # Validate corpus sizes and identify what needs augmentation
    print("\nCorpus adequacy check:")
    languages_to_augment = []
    for lang, min_size in MIN_CORPUS_SIZE.items():
        actual = by_lang.get(lang, 0)
        status = "OK" if actual >= min_size else "NEED MORE"
        print(f"  {lang}: {actual:,} / {min_size:,} - {status}")
        if actual < min_size and lang in AUGMENTATION_DATASETS:
            languages_to_augment.append((lang, min_size - actual))

    # Augment any under-represented languages that have available datasets
    if languages_to_augment:
        print(f"\n" + "=" * 60)
        print(f"AUGMENTING UNDER-REPRESENTED CORPORA")
        print(f"=" * 60)
        print(f"Languages to augment: {[l for l, _ in languages_to_augment]}")

        # Load existing passages
        all_passages = []
        with open("data/processed/passages.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                all_passages.append(json.loads(line))

        # Normalize field names
        for p in all_passages:
            if "lang" not in p and "language" in p:
                p["lang"] = p["language"]
            if "period" not in p and "time_period" in p:
                p["period"] = p["time_period"]

        print(f"Loaded {len(all_passages):,} existing passages")

        from datasets import load_dataset

        for lang, needed in languages_to_augment:
            lang_count = by_lang.get(lang, 0)
            print(f"\n--- Augmenting {lang} (need {needed:,} more) ---")

            for dataset_name, short_name in AUGMENTATION_DATASETS.get(lang, []):
                if lang_count >= MIN_CORPUS_SIZE[lang]:
                    break

                print(f"  Loading {short_name}...")
                try:
                    if dataset_name == "hendrycks/ethics":
                        # ETHICS has multiple categories
                        categories = [
                            "commonsense",
                            "deontology",
                            "justice",
                            "utilitarianism",
                            "virtue",
                        ]
                        for cat in categories:
                            if lang_count >= MIN_CORPUS_SIZE[lang]:
                                break
                            try:
                                ds = load_dataset(
                                    dataset_name, cat, split="train", trust_remote_code=True
                                )
                                cat_count = 0
                                for item in ds:
                                    if lang_count >= MIN_CORPUS_SIZE[lang]:
                                        break
                                    if cat == "commonsense":
                                        text = item.get("input", "")
                                    elif cat == "justice":
                                        text = item.get("scenario", "")
                                    elif cat == "deontology":
                                        text = (
                                            item.get("scenario", "") + " " + item.get("excuse", "")
                                        )
                                    elif cat == "virtue":
                                        text = item.get("scenario", "")
                                    else:
                                        text = (
                                            str(item.get("baseline", ""))
                                            + " vs "
                                            + str(item.get("less_pleasant", ""))
                                        )

                                    if text and len(text) > 30:
                                        all_passages.append(
                                            {
                                                "id": f"ethics_{cat}_{len(all_passages)}",
                                                "text": text[:1000],
                                                "lang": lang,
                                                "language": lang,
                                                "source": f"ETHICS_{cat}",
                                                "period": "MODERN",
                                                "time_period": "MODERN",
                                            }
                                        )
                                        lang_count += 1
                                        cat_count += 1
                                print(f"    {cat}: +{cat_count:,}")
                            except Exception as e:
                                print(f"    {cat} error: {e}")

                    elif dataset_name == "allenai/social_chem_101":
                        ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
                        sc_count = 0
                        for item in ds:
                            if lang_count >= MIN_CORPUS_SIZE[lang]:
                                break
                            action = item.get("action", "")
                            situation = item.get("situation", "")
                            rot = item.get("rot", "")

                            if rot and len(rot) > 20:
                                text = f"{situation} {action}".strip() if situation else action
                                text = f"{text}. {rot}" if text else rot

                                all_passages.append(
                                    {
                                        "id": f"socialchem_{len(all_passages)}",
                                        "text": text[:1000],
                                        "lang": lang,
                                        "language": lang,
                                        "source": "Social_Chemistry_101",
                                        "period": "MODERN",
                                        "time_period": "MODERN",
                                    }
                                )
                                lang_count += 1
                                sc_count += 1
                        print(f"    Social Chemistry: +{sc_count:,}")

                    else:
                        # Generic HuggingFace dataset
                        try:
                            ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
                            gen_count = 0
                            for item in ds:
                                if lang_count >= MIN_CORPUS_SIZE[lang]:
                                    break
                                text = item.get("text", "") or item.get("content", "") or str(item)
                                if text and len(text) > 50:
                                    all_passages.append(
                                        {
                                            "id": f"{short_name.lower()}_{len(all_passages)}",
                                            "text": text[:1000],
                                            "lang": lang,
                                            "language": lang,
                                            "source": short_name,
                                            "period": "MODERN",
                                            "time_period": "MODERN",
                                        }
                                    )
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
        print("\nExtracting bonds for new passages...")
        new_bonds = []
        new_sources = {
            "ETHICS_commonsense",
            "ETHICS_deontology",
            "ETHICS_justice",
            "ETHICS_utilitarianism",
            "ETHICS_virtue",
            "Social_Chemistry_101",
        }

        for p in tqdm(all_passages, desc="Processing"):
            src = p.get("source", "")
            if any(src.startswith(s.split("_")[0]) for s in new_sources) or src in new_sources:
                text_lower = p["text"].lower()
                if any(
                    w in text_lower
                    for w in ["wrong", "bad", "shouldn't", "immoral", "rude", "unethical"]
                ):
                    bond_type = "PROHIBITION"
                elif any(w in text_lower for w in ["should", "must", "duty", "obligat", "need to"]):
                    bond_type = "OBLIGATION"
                elif any(
                    w in text_lower for w in ["okay", "fine", "acceptable", "can", "may", "allowed"]
                ):
                    bond_type = "PERMISSION"
                else:
                    bond_type = "NEUTRAL"

                new_bonds.append(
                    {
                        "passage_id": p["id"],
                        "bond_type": bond_type,
                        "language": p.get("language", p.get("lang")),
                        "time_period": p.get("time_period", p.get("period", "MODERN")),
                        "source": src,
                        "text": p["text"][:500],
                        "context": "prescriptive",
                        "confidence": "high",
                    }
                )

        # Load existing bonds and merge
        existing_bonds = []
        with open("data/processed/bonds.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                existing_bonds.append(json.loads(line))

        all_bonds = existing_bonds + new_bonds
        print(f"Total bonds: {len(all_bonds):,} ({len(new_bonds):,} new)")

        # Save updated passages
        with open("data/processed/passages.jsonl", "w", encoding="utf-8") as f:
            for p in all_passages:
                p_out = {
                    "id": p["id"],
                    "text": p["text"],
                    "language": p.get("language", p.get("lang", "english")),
                    "source": p.get("source", ""),
                    "time_period": p.get("time_period", p.get("period", "MODERN")),
                }
                f.write(json.dumps(p_out, ensure_ascii=False) + "\n")

        # Save updated bonds
        with open("data/processed/bonds.jsonl", "w", encoding="utf-8") as f:
            for b in all_bonds:
                f.write(json.dumps(b, ensure_ascii=False) + "\n")

        print("Saved augmented data")

        # Copy to Drive
        if USE_DRIVE_DATA:
            try:
                shutil.copy("data/processed/passages.jsonl", f"{SAVE_DIR}/passages.jsonl")
                shutil.copy("data/processed/bonds.jsonl", f"{SAVE_DIR}/bonds.jsonl")
                print(f"Updated Drive cache: {SAVE_DIR}")
            except Exception as e:
                print(f"Drive update failed: {e}")

        # Final summary
        print(f"\nFinal corpus sizes:")
        for lang, cnt in sorted(by_lang.items(), key=lambda x: -x[1]):
            target = MIN_CORPUS_SIZE.get(lang, 0)
            status = "OK" if cnt >= target else "LOW"
            print(f"  {lang}: {cnt:,} ({status})")
        n_passages = len(all_passages)

else:
    print("=" * 60)
    print("LOADING CORPORA")
    print(f"GPU Tier: {GPU_TIER}")
    print(f"Max per language: {MAX_PER_LANG:,}")
    print("=" * 60)

    random.seed(42)
    all_passages = []

    # ===== PARALLEL PREFETCH MANAGER =====
    from concurrent.futures import ThreadPoolExecutor, Future
    import threading

    print("Starting parallel prefetch of remote corpora...")
    prefetch_executor = ThreadPoolExecutor(max_workers=12)
    prefetch_results = {}  # url -> Future

    def prefetch_url(url, timeout=60):
        """Fetch URL content in background."""
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.text
        except Exception as e:
            print(f"    Prefetch failed for {url[:50]}...: {e}")
        return None

    # Queue all remote downloads
    PREFETCH_URLS = [
        # Gutenberg - Western Classics
        "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",  # Republic
        "https://www.gutenberg.org/cache/epub/1656/pg1656.txt",  # Apology
        "https://www.gutenberg.org/cache/epub/1657/pg1657.txt",  # Crito
        "https://www.gutenberg.org/cache/epub/1658/pg1658.txt",  # Phaedo
        "https://www.gutenberg.org/cache/epub/3794/pg3794.txt",  # Gorgias
        "https://www.gutenberg.org/cache/epub/1636/pg1636.txt",  # Symposium
        "https://www.gutenberg.org/cache/epub/1726/pg1726.txt",  # Meno
        "https://www.gutenberg.org/cache/epub/8438/pg8438.txt",  # Nicomachean Ethics
        "https://www.gutenberg.org/cache/epub/6762/pg6762.txt",  # Politics
        "https://www.gutenberg.org/cache/epub/2680/pg2680.txt",  # Meditations
        "https://www.gutenberg.org/cache/epub/10661/pg10661.txt",  # Enchiridion
        "https://www.gutenberg.org/cache/epub/3042/pg3042.txt",  # Discourses
        "https://www.gutenberg.org/cache/epub/14988/pg14988.txt",  # De Officiis
        # MIT Classics fallback
        "https://classics.mit.edu/Aristotle/nicomachaen.mb.txt",
        "https://classics.mit.edu/Plato/laws.mb.txt",
        # Bible Parallel Corpus
        "https://raw.githubusercontent.com/christos-c/bible-corpus/master/bibles/English.xml",
        "https://raw.githubusercontent.com/christos-c/bible-corpus/master/bibles/Hebrew.xml",
        "https://raw.githubusercontent.com/christos-c/bible-corpus/master/bibles/Arabic.xml",
        "https://raw.githubusercontent.com/christos-c/bible-corpus/master/bibles/Chinese.xml",
    ]

    for url in PREFETCH_URLS:
        prefetch_results[url] = prefetch_executor.submit(prefetch_url, url)

    print(f"  Queued {len(PREFETCH_URLS)} URLs for background download")

    def get_prefetched(url, timeout=30):
        """Get prefetched content, waiting if necessary."""
        if url in prefetch_results:
            try:
                return prefetch_results[url].result(timeout=timeout)
            except Exception:
                pass
        # Fallback to direct fetch
        return prefetch_url(url)

    # ===== SEFARIA (Hebrew/Aramaic) =====
    print("\nLoading Sefaria...")
    sefaria_path = Path("data/raw/Sefaria-Export/json")

    CATEGORY_TO_PERIOD = {
        "Tanakh": "BIBLICAL",
        "Torah": "BIBLICAL",
        "Prophets": "BIBLICAL",
        "Writings": "BIBLICAL",
        "Mishnah": "TANNAITIC",
        "Tosefta": "TANNAITIC",
        "Sifra": "TANNAITIC",
        "Sifrei": "TANNAITIC",
        "Talmud": "TALMUDIC",
        "Bavli": "TALMUDIC",
        "Yerushalmi": "TALMUDIC",
        "Midrash": "MIDRASHIC",
        "Midrash Rabbah": "MIDRASHIC",
        "Midrash Aggadah": "MIDRASHIC",
        "Halakhah": "MEDIEVAL",
        "Shulchan Arukh": "MEDIEVAL",
        "Mishneh Torah": "MEDIEVAL",
        "Musar": "MODERN",
        "Chasidut": "MODERN",
        "Modern": "MODERN",
    }

    lang_counts = {"hebrew": 0, "aramaic": 0}

    if sefaria_path.exists():
        for json_file in tqdm(list(sefaria_path.rglob("*.json"))[:5000], desc="Sefaria"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, dict) and "text" in data:
                    # Determine period from path
                    path_parts = str(json_file.relative_to(sefaria_path)).split("/")
                    period = "CLASSICAL"
                    for part in path_parts:
                        if part in CATEGORY_TO_PERIOD:
                            period = CATEGORY_TO_PERIOD[part]
                            break

                    # Determine language (heuristic: Talmud is primarily Aramaic)
                    is_talmud = any(t in str(json_file) for t in ["Talmud", "Bavli", "Yerushalmi"])
                    lang = "aramaic" if is_talmud else "hebrew"

                    def extract_texts(obj, texts):
                        if isinstance(obj, str) and len(obj) > 20:
                            texts.append(obj)
                        elif isinstance(obj, list):
                            for item in obj:
                                extract_texts(item, texts)

                    texts = []
                    extract_texts(data["text"], texts)

                    for txt in texts[:50]:  # Limit per file
                        if lang_counts[lang] < MAX_PER_LANG:
                            all_passages.append(
                                {
                                    "id": f"sefaria_{len(all_passages)}",
                                    "text": txt,
                                    "lang": lang,
                                    "source": json_file.stem,
                                    "period": period,
                                }
                            )
                            lang_counts[lang] += 1

            except Exception as e:
                continue
    else:
        print("  Sefaria not found - will download")

    print(f"  Hebrew: {lang_counts['hebrew']:,}, Aramaic: {lang_counts['aramaic']:,}")

    # ===== CLASSICAL CHINESE: Disabled (CText API blocks Colab) =====
    print("  Skipping CText API (blocked from Colab, using Wenyanwen instead)")
    chinese_count = 0  # Initialize counter

    # ===== KAGGLE: Ancient Chinese Wenyanwen (132K texts, 552M chars) =====
    if chinese_count < MAX_PER_LANG:
        print("  Loading from Kaggle Wenyanwen dataset...")
        wenyan_zip_name = "Ancient_Chinese_Text_(wenyanwen)_archive.zip"
        wenyan_csv_name = "cn_wenyan.csv"
        wenyan_local_zip = Path(f"data/raw/{wenyan_zip_name}")
        _drive_ok = "USE_DRIVE_DATA" in dir() and USE_DRIVE_DATA and "SAVE_DIR" in dir()
        wenyan_drive_zip = Path(f"{SAVE_DIR}/{wenyan_zip_name}") if _drive_ok else None
        wenyan_local_csv = Path(f"data/raw/{wenyan_csv_name}")
        wenyan_drive_csv = Path(f"{SAVE_DIR}/{wenyan_csv_name}") if _drive_ok else None

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
                    with zipfile.ZipFile(zip_path, "r") as z:
                        z.extract(wenyan_csv_name, "data/raw/")
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
                with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if chinese_count >= MAX_PER_LANG:
                            break
                        text = row.get("text", "")
                        title = row.get("title", "")
                        # Split long texts into passages (max 2000 chars each)
                        # Use paragraph breaks or every 1500 chars
                        paragraphs = text.split("\n")
                        current_para = ""
                        for para in paragraphs:
                            para = para.strip()
                            if not para:
                                continue
                            if len(current_para) + len(para) < 1500:
                                current_para += para
                            else:
                                if len(current_para) > 50:
                                    all_passages.append(
                                        {
                                            "id": f"wenyan_{len(all_passages)}",
                                            "text": current_para,
                                            "lang": "classical_chinese",
                                            "source": (
                                                title.split("/")[0] if "/" in title else title
                                            ),
                                            "period": "CONFUCIAN",
                                        }
                                    )
                                    chinese_count += 1
                                    wenyan_count += 1
                                    if chinese_count >= MAX_PER_LANG:
                                        break
                                current_para = para
                        # Don't forget last paragraph
                        if current_para and len(current_para) > 50 and chinese_count < MAX_PER_LANG:
                            all_passages.append(
                                {
                                    "id": f"wenyan_{len(all_passages)}",
                                    "text": current_para,
                                    "lang": "classical_chinese",
                                    "source": title.split("/")[0] if "/" in title else title,
                                    "period": "CONFUCIAN",
                                }
                            )
                            chinese_count += 1
                            wenyan_count += 1
                print(f"    Added {wenyan_count:,} passages from Wenyanwen")
            except Exception as e:
                print(f"    Error loading Wenyanwen: {e}")

    print(f"  Total Classical Chinese: {chinese_count:,}")

    # ===== ARABIC/ISLAMIC (Kaggle quran-nlp) =====
    print("\nLoading Arabic from Kaggle quran-nlp...")

    arabic_count = 0
    kaggle_path = Path("data/raw/quran-nlp")

    # Try to download from Kaggle
    if not kaggle_path.exists() and REFRESH_DATA_FROM_SOURCE:
        try:
            import subprocess
            import zipfile

            subprocess.run(["pip", "install", "-q", "kaggle"], check=True)
            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    "alizahidraja/quran-nlp",
                    "-p",
                    "data/raw",
                ],
                check=True,
                timeout=300,
            )

            with zipfile.ZipFile("data/raw/quran-nlp.zip", "r") as z:
                z.extractall(kaggle_path)
            print("  Downloaded from Kaggle!")
        except Exception as e:
            print(f"  Kaggle download failed: {e}")

    # Load if available
    if kaggle_path.exists():
        import pandas as pd

        # Load Quran
        quran_files = list(kaggle_path.rglob("*quran*.csv"))
        for qf in quran_files:
            if arabic_count >= MAX_PER_LANG:
                break
            try:
                df = pd.read_csv(qf, nrows=MAX_PER_LANG - arabic_count)
                for _, row in df.iterrows():
                    text = str(row.get("arabic", row.get("text", row.get("Arabic", ""))))
                    if text and len(text) > 10 and text != "nan":
                        all_passages.append(
                            {
                                "id": f"quran_{len(all_passages)}",
                                "text": text,
                                "lang": "arabic",
                                "source": "Quran",
                                "period": "CLASSICAL",
                            }
                        )
                        arabic_count += 1
            except:
                continue

        # Load Hadith
        hadith_files = list(kaggle_path.rglob("*hadith*.csv"))
        for hf in hadith_files:
            if arabic_count >= MAX_PER_LANG:
                break
            try:
                df = pd.read_csv(hf, nrows=MAX_PER_LANG - arabic_count)
                for _, row in df.iterrows():
                    text = str(row.get("hadith", row.get("text", row.get("Arabic", ""))))
                    if text and len(text) > 10 and text != "nan":
                        all_passages.append(
                            {
                                "id": f"hadith_{len(all_passages)}",
                                "text": text,
                                "lang": "arabic",
                                "source": "Hadith",
                                "period": "CLASSICAL",
                            }
                        )
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
                lines = resp.text.strip().split("\n")
                for line in lines:
                    if "|" in line and arabic_count < MAX_PER_LANG:
                        parts = line.split("|")
                        if len(parts) >= 3:
                            text = parts[2].strip()
                            if len(text) > 10:
                                all_passages.append(
                                    {
                                        "id": f"tanzil_{len(all_passages)}",
                                        "text": text,
                                        "lang": "arabic",
                                        "source": "Quran (Tanzil)",
                                        "period": "CLASSICAL",
                                    }
                                )
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
            all_passages.append(
                {
                    "id": f"arabic_{len(all_passages)}",
                    "text": txt,
                    "lang": "arabic",
                    "source": "Quran/Hadith",
                    "period": "CLASSICAL",
                }
            )
            arabic_count += 1

    print(f"  Arabic: {arabic_count:,}")

    # ===== DEAR ABBY (English) =====
    print("Loading Dear Abby...")

    english_count = 0
    abby_path = Path("data/raw/dear_abby.csv")
    print(f"  Local path exists: {abby_path.exists()}")

    # Check Drive first
    drive_abby = f"{SAVE_DIR}/dear_abby.csv"
    print(f"  Drive path: {drive_abby}")
    print(f"  Drive path exists: {os.path.exists(drive_abby)}")
    if not abby_path.exists() and os.path.exists(drive_abby):
        os.makedirs("data/raw", exist_ok=True)
        shutil.copy(drive_abby, abby_path)
        print("  Copied from Drive")

    if not abby_path.exists() and REFRESH_DATA_FROM_SOURCE:
        try:
            import subprocess

            subprocess.run(["pip", "install", "-q", "kaggle"], check=True)
            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    "thedevastator/20000-dear-abby-questions",
                    "-p",
                    "data/raw",
                    "-f",
                    "dear_abby.csv",
                ],
                check=True,
                timeout=120,
            )
            print("  Downloaded from Kaggle!")
        except Exception as e:
            print(f"  Kaggle download failed: {e}")

    if abby_path.exists():
        import pandas as pd

        df = pd.read_csv(abby_path, nrows=MAX_PER_LANG)
        print(f"  CSV columns: {list(df.columns)}")
        print(f"  CSV rows: {len(df)}")
        for _, row in df.iterrows():
            question = str(row.get("question", ""))
            answer = str(row.get("question_only", ""))
            if len(answer) > 50:
                all_passages.append(
                    {
                        "id": f"abby_{len(all_passages)}",
                        "text": answer,
                        "lang": "english",
                        "source": "Dear Abby",
                        "period": "DEAR_ABBY",
                    }
                )
                english_count += 1
    else:
        print("  Dear Abby not found")

    print(f"  Dear Abby: {english_count:,}")

    # ===== WESTERN CLASSICS (Greek/Roman Philosophy) =====
    print("\nLoading Western Classics (parallel download)...")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Project Gutenberg texts (reliable, plain text)
    GUTENBERG_TEXTS = [
        # Plato - Ethics & Political Philosophy
        (1497, "Republic", "Plato"),
        (1656, "Apology", "Plato"),
        (1657, "Crito", "Plato"),
        (1658, "Phaedo", "Plato"),
        (3794, "Gorgias", "Plato"),
        (1636, "Symposium", "Plato"),
        (1726, "Meno", "Plato"),
        # Aristotle
        (8438, "Nicomachean Ethics", "Aristotle"),
        (6762, "Politics", "Aristotle"),
        # Stoics
        (2680, "Meditations", "Marcus Aurelius"),
        (10661, "Enchiridion", "Epictetus"),
        (3042, "Discourses", "Epictetus"),
        # Cicero
        (14988, "De Officiis", "Cicero"),
    ]

    # MIT Classics fallback
    MIT_TEXTS = [
        (
            "https://classics.mit.edu/Aristotle/nicomachaen.mb.txt",
            "Nicomachean Ethics",
            "Aristotle",
        ),
        ("https://classics.mit.edu/Aristotle/politics.mb.txt", "Politics", "Aristotle"),
        ("https://classics.mit.edu/Plato/republic.mb.txt", "Republic", "Plato"),
        ("https://classics.mit.edu/Plato/laws.mb.txt", "Laws", "Plato"),
        ("https://classics.mit.edu/Antoninus/meditations.mb.txt", "Meditations", "Marcus Aurelius"),
        ("https://classics.mit.edu/Epictetus/epicench.mb.txt", "Enchiridion", "Epictetus"),
        ("https://classics.mit.edu/Cicero/duties.mb.txt", "De Officiis", "Cicero"),
    ]

    western_target = min(MAX_PER_LANG, 15000)

    def fetch_gutenberg(item):
        """Fetch a single Gutenberg text (uses prefetch if available)."""
        gutenberg_id, title, author = item
        try:
            url = f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
            text = get_prefetched(url)
            if text:
                # Skip Gutenberg header/footer
                for marker in ["*** START OF", "***START OF"]:
                    if marker in text:
                        text = text.split(marker, 1)[-1]
                        break
                for marker in ["*** END OF", "***END OF", "End of Project Gutenberg"]:
                    if marker in text:
                        text = text.split(marker, 1)[0]
                        break

                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 100]
                passages = []
                for para in paragraphs:
                    para = re.sub(r"\s+", " ", para).strip()
                    if 50 < len(para) < 2000:
                        passages.append(
                            {
                                "text": para,
                                "source": f"{author}: {title}",
                                "author": author,
                                "title": title,
                            }
                        )
                return (title, author, passages)
        except Exception as e:
            pass
        return (title, author, [])

    def fetch_mit(item):
        """Fetch a single MIT Classics text (uses prefetch if available)."""
        url, title, author = item
        try:
            text = get_prefetched(url)
            if text:
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 100]
                passages = []
                for para in paragraphs[:500]:
                    para = re.sub(r"\s+", " ", para).strip()
                    if 50 < len(para) < 2000:
                        passages.append(
                            {
                                "text": para,
                                "source": f"{author}: {title}",
                                "author": author,
                                "title": title,
                            }
                        )
                return (title, author, passages)
        except:
            pass
        return (title, author, [])

    western_passages = []
    loaded_titles = set()

    # Parallel fetch from Gutenberg
    print("  Fetching from Project Gutenberg (parallel)...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_gutenberg, item): item for item in GUTENBERG_TEXTS}
        for future in as_completed(futures):
            title, author, passages = future.result()
            if passages and title not in loaded_titles:
                western_passages.extend(passages)
                loaded_titles.add(title)
                print(f"    {author}: {title} - {len(passages)} passages")

    # Parallel fetch from MIT for any missing
    missing_mit = [(url, t, a) for url, t, a in MIT_TEXTS if t not in loaded_titles]
    if missing_mit:
        print("  Fetching missing texts from MIT Classics...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(fetch_mit, item): item for item in missing_mit}
            for future in as_completed(futures):
                title, author, passages = future.result()
                if passages and title not in loaded_titles:
                    western_passages.extend(passages)
                    loaded_titles.add(title)
                    print(f"    {author}: {title} - {len(passages)} passages (MIT)")

    # Add to all_passages with proper IDs
    western_count = 0
    for p in western_passages:
        if western_count >= western_target:
            break
        all_passages.append(
            {
                "id": f"western_{len(all_passages)}",
                "text": p["text"],
                "lang": "english",
                "source": p["source"],
                "period": "WESTERN_CLASSICAL",
                "time_period": "WESTERN_CLASSICAL",
            }
        )
        western_count += 1

    print(f"  Total Western Classics: {western_count:,}")
    # ===== UNIMORAL: Disabled (gated dataset requires auth) =====
    print("  Skipping UniMoral (gated HuggingFace dataset)")

    # ===== UN PARALLEL CORPUS (HuggingFace streaming) =====
    print("\nLoading UN Corpus from HuggingFace (streaming)...")
    try:
        from datasets import load_dataset

        pairs = [("ar", "en"), ("en", "zh")]
        un_count = 0
        lang_map = {"ar": "arabic", "zh": "classical_chinese", "en": "english"}

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

                    translation = item.get("translation", {})
                    for lang_code in [src, tgt]:
                        text = translation.get(lang_code, "")
                        if len(text) > 30 and lang_code in lang_map:
                            all_passages.append(
                                {
                                    "id": f"un_{len(all_passages)}",
                                    "text": text,
                                    "lang": lang_map[lang_code],
                                    "source": "UN Corpus",
                                    "period": "MODERN",
                                }
                            )
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
            ("Hebrew.xml", "hebrew"),
            ("Arabic.xml", "arabic"),
            ("Chinese.xml", "classical_chinese"),
        ]

        bible_count = 0
        for filename, lang in bible_files:
            if bible_count >= MAX_PER_LANG * 3:
                break
            try:
                url = f"{base_url}/{filename}"
                text = get_prefetched(url)
                if text:
                    verses = re.findall(r"<seg[^>]*>([^<]+)</seg>", text)
                    file_count = 0
                    for verse in verses:
                        if file_count >= MAX_PER_LANG:
                            break
                        verse = verse.strip()
                        if len(verse) > 10:
                            all_passages.append(
                                {
                                    "id": f"bible_{len(all_passages)}",
                                    "text": verse,
                                    "lang": lang,
                                    "source": "Bible",
                                    "period": "CLASSICAL",
                                }
                            )
                            file_count += 1
                            bible_count += 1
                    print(f"  Bible {lang}: {file_count:,}")
            except Exception as e:
                print(f"  Bible {filename} error: {e}")

        print(f"  Bible total: {bible_count:,}")
    except Exception as e:
        print(f"  Bible error: {e}")

    # ===== NEW v10.9 CORPORA =====
    print("\n--- v10.9 New Corpora (hardcoded) ---")

    # Chinese philosophical traditions
    chinese_corpora = [
        (BUDDHIST_CHINESE, "BUDDHIST", "Buddhist Chinese"),
        (LEGALIST_CHINESE, "LEGALIST", "Legalist Chinese"),
        (MOHIST_CHINESE, "MOHIST", "Mohist Chinese"),
        (NEO_CONFUCIAN_CHINESE, "NEO_CONFUCIAN", "Neo-Confucian"),
    ]

    for corpus, period, label in chinese_corpora:
        count = 0
        for text_content, source_ref, _ in corpus:
            all_passages.append(
                {
                    "id": f"v109_{label.lower().replace(' ', '_')}_{len(all_passages)}",
                    "text": text_content,
                    "lang": "classical_chinese",
                    "source": source_ref,
                    "period": period,
                }
            )
            count += 1
        print(f"  {label}: {count}")

    # Arabic/Islamic traditions
    arabic_corpora = [
        (ISLAMIC_LEGAL_MAXIMS, "FIQH", "Islamic Legal Maxims"),
        (SUFI_ETHICS, "SUFI", "Sufi Ethics"),
        (ARABIC_PHILOSOPHY, "FALSAFA", "Arabic Philosophy"),
    ]

    for corpus, period, label in arabic_corpora:
        count = 0
        for text_content, source_ref, _ in corpus:
            all_passages.append(
                {
                    "id": f"v109_{label.lower().replace(' ', '_')}_{len(all_passages)}",
                    "text": text_content,
                    "lang": "arabic",
                    "source": source_ref,
                    "period": period,
                }
            )
            count += 1
        print(f"  {label}: {count}")

    # Sanskrit tradition
    sanskrit_count = 0
    for text_content, source_ref, period_tag in SANSKRIT_DHARMA:
        all_passages.append(
            {
                "id": f"v109_sanskrit_{len(all_passages)}",
                "text": text_content,
                "lang": "sanskrit",
                "source": source_ref,
                "period": period_tag,
            }
        )
        sanskrit_count += 1
    print(f"  Sanskrit Dharma: {sanskrit_count}")

    # Pali tradition
    pali_count = 0
    for text_content, source_ref, period_tag in PALI_ETHICS:
        all_passages.append(
            {
                "id": f"v109_pali_{len(all_passages)}",
                "text": text_content,
                "lang": "pali",
                "source": source_ref,
                "period": period_tag,
            }
        )
        pali_count += 1
    print(f"  Pali Ethics: {pali_count}")

    # Cleanup prefetch executor
    print("\nWaiting for any remaining prefetch tasks...")
    prefetch_executor.shutdown(wait=False)

    # ===== SUMMARY =====
    print(f"\nTOTAL: {len(all_passages):,}")

    # Count by language
    by_lang = defaultdict(int)
    for p in all_passages:
        by_lang[p["lang"]] += 1
    print("\nBy language:")
    for lang, cnt in sorted(by_lang.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {cnt:,}")

    # ===== EXTRACT BONDS =====
    print("\n" + "=" * 60)
    print("EXTRACTING BONDS")
    print("=" * 60)

    def extract_bond(text, language):
        """Extract bond type with context awareness."""
        tn = normalize_text(text, language)

        for bt, pats in ALL_BOND_PATTERNS.get(language, {}).items():
            for p in pats:
                match = re.search(p, tn)
                if match:
                    # Check context around the match
                    context, marker_type = detect_context(text, language, match.start())
                    confidence = 0.9 if context == "prescriptive" else 0.5
                    return bt, context, confidence
        return None, "unknown", 0.5

    bonds = []
    for p in tqdm(all_passages, desc="Extracting bonds"):
        bt, ctx, conf = extract_bond(p["text"], p["lang"])
        if bt:
            bonds.append(
                {
                    "passage_id": p["id"],
                    "bond_type": bt,
                    "language": p["lang"],
                    "time_period": p["period"],
                    "source": p["source"],
                    "text": p["text"][:500],
                    "context": ctx,
                    "confidence": conf,
                }
            )

    print(f"\nExtracted {len(bonds):,} bonds from {len(all_passages):,} passages")

    # Count by bond type
    by_bond = defaultdict(int)
    for b in bonds:
        by_bond[b["bond_type"]] += 1
    print("\nBy bond type:")
    for bt, cnt in sorted(by_bond.items(), key=lambda x: -x[1]):
        print(f"  {bt}: {cnt:,}")

    # Count by context
    by_ctx = defaultdict(int)
    for b in bonds:
        by_ctx[b["context"]] += 1
    print("\nBy context:")
    for ctx, cnt in sorted(by_ctx.items(), key=lambda x: -x[1]):
        print(f"  {ctx}: {cnt:,}")

    # ===== SAVE =====
    print("\n" + "=" * 60)
    print("SAVING DATA")
    print("=" * 60)

    # Save passages
    with open("data/processed/passages.jsonl", "w", encoding="utf-8") as f:
        for p in all_passages:
            # Normalize field names
            p_out = {
                "id": p["id"],
                "text": p["text"],
                "language": p["lang"],
                "source": p["source"],
                "time_period": p["period"],
            }
            f.write(json.dumps(p_out, ensure_ascii=False) + "\n")
    print(f"  Saved {len(all_passages):,} passages to data/processed/passages.jsonl")

    # Save bonds
    with open("data/processed/bonds.jsonl", "w", encoding="utf-8") as f:
        for b in bonds:
            b_out = {
                **b,
                "bond_type": (
                    b["bond_type"].name if hasattr(b["bond_type"], "name") else str(b["bond_type"])
                ),
            }
            f.write(json.dumps(b_out, ensure_ascii=False) + chr(10))
    print(f"  Saved {len(bonds):,} bonds to data/processed/bonds.jsonl")

    # Copy to Drive if enabled
    if USE_DRIVE_DATA and SAVE_DIR:
        try:
            os.makedirs(SAVE_DIR, exist_ok=True)
            shutil.copy("data/processed/passages.jsonl", f"{SAVE_DIR}/passages.jsonl")
            shutil.copy("data/processed/bonds.jsonl", f"{SAVE_DIR}/bonds.jsonl")
            print(f"  Copied to Drive: {SAVE_DIR}")
        except Exception as e:
            print(f"  Drive copy failed: {e}")

    gc.collect()
    print("\nDone!")
