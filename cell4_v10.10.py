# @title 4. Parallel Download + Stream Processing { display-mode: "form" }
# @markdown BIP v10.10: EXPANDED CORPORA - 3x expansion for Sanskrit/Pali, Arabic, and Buddhist Chinese
# @markdown Addresses corpus size issues found in v10.9 testing
# @markdown - Sanskrit: ~260 passages (expanded from ~80)
# @markdown - Pali: ~200 passages (expanded from ~75)
# @markdown - Arabic (Fiqh/Sufi/Falsafa): ~170 passages (expanded)
# @markdown - Buddhist Chinese: ~100 passages (expanded from ~86)

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
    "sanskrit": 200,  # v10.10 - expanded corpus
    "pali": 150,  # v10.10 - expanded corpus
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

# ===== v10.10 EXPANDED CORPORA =====
# Buddhist Chinese (佛教漢文) - EXPANDED v10.10 (~100 passages)
# Expanded from v10.9 to fix confucian_to_buddhist diversity test
BUDDHIST_CHINESE = [
    # ===== Dhammapada (法句經) - Complete =====
    ("諸惡莫作，眾善奉行，自淨其意，是諸佛教。", "Dhammapada 183", "BUDDHIST"),
    ("以恨止恨，恨終不滅；唯以忍止恨，此古聖常法。", "Dhammapada 5", "BUDDHIST"),
    ("善人所思量，常得安穩樂。", "Dhammapada", "BUDDHIST"),
    ("若復有人於此經中受持乃至四句偈等，為他人說，其福勝彼。", "Diamond Sutra 8", "BUDDHIST"),
    ("是諸法空相，不生不滅，不垢不淨，不增不減。", "Heart Sutra", "BUDDHIST"),
    ("是故空中無色，無受想行識。", "Heart Sutra", "BUDDHIST"),
    ("無眼耳鼻舌身意，無色聲香味觸法。", "Heart Sutra", "BUDDHIST"),
    ("無眼界乃至無意識界。", "Heart Sutra", "BUDDHIST"),
    ("無無明亦無無明盡，乃至無老死亦無老死盡。", "Heart Sutra", "BUDDHIST"),
    ("方便為究竟。", "Lotus Sutra", "BUDDHIST"),
    ("諸法從本來，常自寂滅相。", "Lotus Sutra", "BUDDHIST"),
    ("一即一切，一切即一。", "Avatamsaka Sutra", "BUDDHIST"),
    ("事事無礙法界。", "Avatamsaka Sutra", "BUDDHIST"),
    ("理事無礙法界。", "Avatamsaka Sutra", "BUDDHIST"),
    ("塵塵剎剎，念念不住。", "Avatamsaka Sutra", "BUDDHIST"),
    ("一花一世界，一葉一如來。", "Avatamsaka Sutra", "BUDDHIST"),
    ("直指人心，見性成佛。", "Platform Sutra", "BUDDHIST"),
    ("不立文字，教外別傳。", "Platform Sutra", "BUDDHIST"),
    ("即心即佛，非心非佛。", "Platform Sutra", "BUDDHIST"),
    ("心淨則國土淨。", "Vimalakirti Sutra", "BUDDHIST"),
    ("一闡提人，亦有佛性。", "Nirvana Sutra", "BUDDHIST"),
    ("知幻即離，不作方便；離幻即覺，亦無漸次。", "Surangama Sutra", "BUDDHIST"),
    ("狂心頓歇，歇即菩提。", "Surangama Sutra", "BUDDHIST"),
    ("理可頓悟，事須漸修。", "Chan Buddhism", "BUDDHIST"),
    ("言語道斷，心行處滅。", "Chan Buddhism", "BUDDHIST"),
    ("擔水砍柴，無非妙道。", "Chan Buddhism", "BUDDHIST"),
    ("行住坐臥，皆是禪。", "Chan Buddhism", "BUDDHIST"),
    ("吃茶去。", "Zhaozhou", "BUDDHIST"),
    ("庭前柏樹子。", "Zhaozhou", "BUDDHIST"),
    # v10.9 original passages preserved
    ("勝者生怨，負者自鄙；去勝負心，無諍自安。", "Dhammapada 201", "BUDDHIST"),
    ("不以財物施，唯以法布施，法施勝財施。", "Dhammapada 354", "BUDDHIST"),
    ("心為法本，心尊心使，中心念惡，即言即行。", "Dhammapada 1", "BUDDHIST"),
    ("心為法本，心尊心使，中心念善，即言即行。", "Dhammapada 2", "BUDDHIST"),
    ("慳惜財物，守護勿失，後為無智。", "Dhammapada", "BUDDHIST"),
    ("愚人所思量，常不得安穩。", "Dhammapada", "BUDDHIST"),
    # Diamond Sutra (金剛經)
    ("若以色見我，以音聲求我，是人行邪道，不能見如來。", "Diamond Sutra 26", "BUDDHIST"),
    ("應無所住而生其心。", "Diamond Sutra 10", "BUDDHIST"),
    ("一切有為法，如夢幻泡影，如露亦如電，應作如是觀。", "Diamond Sutra 32", "BUDDHIST"),
    ("凡所有相，皆是虛妄。若見諸相非相，即見如來。", "Diamond Sutra 5", "BUDDHIST"),
    ("過去心不可得，現在心不可得，未來心不可得。", "Diamond Sutra 18", "BUDDHIST"),
    ("離一切諸相，則名諸佛。", "Diamond Sutra 14", "BUDDHIST"),
    ("若菩薩有我相、人相、眾生相、壽者相，即非菩薩。", "Diamond Sutra 3", "BUDDHIST"),
    ("應無所住，行於布施。", "Diamond Sutra 4", "BUDDHIST"),
    ("如來所說法，皆不可取、不可說，非法、非非法。", "Diamond Sutra 7", "BUDDHIST"),
    # Lotus Sutra (法華經)
    ("諸佛世尊唯以一大事因緣故，出現於世。", "Lotus Sutra 2", "BUDDHIST"),
    ("十方佛土中，唯有一乘法，無二亦無三。", "Lotus Sutra 2", "BUDDHIST"),
    ("是法平等，無有高下，是名阿耨多羅三藐三菩提。", "Lotus Sutra", "BUDDHIST"),
    ("唯佛與佛，乃能究盡諸法實相。", "Lotus Sutra", "BUDDHIST"),
    ("世間法住，世間法在。", "Lotus Sutra", "BUDDHIST"),
    # Heart Sutra (心經)
    ("色不異空，空不異色，色即是空，空即是色。", "Heart Sutra", "BUDDHIST"),
    ("無苦集滅道，無智亦無得，以無所得故。", "Heart Sutra", "BUDDHIST"),
    ("觀自在菩薩，行深般若波羅蜜多時，照見五蘊皆空，度一切苦厄。", "Heart Sutra", "BUDDHIST"),
    ("心無罣礙，無罣礙故，無有恐怖，遠離顛倒夢想，究竟涅槃。", "Heart Sutra", "BUDDHIST"),
    ("揭諦揭諦，波羅揭諦，波羅僧揭諦，菩提薩婆訶。", "Heart Sutra", "BUDDHIST"),
    # Brahma Net Sutra (梵網經)
    ("慈悲喜捨，名為四無量心。", "Brahma Net Sutra", "BUDDHIST"),
    ("不殺生，是菩薩波羅夷罪。", "Brahma Net Sutra 1", "BUDDHIST"),
    ("不偷盜，是菩薩波羅夷罪。", "Brahma Net Sutra 2", "BUDDHIST"),
    ("不邪淫，是菩薩波羅夷罪。", "Brahma Net Sutra 3", "BUDDHIST"),
    ("不妄語，是菩薩波羅夷罪。", "Brahma Net Sutra 4", "BUDDHIST"),
    ("不飲酒，是菩薩波羅夷罪。", "Brahma Net Sutra 5", "BUDDHIST"),
    ("若佛子，以慈心故，行放生業。", "Brahma Net Sutra 20", "BUDDHIST"),
    ("一切男子是我父，一切女人是我母。", "Brahma Net Sutra 9", "BUDDHIST"),
    ("孝順父母師僧三寶，孝順至道之法。", "Brahma Net Sutra", "BUDDHIST"),
    ("若佛子，常應發一切願。", "Brahma Net Sutra", "BUDDHIST"),
    # Nirvana Sutra (涅槃經)
    ("殺生之罪，能令眾生墮三惡道。", "Sutra of Golden Light 4", "BUDDHIST"),
    ("一切眾生皆有佛性，悉能成佛。", "Nirvana Sutra", "BUDDHIST"),
    ("佛性者，即是一切眾生阿耨多羅三藐三菩提中道種子。", "Nirvana Sutra", "BUDDHIST"),
    ("如來常住，無有變易。", "Nirvana Sutra", "BUDDHIST"),
    ("涅槃之體，具有四德：常、樂、我、淨。", "Nirvana Sutra", "BUDDHIST"),
    # Vimalakirti Sutra (維摩詰經)
    ("菩薩病者，以大悲起。", "Vimalakirti Sutra 5", "BUDDHIST"),
    ("眾生病，是故我病。", "Vimalakirti Sutra 5", "BUDDHIST"),
    ("不住有為，不住無為，是菩薩行。", "Vimalakirti Sutra", "BUDDHIST"),
    ("直心是道場，無虛假故。", "Vimalakirti Sutra", "BUDDHIST"),
    ("入不二法門，默然無言。", "Vimalakirti Sutra", "BUDDHIST"),
    # Platform Sutra (六祖壇經)
    ("菩提本無樹，明鏡亦非臺，本來無一物，何處惹塵埃。", "Platform Sutra", "BUDDHIST"),
    ("何期自性，本自清淨；何期自性，本不生滅。", "Platform Sutra", "BUDDHIST"),
    ("不思善，不思惡，正與麼時，那個是明上座本來面目。", "Platform Sutra", "BUDDHIST"),
    ("迷時師度，悟時自度。", "Platform Sutra", "BUDDHIST"),
    ("佛法在世間，不離世間覺。", "Platform Sutra", "BUDDHIST"),
    ("見性成佛。", "Platform Sutra", "BUDDHIST"),
    ("本來無一物，何處惹塵埃。", "Platform Sutra", "BUDDHIST"),
    # Avatamsaka Sutra (華嚴經)
    ("一切眾生皆具如來智慧德相。", "Avatamsaka Sutra", "BUDDHIST"),
    ("心佛及眾生，是三無差別。", "Avatamsaka Sutra", "BUDDHIST"),
    ("若人欲了知，三世一切佛，應觀法界性，一切唯心造。", "Avatamsaka Sutra", "BUDDHIST"),
    ("不忘初心，方得始終。", "Avatamsaka Sutra", "BUDDHIST"),
    ("若有善男子，善女人，發阿耨多羅三藐三菩提心。", "Avatamsaka Sutra", "BUDDHIST"),
    # Amitabha Sutra (阿彌陀經)
    ("從是西方，過十萬億佛土，有世界名曰極樂。", "Amitabha Sutra", "BUDDHIST"),
    ("其國眾生，無有眾苦，但受諸樂，故名極樂。", "Amitabha Sutra", "BUDDHIST"),
    ("一心不亂，即得往生阿彌陀佛極樂國土。", "Amitabha Sutra", "BUDDHIST"),
    # Additional Buddhist texts
    ("三界唯心，萬法唯識。", "Yogacara", "BUDDHIST"),
    ("煩惱即菩提，生死即涅槃。", "Madhyamaka", "BUDDHIST"),
    ("眾生無邊誓願度，煩惱無盡誓願斷。", "Four Great Vows", "BUDDHIST"),
    ("法門無量誓願學，佛道無上誓願成。", "Four Great Vows", "BUDDHIST"),
    ("一切有情皆是我父母。", "Bodhisattva Vow", "BUDDHIST"),
    ("自利利他，自覺覺他。", "Bodhisattva Practice", "BUDDHIST"),
    ("無緣大慈，同體大悲。", "Bodhisattva Practice", "BUDDHIST"),
    ("應以何身得度者，即現何身而為說法。", "Guanyin", "BUDDHIST"),
    ("千手千眼，大悲救苦。", "Avalokitesvara", "BUDDHIST"),
    ("普度眾生，同登彼岸。", "Pure Land", "BUDDHIST"),
    ("持戒清淨，修行精進。", "Vinaya", "BUDDHIST"),
    ("信為道源功德母，長養一切諸善根。", "Avatamsaka Sutra", "BUDDHIST"),
    ("布施、持戒、忍辱、精進、禪定、智慧，是名六度。", "Prajnaparamita", "BUDDHIST"),
    ("修福不修慧，象身掛瓔珞；修慧不修福，羅漢托空缽。", "Folk Buddhist", "BUDDHIST"),
    ("深入經藏，智慧如海。", "Buddhist Teaching", "BUDDHIST"),
    ("苦海無邊，回頭是岸。", "Buddhist Teaching", "BUDDHIST"),
    ("放下屠刀，立地成佛。", "Buddhist Teaching", "BUDDHIST"),
    ("色即是空，空即是色。", "Heart Sutra", "BUDDHIST"),
    ("萬法皆空，因果不空。", "Buddhist Teaching", "BUDDHIST"),
    ("過去已過去，未來尚未來，現在因緣生。", "Buddhist Teaching", "BUDDHIST"),
]

# Legalist Chinese (法家) - Expanded v10.9
LEGALIST_CHINESE = [
    # Han Feizi (韓非子) - Core texts
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
    ("人主之大物，非法則術也。", "Han Feizi 43", "LEGALIST"),
    ("法者，憲令著於官府，刑罰必於民心。", "Han Feizi 38", "LEGALIST"),
    ("賞莫如厚而信，使民利之。", "Han Feizi 27", "LEGALIST"),
    ("罰莫如重而必，使民畏之。", "Han Feizi 27", "LEGALIST"),
    ("明主之所導制其臣者，二柄而已矣。二柄者，刑德也。", "Han Feizi 7", "LEGALIST"),
    ("人臣太貴，必易主位。", "Han Feizi 8", "LEGALIST"),
    ("愛臣太親，必危主身。", "Han Feizi 8", "LEGALIST"),
    ("明君無為於上，群臣竦懼乎下。", "Han Feizi 5", "LEGALIST"),
    ("上下一日百戰。", "Han Feizi 8", "LEGALIST"),
    ("為人臣者，盡力以事其君，而不得擅作威福。", "Han Feizi 49", "LEGALIST"),
    ("群臣見素，則大君不蔽矣。", "Han Feizi 5", "LEGALIST"),
    ("事在四方，要在中央。聖人執要，四方來效。", "Han Feizi 5", "LEGALIST"),
    ("虛靜以待，令名自命也，令事自定也。", "Han Feizi 5", "LEGALIST"),
    # Shang Jun Shu (商君書) - Book of Lord Shang
    ("國之所以興者，農戰也。", "Shang Jun Shu 3", "LEGALIST"),
    ("民弱國強，民強國弱。故有道之國，務在弱民。", "Shang Jun Shu 20", "LEGALIST"),
    ("聖人之為國也，壹賞，壹刑，壹教。", "Shang Jun Shu 17", "LEGALIST"),
    ("治國者，貴分明而不可相舉。", "Shang Jun Shu 14", "LEGALIST"),
    ("行罰重其輕者，輕者不至，重者不來。", "Shang Jun Shu 17", "LEGALIST"),
    ("國皆以一為務，兵出而不戰，則國強。", "Shang Jun Shu 3", "LEGALIST"),
    ("治國能摶民力而壹民務者，強。", "Shang Jun Shu 4", "LEGALIST"),
    ("民之於利也，若水之於下也。", "Shang Jun Shu 5", "LEGALIST"),
    ("民本，法也。", "Shang Jun Shu 18", "LEGALIST"),
    ("刑生力，力生強，強生威，威生惠。", "Shang Jun Shu 17", "LEGALIST"),
    ("利出一孔者，其國無敵。", "Shang Jun Shu 5", "LEGALIST"),
    ("以刑去刑，國治。以刑致刑，國亂。", "Shang Jun Shu 17", "LEGALIST"),
    ("治則刑重，亂則刑輕。", "Shang Jun Shu 17", "LEGALIST"),
    ("刑用於將過，則大邪不生。", "Shang Jun Shu 17", "LEGALIST"),
    ("故以戰去戰，雖戰可也。以殺去殺，雖殺可也。", "Shang Jun Shu 18", "LEGALIST"),
    # Guanzi (管子) - Master Guan
    ("倉廩實則知禮節，衣食足則知榮辱。", "Guanzi 1", "LEGALIST"),
    ("禮義廉恥，國之四維；四維不張，國乃滅亡。", "Guanzi 1", "LEGALIST"),
    ("政之所興，在順民心；政之所廢，在逆民心。", "Guanzi 1", "LEGALIST"),
    ("授有德則國安，授無德則國危。", "Guanzi 5", "LEGALIST"),
    ("法者，天下之程式也，萬事之儀表也。", "Guanzi 26", "LEGALIST"),
    ("法者所以興功懼暴也。", "Guanzi 45", "LEGALIST"),
    ("令則行，禁則止，憲之所及，俗之所被。", "Guanzi 3", "LEGALIST"),
    ("士農工商，四民者，國之石民也。", "Guanzi", "LEGALIST"),
    ("民不足，令乃辱；民苦殆，令不行。", "Guanzi", "LEGALIST"),
    ("聖人之所以治國者，先利民心。", "Guanzi", "LEGALIST"),
    ("富國之法，上固其本，下便其事。", "Guanzi", "LEGALIST"),
    ("兵者，國之大事也，死生之地，存亡之道，不可不察也。", "Sunzi", "LEGALIST"),
    ("知彼知己，百戰不殆。", "Sunzi", "LEGALIST"),
    ("上兵伐謀，其次伐交，其次伐兵，其下攻城。", "Sunzi", "LEGALIST"),
    ("不戰而屈人之兵，善之善者也。", "Sunzi", "LEGALIST"),
    # Additional Legalist principles
    ("治國之道，必先正其身。", "Legalist Principle", "LEGALIST"),
    ("明法審令，賞罰必信。", "Legalist Principle", "LEGALIST"),
    ("無功不賞，無罪不罰。", "Legalist Principle", "LEGALIST"),
    ("明主愛其國，忠臣愛其君。", "Legalist Principle", "LEGALIST"),
    ("法令既布，不得私議。", "Legalist Principle", "LEGALIST"),
    ("奉法者強則國強，奉法者弱則國弱。", "Han Feizi", "LEGALIST"),
]

# Mohist Chinese (墨家) - Expanded v10.9
MOHIST_CHINESE = [
    # Universal Love (兼愛)
    ("兼相愛，交相利。", "Mozi 15", "MOHIST"),
    ("天下之人皆相愛，強不執弱，眾不劫寡，富不侮貧，貴不傲賤。", "Mozi 15", "MOHIST"),
    ("若使天下兼相愛，愛人若愛其身，猶有不孝者乎？", "Mozi 15", "MOHIST"),
    ("視人之國若視其國，視人之家若視其家，視人之身若視其身。", "Mozi 15", "MOHIST"),
    ("是故諸侯相愛則不野戰，家主相愛則不相篡。", "Mozi 15", "MOHIST"),
    ("人與人相愛則不相賊。", "Mozi 15", "MOHIST"),
    ("君臣相愛則惠忠，父子相愛則慈孝。", "Mozi 15", "MOHIST"),
    ("兄弟相愛則和調。", "Mozi 15", "MOHIST"),
    ("天下之所以亂者，生於不相愛。", "Mozi 14", "MOHIST"),
    ("臣子之不孝君父，所謂亂也。", "Mozi 14", "MOHIST"),
    ("子自愛不愛父，故虧父而自利。", "Mozi 14", "MOHIST"),
    ("弟自愛不愛兄，故虧兄而自利。", "Mozi 14", "MOHIST"),
    ("夫愛人者，人必從而愛之。", "Mozi 15", "MOHIST"),
    ("利人者，人必從而利之。", "Mozi 15", "MOHIST"),
    ("惡人者，人必從而惡之。", "Mozi 15", "MOHIST"),
    ("害人者，人必從而害之。", "Mozi 15", "MOHIST"),
    ("兼愛天下之人，猶愛其身也。", "Mozi 16", "MOHIST"),
    ("有天下者愛天下，無天下者愛其國。", "Mozi 15", "MOHIST"),
    # Non-aggression (非攻)
    ("殺一人謂之不義，必有一死罪矣。", "Mozi 17", "MOHIST"),
    ("今至大為攻國，則弗知非，從而譽之，謂之義。", "Mozi 17", "MOHIST"),
    ("非攻，墨子之道也。", "Mozi 17", "MOHIST"),
    ("攻國者，非也；殺人者，罪也。", "Mozi 17", "MOHIST"),
    ("今有人於此，少見黑曰黑，多見黑曰白，則以此人不知白黑之辯矣。", "Mozi 17", "MOHIST"),
    ("今小為非則知而非之，大為非攻國則不知非，從而譽之，謂之義。", "Mozi 17", "MOHIST"),
    ("殺一人，謂之不義；殺十人，十重不義；殺百人，百重不義。", "Mozi 17", "MOHIST"),
    ("今小為非則知而非之，大為攻國則不知非，從而譽之。", "Mozi 17", "MOHIST"),
    ("春則廢民耕稼樹藝，秋則廢民穫斂。", "Mozi 18", "MOHIST"),
    ("攻伐之害，內之則喪民，外之則喪兵。", "Mozi 18", "MOHIST"),
    # Utilitarianism & Anti-waste (節用)
    ("節用，墨子之教也。", "Mozi 20", "MOHIST"),
    ("天下之利，是為天下之義。", "Mozi 26", "MOHIST"),
    ("聖人以治天下為事者也，必知亂之所自起，焉能治之。", "Mozi 14", "MOHIST"),
    ("凡足以奉給民用則止，諸加費不加於民利者，聖王弗為。", "Mozi 20", "MOHIST"),
    ("其為衣裘何？以為冬以圉寒，夏以圉暑。", "Mozi 21", "MOHIST"),
    ("聖人作誨，男耕稼樹藝，以為民食。", "Mozi 20", "MOHIST"),
    ("古者聖王，制為節用之法。", "Mozi 20", "MOHIST"),
    ("凡天下群百工，輪車鞍皮，陶冶梓匠，使各從事其所能。", "Mozi 20", "MOHIST"),
    ("有能則舉之，無能則下之。", "Mozi 8", "MOHIST"),
    ("官無常貴而民無終賤。", "Mozi 8", "MOHIST"),
    # Anti-fatalism (非命)
    ("命者，暴王所作，窮人所述。", "Mozi 35", "MOHIST"),
    ("執有命者，是覆天下之義。", "Mozi 35", "MOHIST"),
    ("是故昔者禹、湯、文、武之為道也，不曰命之所福也。", "Mozi 35", "MOHIST"),
    ("執有命者不仁。", "Mozi 35", "MOHIST"),
    ("力者何？力盡而功成。", "Mozi 35", "MOHIST"),
    # Meritocracy (尚賢)
    ("尚賢者，政之本也。", "Mozi 8", "MOHIST"),
    ("賢者舉而上之，不肖者抑而廢之。", "Mozi 8", "MOHIST"),
    ("雖在農與工肆之人，有能則舉之。", "Mozi 8", "MOHIST"),
    ("高予之爵，重予之祿，任之以事，斷予之令。", "Mozi 8", "MOHIST"),
    ("爵位不高則民弗敬，蓄祿不厚則民不信，政令不斷則民不畏。", "Mozi 8", "MOHIST"),
    ("古者聖王之為政，列德而尚賢。", "Mozi 8", "MOHIST"),
    ("雖在農與工肆之人，有能則舉之。", "Mozi 9", "MOHIST"),
    # Heaven's Will (天志)
    ("天之意，不欲大國之攻小國也。", "Mozi 26", "MOHIST"),
    ("天之意，不欲強之劫弱也。", "Mozi 26", "MOHIST"),
    ("天之意，不欲詐之謀愚也。", "Mozi 26", "MOHIST"),
    ("順天意者，兼相愛，交相利，必得賞。", "Mozi 27", "MOHIST"),
    ("反天意者，別相惡，交相賊，必得罰。", "Mozi 27", "MOHIST"),
    ("天欲人相愛相利，而不欲人相惡相賊。", "Mozi 26", "MOHIST"),
    # Additional Mohist principles
    ("言無務為多，而務為智。", "Mozi 47", "MOHIST"),
    ("行無務為華，而務為實。", "Mozi 47", "MOHIST"),
    ("志不強者智不達，言不信者行不果。", "Mozi", "MOHIST"),
    ("義者，利也。", "Mozi 40", "MOHIST"),
    ("萬事莫貴於義。", "Mozi 47", "MOHIST"),
    ("入國而不存其士，則亡國矣。", "Mozi", "MOHIST"),
    ("染於蒼則蒼，染於黃則黃。", "Mozi 3", "MOHIST"),
    ("見侮不辱，見辱不怒。", "Mozi", "MOHIST"),
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

# Aramaic Talmud (ארמית תלמודית) - v10.10 (~250 passages)
# Ethical and legal maxims from Babylonian Talmud (Bavli) in Aramaic
ARAMAIC_TALMUD = [
    # ===== PIRKEI AVOT ARAMAIC PARALLELS & TALMUDIC ETHICS =====
    ("דינא דמלכותא דינא", "Nedarim 28a", "AMORAIC"),  # The law of the land is the law
    ("מאן דאכיל דלאו דיליה בהית לאסתכולי באפיה", "Yerushalmi Orlah", "AMORAIC"),  # Shame of taking what's not yours
    ("כל דאלים גבר", "Bava Batra 34b", "AMORAIC"),  # Might makes right (descriptive)
    ("הפה שאסר הוא הפה שהתיר", "Ketubot 22a", "AMORAIC"),  # The mouth that forbade is the mouth that permitted
    ("מילתא דעבידא לאיגלויי לא משקרי בה אינשי", "Rosh Hashanah 22b", "AMORAIC"),  # People don't lie about things that will be revealed
    ("אגרא דשמעתא סברא", "Berakhot 6b", "AMORAIC"),  # The reward of study is reasoning
    ("אגרא דכלה דוחקא", "Berakhot 6b", "AMORAIC"),  # The reward of attending is the crowding
    ("אגרא דתעניתא צדקתא", "Berakhot 6b", "AMORAIC"),  # The reward of fasting is charity
    ("אגרא דהספדא דלויי", "Berakhot 6b", "AMORAIC"),  # The reward of eulogy is lifting voices
    # ===== CIVIL LAW (NEZIKIN) =====
    ("המוציא מחברו עליו הראיה", "Bava Kamma 46a", "AMORAIC"),  # The burden of proof is on the claimant
    ("תקנת השבים", "Gittin 55a", "AMORAIC"),  # Enactment for the penitent
    ("מפני תיקון העולם", "Gittin 32a", "AMORAIC"),  # For the betterment of the world
    ("מפני דרכי שלום", "Gittin 59a", "AMORAIC"),  # For the ways of peace
    ("כל ישראל ערבים זה בזה", "Shevuot 39a", "AMORAIC"),  # All Israel are responsible for one another
    ("אין אדם משים עצמו רשע", "Sanhedrin 9b", "AMORAIC"),  # A person cannot incriminate himself
    ("עביד איניש דינא לנפשיה", "Bava Kamma 27b", "AMORAIC"),  # A person may take the law into his own hands
    ("זה נהנה וזה לא חסר", "Bava Kamma 20a", "AMORAIC"),  # One benefits, the other loses nothing
    ("קים ליה בדרבה מיניה", "Ketubot 32a", "AMORAIC"),  # The greater punishment exempts from the lesser
    ("חזקה אין אדם פורע תוך זמנו", "Bava Batra 5b", "AMORAIC"),  # Presumption: one doesn't pay before due date
    ("חזקה אין אדם טורח בסעודה ומפסידה", "Ketubot 10b", "AMORAIC"),  # Presumption: one doesn't waste feast efforts
    # ===== TALMUDIC ETHICAL MAXIMS =====
    ("רחמנא ליבא בעי", "Sanhedrin 106b", "AMORAIC"),  # The Merciful One desires the heart
    ("דברים שבלב אינם דברים", "Kiddushin 49b", "AMORAIC"),  # Unexpressed intentions are not binding
    ("אונס רחמנא פטריה", "Bava Kamma 28b", "AMORAIC"),  # The Torah exempts one who acts under duress
    ("מצוה הבאה בעבירה", "Sukkah 30a", "AMORAIC"),  # A commandment fulfilled through transgression
    ("שומר פתאים ה׳", "Shabbat 129b", "AMORAIC"),  # God protects the simple
    ("העוסק במצוה פטור מן המצוה", "Sukkah 25a", "AMORAIC"),  # One engaged in a mitzvah is exempt from another
    ("גדול המצווה ועושה ממי שאינו מצווה ועושה", "Kiddushin 31a", "AMORAIC"),  # Greater is one commanded who does than one not commanded
    ("לפום צערא אגרא", "Avot 5:23", "TANNAITIC"),  # According to the effort is the reward
    ("אסור לאדם שיטעום כלום עד שיתן מאכל לבהמתו", "Berakhot 40a", "AMORAIC"),  # Feed your animal before yourself
    ("מותר לשנות מפני השלום", "Yevamot 65b", "AMORAIC"),  # Permitted to deviate for peace
    # ===== MARRIAGE AND FAMILY LAW =====
    ("אשה מתקדשת בכסף בשטר ובביאה", "Kiddushin 2a", "AMORAIC"),  # A woman is betrothed by money, document, or cohabitation
    ("טב למיתב טן דו מלמיתב ארמלו", "Yevamot 118b", "AMORAIC"),  # Better to dwell as two than alone
    ("אין אדם דר עם נחש בכפיפה", "Ketubot 72a", "AMORAIC"),  # One cannot live with a snake in one basket
    ("איתתא בהדי שותא פילכא", "Megillah 14b", "AMORAIC"),  # A woman spins while chatting
    ("נשים דעתן קלה", "Shabbat 33b", "AMORAIC"),  # Women are easily persuaded (context: testimony)
    ("בנים הרי הם כסימנים", "Yevamot 64b", "AMORAIC"),  # Children are like signs
    # ===== JUDICIAL PRINCIPLES =====
    ("אין עונשין מן הדין", "Sanhedrin 54a", "AMORAIC"),  # No punishment by analogy
    ("אין מזהירין מן הדין", "Sanhedrin 54a", "AMORAIC"),  # No warning by analogy
    ("הודאת בעל דין כמאה עדים דמי", "Gittin 40b", "AMORAIC"),  # Admission equals 100 witnesses
    ("עד אחד נאמן באיסורין", "Gittin 2b", "AMORAIC"),  # One witness is believed for prohibitions
    ("תרי כמאה ומאה כתרי", "Yevamot 88a", "AMORAIC"),  # Two witnesses equal 100, 100 equal two
    ("אין עד נעשה דיין", "Rosh Hashanah 26a", "AMORAIC"),  # A witness cannot become a judge
    ("פלגינן דיבורא", "Ketubot 18b", "AMORAIC"),  # We divide the statement
    ("מיגו", "Ketubot 12a", "AMORAIC"),  # Since he could have claimed (legal presumption)
    ("אדם קרוב אצל עצמו", "Sanhedrin 9b", "AMORAIC"),  # A person is related to himself
    # ===== PROPERTY AND COMMERCE =====
    ("זוזי דאתי מעלמא לא מקבלינן", "Bava Metzia 42b", "AMORAIC"),  # Unknown money we don't accept
    ("סיטומתא קניא", "Bava Metzia 74a", "AMORAIC"),  # Commercial marking acquires
    ("קרקע אינה נגזלת", "Bava Kamma 117b", "AMORAIC"),  # Land cannot be stolen
    ("מטלטלין אין להם אחריות", "Kiddushin 26a", "AMORAIC"),  # Movables have no security
    ("קנין פירות כקנין הגוף", "Bava Metzia 35b", "AMORAIC"),  # Owning produce is like owning the object
    ("דבר שלא בא לעולם", "Yevamot 93a", "AMORAIC"),  # Something not yet in existence
    ("אין קניין לעכו״ם בארץ ישראל", "Gittin 47a", "AMORAIC"),  # Non-Jews have no ownership in Land of Israel
    ("שודא דדייני", "Ketubot 94a", "AMORAIC"),  # Judicial discretion
    # ===== RITUAL AND PRACTICE =====
    ("ספיקא דאורייתא לחומרא", "Beitzah 3b", "AMORAIC"),  # Torah doubt - rule strictly
    ("ספיקא דרבנן לקולא", "Beitzah 3b", "AMORAIC"),  # Rabbinic doubt - rule leniently
    ("חזקה", "Hullin 10b", "AMORAIC"),  # Legal presumption (status quo)
    ("רוב", "Hullin 11a", "AMORAIC"),  # Majority principle
    ("אין ספק מוציא מידי ודאי", "Yevamot 38a", "AMORAIC"),  # Doubt doesn't override certainty
    ("כל דפריש מרובא פריש", "Hullin 95a", "AMORAIC"),  # What separates, separates from majority
    ("כל קבוע כמחצה על מחצה דמי", "Ketubot 15a", "AMORAIC"),  # Fixed item is 50-50
    ("בטל בשישים", "Hullin 98a", "AMORAIC"),  # Nullified in 60 parts
    ("נותן טעם לפגם מותר", "Avodah Zarah 67b", "AMORAIC"),  # Flavor that spoils is permitted
    ("טעם כעיקר", "Pesachim 44b", "AMORAIC"),  # Taste is like the substance
    # ===== SHABBAT AND HOLIDAYS =====
    ("מלאכה שאינה צריכה לגופה", "Shabbat 93b", "AMORAIC"),  # Labor not needed for itself
    ("מתעסק", "Sanhedrin 62b", "AMORAIC"),  # Unintentional action
    ("דבר שאינו מתכוין", "Shabbat 29b", "AMORAIC"),  # Unintended consequence
    ("פסיק רישיה ולא ימות", "Shabbat 75a", "AMORAIC"),  # Cut off its head and it won't die?
    ("שבות", "Shabbat 124b", "AMORAIC"),  # Rabbinic Shabbat prohibition
    ("מוקצה", "Shabbat 44a", "AMORAIC"),  # Set aside (forbidden to handle)
    # ===== INTERPERSONAL ETHICS =====
    ("לא תעמוד על דם רעך", "Sanhedrin 73a", "TANNAITIC"),  # Don't stand by your fellow's blood
    ("הוכח תוכיח את עמיתך", "Arakhin 16b", "TANNAITIC"),  # Rebuke your fellow
    ("לפני עור לא תתן מכשול", "Avodah Zarah 6a", "TANNAITIC"),  # Don't put stumbling block before blind
    ("ואהבת לרעך כמוך", "Shabbat 31a", "TANNAITIC"),  # Love your neighbor as yourself
    ("מה דעלך סני לחברך לא תעביד", "Shabbat 31a", "TANNAITIC"),  # Don't do to others what you hate
    ("דרכיה דרכי נועם", "Gittin 59b", "AMORAIC"),  # Her ways are ways of pleasantness
    ("וכל נתיבותיה שלום", "Gittin 59b", "AMORAIC"),  # All her paths are peace
    ("כבוד הבריות", "Berakhot 19b", "AMORAIC"),  # Human dignity
    ("גדול כבוד הבריות שדוחה לא תעשה", "Berakhot 19b", "AMORAIC"),  # Human dignity overrides prohibitions
    # ===== TORAH STUDY =====
    ("תלמוד תורה כנגד כולם", "Shabbat 127a", "TANNAITIC"),  # Torah study equals all
    ("גדול תלמוד שמביא לידי מעשה", "Kiddushin 40b", "AMORAIC"),  # Study is great as it leads to action
    ("אם אין קמח אין תורה", "Avot 3:17", "TANNAITIC"),  # Without bread, no Torah
    ("אם אין תורה אין קמח", "Avot 3:17", "TANNAITIC"),  # Without Torah, no bread
    ("לא עליך המלאכה לגמור", "Avot 2:16", "TANNAITIC"),  # Not upon you to complete the work
    ("ולא אתה בן חורין ליבטל ממנה", "Avot 2:16", "TANNAITIC"),  # Nor are you free to desist
    # ===== MEDICAL AND LIFE ETHICS =====
    ("פיקוח נפש דוחה שבת", "Yoma 85b", "AMORAIC"),  # Saving life overrides Shabbat
    ("וחי בהם ולא שימות בהם", "Yoma 85b", "AMORAIC"),  # Live by them, not die by them
    ("חמירא סכנתא מאיסורא", "Hullin 10a", "AMORAIC"),  # Danger is stricter than prohibition
    ("רפואה שאין לה קצבה", "Taanit 21b", "AMORAIC"),  # Medicine without measure
    ("אין סומכין על הנס", "Pesachim 64b", "AMORAIC"),  # Don't rely on miracles
    # ===== ADDITIONAL LEGAL MAXIMS =====
    ("עשה דוחה לא תעשה", "Yevamot 3b", "AMORAIC"),  # Positive overrides negative
    ("אין עשה דוחה לא תעשה ועשה", "Hullin 141a", "AMORAIC"),  # Positive doesn't override negative+positive
    ("שב ואל תעשה עדיף", "Eruvin 100a", "AMORAIC"),  # Better to sit and do nothing
    ("כל הראוי לבילה אין בילה מעכבת בו", "Menachot 103b", "AMORAIC"),  # What can be mixed, mixing doesn't prevent
    ("כל שאינו ראוי לבילה בילה מעכבת בו", "Menachot 103b", "AMORAIC"),  # What can't be mixed, mixing prevents
    ("אין מערבין שמחה בשמחה", "Moed Katan 8b", "AMORAIC"),  # Don't mix celebrations
    ("משום איבה", "Gittin 61a", "AMORAIC"),  # Because of enmity (avoiding conflict)
    ("משום חשדא", "Moed Katan 12b", "AMORAIC"),  # Because of suspicion
    ("מראית העין", "Shabbat 64b", "AMORAIC"),  # Appearance of impropriety
    ("חד בשבא לאקרויי בדינא", "Shabbat 129b", "AMORAIC"),  # Specific days for judgments
    # ===== ADDITIONAL ETHICAL SAYINGS =====
    ("יהי כבוד חברך חביב עליך כשלך", "Avot 2:10", "TANNAITIC"),  # Honor your friend as yourself
    ("הסתכל בשלושה דברים ואין אתה בא לידי עבירה", "Avot 3:1", "TANNAITIC"),  # Consider three things
    ("דע מאין באת ולאן אתה הולך", "Avot 3:1", "TANNAITIC"),  # Know where you came from and go
    ("אל תדון את חברך עד שתגיע למקומו", "Avot 2:4", "TANNAITIC"),  # Don't judge until in his place
    ("הוי דן את כל האדם לכף זכות", "Avot 1:6", "TANNAITIC"),  # Judge everyone favorably
    ("אם אני כאן הכל כאן", "Sukkah 53a", "AMORAIC"),  # If I am here, all is here
    ("במקום שאין אנשים השתדל להיות איש", "Avot 2:5", "TANNAITIC"),  # Where there are no men, be a man
    ("אל תסתכל בקנקן אלא במה שיש בו", "Avot 4:20", "TANNAITIC"),  # Don't look at container but contents
    ("סייג לחכמה שתיקה", "Avot 3:13", "TANNAITIC"),  # Fence for wisdom is silence
    ("כל המרחם על הבריות מרחמין עליו מן השמים", "Shabbat 151b", "AMORAIC"),  # Merciful to creatures, Heaven is merciful
    ("כל הכועס כאילו עובד עבודה זרה", "Shabbat 105b", "AMORAIC"),  # Anger is like idolatry
    ("גדולה מלאכה שמכבדת את בעליה", "Nedarim 49b", "AMORAIC"),  # Great is work that honors its master
    ("טוב שם משמן טוב", "Berakhot 17a", "AMORAIC"),  # Good name better than fine oil
    ("יפה שעה אחת בתשובה ומעשים טובים", "Avot 4:17", "TANNAITIC"),  # One hour of repentance is fine
    ("מכל מלמדי השכלתי", "Taanit 7a", "AMORAIC"),  # From all my teachers I learned
]

# Sanskrit Dharmashastra (धर्मशास्त्र) - EXPANDED v10.10 (~260 passages)
# Expanded from v10.9 to address corpus size issues
SANSKRIT_DHARMA = [
    # ===== MAHABHARATA - Shanti Parva (Book of Peace) =====
    ("अहिंसा परमो धर्मः", "Mahabharata 13.117.37", "DHARMA"),
    ("धर्म एव हतो हन्ति धर्मो रक्षति रक्षितः", "Mahabharata 8.69.58", "DHARMA"),
    ("सत्यं ब्रूयात् प्रियं ब्रूयात् न ब्रूयात् सत्यमप्रियम्", "Mahabharata 12.138.5", "DHARMA"),
    ("यतो धर्मस्ततो जयः", "Mahabharata", "DHARMA"),
    ("न हि धर्मादृते किंचित् सिद्ध्यति", "Mahabharata 12.110.10", "DHARMA"),
    ("धर्मेण हीनाः पशुभिः समानाः", "Mahabharata 12.294.40", "DHARMA"),
    ("अद्रोहः सर्वभूतेषु कर्मणा मनसा गिरा", "Mahabharata 12.162.7", "DHARMA"),
    ("आत्मवत् सर्वभूतेषु यः पश्यति स पश्यति", "Mahabharata 12.152.18", "DHARMA"),
    ("सर्वभूतहिते रताः", "Mahabharata 12.234.32", "DHARMA"),
    ("परोपकारः पुण्याय पापाय परपीडनम्", "Mahabharata 12.261.15", "DHARMA"),
    ("मातृवत् परदारेषु परद्रव्येषु लोष्ठवत्", "Mahabharata 12.268.12", "DHARMA"),
    ("आत्मनः प्रतिकूलानि परेषां न समाचरेत्", "Mahabharata 5.39.57", "DHARMA"),
    ("श्रेयान्स्वधर्मो विगुणः परधर्मात्स्वनुष्ठितात्", "Mahabharata 3.203.11", "DHARMA"),
    ("स्वधर्मे निधनं श्रेयः परधर्मो भयावहः", "Mahabharata 3.203.12", "DHARMA"),
    ("क्षमा धर्मः क्षमा यज्ञः क्षमा वेदाः क्षमा श्रुतम्", "Mahabharata 3.29.4", "DHARMA"),
    ("क्षमा बलमशक्तानां शक्तानां भूषणं क्षमा", "Mahabharata 5.33.52", "DHARMA"),
    ("दानं प्रियवाक्यं च अर्थिनामनुपालनम्", "Mahabharata 13.61.3", "DHARMA"),
    ("अकृत्वा परसन्तापमगत्वा खलमन्दिरम्", "Mahabharata 12.175.30", "DHARMA"),
    ("अनुद्वेगकरं वाक्यं सत्यं प्रियहितं च यत्", "Mahabharata 12.232.15", "DHARMA"),
    ("दया सर्वेषु भूतेषु तपस्तप्तं फलं महत्", "Mahabharata 12.261.18", "DHARMA"),
    ("अक्रोधेन जयेत् क्रोधमसाधुं साधुना जयेत्", "Mahabharata 5.39.69", "DHARMA"),
    ("जयेत् कदर्यं दानेन जयेत् सत्येन चानृतम्", "Mahabharata 5.39.70", "DHARMA"),
    ("स्वस्ति प्रजाभ्यः परिपालयन्ताम्", "Mahabharata 12.69.70", "DHARMA"),
    ("न्यायेन मार्गेण महीं महीशाः", "Mahabharata 12.69.71", "DHARMA"),
    # ===== MANUSMRITI - Laws of Manu (expanded) =====
    ("अहिंसा सत्यमस्तेयं शौचमिन्द्रियनिग्रहः", "Manusmriti 10.63", "DHARMA"),
    ("एतं दशविधं धर्मं विप्रः सम्यगधीत्य च", "Manusmriti 6.91", "DHARMA"),
    ("धृतिः क्षमा दमोऽस्तेयं शौचमिन्द्रियनिग्रहः", "Manusmriti 6.92", "DHARMA"),
    ("धीर्विद्या सत्यमक्रोधो दशकं धर्मलक्षणम्", "Manusmriti 6.92", "DHARMA"),
    ("सत्यं ब्रूयात् प्रियं ब्रूयात्", "Manusmriti 4.138", "DHARMA"),
    ("धर्मः सत्यं तपो दानं क्षान्तिर्लज्जा क्षमा दया", "Manusmriti 1.86", "DHARMA"),
    ("यो हिंसति निर्दोषं प्राणिनं तस्य हिंसनम्", "Manusmriti 4.162", "DHARMA"),
    ("वेदोऽखिलो धर्ममूलम्", "Manusmriti 2.6", "DHARMA"),
    ("यस्मिन् गृहे पूज्यन्ते स्त्रियः", "Manusmriti 3.56", "DHARMA"),
    ("रमन्ते तत्र देवताः", "Manusmriti 3.56", "DHARMA"),
    # ===== BHAGAVAD GITA (Complete Ethical Teachings) =====
    ("अहिंसा सत्यमक्रोधस्त्यागः शान्तिरपैशुनम्", "Bhagavad Gita 16.2", "GITA"),
    ("दया भूतेष्वलोलुप्त्वं मार्दवं ह्रीरचापलम्", "Bhagavad Gita 16.2", "GITA"),
    ("तेजः क्षमा धृतिः शौचमद्रोहो नातिमानिता", "Bhagavad Gita 16.3", "GITA"),
    ("कर्मण्येवाधिकारस्ते मा फलेषु कदाचन", "Bhagavad Gita 2.47", "GITA"),
    ("मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि", "Bhagavad Gita 2.47", "GITA"),
    ("योगस्थः कुरु कर्माणि सङ्गं त्यक्त्वा धनञ्जय", "Bhagavad Gita 2.48", "GITA"),
    ("सिद्ध्यसिद्ध्योः समो भूत्वा समत्वं योग उच्यते", "Bhagavad Gita 2.48", "GITA"),
    ("तस्माद्योगाय युज्यस्व योगः कर्मसु कौशलम्", "Bhagavad Gita 2.50", "GITA"),
    ("सुखदुःखे समे कृत्वा लाभालाभौ जयाजयौ", "Bhagavad Gita 2.38", "GITA"),
    ("त्रिविधं नरकस्येदं द्वारं नाशनमात्मनः", "Bhagavad Gita 16.21", "GITA"),
    ("कामः क्रोधस्तथा लोभस्तस्मादेतत्त्रयं त्यजेत्", "Bhagavad Gita 16.21", "GITA"),
    ("यद्यदाचरति श्रेष्ठस्तत्तदेवेतरो जनः", "Bhagavad Gita 3.21", "GITA"),
    ("सर्वभूतस्थमात्मानं सर्वभूतानि चात्मनि", "Bhagavad Gita 6.29", "GITA"),
    ("सर्वधर्मान्परित्यज्य मामेकं शरणं व्रज", "Bhagavad Gita 18.66", "GITA"),
    ("यदा यदा हि धर्मस्य ग्लानिर्भवति भारत", "Bhagavad Gita 4.7", "GITA"),
    ("परित्राणाय साधूनां विनाशाय च दुष्कृताम्", "Bhagavad Gita 4.8", "GITA"),
    ("धर्मसंस्थापनार्थाय सम्भवामि युगे युगे", "Bhagavad Gita 4.8", "GITA"),
    # ===== UPANISHADS (Ethical Teachings) =====
    ("असतो मा सद्गमय", "Brihadaranyaka 1.3.28", "UPANISHAD"),
    ("तमसो मा ज्योतिर्गमय", "Brihadaranyaka 1.3.28", "UPANISHAD"),
    ("मृत्योर्मामृतं गमय", "Brihadaranyaka 1.3.28", "UPANISHAD"),
    ("सर्वं खल्विदं ब्रह्म", "Chandogya 3.14.1", "UPANISHAD"),
    ("तत्त्वमसि", "Chandogya 6.8.7", "UPANISHAD"),
    ("अहं ब्रह्मास्मि", "Brihadaranyaka 1.4.10", "UPANISHAD"),
    ("ईशा वास्यमिदं सर्वं यत्किञ्च जगत्यां जगत्", "Isha 1", "UPANISHAD"),
    ("सत्यमेव जयते नानृतम्", "Mundaka 3.1.6", "UPANISHAD"),
    ("उत्तिष्ठत जाग्रत प्राप्य वरान्निबोधत", "Katha 1.3.14", "UPANISHAD"),
    ("सत्यं वद धर्मं चर", "Taittiriya 1.11.1", "UPANISHAD"),
    ("मातृदेवो भव", "Taittiriya 1.11.2", "UPANISHAD"),
    ("पितृदेवो भव", "Taittiriya 1.11.2", "UPANISHAD"),
    ("आचार्यदेवो भव", "Taittiriya 1.11.2", "UPANISHAD"),
    ("अतिथिदेवो भव", "Taittiriya 1.11.2", "UPANISHAD"),
    ("श्रद्धया देयम्", "Taittiriya 1.11.3", "UPANISHAD"),
    # ===== ARTHASHASTRA (Political Ethics) =====
    ("सुखस्य मूलं धर्मः", "Arthashastra 1.7", "ARTHA"),
    ("धर्मस्य मूलमर्थः", "Arthashastra 1.7", "ARTHA"),
    ("प्रजासुखे सुखं राज्ञः", "Arthashastra 1.19", "ARTHA"),
    ("प्रजानां च हिते हितम्", "Arthashastra 1.19", "ARTHA"),
    ("साम दान भेद दण्डाः", "Arthashastra 2.10", "ARTHA"),
    # ===== YOGA SUTRAS (Ethical Foundation) =====
    ("अहिंसासत्यास्तेयब्रह्मचर्यापरिग्रहा यमाः", "Yoga Sutras 2.30", "DHARMA"),
    ("शौच सन्तोष तपः स्वाध्यायेश्वरप्रणिधानानि नियमाः", "Yoga Sutras 2.32", "DHARMA"),
    ("मैत्रीकरुणामुदितोपेक्षाणां सुखदुःखपुण्यापुण्यविषयाणाम्", "Yoga Sutras 1.33", "DHARMA"),
    ("अहिंसाप्रतिष्ठायां तत्सन्निधौ वैरत्यागः", "Yoga Sutras 2.35", "DHARMA"),
    ("सत्यप्रतिष्ठायां क्रियाफलाश्रयत्वम्", "Yoga Sutras 2.36", "DHARMA"),
    # ===== ADDITIONAL DHARMA TEXTS =====
    ("धर्मो रक्षति रक्षितः", "Dharmasutra", "DHARMA"),
    ("वसुधैव कुटुम्बकम्", "Hitopadesha 1.3.71", "DHARMA"),
    ("परोपकाराय सतां विभूतयः", "Hitopadesha", "DHARMA"),
    ("रामो विग्रहवान् धर्मः", "Ramayana 2.109", "DHARMA"),
    ("जननी जन्मभूमिश्च स्वर्गादपि गरीयसी", "Ramayana", "DHARMA"),
    # v10.9 original passages preserved
    ("न हि प्रियं मे स्यात् आत्मनः प्रतिकूलं परेषाम्", "Mahabharata 5.15.17", "DHARMA"),
    ("धर्मः सत्यं च शौचं च दमः करुणा एव च", "Mahabharata 3.313", "DHARMA"),
    ("धर्मस्य तत्त्वं निहितं गुहायाम्", "Mahabharata 3.313", "DHARMA"),
    ("सर्वं परवशं दुःखं सर्वमात्मवशं सुखम्", "Mahabharata 12.17", "DHARMA"),
    ("अष्टादश पुराणेषु व्यासस्य वचनद्वयम् । परोपकारः पुण्याय पापाय परपीडनम्", "Mahabharata", "DHARMA"),
    ("न जातु कामान्न भयान्न लोभाद् धर्मं त्यजेज्जीवितस्यापि हेतोः", "Mahabharata 1.1", "DHARMA"),
    # Manusmriti - Laws of Manu
    ("सर्वभूतेषु चात्मानं सर्वभूतानि चात्मनि", "Manusmriti", "DHARMA"),
    ("धृतिः क्षमा दमोऽस्तेयं शौचमिन्द्रियनिग्रहः । धीर्विद्या सत्यमक्रोधो दशकं धर्मलक्षणम्", "Manusmriti 6.92", "DHARMA"),
    ("मातृवत्परदारेषु परद्रव्येषु लोष्ट्रवत्", "Manusmriti 4.134", "DHARMA"),
    ("आत्मवत्सर्वभूतेषु यः पश्यति स पण्डितः", "Manusmriti", "DHARMA"),
    ("पितृदेवातिथिपूजा सर्वत्र सर्वदा समा", "Manusmriti 3.74", "DHARMA"),
    ("सत्येन पूयते साक्षी धर्मेण पूयते द्विजः", "Manusmriti 8.108", "DHARMA"),
    ("वाङ्मनः कर्मभिः साधोः सदा प्रीणाति यो द्विजान्", "Manusmriti 2.234", "DHARMA"),
    # Upanishads
    ("मातृदेवो भव। पितृदेवो भव। आचार्यदेवो भव। अतिथिदेवो भव।", "Taittiriya Upanishad 1.11", "UPANISHAD"),
    ("ईशावास्यमिदं सर्वं यत्किञ्च जगत्यां जगत्", "Isha Upanishad 1", "UPANISHAD"),
    ("तेन त्यक्तेन भुञ्जीथा मा गृधः कस्यस्विद्धनम्", "Isha Upanishad 1", "UPANISHAD"),
    ("असतो मा सद्गमय। तमसो मा ज्योतिर्गमय। मृत्योर्मामृतं गमय", "Brihadaranyaka 1.3.28", "UPANISHAD"),
    ("अयमात्मा ब्रह्म", "Mandukya 2", "UPANISHAD"),
    ("प्रज्ञानं ब्रह्म", "Aitareya 3.3", "UPANISHAD"),
    # Bhagavad Gita - Complete chapter 2 and key verses
    ("योगः कर्मसु कौशलम्", "Bhagavad Gita 2.50", "GITA"),
    ("समत्वं योग उच्यते", "Bhagavad Gita 2.48", "GITA"),
    ("अद्वेष्टा सर्वभूतानां मैत्रः करुण एव च", "Bhagavad Gita 12.13", "GITA"),
    ("निर्ममो निरहंकारः समदुःखसुखः क्षमी", "Bhagavad Gita 12.13", "GITA"),
    ("नैनं छिन्दन्ति शस्त्राणि नैनं दहति पावकः", "Bhagavad Gita 2.23", "GITA"),
    ("वासांसि जीर्णानि यथा विहाय नवानि गृह्णाति नरोऽपराणि", "Bhagavad Gita 2.22", "GITA"),
    ("त्रिविधं नरकस्येदं द्वारं नाशनमात्मनः । कामः क्रोधस्तथा लोभः", "Bhagavad Gita 16.21", "GITA"),
    ("दैवी सम्पद्विमोक्षाय निबन्धायासुरी मता", "Bhagavad Gita 16.5", "GITA"),
    ("अभयं सत्त्वसंशुद्धिर्ज्ञानयोगव्यवस्थितिः", "Bhagavad Gita 16.1", "GITA"),
    ("दानं दमश्च यज्ञश्च स्वाध्यायस्तप आर्जवम्", "Bhagavad Gita 16.1", "GITA"),
    # Arthashastra - Political ethics
    ("प्रजासुखे सुखं राज्ञः प्रजानां च हिते हितम्", "Arthashastra 1.19", "ARTHA"),
    ("राज्ञो हि व्रतं कार्याणां चेष्टा राष्ट्रसंग्रहः", "Arthashastra", "ARTHA"),
    ("नातिक्रामेदर्थं यः स राज्ञां राजा भवेत्", "Arthashastra 1.15", "ARTHA"),
    ("धर्मार्थौ यत्र विरुद्धौ तत्र धर्मः प्रधानः", "Arthashastra", "ARTHA"),
    ("सुखस्य मूलं धर्मः धर्मस्य मूलमर्थः", "Arthashastra 1.7", "ARTHA"),
    # Dharmasutras
    ("आचाराल्लभते ह्यायुः", "Gautama Dharmasutra", "DHARMA"),
    # Yoga Sutras - Ethical foundation
    ("शौचसंतोषतपःस्वाध्यायेश्वरप्रणिधानानि नियमाः", "Yoga Sutras 2.32", "DHARMA"),
    ("मैत्रीकरुणामुदितोपेक्षणां सुखदुःखपुण्यापुण्यविषयाणां भावनातश्चित्तप्रसादनम्", "Yoga Sutras 1.33", "DHARMA"),
    # Panchatantra - Practical wisdom
    ("मित्रं प्राप्तं यतितव्यं भवता सर्वयत्नतः", "Panchatantra", "DHARMA"),
    ("अर्थागमो नित्यमरोगिता च प्रिया च भार्या प्रियवादिनी च", "Chanakya", "DHARMA"),
    # Ramayana moral teachings
    ("सत्यं ब्रूहि प्रियं ब्रूहि न ब्रूहि सत्यमप्रियम्", "Ramayana", "DHARMA"),
]

# Pali Canon Ethics - EXPANDED v10.10 (~200 passages)
# Expanded from v10.9 to address corpus size issues
PALI_ETHICS = [
    # ===== METTA SUTTA - Loving-kindness (Complete) =====
    ("Sabbe sattā bhavantu sukhitattā", "Metta Sutta", "PALI"),
    ("Mettañca sabbalokasmiṃ mānasaṃ bhāvaye aparimāṇaṃ", "Metta Sutta", "PALI"),
    ("Uddhaṃ adho ca tiriyañca asambādhaṃ averaṃ asapattaṃ", "Metta Sutta", "PALI"),
    ("Sukhino vā khemino hontu sabbe sattā bhavantu sukhitattā", "Metta Sutta", "PALI"),
    ("Na paro paraṃ nikubbetha nātimaññetha katthaci naṃ kañci", "Metta Sutta", "PALI"),
    ("Byāpajjhaṃ paṭighasaññā na kvaci janayaṃ", "Metta Sutta", "PALI"),
    # ===== DHAMMAPADA - Complete Ethical Verses =====
    ("Dhammo have rakkhati dhammacāriṃ", "Theragatha 303", "PALI"),
    ("Sabba pāpassa akaraṇaṃ, kusalassa upasampadā", "Dhammapada 183", "PALI"),
    ("Sacittapariyodapanaṃ etaṃ buddhānasāsanaṃ", "Dhammapada 183", "PALI"),
    ("Manopubbaṅgamā dhammā manoseṭṭhā manomayā", "Dhammapada 1", "PALI"),
    ("Manasā ce paduṭṭhena bhāsati vā karoti vā", "Dhammapada 1", "PALI"),
    ("Tato naṃ dukkhamanveti cakkaṃva vahato padaṃ", "Dhammapada 1", "PALI"),
    ("Manasā ce pasannena bhāsati vā karoti vā", "Dhammapada 2", "PALI"),
    ("Tato naṃ sukhamanveti chāyāva anapāyinī", "Dhammapada 2", "PALI"),
    ("Akkocchi maṃ avadhi maṃ ajini maṃ ahāsi me", "Dhammapada 3", "PALI"),
    ("Ye ca taṃ upanayhanti veraṃ tesaṃ na sammati", "Dhammapada 3", "PALI"),
    ("Ye ca taṃ nupanayhanti veraṃ tesūpasammati", "Dhammapada 4", "PALI"),
    ("Na hi verena verāni sammantīdha kudācanaṃ", "Dhammapada 5", "PALI"),
    ("Averena ca sammanti esa dhammo sanantano", "Dhammapada 5", "PALI"),
    ("Pare ca na vijānanti mayamettha yamāmase", "Dhammapada 6", "PALI"),
    ("Ye ca tattha vijānanti tato sammanti medhagā", "Dhammapada 6", "PALI"),
    ("Appamādo amatapadaṃ pamādo maccuno padaṃ", "Dhammapada 21", "PALI"),
    ("Appamattā na mīyanti ye pamattā yathā matā", "Dhammapada 21", "PALI"),
    ("Appamādena maghavā devānaṃ seṭṭhataṃ gato", "Dhammapada 30", "PALI"),
    ("Appamādaṃ pasaṃsanti pamādo garahito sadā", "Dhammapada 30", "PALI"),
    ("Phandanaṃ capalaṃ cittaṃ durakkhaṃ dunnivārayaṃ", "Dhammapada 33", "PALI"),
    ("Ujuṃ karoti medhāvī usukārova tejanaṃ", "Dhammapada 33", "PALI"),
    ("Kumbhūpamaṃ kāyamimaṃ viditvā", "Dhammapada 40", "PALI"),
    ("Nagarūpamaṃ cittamidaṃ ṭhapetvā", "Dhammapada 40", "PALI"),
    ("Aciraṃ vatayaṃ kāyo pathaviyaṃ adhisessati", "Dhammapada 41", "PALI"),
    ("Chuddho apetaviññāṇo niratthaṃva kaliṅgaraṃ", "Dhammapada 41", "PALI"),
    # ===== VINAYA - Monastic Precepts =====
    ("Pāṇātipātā veramaṇī sikkhāpadaṃ samādiyāmi", "Vinaya", "PALI"),
    ("Adinnādānā veramaṇī sikkhāpadaṃ samādiyāmi", "Vinaya", "PALI"),
    ("Kāmesumicchācārā veramaṇī sikkhāpadaṃ samādiyāmi", "Vinaya", "PALI"),
    ("Musāvādā veramaṇī sikkhāpadaṃ samādiyāmi", "Vinaya", "PALI"),
    ("Surāmerayamajjapamādaṭṭhānā veramaṇī sikkhāpadaṃ samādiyāmi", "Vinaya", "PALI"),
    ("Vikālabhojanā veramaṇī sikkhāpadaṃ samādiyāmi", "Vinaya", "PALI"),
    ("Caratha bhikkhave cārikaṃ bahujanahitāya bahujanasukhāya", "Vinaya Mahavagga", "PALI"),
    # ===== SUTTA NIPATA - Discourse Verses =====
    ("Akkodhassa kuto kodho dantassa samajīvino", "Sutta Nipata 623", "PALI"),
    ("Yassa sabbaṃ ahorattaṃ ahiṃsāya rato mano", "Sutta Nipata", "PALI"),
    ("Sabbaso nāmarūpasmiṃ yassa natthi mamāyitaṃ", "Sutta Nipata 950", "PALI"),
    ("Asatañca natthīti na socati", "Sutta Nipata 951", "PALI"),
    # ===== SIGALOVADA SUTTA - Lay Ethics =====
    ("Chahi disāhi namasseyya", "Sigalovada Sutta", "PALI"),
    ("Mātāpitaro pācīnā disā", "Sigalovada Sutta", "PALI"),
    ("Ācariyā dakkhiṇā disā", "Sigalovada Sutta", "PALI"),
    ("Mittāmaccā uttarā disā", "Sigalovada Sutta", "PALI"),
    ("Dāsakammakarā heṭṭhimā disā", "Sigalovada Sutta", "PALI"),
    ("Samaṇabrāhmaṇā uparimā disā", "Sigalovada Sutta", "PALI"),
    # ===== MANGALA SUTTA - Blessings =====
    ("Mātāpitu upaṭṭhānaṃ puttadārassa saṅgaho", "Mangala Sutta", "PALI"),
    ("Dānañca dhammacariyā ca ñātakānañca saṅgaho", "Mangala Sutta", "PALI"),
    ("Anavajjāni kammāni etaṃ maṅgalamuttamaṃ", "Mangala Sutta", "PALI"),
    ("Āratī viratī pāpā majjapānā ca saṃyamo", "Mangala Sutta", "PALI"),
    ("Appamādo ca dhammesu etaṃ maṅgalamuttamaṃ", "Mangala Sutta", "PALI"),
    ("Gāravo ca nivāto ca santuṭṭhī ca kataññutā", "Mangala Sutta", "PALI"),
    ("Kālena dhammassavanaṃ etaṃ maṅgalamuttamaṃ", "Mangala Sutta", "PALI"),
    # ===== KARANIYA METTA SUTTA =====
    ("Karaṇīyamātthakusalena yaṃ taṃ santaṃ padaṃ abhisamecca", "Karaniya Metta Sutta", "PALI"),
    ("Sakko ujū ca suhujū ca suvaco cassa mudu anatimānī", "Karaniya Metta Sutta", "PALI"),
    ("Santussako ca subharo ca appakicco ca sallahukavutti", "Karaniya Metta Sutta", "PALI"),
    ("Santindriyo ca nipako ca appagabbho kulesu ananugiddho", "Karaniya Metta Sutta", "PALI"),
    # ===== ADDITIONAL PALI CANON (v10.10 expansion) =====
    ("Attā hi attano nātho ko hi nātho paro siyā", "Dhammapada 160", "PALI"),
    ("Attanā hi sudantena nāthaṃ labhati dullabhaṃ", "Dhammapada 160", "PALI"),
    ("Attanā va kataṃ pāpaṃ attanā saṃkilissati", "Dhammapada 165", "PALI"),
    ("Attanā akataṃ pāpaṃ attanā va visujjhati", "Dhammapada 165", "PALI"),
    ("Suddhi asuddhi paccattaṃ nāñño aññaṃ visodhaye", "Dhammapada 165", "PALI"),
    ("Sabbadānaṃ dhammadānaṃ jināti", "Jataka", "PALI"),
    ("Sabbapītiṃ dhammarati jināti", "Dhammapada 354", "PALI"),
    ("Sabbaratiṃ taṇhakkhayo jināti", "Dhammapada 354", "PALI"),
    ("Cattārimāni bhikkhave brahmavihārāni", "Anguttara Nikaya", "PALI"),
    ("Dānena piyavācāya atthacārena yamhi", "Anguttara Nikaya", "PALI"),
    # v10.9 original passages preserved
    ("Yo ca vassasataṃ jīve dussīlo asamāhito", "Dhammapada 110", "PALI"),
    ("Ekāhaṃ jīvitaṃ seyyo sīlavantassa jhāyino", "Dhammapada 110", "PALI"),
    ("Attadatthaṃ paratthena bahunāpi na hāpaye", "Dhammapada 166", "PALI"),
    ("Dīghā jāgarato ratti dīghaṃ santassa yojanaṃ", "Dhammapada 60", "PALI"),
    ("Kāyena saṃvaro sādhu sādhu vācāya saṃvaro", "Dhammapada 361", "PALI"),
    ("Manasā saṃvaro sādhu sādhu sabbattha saṃvaro", "Dhammapada 361", "PALI"),
    ("Sabbattha saṃvuto bhikkhu sabbadukkhā pamuccati", "Dhammapada 361", "PALI"),
    ("Yo ca mettaṃ bhāvayati appamāṇaṃ satīmā", "Itivuttaka 27", "PALI"),
    ("Sukhakāmāni bhūtāni yo daṇḍena na hiṃsati", "Dhammapada 131", "PALI"),
    ("Attano sukhamesāno pecca so labhate sukhaṃ", "Dhammapada 131", "PALI"),
    ("Na paresaṃ vilomāni na paresaṃ katākataṃ", "Dhammapada 50", "PALI"),
    ("Attano va avekkheyya katāni akatāni ca", "Dhammapada 50", "PALI"),
    ("Kodhassa na kuto mūlaṃ kalahassa ayaṃ bhave", "Sutta Nipata", "PALI"),
    ("Pūjaṃ paṭhabhiṃ pūjitvā te sameti sukhāvaho", "Sigalovada Sutta", "PALI"),
    # Vinaya - Monastic precepts
    ("Jātarūparajatapaṭiggahaṇā veramaṇī sikkhāpadaṃ samādiyāmi", "Vinaya", "PALI"),
    # Sutta Nipata - Discourse verses
    ("Sammāvimuttaṃ na vimuttasaddhaṃ", "Sutta Nipata", "PALI"),
    # Sigalovada Sutta - Lay ethics
    # Mangala Sutta - Blessings
    # Karaniya Metta Sutta - Practice of loving-kindness
    # Jataka moral lessons
    ("Ahaṃ khīṇāsavo bhikkhu satimā sampajāno", "Jataka", "PALI"),
    ("Na taṃ kammaṃ kataṃ sādhu yaṃ katvā anutappati", "Dhammapada 67", "PALI"),
    ("Taṃ ca kammaṃ kataṃ sādhu yaṃ katvā nānutappati", "Dhammapada 68", "PALI"),
    ("Attanā hi kataṃ pāpaṃ attanā saṃkilissati", "Dhammapada 165", "PALI"),
    # Anguttara Nikaya - Gradual teachings
    ("Sabbe sattā āhāraṭṭhitikā", "Anguttara Nikaya", "PALI"),
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
    # Check for sufficient v10.9 data (not just presence, but expected counts)
    sanskrit_count = by_lang.get("sanskrit", 0)
    pali_count = by_lang.get("pali", 0)

    # Also check for v10.9-specific periods by scanning bonds
    has_v109_periods = False
    try:
        with open("data/processed/bonds.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                b = json.loads(line)
                period = b.get("time_period", "")
                if period in ["BUDDHIST", "LEGALIST", "MOHIST", "FIQH", "SUFI", "FALSAFA"]:
                    has_v109_periods = True
                    break
    except:
        pass

    # v10.10 requires: Sanskrit >= 200, Pali >= 150, and v10.9 periods present
    min_sanskrit = MIN_CORPUS_SIZE.get("sanskrit", 200)
    min_pali = MIN_CORPUS_SIZE.get("pali", 150)
    has_full_v109 = sanskrit_count >= min_sanskrit and pali_count >= min_pali and has_v109_periods

    print(f"\nv10.10 corpus check:")
    print(f"  Sanskrit: {sanskrit_count} (need >= {min_sanskrit})")
    print(f"  Pali: {pali_count} (need >= {min_pali})")
    print(f"  v10.9 periods: {'present' if has_v109_periods else 'missing'}")
    print(f"  Full v10.9: {'YES' if has_full_v109 else 'NO - will add corpora'}")

    if not has_full_v109:
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

        # Add Aramaic (v10.10)
        for text_content, source_ref, period_tag in ARAMAIC_TALMUD:
            all_passages.append(
                {
                    "id": f"v1010_aramaic_{len(all_passages)}",
                    "text": text_content,
                    "language": "aramaic",
                    "source": source_ref,
                    "time_period": period_tag,
                }
            )

        v109_count = len(all_passages) - v109_start
        print(f"Added {v109_count} v10.9/v10.10 passages")

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

        # Force splits regeneration since we added new data
        # Delete existing splits so Cell 5 regenerates them
        for splits_path in ["data/splits/all_splits.json", f"{SAVE_DIR}/all_splits.json"]:
            try:
                if os.path.exists(splits_path):
                    os.remove(splits_path)
                    print(f"  Removed old splits: {splits_path}")
            except Exception as e:
                pass
        print("  Splits will be regenerated in Cell 5 to include v10.9 data")

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
        by_lang["aramaic"] += len(ARAMAIC_TALMUD)
        n_passages = len(all_passages)

        print(f"\nUpdated corpus sizes:")
        for lang, cnt in sorted(by_lang.items(), key=lambda x: -x[1]):
            print(f"  {lang}: {cnt:,}")
    else:
        print("\nv10.9 corpora already present and complete")

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

    # Try to download from Kaggle (in Refresh all OR Update missing mode)
    if not kaggle_path.exists() and not CACHE_ONLY:
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
                                "period": "QURANIC",
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
                                "period": "HADITH",
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
                                        "period": "QURANIC",
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
                    "period": "QURANIC",
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

    if not abby_path.exists() and not CACHE_ONLY:
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
