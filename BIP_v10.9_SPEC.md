# BIP v10.9 Engineering Specification

**Bond Invariance Principle - Native Language Moral Pattern Transfer**

**Version**: 10.9  
**Author**: Andrew Bond  
**Date**: 2026-01-14  
**Status**: Draft  

---

## 1. Executive Summary

### 1.1 Background

BIP v10.8 demonstrated successful cross-linguistic moral concept transfer:
- `ancient_to_modern`: F1 = 0.597 (6× chance), Chinese = 0.68
- `mixed_baseline`: F1 = 0.839 (8.4× chance)
- Language invariance: 0.02% - 0.46% accuracy (near-random)

### 1.2 v10.9 Objectives

1. **Explain Chinese outperformance** — diversify Chinese corpus to test if transfer is genuine or corpus artifact
2. **Improve Arabic performance** — add structured legal/philosophical sources
3. **Add independence test** — Sanskrit/Pali as truly unrelated language family
4. **Geometric discovery** — new analysis tools to probe latent space structure
5. **Scalable architecture** — support for larger corpora and longer training

### 1.3 Success Criteria

| Metric | v10.8 | v10.9 Target |
|--------|-------|--------------|
| `ancient_to_modern` F1 | 0.597 | ≥ 0.55 (maintained with diverse corpus) |
| Arabic F1 | 0.36 | ≥ 0.50 |
| Language invariance | 0.02-0.46% | < 5% |
| New language (Sanskrit) transfer | N/A | ≥ 0.40 F1 |
| Geometric structure found | N/A | ≥ 2 interpretable axes |

---

## 2. Corpus Specification

### 2.1 Current Corpus Summary (v10.8)

| Language | Sources | ~Passages | Issues |
|----------|---------|-----------|--------|
| Hebrew | Sefaria | ~50,000 | Good coverage |
| Aramaic | Sefaria | ~20,000 | Good coverage |
| Classical Chinese | Analects, Mencius, Daodejing, Daxue, Zhongyong, Wenyanwen | ~50,000 | Narrow philosophical tradition |
| Arabic | Quran, Hadith | ~180 | Very small, religiously focused |
| English | Dear Abby, ETHICS, Social Chemistry | ~50,000 | Different register |

### 2.2 New Corpus: Classical Chinese Expansion

**Goal**: Test whether Chinese performance is due to (a) universal structure, (b) Confucian moral explicitness, or (c) corpus homogeneity.

#### 2.2.1 Buddhist Chinese (佛教漢文)
```python
BUDDHIST_CHINESE = [
    # === DHAMMAPADA (法句經) ===
    ("諸惡莫作，眾善奉行，自淨其意，是諸佛教。", "Dhammapada 183", "BUDDHIST", 3),
    ("以恨止恨，恨終不滅；唯以忍止恨，此古聖常法。", "Dhammapada 5", "BUDDHIST", 3),
    ("勝者生怨，負者自鄙；去勝負心，無諍自安。", "Dhammapada 201", "BUDDHIST", 3),
    ("不以財物施，唯以法布施，法施勝財施。", "Dhammapada 354", "BUDDHIST", 3),
    
    # === DIAMOND SUTRA (金剛經) ===
    ("若以色見我，以音聲求我，是人行邪道，不能見如來。", "Diamond Sutra 26", "BUDDHIST", 5),
    ("應無所住而生其心。", "Diamond Sutra 10", "BUDDHIST", 5),
    ("一切有為法，如夢幻泡影，如露亦如電，應作如是觀。", "Diamond Sutra 32", "BUDDHIST", 5),
    ("凡所有相，皆是虛妄。若見諸相非相，即見如來。", "Diamond Sutra 5", "BUDDHIST", 5),
    
    # === LOTUS SUTRA (法華經) ===
    ("諸佛世尊唯以一大事因緣故，出現於世。", "Lotus Sutra 2", "BUDDHIST", 3),
    ("十方佛土中，唯有一乘法，無二亦無三。", "Lotus Sutra 2", "BUDDHIST", 3),
    
    # === HEART SUTRA (心經) ===
    ("色不異空，空不異色，色即是空，空即是色。", "Heart Sutra", "BUDDHIST", 7),
    ("無苦集滅道，無智亦無得，以無所得故。", "Heart Sutra", "BUDDHIST", 7),
    
    # === SUTRA OF GOLDEN LIGHT (金光明經) ===
    ("殺生之罪，能令眾生墮三惡道。", "Sutra of Golden Light 4", "BUDDHIST", 5),
    ("一切眾生皆有佛性，悉能成佛。", "Nirvana Sutra", "BUDDHIST", 5),
    
    # === BRAHMA NET SUTRA (梵網經) — PRECEPTS ===
    ("慈悲喜捨，名為四無量心。", "Brahma Net Sutra", "BUDDHIST", 5),
    ("不殺生，是菩薩波羅夷罪。", "Brahma Net Sutra 1", "BUDDHIST", 5),
    ("不偷盜，是菩薩波羅夷罪。", "Brahma Net Sutra 2", "BUDDHIST", 5),
    ("不邪淫，是菩薩波羅夷罪。", "Brahma Net Sutra 3", "BUDDHIST", 5),
    ("不妄語，是菩薩波羅夷罪。", "Brahma Net Sutra 4", "BUDDHIST", 5),
    ("不飲酒，是菩薩波羅夷罪。", "Brahma Net Sutra 5", "BUDDHIST", 5),
    ("若佛子，以慈心故，行放生業。", "Brahma Net Sutra 20", "BUDDHIST", 5),
    ("一切男子是我父，一切女人是我母。", "Brahma Net Sutra 9", "BUDDHIST", 5),
    
    # === VIMALAKIRTI SUTRA (維摩經) ===
    ("菩薩病者，以大悲起。", "Vimalakirti Sutra 5", "BUDDHIST", 5),
    ("眾生病，是故我病。", "Vimalakirti Sutra 5", "BUDDHIST", 5),
    
    # === PLATFORM SUTRA (六祖壇經) ===
    ("菩提本無樹，明鏡亦非臺，本來無一物，何處惹塵埃。", "Platform Sutra", "BUDDHIST", 8),
    ("何期自性，本自清淨；何期自性，本不生滅。", "Platform Sutra", "BUDDHIST", 8),
]
# Target: 50+ passages
```

#### 2.2.2 Legalist Chinese (法家)
```python
LEGALIST_CHINESE = [
    # === HAN FEIZI (韓非子) ===
    ("法不阿貴，繩不撓曲。", "Han Feizi 6", "LEGALIST", -3),
    ("刑過不避大臣，賞善不遺匹夫。", "Han Feizi 50", "LEGALIST", -3),
    ("以法為教，以吏為師。", "Han Feizi 49", "LEGALIST", -3),
    ("明主之國，無書簡之文，以法為教。", "Han Feizi 49", "LEGALIST", -3),
    ("法者，編著之圖籍，設之於官府，而布之於百姓者也。", "Han Feizi 38", "LEGALIST", -3),
    ("術者，藏之於胸中，以偶眾端，而潛御群臣者也。", "Han Feizi 38", "LEGALIST", -3),
    ("法莫如顯，而術不欲見。", "Han Feizi 38", "LEGALIST", -3),
    ("賞罰不信，則禁令不行。", "Han Feizi 46", "LEGALIST", -3),
    ("刑重則不敢以惡犯，罰輕則民不畏。", "Han Feizi 46", "LEGALIST", -3),
    ("夫嚴刑重罰者，民之所惡也；而國之所以治也。", "Han Feizi 49", "LEGALIST", -3),
    ("法之所加，智者弗能辭，勇者弗敢爭。", "Han Feizi 6", "LEGALIST", -3),
    ("一民之軌，莫如法。", "Han Feizi 6", "LEGALIST", -3),
    ("故明主使法擇人，不自舉也。", "Han Feizi 6", "LEGALIST", -3),
    ("使法量功，不自度也。", "Han Feizi 6", "LEGALIST", -3),
    
    # === SHANG JUN SHU (商君書) ===
    ("國之所以興者，農戰也。", "Shang Jun Shu 3", "LEGALIST", -4),
    ("民弱國強，民強國弱。故有道之國，務在弱民。", "Shang Jun Shu 20", "LEGALIST", -4),
    ("聖人之為國也，壹賞，壹刑，壹教。", "Shang Jun Shu 17", "LEGALIST", -4),
    ("治國者，貴分明而不可相舉。", "Shang Jun Shu 14", "LEGALIST", -4),
    ("令民無得擅徙，使民無得不耕。", "Shang Jun Shu 15", "LEGALIST", -4),
    ("行罰重其輕者，輕者不至，重者不來。", "Shang Jun Shu 17", "LEGALIST", -4),
    
    # === GUANZI (管子) ===
    ("倉廩實則知禮節，衣食足則知榮辱。", "Guanzi 1", "LEGALIST", -7),
    ("禮義廉恥，國之四維；四維不張，國乃滅亡。", "Guanzi 1", "LEGALIST", -7),
    ("政之所興，在順民心；政之所廢，在逆民心。", "Guanzi 1", "LEGALIST", -7),
]
# Target: 40+ passages
```

#### 2.2.3 Mohist Chinese (墨家)
```python
MOHIST_CHINESE = [
    # === MOZI (墨子) ===
    ("兼相愛，交相利。", "Mozi 15", "MOHIST", -5),
    ("天下之人皆相愛，強不執弱，眾不劫寡，富不侮貧，貴不傲賤。", "Mozi 15", "MOHIST", -5),
    ("殺一人謂之不義，必有一死罪矣。", "Mozi 17", "MOHIST", -5),
    ("今至大為攻國，則弗知非，從而譽之，謂之義。", "Mozi 17", "MOHIST", -5),
    ("天下之利，是為天下之義。", "Mozi 26", "MOHIST", -5),
    ("非攻，墨子之道也。", "Mozi 17", "MOHIST", -5),
    ("節用，墨子之教也。", "Mozi 20", "MOHIST", -5),
    ("聖人以治天下為事者也，必知亂之所自起，焉能治之。", "Mozi 14", "MOHIST", -5),
    ("天下之所以亂者，生於不相愛。", "Mozi 14", "MOHIST", -5),
    ("臣子之不孝君父，所謂亂也。", "Mozi 14", "MOHIST", -5),
    ("子自愛不愛父，故虧父而自利。", "Mozi 14", "MOHIST", -5),
    ("弟自愛不愛兄，故虧兄而自利。", "Mozi 14", "MOHIST", -5),
    ("若使天下兼相愛，愛人若愛其身，猶有不孝者乎？", "Mozi 15", "MOHIST", -5),
    ("視人之國若視其國，視人之家若視其家，視人之身若視其身。", "Mozi 15", "MOHIST", -5),
    ("是故諸侯相愛則不野戰，家主相愛則不相篡。", "Mozi 15", "MOHIST", -5),
    ("人與人相愛則不相賊。", "Mozi 15", "MOHIST", -5),
    ("君臣相愛則惠忠，父子相愛則慈孝。", "Mozi 15", "MOHIST", -5),
    ("兄弟相愛則和調。", "Mozi 15", "MOHIST", -5),
]
# Target: 30+ passages
```

#### 2.2.4 Neo-Confucian Chinese (宋明理學)
```python
NEO_CONFUCIAN_CHINESE = [
    # === ZHU XI (朱熹) ===
    ("存天理，滅人欲。", "Zhu Xi - Analects Commentary", "NEO_CONFUCIAN", 12),
    ("格物致知，誠意正心。", "Zhu Xi - Great Learning Commentary", "NEO_CONFUCIAN", 12),
    ("天理人欲，同行異情。", "Zhu Xi - Classified Conversations", "NEO_CONFUCIAN", 12),
    ("聖人千言萬語，只是教人明天理，滅人欲。", "Zhu Xi - Classified Conversations", "NEO_CONFUCIAN", 12),
    ("敬者，聖學之所以成始而成終者也。", "Zhu Xi - Collected Writings", "NEO_CONFUCIAN", 12),
    ("窮理以致其知，反躬以踐其實。", "Zhu Xi - Collected Writings", "NEO_CONFUCIAN", 12),
    ("涵養須用敬，進學則在致知。", "Zhu Xi - Classified Conversations", "NEO_CONFUCIAN", 12),
    
    # === WANG YANGMING (王陽明) ===
    ("知行合一。", "Wang Yangming - Instructions for Practical Living", "NEO_CONFUCIAN", 16),
    ("致良知。", "Wang Yangming - Instructions for Practical Living", "NEO_CONFUCIAN", 16),
    ("無善無惡心之體，有善有惡意之動。", "Wang Yangming - Four Maxims", "NEO_CONFUCIAN", 16),
    ("知善知惡是良知，為善去惡是格物。", "Wang Yangming - Four Maxims", "NEO_CONFUCIAN", 16),
    ("心即理也。", "Wang Yangming - Instructions for Practical Living", "NEO_CONFUCIAN", 16),
    ("吾心之良知，即所謂天理也。", "Wang Yangming - Instructions for Practical Living", "NEO_CONFUCIAN", 16),
    ("知是行之始，行是知之成。", "Wang Yangming - Instructions for Practical Living", "NEO_CONFUCIAN", 16),
    ("知而不行，只是未知。", "Wang Yangming - Instructions for Practical Living", "NEO_CONFUCIAN", 16),
    ("破山中賊易，破心中賊難。", "Wang Yangming - Letters", "NEO_CONFUCIAN", 16),
    
    # === ZHOU DUNYI (周敦頤) ===
    ("誠者，聖人之本。", "Zhou Dunyi - Tongshu", "NEO_CONFUCIAN", 11),
    ("誠，五常之本，百行之源也。", "Zhou Dunyi - Tongshu", "NEO_CONFUCIAN", 11),
    
    # === ZHANG ZAI (張載) ===
    ("民吾同胞，物吾與也。", "Zhang Zai - Western Inscription", "NEO_CONFUCIAN", 11),
    ("為天地立心，為生民立命，為往聖繼絕學，為萬世開太平。", "Zhang Zai - Attributed", "NEO_CONFUCIAN", 11),
]
# Target: 30+ passages
```

### 2.3 New Corpus: Arabic Expansion

#### 2.3.1 Islamic Legal Maxims (قواعد فقهية)
```python
ISLAMIC_LEGAL_MAXIMS = [
    # === THE FIVE MAJOR MAXIMS (الكليات الخمس) ===
    ("الأمور بمقاصدها", "Al-Qawa'id - Major 1", "FIQH", 8),  # Acts by intentions
    ("اليقين لا يزول بالشك", "Al-Qawa'id - Major 2", "FIQH", 8),  # Certainty not removed by doubt
    ("المشقة تجلب التيسير", "Al-Qawa'id - Major 3", "FIQH", 8),  # Hardship brings ease
    ("الضرر يزال", "Al-Qawa'id - Major 4", "FIQH", 8),  # Harm is removed
    ("العادة محكمة", "Al-Qawa'id - Major 5", "FIQH", 8),  # Custom is authoritative
    
    # === HARM PREVENTION MAXIMS ===
    ("لا ضرر ولا ضرار", "Al-Qawa'id", "FIQH", 8),  # No harm, no reciprocal harm
    ("الضرر لا يزال بالضرر", "Al-Qawa'id", "FIQH", 8),  # Harm not removed by harm
    ("الضرر الأشد يزال بالضرر الأخف", "Al-Qawa'id", "FIQH", 8),  # Greater harm removed by lesser
    ("درء المفاسد أولى من جلب المصالح", "Al-Qawa'id", "FIQH", 8),  # Preventing harm > gaining benefit
    ("يتحمل الضرر الخاص لدفع الضرر العام", "Al-Qawa'id", "FIQH", 8),  # Private harm for public good
    ("إذا تعارضت مفسدتان روعي أعظمهما ضررا بارتكاب أخفهما", "Al-Qawa'id", "FIQH", 8),
    
    # === PERMISSION AND PROHIBITION ===
    ("الأصل في الأشياء الإباحة", "Al-Qawa'id", "FIQH", 8),  # Default is permissibility
    ("الأصل في العقود الصحة", "Al-Qawa'id", "FIQH", 8),  # Default validity of contracts
    ("الأصل بقاء ما كان على ما كان", "Al-Qawa'id", "FIQH", 8),  # Status quo presumption
    ("ما حرم أخذه حرم إعطاؤه", "Al-Qawa'id", "FIQH", 8),  # Forbidden to take = forbidden to give
    ("ما حرم فعله حرم طلبه", "Al-Qawa'id", "FIQH", 8),  # Forbidden to do = forbidden to request
    
    # === NECESSITY AND EXCEPTION ===
    ("الضرورات تبيح المحظورات", "Al-Qawa'id", "FIQH", 8),  # Necessity permits forbidden
    ("الضرورة تقدر بقدرها", "Al-Qawa'id", "FIQH", 8),  # Necessity measured by extent
    ("ما أبيح للضرورة يقدر بقدرها", "Al-Qawa'id", "FIQH", 8),
    ("الحاجة تنزل منزلة الضرورة عامة كانت أو خاصة", "Al-Qawa'id", "FIQH", 8),
    ("إذا ضاق الأمر اتسع", "Al-Qawa'id", "FIQH", 8),  # When matters narrow, they widen
    
    # === AGENCY AND RESPONSIBILITY ===
    ("الجواز الشرعي ينافي الضمان", "Al-Qawa'id", "FIQH", 8),  # Legal permission negates liability
    ("المباشر ضامن وإن لم يتعمد", "Al-Qawa'id", "FIQH", 8),  # Direct actor liable even without intent
    ("المتسبب لا يضمن إلا بالتعمد", "Al-Qawa'id", "FIQH", 8),  # Indirect cause liable only with intent
    ("إذا اجتمع المباشر والمتسبب يضاف الحكم إلى المباشر", "Al-Qawa'id", "FIQH", 8),
    ("الإذن العام كالإذن الخاص", "Al-Qawa'id", "FIQH", 8),  # General permission = specific
    
    # === INTENTIONS AND SPEECH ACTS ===
    ("لا عبرة بالدلالة في مقابلة التصريح", "Al-Qawa'id", "FIQH", 8),
    ("إعمال الكلام أولى من إهماله", "Al-Qawa'id", "FIQH", 8),  # Interpret speech, don't ignore
    ("الأصل في الكلام الحقيقة", "Al-Qawa'id", "FIQH", 8),  # Speech interpreted literally first
    ("ذكر بعض ما لا يتجزأ كذكر كله", "Al-Qawa'id", "FIQH", 8),
]
# Target: 50+ passages
```

#### 2.3.2 Sufi Ethics (الأخلاق الصوفية)
```python
SUFI_ETHICS = [
    # === AL-GHAZALI (الغزالي) - Ihya Ulum al-Din ===
    ("التصوف كله أخلاق", "Al-Ghazali - Ihya", "SUFI", 11),
    ("من لم يؤثر فيه علم أخلاقه فقد غفل عن الفقه", "Al-Ghazali - Ihya", "SUFI", 11),
    ("الخلق الحسن جماع الدين كله", "Al-Ghazali - Ihya", "SUFI", 11),
    ("أصل الأخلاق المحمودة كلها أربعة: الحكمة والشجاعة والعفة والعدل", "Al-Ghazali - Ihya", "SUFI", 11),
    ("العلم بلا عمل جنون، والعمل بغير علم لا يكون", "Al-Ghazali - Ihya", "SUFI", 11),
    ("من عرف نفسه عرف ربه", "Al-Ghazali - Attributed", "SUFI", 11),
    ("قلب المؤمن بين أصبعين من أصابع الرحمن", "Al-Ghazali - Ihya", "SUFI", 11),
    
    # === AL-JUNAYD (الجنيد) ===
    ("التصوف هو الخلق، فمن زاد عليك في الخلق زاد عليك في التصوف", "Al-Junayd", "SUFI", 9),
    ("الصوفي من صفا قلبه لله", "Al-Junayd", "SUFI", 9),
    ("أفضل الأعمال مخالفة النفس والهوى", "Al-Junayd", "SUFI", 9),
    
    # === AL-QUSHAYRI (القشيري) ===
    ("من لم يزن أفعاله وأحواله في كل وقت بالكتاب والسنة فلا تعده في ديوان الرجال", "Al-Qushayri - Risala", "SUFI", 11),
    ("الصدق سيف الله في أرضه، ما وضع على شيء إلا قطعه", "Al-Qushayri - Risala", "SUFI", 11),
    
    # === RUMI (جلال الدين الرومي) ===
    ("ما خلقت الخلق إلا ليعرفوني", "Rumi - Attributed", "SUFI", 13),
    ("كن كالشمس للرحمة والشفقة، وكالليل في ستر عيوب الغير", "Rumi - Masnavi", "SUFI", 13),
    
    # === IBN ARABI (ابن عربي) ===
    ("من عرف نفسه فقد عرف ربه", "Ibn Arabi - Fusus", "SUFI", 12),
    ("الإنسان الكامل مرآة الحق", "Ibn Arabi - Fusus", "SUFI", 12),
]
# Target: 30+ passages
```

#### 2.3.3 Arabic Philosophy (الفلسفة العربية)
```python
ARABIC_PHILOSOPHY = [
    # === AL-FARABI (الفارابي) ===
    ("الإنسان مدني بالطبع", "Al-Farabi - Ara Ahl al-Madina", "FALSAFA", 10),
    ("السعادة هي الخير المطلوب لذاته", "Al-Farabi - Tahsil al-Sa'ada", "FALSAFA", 10),
    ("الفضيلة هي الحال التي بها يفعل الإنسان الأفعال الجميلة", "Al-Farabi - Fusul", "FALSAFA", 10),
    
    # === IBN SINA (ابن سينا) ===
    ("العقل العملي هو الذي يدبر البدن", "Ibn Sina - Shifa", "FALSAFA", 11),
    ("النفس جوهر روحاني", "Ibn Sina - Shifa", "FALSAFA", 11),
    
    # === IBN RUSHD (ابن رشد) ===
    ("العدل هو فضيلة من الفضائل العامة", "Ibn Rushd - Commentary on Republic", "FALSAFA", 12),
    ("الحكمة والشريعة أختان رضيعتان", "Ibn Rushd - Fasl al-Maqal", "FALSAFA", 12),
    ("الحق لا يضاد الحق بل يوافقه ويشهد له", "Ibn Rushd - Fasl al-Maqal", "FALSAFA", 12),
    
    # === IBN KHALDUN (ابن خلدون) ===
    ("الإنسان مدني بالطبع، أي لا بد له من الاجتماع", "Ibn Khaldun - Muqaddima", "FALSAFA", 14),
    ("العصبية هي الرابطة الاجتماعية", "Ibn Khaldun - Muqaddima", "FALSAFA", 14),
    ("الظلم مؤذن بخراب العمران", "Ibn Khaldun - Muqaddima", "FALSAFA", 14),
]
# Target: 20+ passages
```

### 2.4 New Language: Sanskrit/Pali

**Rationale**: Indo-European, but different branch from Greek/Latin. Rich ethical tradition, completely independent development from Semitic and Sinitic.

```python
# === DHARMASHASTRA (धर्मशास्त्र) ===
SANSKRIT_DHARMA = [
    # === MAHABHARATA ===
    ("अहिंसा परमो धर्मः", "Mahabharata 13.117.37", "DHARMA", -4),  # Non-violence highest dharma
    ("धर्म एव हतो हन्ति धर्मो रक्षति रक्षितः", "Mahabharata 8.69.57", "DHARMA", -4),  # Dharma destroys destroyer
    ("न हि प्रियं मे स्यात् आत्मनः प्रतिकूलं परेषाम्", "Mahabharata 5.15.17", "DHARMA", -4),  # Golden Rule
    ("सत्यं ब्रूयात् प्रियं ब्रूयात्", "Mahabharata", "DHARMA", -4),  # Speak truth pleasantly
    ("आत्मनः प्रतिकूलानि परेषां न समाचरेत्", "Mahabharata 5.15.17", "DHARMA", -4),  # Don't do to others...
    
    # === MANUSMRITI ===
    ("अहिंसा सत्यमस्तेयं शौचमिन्द्रियनिग्रहः", "Manusmriti 10.63", "DHARMA", 2),  # Five virtues
    ("सर्वभूतेषु चात्मानं सर्वभूतानि चात्मनि", "Manusmriti", "DHARMA", 2),  # See self in all
    
    # === UPANISHADS ===
    ("सत्यं वद धर्मं चर", "Taittiriya Upanishad 1.11", "UPANISHAD", -6),  # Speak truth, follow dharma
    ("मातृदेवो भव। पितृदेवो भव। आचार्यदेवो भव। अतिथिदेवो भव।", "Taittiriya Upanishad 1.11", "UPANISHAD", -6),
    ("ईशावास्यमिदं सर्वं यत्किञ्च जगत्यां जगत्", "Isha Upanishad 1", "UPANISHAD", -6),
    ("तेन त्यक्तेन भुञ्जीथा मा गृधः कस्यस्विद्धनम्", "Isha Upanishad 1", "UPANISHAD", -6),
    
    # === BHAGAVAD GITA ===
    ("कर्मण्येवाधिकारस्ते मा फलेषु कदाचन", "Bhagavad Gita 2.47", "GITA", -3),  # Right to action, not fruits
    ("योगः कर्मसु कौशलम्", "Bhagavad Gita 2.50", "GITA", -3),  # Yoga is skill in action
    ("समत्वं योग उच्यते", "Bhagavad Gita 2.48", "GITA", -3),  # Equanimity is yoga
    ("सर्वधर्मान्परित्यज्य मामेकं शरणं व्रज", "Bhagavad Gita 18.66", "GITA", -3),
    ("अद्वेष्टा सर्वभूतानां मैत्रः करुण एव च", "Bhagavad Gita 12.13", "GITA", -3),
    
    # === ARTHASHASTRA ===
    ("प्रजासुखे सुखं राज्ञः प्रजानां च हिते हितम्", "Arthashastra 1.19", "ARTHA", -3),  # King's happiness in people's
    ("राज्ञो हि व्रतं कार्याणां चेष्टा राष्ट्रसंग्रहः", "Arthashastra", "ARTHA", -3),
]

# === PALI CANON ===
PALI_ETHICS = [
    # === DHAMMAPADA ===
    ("Sabbe sattā bhavantu sukhitattā", "Metta Sutta", "PALI", -3),  # May all beings be happy
    ("Dhammo have rakkhati dhammacāriṃ", "Theragatha 303", "PALI", -3),  # Dhamma protects follower
    ("Sabba pāpassa akaraṇaṃ, kusalassa upasampadā", "Dhammapada 183", "PALI", -3),
    ("Manopubbaṅgamā dhammā manoseṭṭhā manomayā", "Dhammapada 1", "PALI", -3),  # Mind forerunner
    ("Na hi verena verāni sammantīdha kudācanaṃ", "Dhammapada 5", "PALI", -3),  # Hatred not by hatred
    ("Averena ca sammanti esa dhammo sanantano", "Dhammapada 5", "PALI", -3),
    ("Attā hi attano nātho ko hi nātho paro siyā", "Dhammapada 160", "PALI", -3),  # Self is own refuge
    
    # === VINAYA PITAKA (Monastic Rules) ===
    ("Pāṇātipātā veramaṇī sikkhāpadaṃ samādiyāmi", "Vinaya", "PALI", -3),  # I undertake not to kill
    ("Adinnādānā veramaṇī sikkhāpadaṃ samādiyāmi", "Vinaya", "PALI", -3),  # Not to steal
    ("Kāmesumicchācārā veramaṇī sikkhāpadaṃ samādiyāmi", "Vinaya", "PALI", -3),  # Sexual misconduct
    ("Musāvādā veramaṇī sikkhāpadaṃ samādiyāmi", "Vinaya", "PALI", -3),  # False speech
    ("Surāmerayamajjapamādaṭṭhānā veramaṇī sikkhāpadaṃ samādiyāmi", "Vinaya", "PALI", -3),  # Intoxicants
    
    # === SUTTA NIPATA ===
    ("Mettañca sabbalokasmiṃ mānasaṃ bhāvaye aparimāṇaṃ", "Metta Sutta", "PALI", -3),
]
# Target: 50+ passages total
```

### 2.5 Period Index Updates

```python
PERIOD_TO_IDX = {
    # Semitic
    'BIBLICAL': 0,
    'TANNAITIC': 1,
    'AMORAIC': 2,
    'RISHONIM': 3,
    'ACHRONIM': 4,
    
    # Chinese
    'CONFUCIAN': 5,
    'DAOIST': 6,
    'MOHIST': 7,       # NEW
    'LEGALIST': 8,     # NEW
    'BUDDHIST': 9,     # NEW (Chinese Buddhism)
    'NEO_CONFUCIAN': 10,  # NEW
    
    # Arabic
    'QURANIC': 11,
    'HADITH': 12,
    'FIQH': 13,        # NEW
    'SUFI': 14,        # NEW
    'FALSAFA': 15,     # NEW
    
    # Sanskrit/Pali
    'DHARMA': 16,      # NEW
    'UPANISHAD': 17,   # NEW
    'GITA': 18,        # NEW
    'ARTHA': 19,       # NEW
    'PALI': 20,        # NEW
    
    # Western
    'WESTERN_CLASSICAL': 21,
    
    # Modern
    'DEAR_ABBY': 22,
    'MODERN': 23,
    'CLASSICAL': 24,   # Generic classical
    'MEDIEVAL': 25,
}

LANG_TO_IDX = {
    'hebrew': 0,
    'aramaic': 1,
    'classical_chinese': 2,
    'arabic': 3,
    'english': 4,
    'sanskrit': 5,     # NEW
    'pali': 6,         # NEW
    'greek': 7,        # FUTURE
}
```

### 2.6 Bond Pattern Updates

```python
# Add to ALL_BOND_PATTERNS:

'classical_chinese': {
    # Existing patterns plus:
    BondType.HARM_PREVENTION: [..., r'暴', r'虐', r'殘', r'攻', r'戒', r'殺生'],
    BondType.RECIPROCITY: [..., r'恩', r'德', r'施', r'受', r'報應'],
    BondType.AUTHORITY: [..., r'法', r'刑', r'律', r'禁', r'罰', r'賞'],
    BondType.CARE: [..., r'慈', r'悲', r'捨', r'度', r'救'],
    BondType.FAIRNESS: [..., r'兼', r'利', r'功', r'過'],
    BondType.CONTRACT: [..., r'戒', r'律', r'願'],
},

'arabic': {
    # Existing patterns plus:
    BondType.AUTHORITY: [..., r'قاعد', r'أصل', r'محكم'],
    BondType.HARM_PREVENTION: [..., r'ضرورة', r'مفسد', r'مصلح'],
    BondType.CONTRACT: [..., r'ضمان', r'عقد', r'شرط'],
    BondType.FAIRNESS: [..., r'يقين', r'شك', r'أصل'],
},

'sanskrit': {
    BondType.HARM_PREVENTION: [r'अहिंसा', r'हिंस', r'रक्ष', r'पाप'],
    BondType.RECIPROCITY: [r'कर्म', r'फल', r'प्रतिकूल'],
    BondType.AUTONOMY: [r'स्व', r'आत्म', r'मोक्ष', r'स्वतन्त्र'],
    BondType.FAMILY: [r'पितृ', r'मातृ', r'पुत्र', r'कुल', r'गृह'],
    BondType.AUTHORITY: [r'राज', r'धर्म', r'नियम', r'दण्ड'],
    BondType.CARE: [r'करुण', r'दया', r'अनुकम्पा', r'मैत्र'],
    BondType.FAIRNESS: [r'न्याय', r'सम', r'धर्म', r'ऋत'],
    BondType.CONTRACT: [r'व्रत', r'शपथ', r'प्रतिज्ञा', r'सत्य'],
},

'pali': {
    BondType.HARM_PREVENTION: [r'ahiṃsā', r'pāṇātipāt', r'veramaṇ'],
    BondType.RECIPROCITY: [r'kamma', r'vipāka', r'phala'],
    BondType.AUTONOMY: [r'attā', r'nātho', r'vimutti'],
    BondType.FAMILY: [r'mātā', r'pitā', r'putta', r'kula'],
    BondType.AUTHORITY: [r'vinaya', r'sikkhāpada', r'dhamma'],
    BondType.CARE: [r'mettā', r'karuṇā', r'muditā', r'upekkhā'],
    BondType.FAIRNESS: [r'samma', r'ujuka', r'dhamma'],
    BondType.CONTRACT: [r'sīla', r'vata', r'sacca'],
},
```

---

## 3. Architecture Updates

### 3.1 Model Architecture (Unchanged Core)

```python
class BIPModel(nn.Module):
    """
    v10.9: Same architecture, updated heads for new languages/periods.
    
    Components:
    - encoder: LaBSE (768-dim) or specified backbone
    - z_proj: 768 → 512 → 64 (bond-invariant latent space)
    - bond_head: 64 → 10 (bond types)
    - hohfeld_head: 64 → 4 (Hohfeld states)
    - language_head: 64 → 8 (languages, adversarial)  # Was 5
    - period_head: 64 → 26 (periods, adversarial)     # Was 14
    - context_head: 64 → 3 (prescriptive/descriptive/unknown)
    """
    
    def __init__(self, model_name=MODEL_NAME, hidden_size=BACKBONE_HIDDEN, z_dim=64):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Projection unchanged
        proj_hidden = min(512, hidden_size)
        self.z_proj = nn.Sequential(
            nn.Linear(hidden_size, proj_hidden),
            nn.LayerNorm(proj_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_hidden, z_dim),
        )
        
        # Task heads
        self.bond_head = nn.Linear(z_dim, len(BondType))       # 10 types
        self.hohfeld_head = nn.Linear(z_dim, len(HohfeldState)) # 4 states
        
        # Adversarial heads - UPDATED SIZES
        self.language_head = nn.Linear(z_dim, len(LANG_TO_IDX))   # 8 languages
        self.period_head = nn.Linear(z_dim, len(PERIOD_TO_IDX))   # 26 periods
        
        # Auxiliary
        self.context_head = nn.Linear(z_dim, len(CONTEXT_TO_IDX)) # 3 contexts
```

### 3.2 New: Geometric Analysis Module

```python
class GeometricAnalyzer:
    """
    Probe the latent space geometry to discover moral structure.
    """
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    @torch.no_grad()
    def get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", 
                                truncation=True, max_length=128, padding="max_length")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        z = self.model.get_bond_embedding(inputs['input_ids'], inputs['attention_mask'])
        return z.cpu().numpy().flatten()
    
    def find_direction(self, positive_texts: List[str], negative_texts: List[str]) -> np.ndarray:
        """
        Find the direction in z-space that separates two concepts.
        E.g., obligation vs permission, harm vs care.
        """
        pos_embs = np.array([self.get_embedding(t) for t in positive_texts])
        neg_embs = np.array([self.get_embedding(t) for t in negative_texts])
        
        pos_mean = pos_embs.mean(axis=0)
        neg_mean = neg_embs.mean(axis=0)
        
        direction = pos_mean - neg_mean
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        return direction
    
    def test_direction_transfer(self, direction: np.ndarray, 
                                 test_pairs: List[Tuple[str, str]]) -> float:
        """
        Test if a direction generalizes to new examples.
        Returns correlation between direction projection and expected ordering.
        """
        scores = []
        for pos_text, neg_text in test_pairs:
            pos_proj = np.dot(self.get_embedding(pos_text), direction)
            neg_proj = np.dot(self.get_embedding(neg_text), direction)
            scores.append(1.0 if pos_proj > neg_proj else 0.0)
        return np.mean(scores)
    
    def pca_on_pairs(self, concept_pairs: Dict[str, List[Tuple[str, str]]]) -> Dict:
        """
        Run PCA on difference vectors to find dominant axes.
        
        concept_pairs: {"obligation_permission": [(obl1, perm1), ...], ...}
        """
        all_diffs = []
        labels = []
        
        for concept, pairs in concept_pairs.items():
            for pos, neg in pairs:
                diff = self.get_embedding(pos) - self.get_embedding(neg)
                all_diffs.append(diff)
                labels.append(concept)
        
        X = np.array(all_diffs)
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(10, len(X)))
        pca.fit(X)
        
        return {
            'components': pca.components_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'labels': labels,
            'transformed': pca.transform(X),
        }
    
    def role_swap_analysis(self, agent_patient_pairs: List[Tuple[str, str]]) -> Dict:
        """
        Test if swapping agent/patient produces consistent transformation.
        
        agent_patient_pairs: [("A harmed B", "B harmed A"), ...]
        """
        transformations = []
        
        for original, swapped in agent_patient_pairs:
            orig_emb = self.get_embedding(original)
            swap_emb = self.get_embedding(swapped)
            transformations.append(swap_emb - orig_emb)
        
        T = np.array(transformations)
        
        # Check consistency: are all transformations similar?
        mean_transform = T.mean(axis=0)
        cosines = [np.dot(t, mean_transform) / (np.linalg.norm(t) * np.linalg.norm(mean_transform) + 1e-9)
                   for t in T]
        
        # Check involutory: does double-swap return to origin?
        # (T applied twice should ≈ identity)
        
        return {
            'mean_transform': mean_transform,
            'consistency': np.mean(cosines),
            'consistency_std': np.std(cosines),
        }
```

### 3.3 New: Contrastive Fuzz Testing

```python
class StructuralFuzzTest:
    """
    Extended fuzz testing with cross-lingual pairs.
    """
    
    # Structural perturbation templates (language-agnostic concepts)
    STRUCTURAL_PAIRS = {
        'obligation_to_permission': [
            # English
            ("You must help the elderly", "You may help the elderly"),
            ("He is required to pay", "He is allowed to pay"),
            ("Parents must protect children", "Parents may protect children"),
            # Chinese
            ("君子必孝", "君子可孝"),  # Gentleman must/may be filial
            ("民必從法", "民可從法"),  # People must/may follow law
            # Arabic
            ("يجب عليك أن تساعد", "يجوز لك أن تساعد"),  # You must/may help
            # Hebrew
            ("חייב לכבד", "מותר לכבד"),  # Obligated/permitted to honor
        ],
        
        'harm_to_care': [
            ("He injured the child", "He protected the child"),
            ("殺人者", "救人者"),  # One who kills / one who saves
            ("ظلم الضعيف", "رحم الضعيف"),  # Oppressed / showed mercy to the weak
        ],
        
        'role_swap': [
            ("The master commands the servant", "The servant commands the master"),
            ("君命臣", "臣命君"),  # Lord commands minister / minister commands lord
            ("الأب يأمر الابن", "الابن يأمر الأب"),  # Father commands son / son commands father
        ],
        
        'violation_to_fulfillment': [
            ("He broke his promise", "He kept his promise"),
            ("違約", "守約"),  # Violate contract / keep contract
            ("نقض العهد", "وفى بالعهد"),  # Broke covenant / fulfilled covenant
        ],
    }
    
    # Surface perturbation templates (should NOT move embeddings)
    SURFACE_PERTURBATIONS = {
        'name_change': lambda t: t.replace("John", "Michael").replace("Mary", "Lisa"),
        'irrelevant_detail': lambda t: t + " It was Tuesday.",
        'passive_voice': lambda t: t,  # Would need NLP for actual transformation
    }
    
    def run_comprehensive_test(self, analyzer: GeometricAnalyzer) -> Dict:
        """
        Run full structural vs surface test battery.
        """
        results = {}
        
        for perturbation_type, pairs in self.STRUCTURAL_PAIRS.items():
            distances = []
            for text1, text2 in pairs:
                emb1 = analyzer.get_embedding(text1)
                emb2 = analyzer.get_embedding(text2)
                dist = 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-9)
                distances.append(dist)
            
            results[f'structural_{perturbation_type}'] = {
                'mean_distance': np.mean(distances),
                'std': np.std(distances),
                'n': len(distances),
            }
        
        # Surface perturbations on base sentences
        base_sentences = [
            "John borrowed money from Mary and must repay it.",
            "The doctor has a duty to help patients.",
            "Parents should protect their children.",
        ]
        
        surface_distances = []
        for base in base_sentences:
            base_emb = analyzer.get_embedding(base)
            for name, perturb_fn in self.SURFACE_PERTURBATIONS.items():
                perturbed = perturb_fn(base)
                if perturbed != base:
                    perturbed_emb = analyzer.get_embedding(perturbed)
                    dist = 1 - np.dot(base_emb, perturbed_emb) / (np.linalg.norm(base_emb) * np.linalg.norm(perturbed_emb) + 1e-9)
                    surface_distances.append(dist)
        
        results['surface_all'] = {
            'mean_distance': np.mean(surface_distances),
            'std': np.std(surface_distances),
            'n': len(surface_distances),
        }
        
        # Statistical comparison
        structural_all = []
        for k, v in results.items():
            if k.startswith('structural_'):
                structural_all.extend([v['mean_distance']] * v['n'])
        
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(structural_all, surface_distances)
        
        results['comparison'] = {
            'structural_mean': np.mean(structural_all),
            'surface_mean': np.mean(surface_distances),
            'ratio': np.mean(structural_all) / (np.mean(surface_distances) + 1e-9),
            't_statistic': t_stat,
            'p_value': p_value,
        }
        
        return results
```

---

## 4. Training Protocol

### 4.1 Data Splits

```python
SPLITS_V10_9 = {
    # Core transfer tests (from v10.8)
    'hebrew_to_others': {
        'train_langs': ['hebrew', 'aramaic'],
        'test_langs': ['classical_chinese', 'arabic'],
        'description': 'Semitic → non-Semitic transfer',
    },
    
    'semitic_to_non_semitic': {
        'train_langs': ['hebrew', 'aramaic', 'arabic'],
        'test_langs': ['classical_chinese'],
        'description': 'All Semitic → Chinese transfer',
    },
    
    'ancient_to_modern': {
        'train_periods': ['BIBLICAL', 'TANNAITIC', 'CONFUCIAN', 'DAOIST', 'QURANIC'],
        'test_periods': ['DEAR_ABBY', 'MODERN'],
        'description': 'Ancient texts → modern English',
    },
    
    'mixed_baseline': {
        'train_split': 0.8,
        'test_split': 0.2,
        'stratify_by': ['language', 'bond_type'],
        'description': 'Stratified random split (sanity check)',
    },
    
    # NEW: Chinese diversity test
    'confucian_to_buddhist': {
        'train_periods': ['CONFUCIAN', 'DAOIST'],
        'test_periods': ['BUDDHIST'],
        'train_langs': ['classical_chinese'],
        'test_langs': ['classical_chinese'],
        'description': 'Test if Chinese performance is tradition-specific',
    },
    
    'confucian_to_legalist': {
        'train_periods': ['CONFUCIAN'],
        'test_periods': ['LEGALIST', 'MOHIST'],
        'train_langs': ['classical_chinese'],
        'test_langs': ['classical_chinese'],
        'description': 'Virtue ethics → consequentialist/legalist',
    },
    
    # NEW: Sanskrit independence test
    'all_to_sanskrit': {
        'train_langs': ['hebrew', 'aramaic', 'classical_chinese', 'arabic', 'english'],
        'test_langs': ['sanskrit', 'pali'],
        'description': 'Ultimate transfer test: completely held-out language family',
    },
    
    'semitic_to_indic': {
        'train_langs': ['hebrew', 'aramaic', 'arabic'],
        'test_langs': ['sanskrit', 'pali'],
        'description': 'Semitic → Indo-Aryan transfer',
    },
    
    # NEW: Arabic improvement test
    'quran_to_fiqh': {
        'train_periods': ['QURANIC', 'HADITH'],
        'test_periods': ['FIQH', 'SUFI', 'FALSAFA'],
        'train_langs': ['arabic'],
        'test_langs': ['arabic'],
        'description': 'Religious → legal/philosophical Arabic',
    },
}
```

### 4.2 Training Hyperparameters

```python
TRAINING_CONFIG_V10_9 = {
    # Unchanged from v10.8
    'backbone': 'LaBSE',
    'z_dim': 64,
    'max_len': 128,
    'lr': 2e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'max_epochs': 10,
    'patience': 3,
    
    # Adjusted for larger corpus
    'batch_size': {
        'L4/A100': 256,
        'T4': 128,
        '2xT4': 256,
        'SMALL': 64,
        'MINIMAL/CPU': 32,
    },
    
    # Adversarial schedule (unchanged)
    'adv_lambda_schedule': 'linear',  # 0 → 1 over first 2 epochs
    'adv_lambda_max': 1.0,
    
    # Loss weights
    'loss_weights': {
        'bond': 1.0,
        'hohfeld': 0.5,
        'language': 0.3,  # Adversarial
        'period': 0.2,    # Adversarial
        'context': 0.1,   # Auxiliary
    },
    
    # NEW: Minimum samples per category
    'min_samples_per_lang': 100,
    'min_samples_per_period': 50,
    'min_test_samples': 500,
}
```

### 4.3 Evaluation Protocol

```python
def evaluate_split_v10_9(model, test_loader, split_name):
    """
    Comprehensive evaluation with geometric analysis.
    """
    results = {}
    
    # Standard metrics (from v10.8)
    bond_preds, bond_labels = [], []
    language_preds, language_labels = [], []
    all_z = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['input_ids'], batch['attention_mask'])
            
            bond_preds.extend(outputs['bond_pred'].argmax(dim=1).cpu().tolist())
            bond_labels.extend(batch['bond_label'].cpu().tolist())
            language_preds.extend(outputs['language_pred'].argmax(dim=1).cpu().tolist())
            language_labels.extend(batch['language_label'].cpu().tolist())
            all_z.append(outputs['z'].cpu().numpy())
    
    # Classification metrics
    results['bond_f1_macro'] = f1_score(bond_labels, bond_preds, average='macro')
    results['bond_accuracy'] = accuracy_score(bond_labels, bond_preds)
    results['language_accuracy'] = accuracy_score(language_labels, language_preds)
    
    # Per-language breakdown
    results['per_language_f1'] = {}
    for lang_idx in set(language_labels):
        mask = [l == lang_idx for l in language_labels]
        if sum(mask) > 0:
            lang_name = IDX_TO_LANG[lang_idx]
            lang_f1 = f1_score(
                [b for b, m in zip(bond_labels, mask) if m],
                [p for p, m in zip(bond_preds, mask) if m],
                average='macro'
            )
            results['per_language_f1'][lang_name] = {
                'f1': lang_f1,
                'n': sum(mask),
            }
    
    # NEW: Geometric metrics
    Z = np.vstack(all_z)
    
    # Clustering quality
    from sklearn.metrics import silhouette_score
    if len(set(bond_labels)) > 1:
        results['bond_silhouette'] = silhouette_score(Z, bond_labels)
    
    # Language separability (lower is better for invariance)
    if len(set(language_labels)) > 1:
        results['language_silhouette'] = silhouette_score(Z, language_labels)
    
    return results
```

---

## 5. Geometric Discovery Protocol

### 5.1 Axis Discovery

Run after training completes successfully:

```python
def discover_moral_axes(model, tokenizer, device):
    """
    Attempt to find interpretable axes in the z-space.
    """
    analyzer = GeometricAnalyzer(model, tokenizer, device)
    
    # 1. Obligation/Permission axis
    obligation_permission_pairs = [
        # English
        ("You must repay the debt", "You may repay the debt"),
        ("He is obligated to help", "He is permitted to help"),
        ("They are required to attend", "They are allowed to attend"),
        # Chinese
        ("必須守信", "可以守信"),
        ("當報恩", "可報恩"),
        # Arabic
        ("يجب أن يساعد", "يجوز أن يساعد"),
        # Hebrew
        ("חייב לשלם", "מותר לשלם"),
    ]
    
    obl_texts = [p[0] for p in obligation_permission_pairs]
    perm_texts = [p[1] for p in obligation_permission_pairs]
    
    obl_perm_axis = analyzer.find_direction(obl_texts, perm_texts)
    
    # Test generalization
    test_pairs = [
        ("Parents must feed children", "Parents may feed children"),
        ("君子必仁", "君子可仁"),
        ("يجب الصدق", "يجوز الصدق"),
    ]
    obl_perm_transfer = analyzer.test_direction_transfer(obl_perm_axis, test_pairs)
    
    # 2. Harm/Care axis
    harm_care_pairs = [
        ("He hurt the child", "He helped the child"),
        ("傷人", "助人"),
        ("ظلم", "رحم"),
    ]
    
    harm_texts = [p[0] for p in harm_care_pairs]
    care_texts = [p[1] for p in harm_care_pairs]
    
    harm_care_axis = analyzer.find_direction(harm_texts, care_texts)
    
    # 3. Agent/Patient swap
    role_pairs = [
        ("John helped Mary", "Mary helped John"),
        ("A借B錢", "B借A錢"),
        ("الأب ضرب الابن", "الابن ضرب الأب"),
    ]
    
    role_analysis = analyzer.role_swap_analysis(role_pairs)
    
    # 4. PCA on all structural pairs
    all_pairs = {
        'obligation_permission': obligation_permission_pairs,
        'harm_care': harm_care_pairs,
    }
    
    pca_results = analyzer.pca_on_pairs(all_pairs)
    
    return {
        'obligation_permission': {
            'axis': obl_perm_axis,
            'transfer_accuracy': obl_perm_transfer,
        },
        'harm_care': {
            'axis': harm_care_axis,
        },
        'role_swap': role_analysis,
        'pca': {
            'explained_variance': pca_results['explained_variance_ratio'],
            'n_components_90pct': np.argmax(np.cumsum(pca_results['explained_variance_ratio']) > 0.9) + 1,
        },
    }
```

### 5.2 Interpretation Criteria

| Finding | Interpretation |
|---------|----------------|
| `obl_perm_transfer > 0.8` | Strong deontic axis discovered |
| `role_swap.consistency > 0.9` | Consistent agent/patient transformation |
| `pca.n_components_90pct <= 3` | Low-dimensional moral structure |
| `language_silhouette < 0.1` | Language successfully erased |
| `bond_silhouette > 0.3` | Bond types form distinct clusters |

---

## 6. Implementation Checklist

### 6.1 Corpus Expansion

- [ ] Add Buddhist Chinese passages (50+)
- [ ] Add Legalist Chinese passages (40+)
- [ ] Add Mohist Chinese passages (30+)
- [ ] Add Neo-Confucian Chinese passages (30+)
- [ ] Add Islamic Legal Maxims (50+)
- [ ] Add Sufi Ethics passages (30+)
- [ ] Add Arabic Philosophy passages (20+)
- [ ] Add Sanskrit/Dharmashastra passages (30+)
- [ ] Add Pali Canon passages (20+)
- [ ] Update PERIOD_TO_IDX (26 periods)
- [ ] Update LANG_TO_IDX (8 languages)
- [ ] Update bond patterns for new languages

### 6.2 Architecture Updates

- [ ] Update language_head output size (5 → 8)
- [ ] Update period_head output size (14 → 26)
- [ ] Add GeometricAnalyzer class
- [ ] Add StructuralFuzzTest class

### 6.3 Training Updates

- [ ] Add new data splits (confucian_to_buddhist, all_to_sanskrit, etc.)
- [ ] Update minimum sample thresholds
- [ ] Add geometric metrics to evaluation

### 6.4 Analysis Pipeline

- [ ] Implement axis discovery protocol
- [ ] Implement cross-lingual fuzz test
- [ ] Add visualization (t-SNE/UMAP colored by bond type, language)
- [ ] Generate interpretation report

---

## 7. Expected Outcomes

### 7.1 If BIP Hypothesis Holds

1. **Chinese diversity test passes**: Confucian→Buddhist and Confucian→Legalist maintain F1 ≥ 0.5
2. **Sanskrit transfer works**: all_to_sanskrit achieves F1 ≥ 0.4 despite held-out language family
3. **Arabic improves**: quran_to_fiqh achieves F1 ≥ 0.5 (up from 0.36)
4. **Geometric structure found**: ≤3 principal components explain 90% of moral-relevant variance
5. **Interpretable axes**: obligation/permission axis transfers with >80% accuracy

### 7.2 If BIP Hypothesis Fails

1. Chinese performance was corpus artifact (Confucian→Buddhist drops below 0.3)
2. Sanskrit transfer fails (F1 < 0.2) → language-family-specific structure
3. No consistent geometric structure → moral concepts are language-embedded

### 7.3 Publication Path

**Positive results**: Submit to ACL/EMNLP as "Cross-Linguistic Moral Concept Transfer Without Translation Bridges"

**Mixed results**: Workshop paper on "Limits of Universal Moral Structure: Evidence from Multilingual NLP"

---

## 8. Timeline

| Week | Milestone |
|------|-----------|
| 1 | Corpus expansion complete, patterns updated |
| 2 | Architecture updates, new splits implemented |
| 3 | Training runs (all splits) |
| 4 | Geometric analysis, fuzz testing |
| 5 | Interpretation, visualization, writeup |

**Total GPU time estimate**: ~40 hours (L4/A100)

---

## Appendix A: Data Sources

| Corpus | Source | License |
|--------|--------|---------|
| Sefaria | github.com/Sefaria/Sefaria-Export | CC-BY-NC |
| Wenyanwen | Kaggle (Ancient Chinese Text) | CC0 |
| Quran | Tanzil.net | Open |
| Hadith | Various (Bukhari, Muslim) | Public Domain |
| Dear Abby | Kaggle | Research |
| ETHICS | HuggingFace (hendrycks/ethics) | MIT |
| Buddhist Chinese | CBETA | Academic |
| Sanskrit | GRETIL, Wisdom Library | Academic |
| Pali | Tipitaka.org, SuttaCentral | CC0 |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| BIP | Bond Invariance Principle — hypothesis that moral concepts have universal mathematical structure |
| Bond | A moral/normative relationship between agents (obligation, permission, care, harm, etc.) |
| Hohfeld State | Legal/deontic classification (right, duty, privilege, no-right) |
| z-space | The 64-dimensional latent space learned by the model |
| Adversarial invariance | Training the model to NOT encode language/period while encoding bond type |
| Gradient reversal | Technique for adversarial training — reverses gradients on nuisance predictors |

---

*End of Specification*
